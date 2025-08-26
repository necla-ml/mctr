import os
import json
import random
import copy 

# third party
import torch
import torch.optim
from torch import nn
import torch.utils.data
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from trainer import *
from data.mmptracking import *
from data.mtmc_nvidia import *
from utils.io import save_checkpoint, logdir, write_scores, write_stats, write_cfg
from utils.misc import init_distributed_mode

#from icecream import install
#install()

best_loss = math.inf

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def collate_fn(batch):
    batch_clips, batch_targets, batch_new_segment, batch_frames, batch_metadata = [], [], [], [], []
    # use deepcopy so the shared memory can be closed. Otherwise there are files left out in /dev/shm
    i = 0
    for clips, targets, new_segment, frames, metadata in batch:
        batch_clips.append(clips)
        batch_targets.append(copy.deepcopy(targets))
        batch_new_segment.append(copy.deepcopy(new_segment))
        batch_frames.append(copy.deepcopy(frames))
        batch_metadata.append(copy.deepcopy(metadata))
        i = i + 1

    batch_clips = torch.stack(batch_clips) # B, F, Cm, T, C, H, W
    assert batch_clips.shape[3] == 1
    batch_clips = batch_clips.squeeze(dim=3)  # B, F, Cm, C, H, W
    batch_frames = torch.tensor(batch_frames, dtype=torch.long)

    return batch_clips, batch_targets, batch_new_segment, batch_frames, batch_metadata


def _build_dataset(cfg, build_train = False):
    cfg_dataset = cfg.DATASET
    if "MMPTracking" in cfg_dataset.ROOT:
        dataset = MMPTrackClipsFull
    elif  "MTMC_Tracking_AIC23_Track1" in cfg_dataset.ROOT:
        dataset = MTMCClipsFull
    else:
        raise ValueError(f'Invalid dataset {cfg_dataset.ROOT}')
    
    val_dataset = dataset(cfg_dataset, split='validation')
    train_dataset = None
    if cfg.CMD == 'train' or build_train:
        train_dataset = dataset(cfg_dataset, split='train')
    return val_dataset, train_dataset


def _process_data(data, dev):
    clips, targets, new_segment, frames, metadata = data

    # TODO: multi frame: use single frame for now
    clips = clips.to(dev, non_blocking=True) # B, F, Cm, C, H, W
    targets = [[[{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in tv] for tv in tf] for tf in targets]

    new_segment = torch.tensor(new_segment, device=dev)
    for tf in targets:
        for tv in tf:
            for t in tv: 
                t.update({'labels': torch.zeros((len(t['boxes']),), dtype=torch.int64, device=dev)})

    return clips, targets, new_segment, frames, metadata


def _build_model(cfg, ncams):
    if cfg.TRAINING.MODEL == 'PAIRWISE':

        from models.detr.pairwise_model import build_pairwise
        from models.detr.pairwise_losses import build_criterion
        from models.detr.pairwise_postprocessor import build_postprocessor
        # from models.detr.pairwise_postprocessor_with_tiebreak import build_postprocessor
        from models.detr.postprocess_loss import build_postprocess_loss

        from models.detr.config import CONFIGS

        model_cfg = cfg.MODELS[cfg.TRAINING.MODEL]

        args = CONFIGS[model_cfg.BACKBONE]
        args.num_classes = model_cfg.NUM_CLASSES
        args.device = 'cpu'
        args.num_queries = model_cfg.NUM_QUERIES
        args.pw_queries = model_cfg.PW_QUERIES
        args.pw_layers = model_cfg.PW_LAYERS
        args.nviews = ncams
        args.pw_passthrough_src = model_cfg.PW_PASSTHROUGH_SRC
        if model_cfg.PW_PROJ_DIM_SCALE:
            args.proj_dim_scale = model_cfg.PW_PROJ_DIM_SCALE
        else:
            args.proj_dim_scale = 1
        args.no_proj = model_cfg.PW_NO_PROJ
        args.detached_until = model_cfg.DETACHED_UNTIL
        args.add_bboxes = model_cfg.ADD_BBOXES
        args.pairwise_loss = model_cfg.PAIRWISE_LOSS
        args.pred_track_boxes = model_cfg.PRED_TRACK_BOXES
        args.embedding_loss = model_cfg.EMBEDDING_LOSS
        args.pred_track_boxes_time = model_cfg.PRED_TRACK_BOXES_TIME
        args.pred_topdown = model_cfg.PRED_TOPDOWN or False
        args.scaled_costs = model_cfg.SCALED_COSTS
        args.tgt_as_pos = model_cfg.TGT_AS_POS
        args.zero_qe = model_cfg.ZERO_QE or False
        args.weighted_views = model_cfg.WEIGHTED_VIEWS or False
        args.self_attention_before = True if model_cfg.SELF_ATTENTION_BEFORE is None else model_cfg.SELF_ATTENTION_BEFORE
        args.self_attention_after = model_cfg.SELF_ATTENTION_AFTER or False
        args.context_size = model_cfg.CONTEXT_SIZE or 1

        checkpoint_pth = model_cfg.CHECKPOINT

        model = build_pairwise(args)
        criterion = build_criterion(args)
        post_processor = build_postprocessor(args)
        postprocessors = {"tracks":post_processor}
        postprocess_loss = build_postprocess_loss(args)
        postprocess_losses = {"tracks": postprocess_loss}

        assert Path(checkpoint_pth).exists(), f'{checkpoint_pth=} does not exist'
        chkpt = torch.load(checkpoint_pth, map_location='cpu')
        state_dict = chkpt['model']
        for k, v in state_dict.items():
            if k == 'class_embed.weight':
                state_dict[k] = v[[0, -1]]
            elif k == 'class_embed.bias':
                state_dict[k] = v[[0, -1]]
            elif k == 'query_embed.weight':
                state_dict[k] = v[:args.num_queries]

        res = model.detr.load_state_dict(state_dict, strict=False)
        print(f'Loaded {checkpoint_pth=} with {res=}')

        if model_cfg.FREEZE_DETR:
            for p in model.detr.parameters():
                p.requires_grad = False
        elif model_cfg.FREEZE_DETR_BACKBONE:
            for p in model.detr.backbone.parameters():
                p.requires_grad = False
   
    else:
        raise ValueError(f'Invalid value for {cfg.TRAINING.MODEL=}')
    
    return model, criterion, postprocessors, postprocess_losses


def _build_optmizer_scheduler(cfg, model, train_loader):
    training = cfg.TRAINING
    # XXX: different learning rate for backbone and rest of the model
    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "detr" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "detr" in n and p.requires_grad], "lr": (training.LR_DETR_MULT or 1)*training.LR}
    ]
    #grouped_parameters = model.parameters()
    if training.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(grouped_parameters, training.LR,
                                    momentum=training.MOMENTUM,
                                    weight_decay=training.WEIGHT_DECAY,
                                    nesterov=training.NESTEROV == 1)
    elif training.OPTIMIZER in 'adam':
        optimizer = torch.optim.Adam(grouped_parameters,
                                     lr=training.LR,
                                     weight_decay=training.WEIGHT_DECAY,
                                     eps=training.ADAM_EPS)
    elif training.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(grouped_parameters,
                                      lr=training.LR,
                                      weight_decay=training.WEIGHT_DECAY,
                                      eps=training.ADAM_EPS,
                                      )
    else:
        raise ValueError(f'optimizer {training.OPTIMIZER} not found.')

    scheduler = training.SCHEDULER
    if scheduler is None:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=training.LR_DECAY_RATIO, patience=training.PATIENCE, min_lr=1e-7)
    else:
        num_training_steps = len(train_loader) * training.MAX_NUM_EPOCHS
        num_warmup_steps = 2*len(train_loader) #0.1 * num_training_steps
        print(f"NUM_WARMUP_STEPS={num_warmup_steps}, NUM_TRAINING_STEP={num_training_steps}")
        if scheduler == 'cosine':
            # from utils.optim import CosineWarmupScheduler
            # from torch.optim.lr_scheduler import CosineAnnealingLR
            # scheduler = CosineWarmupScheduler(optimizer, num_warmup_steps, num_training_steps)
            # scheduler = CosineAnnealingLR(optimizer, T_max=training.max_num_epochs, eta_min=1e-7)
            from utils.optim import WarmUpCosineAnnealingLR
            scheduler = WarmUpCosineAnnealingLR(
                optimizer,
                warm_multiplier=training.WARMUP_MULTIPLIER or 100,
                warm_duration=num_warmup_steps,
                cos_duration=num_training_steps - num_warmup_steps,
                eta_min=1e-7
            )
        elif scheduler == 'linear':
            from utils.optim import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_training_steps)
        elif scheduler == 'step':
            scheduler = StepLR(optimizer, step_size = training.STEP_SIZE*len(train_loader), gamma = training.LR_DECAY_RATIO)
        else:
            raise ValueError(f'Scheduler: {scheduler} is not available')

    return optimizer, scheduler


def _build_trainer(cfg, model, criterion, postprocessor, postprocess_loss, optimizer, scheduler, scaler, train_loader, val_loader, process_data):
    return TrainerPairwise(cfg, model, criterion, postprocessor, postprocess_loss, optimizer, scheduler, scaler, train_loader, val_loader, process_data)


def main(cfg):

    init_distributed_mode(cfg)
    print(f"Working with cfg: \n{cfg}")

    cfg_training = cfg.TRAINING
    
    global best_loss
    if cfg.SEED:
        random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
    torch.backends.cudnn.benchmark = cfg_training.CUDNN_BENCHMARK

    # XXX: create dataset
    val_dataset, train_dataset = _build_dataset(cfg)


    # XXX: create model
    model, criterion, postprocessors, postprocess_losses = _build_model(cfg, val_dataset.ncams)
    print(model)

    assert val_dataset.ncams == model.num_views, "Number of cameras in the data and the model does not match"

    train_loader = optimizer = scheduler = None

    # XXX: setup dist
    if cfg.DIST:
        if not cfg.DIST_NO_SYNC_BN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        cfg.DEV = torch.cuda.default_stream().device
        if cfg.GPU is not None:
            cfg_training.BATCH_SIZE //= cfg.WORLD_SIZE
            print(f"Reset batch size to {cfg_training.BATCH_SIZE} in a world of size {cfg.WORLD_SIZE}")
            model = model.to(cfg.DEV)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.GPU], find_unused_parameters=cfg_training.FIND_UNUSED_PARAM)
        else:
            model = model.to(cfg.dev)
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=cfg_training.FIND_UNUSED_PARAM)
    else:
        if cfg.GPU:
            cfg.DEV = torch.cuda.default_stream().device
            model = model.to(cfg.DEV)
            if len(cfg.GPU) > 1:
                model = nn.DataParallel(model)
        else:
            raise NotImplementedError('Training on CPU will be utterly slow, go buy a GPU!')

    criterion = criterion.to(cfg.DEV)
    criterion = {'train': criterion, 'val': criterion}

    # XXX: setup dataset samplers
    if cfg_training.LOADER_NUM_WORKERS is None:
        loader_num_workers = max((torch.get_num_threads() // cfg.WORLD_SIZE if cfg.DIST else torch.get_num_threads()) - 1, 1)
    else:
        loader_num_workers = cfg_training.LOADER_NUM_WORKERS

    # XXX: init dataloader, optimizer and scheduler
    if not cfg.MODELS.PAIRWISE.RESET_TGT:
        assert cfg.DATASET.SUBSAMPLE_RATE == cfg.DATASET.FRAMES_PER_SAMPLE *  cfg.DATASET.FRAME_STRIDE
        sampler_val = MMPTrackClipsFullBatchSampler(cfg_training.BATCH_SIZE, 1, val_dataset.__len__(), cfg_training.ITS_PER_EPOCH)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=loader_num_workers,
            pin_memory=cfg_training.PIN_MEMORY,
            batch_sampler=sampler_val,
            worker_init_fn=set_worker_sharing_strategy,
            collate_fn=collate_fn)
    else:
        if cfg.DIST:
            sampler_val = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(val_dataset)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg_training.BATCH_SIZE,
            num_workers=loader_num_workers,
            pin_memory=cfg_training.PIN_MEMORY,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=collate_fn,
            worker_init_fn=set_worker_sharing_strategy,
            shuffle=False)

    if cfg.CMD == 'train':
        sampler_train = None
        shuffle = cfg.MODELS.PAIRWISE.RESET_TGT
        cfg.MODELS.PAIRWISE.CLIP_LENGTH_PROGRESSION = cfg.MODELS.PAIRWISE.CLIP_LENGTH_PROGRESSION or 'linear'
        if not cfg.MODELS.PAIRWISE.RESET_TGT:
            assert cfg.DATASET.SUBSAMPLE_RATE == cfg.DATASET.FRAMES_PER_SAMPLE *  cfg.DATASET.FRAME_STRIDE
            sampler_train = MMPTrackClipsFullBatchSampler(cfg_training.BATCH_SIZE, 1, train_dataset.__len__(), cfg_training.ITS_PER_EPOCH)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=sampler_train,
                num_workers=loader_num_workers,
                pin_memory=cfg_training.PIN_MEMORY,
                worker_init_fn=set_worker_sharing_strategy,
                collate_fn=collate_fn)
        else:
            if cfg.DIST:
                sampler_train = torch.utils.data.DistributedSampler(train_dataset, shuffle=cfg.MODELS.PAIRWISE.RESET_TGT)
                shuffle = None
                #sampler_train = torch.utils.data.DistributedSampler(train_dataset, shuffle=False)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=cfg_training.BATCH_SIZE,
                sampler=sampler_train,
                num_workers=loader_num_workers,
                pin_memory=cfg_training.PIN_MEMORY,
                worker_init_fn=set_worker_sharing_strategy,
                collate_fn=collate_fn,
                shuffle=shuffle)
    #            shuffle=False)


        optimizer, scheduler = _build_optmizer_scheduler(cfg, model, train_loader)

        print(f"OPTIMIZER={optimizer}, SCHEDULER={scheduler}")

    # XXX: init scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg_training.AMP)

    # loading a model if provided. This is not resuming from that model,just initializes parameters
    print("INITIALIZE", cfg.INITIALIZE)
    if cfg.INITIALIZE:
        if os.path.isfile(cfg.INITIALIZE):
            print(f"Loading checkpoint '{cfg.INITIALIZE}'")
            if cfg.DEV is None:
                checkpoint = torch.load(cfg.INITIALIZE, map_location='cpu')
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(cfg.INITIALIZE, map_location=cfg.DEV)

            # check that the cfg matches wit hteh initialization cfg
            init_cfg = checkpoint["cfg"]
            #assert init_cfg.DATASET.LOCATIONS == cfg.DATASET.LOCATIONS, f"Locations do not match. Init: {init_cfg.DATASET.LOCATIONS}, cfg: {cfg.DATASET.LOCATIONS}"
            assert init_cfg.DATASET.CAMS == cfg.DATASET.CAMS, f"Cameras do not match. Init: {init_cfg.DATASET.CAMS}, cfg: {cfg.DATASET.CAMS}"
            assert init_cfg.MODELS.PAIRWISE.NUM_CLASSES == cfg.MODELS.PAIRWISE.NUM_CLASSES, f"NUM_CLASSES do not match. Init: {init_cfg.MODELS.PAIRWISE.NUM_CLASSES}, cfg: {cfg.MODELS.PAIRWISE.NUM_CLASSES}"
            assert init_cfg.MODELS.PAIRWISE.NUM_QUERIES == cfg.MODELS.PAIRWISE.NUM_QUERIES, f"NUM_QUERIES do not match. Init: {init_cfg.MODELS.PAIRWISE.NUM_QUERIES}, cfg: {cfg.MODELS.PAIRWISE.NUM_QUERIES}"
            assert init_cfg.MODELS.PAIRWISE.PW_QUERIES == cfg.MODELS.PAIRWISE.PW_QUERIES, f"PW_QUERIES do not match. Init: {init_cfg.MODELS.PAIRWISE.PW_QUERIES}, cfg: {cfg.MODELS.PAIRWISE.PW_QUERIES}"

            state_dict = checkpoint['state_dict']
            consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')
            if cfg.DIST:
                model.module.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint '{cfg.INITIALIZE}'")

            # put here checks on same cameras. same locations

        else:
            print(f"No checkpoint found at '{cfg.INITIALIZE}'")
            exit()

    
    # XXX: load checkpoint if --resume
    if cfg.RESUME:
        if os.path.isfile(cfg.RESUME):
            print(f"Loading checkpoint '{cfg.RESUME}'")
            if cfg.DEV is None:
                checkpoint = torch.load(cfg.RESUME, map_location='cpu')
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(cfg.RESUME, map_location=cfg.DEV)

            init_cfg = checkpoint["cfg"]
            # assert init_cfg.DATASET.LOCATIONS == cfg.DATASET.LOCATIONS, f"Locations do not match. Init: {init_cfg.DATASET.LOCATIONS}, cfg: {cfg.DATASET.LOCATIONS}"
            assert init_cfg.DATASET.CAMS == cfg.DATASET.CAMS, f"Cameras do not match. Init: {init_cfg.DATASET.CAMS}, cfg: {cfg.DATASET.CAMS}"
            # assert init_cfg.MODELS.PAIRWISE == cfg.MODELS.PAIRWISE, f"Model params do not match. Init: {init_cfg.MODELS.PAIRWISE.NUM_CLASSES}, cfg: {cfg.MODELS.PAIRWISE.NUM_CLASSES}"
    
            if 'loss' in checkpoint:
                best_loss = checkpoint['loss']
            if 'epoch' in checkpoint:
                cfg_training.START_EPOCH = checkpoint['epoch'] + 1
            state_dict = checkpoint['state_dict']
            consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')
            if cfg.DIST:
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            if cfg.CMD == 'train':
                for key in ['optimizer', 'scheduler', 'scaler']:
                    if key in checkpoint:
                        eval(key).load_state_dict(checkpoint[key])
                        print(f'Loaded {key} from checkpoint')
                    else:
                        print(f'Skipping {key} not found in checkpoint')
            print(f"Loaded checkpoint '{cfg.RESUME}' (epoch {checkpoint['epoch']}) (best_loss {best_loss})")

            #put here checks on same camera same location same params (Except max epochs)

        else:
            print(f"No checkpoint found at '{cfg.RESUME}")
            exit()

    # XXX: init tensorboard
    out_dir = f'frz_dtr{cfg.MODELS.PAIRWISE.FREEZE_DETR}-res_unused{cfg.MODELS.PAIRWISE.RESET_UNUSED}-res_tgt{cfg.MODELS.PAIRWISE.RESET_TGT}-clp{cfg.MODELS.PAIRWISE.CLIP_LENGTH_PROGRESSION}-{cfg_training.COMMENT}'
    # get a more informative log_dir 
    tensorboard = None if cfg.NOTENSORBOARD or (cfg.DIST and cfg.RANK != 0) else SummaryWriter(log_dir=logdir(logdir=cfg.LOGDIR, log_prefix=cfg.LOG_PREFIX, comment=out_dir))
    cfgS = str(cfg)
    cfgS = cfgS.replace("\n","  \n")
    cfgS = cfgS.replace("    ","&nbsp;")
    if tensorboard:
        tensorboard.add_text('config', cfgS)
        print(f'Tensorboard logging to {tensorboard.log_dir}')
        print('Start Tensorboard with "tensorboard --logdir=runs --bind_all --reload_multifile True"')
        write_cfg(tensorboard.log_dir, cfg)
    out_dir = tensorboard and tensorboard.log_dir or out_dir        

    # XXX: init trainer
    trainer = _build_trainer(cfg, model, criterion, postprocessors, postprocess_losses, optimizer, scheduler, scaler, train_loader, val_loader, _process_data)

    if cfg.CMD == 'eval':
        loss = trainer.eval(cfg_training.START_EPOCH, tensorboard)
        print(f"Best Loss: {loss:.3f}")
        if not cfg.DIST or cfg.RANK == 0:
            write_stats({'epoch': 0, 'loss': loss}, out_dir)
        return


    # XXX: start training
    for epoch in range(cfg_training.START_EPOCH, cfg_training.MAX_NUM_EPOCHS):
        if not cfg.MODELS.PAIRWISE.RESET_TGT:
            if cfg.MODELS.PAIRWISE.CLIP_LENGTH_PROGRESSION == 'linear':
                sampler_train.set_p_reset(1/(2*epoch+1))
                sampler_val.set_p_reset(1/(2*epoch+1))
            elif cfg.MODELS.PAIRWISE.CLIP_LENGTH_PROGRESSION == 'quadratic':
                sampler_train.set_p_reset(1/(epoch*epoch+1))
                sampler_val.set_p_reset(1/(epoch*epoch+1))
            elif cfg.MODELS.PAIRWISE.CLIP_LENGTH_PROGRESSION == 'constant':
                sampler_train.set_p_reset(1)
                sampler_val.set_p_reset(1)
            else:
                raise ValueError(f'CLIP_LENGTH_PROGRESSION: {cfg.MODELS.PAIRWISE.CLIP_LENGTH_PROGRESSION} is not implemented')

        if cfg.DIST and cfg.MODELS.PAIRWISE.RESET_TGT:
            sampler_train.set_epoch(epoch)
        trainer.train(epoch, tensorboard) 
        if ((epoch+1)%5 == 0):       
            loss = trainer.eval(epoch, tensorboard)
        else:
            loss = 100


        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(loss)

        # XXX: save checkpoint
        if not cfg.DIST or cfg.RANK == 0:
            chkpt = {
                'cfg': cfg,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'loss': loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }
            name = f'{cfg_training.MODEL}'
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            save_checkpoint(chkpt, is_best, dir=out_dir, name=name)

            if (epoch + 1) % cfg.TRAINING.SAVE_EVERY == 0: 
                name = f'{name}.epoch{epoch}'
                save_checkpoint(chkpt, False, dir=out_dir, name=name)


            write_stats({'epoch': epoch, 'loss': loss}, out_dir)

    if tensorboard:
        tensorboard.close()

    print("Done training")


if __name__ == '__main__':
    from utils.args import parse_args
    cfg = parse_args()
    main(cfg)
