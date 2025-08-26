import math
import copy

# third party
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision.ops import sigmoid_focal_loss

from ml import plot

from utils.metrics import (
    accuracy_binary as accuracy,
    classification_report,
    SmoothedValue,
    MetricLogger
)
from utils.misc import all_gather, sync_tensor_list_dist, is_main_process, reduce_dict

class TrainerPairwise():
    def __init__(self, args, model, criterion, 
                postprocessor, postprocess_loss,
                optimizer, scheduler, scaler,
                train_loader,
                val_loader,
                process_data):
        self.args = args
        self.model = model
        self.scaler = scaler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.postprocess_loss = postprocess_loss
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.process_data = process_data

        # from utils.misc import nan_hook, inf_hook
        # for submodule in model.modules():
        #     submodule.register_forward_hook(nan_hook)
        #     submodule.register_forward_hook(inf_hook)

        # for submodule in criterion.modules():
        #     submodule.register_forward_hook(nan_hook)
        #     submodule.register_forward_hook(inf_hook)


    def train(self, epoch, tensorboard):

        criterion = self.criterion['train']

        # init metrics logger
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = self.args.TRAINING.PRINT_FREQ

        # switch to train mode
        self.model.train()
        criterion.train()

        dev = self.args.DEV
        track_unused_since = torch.zeros((int(self.args.TRAINING.BATCH_SIZE), int(self.args.MODELS.PAIRWISE.PW_QUERIES)), dtype=torch.int, device=dev)

        tgt = None
        prev_frames = None
        for iter_step, data in enumerate(metric_logger.log_every(self.train_loader, print_freq, header)): 
            if self.args.DRYRUN and iter_step == 1:
                break

            clips, targets, new_segment, frames, metadata = self.process_data(data, dev)

            self.optimizer.zero_grad()

            if (tgt is None) or self.args.MODELS.PAIRWISE.RESET_TGT or (epoch == 0 and iter_step < self.args.MODELS.PAIRWISE.RESET_TGT_UNTIL):
                tgt=None     
                reset_tgt = None      
            else: 
                # only do this when not resetting the target
                # othewiese the batches could be different creating errors
                if prev_frames is not None:
                    new_segment = frames[:,0] != (prev_frames + self.args.DATASET.FRAME_STRIDE) 
                else:
                    new_segment = torch.zeros(frames.shape[0], dtype=torch.bool)

                if self.args.MODELS.PAIRWISE.RESET_UNUSED > 0:                
                    reset_tgt = new_segment.unsqueeze(1).repeat(1,self.args.MODELS.PAIRWISE.PW_QUERIES)
                    track_unused_since[reset_tgt] = 0
                    reset_tgt[track_unused_since >= self.args.MODELS.PAIRWISE.RESET_UNUSED] = True
                else:
                    reset_tgt = new_segment

            prev_frames = copy.deepcopy(frames[:,-1])

                # compute output
            with torch.cuda.amp.autocast(enabled=self.args.TRAINING.AMP):

                outputs, costs, tgt = self.model(clips, tgt, reset_tgt)
                # calculate loss
                loss_dict = criterion(outputs, costs, tgt, targets)
                # decode tracks and compute loss
                preds, _ , _= self.postprocessor["tracks"](outputs, costs, keep_prob=0.9)
                loss_dict_track = self.postprocess_loss["tracks"](preds, targets)

            if not self.args.MODELS.PAIRWISE.RESET_TGT and self.args.MODELS.PAIRWISE.RESET_UNUSED>0:
                # this does not make sense if the targets are reset anyway 
                # or if the batches are not of constant size.                 
                used_tracks = [torch.cat([predv["tracks"].long() for predf in predb for predv in predf]).unique() for predb in preds]
                track_unused_since += 1
                for b in range(track_unused_since.shape[0]):
                    track_unused_since[b,used_tracks[b]] = 0

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            loss_dict_track_reduced = reduce_dict(loss_dict_track)

            # backward   
            self.scaler.scale(losses).backward()
            if self.args.TRAINING.MAX_GRAD_NORM:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.TRAINING.MAX_GRAD_NORM, error_if_nonfinite=True)

            # forward
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            # metric calculation and logging
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training, {loss_dict}")
                assert math.isfinite(loss_value)

            lr = self.optimizer.param_groups[0]['lr']
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled, **loss_dict_track_reduced)
            metric_logger.update(lr=lr)
            if tensorboard is not None:
                current_step = (epoch - 1) * len(self.train_loader) + iter_step
                tensorboard.add_scalar('train/loss_train', loss_value, current_step)
                for loss in ["loss_pw_unscaled", "loss_pwc_unscaled","loss_ce_unscaled",
                            "loss_bbox_unscaled", "loss_giou_unscaled", 
                            "loss_track_bbox_unscaled", "loss_track_giou_prev_unscaled",
                            "loss_track_next_unscaled", "loss_track_giou_next_unscaled", "loss_emb"]:
                    if loss in loss_dict_reduced_unscaled:
                        tensorboard.add_scalar(f'eval/{loss}', loss_dict_reduced_unscaled[loss], current_step)
                for loss in ["precision", "recall", "f1", 
                            "track_pwc_accuracy", "correct_pwc_tracks",
                            "track_pwf_accuracy", "correct_pwf_tracks",
                            "full_track_accuracy"]:
                    if loss in loss_dict_track_reduced:
                        tensorboard.add_scalar(f'eval/{loss}', loss_dict_track_reduced[loss], current_step)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger, flush=True)


        if tensorboard is not None:
            loss_avg = metric_logger.loss.global_avg
            tensorboard.add_scalar('train_epoch/loss_avg_eval', loss_avg, epoch)

            for loss in ["loss_pw_unscaled", "loss_pwc_unscaled","loss_ce_unscaled",
                         "loss_bbox_unscaled", "loss_giou_unscaled", 
                         "loss_track_bbox_unscaled", "loss_track_giou_unscaled",
                         "loss_track_bbox_prev_unscaled", "loss_track_giou_prev_unscaled",
                         "loss_track_bbox_next_unscaled", "loss_track_giou_next_unscaled",
                         "loss_emb"]:
                if loss in loss_dict_reduced_unscaled:
                    tensorboard.add_scalar(f'train_epoch/{loss}_avg_eval', metric_logger[loss].global_avg, epoch)
            for loss in ["precision", "recall", "f1", 
                         "track_pwc_accuracy", "correct_pwc_tracks",
                         "track_pwf_accuracy", "correct_pwf_tracks",
                         "full_track_accuracy"]:
                if loss in loss_dict_track_reduced:
                    tensorboard.add_scalar(f'train_epoch/{loss}_avg_eval', metric_logger[loss].global_avg, epoch)


    @torch.no_grad()
    def eval(self, epoch, tensorboard):

        criterion = self.criterion['val']

        # init metrics logger
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Val:'

        # switch to eval mode
        self.model.eval()
        criterion.eval()

        dev = self.args.DEV
        track_unused_since = torch.zeros((int(self.args.TRAINING.BATCH_SIZE), int(self.args.MODELS.PAIRWISE.PW_QUERIES)), dtype=torch.int, device=dev)

        tgt = None
        prev_frames = None
        for iter_step, data in enumerate(metric_logger.log_every(self.val_loader, 100, header)):

            if self.args.DRYRUN and iter_step == 1:
                break

            clips, targets, new_segment, frames, metadata = self.process_data(data, dev)

            if (tgt is None) or self.args.MODELS.PAIRWISE.RESET_TGT or (epoch == 0 and iter_step < self.args.MODELS.PAIRWISE.RESET_TGT_UNTIL):
                tgt=None
                reset_tgt=None
            else:
                # only do this when not resetting the target
                # othewiese the batches could be different creating errors
                new_segment = frames[:,0] != (prev_frames + self.args.DATASET.FRAME_STRIDE) 


                if new_segment is not None and self.args.MODELS.PAIRWISE.RESET_UNUSED > 0:                
                    reset_tgt = new_segment.unsqueeze(1).repeat(1,self.args.MODELS.PAIRWISE.PW_QUERIES)
                    track_unused_since[reset_tgt] = 0
                    reset_tgt[track_unused_since >= self.args.MODELS.PAIRWISE.RESET_UNUSED] = True
                else:
                    reset_tgt = new_segment

            prev_frames = frames[:,-1]  

            # compute output
            with torch.cuda.amp.autocast(enabled=self.args.TRAINING.AMP):
                outputs, costs, tgt = self.model(clips, tgt, reset_tgt)
                # calculate loss
                loss_dict = criterion(outputs, costs, tgt, targets)
                # decode tracks and compute loss
                preds, _ , _= self.postprocessor["tracks"](outputs, costs, keep_prob=0.9)
                loss_dict_track = self.postprocess_loss["tracks"](preds, targets)

            if not self.args.MODELS.PAIRWISE.RESET_TGT and self.args.MODELS.PAIRWISE.RESET_UNUSED>0:
                # this does not make sense if the targets are reset anyway 
                # or if the batches are not of constant size.  
                used_tracks = [torch.cat([predv["tracks"].long() for predf in predb for predv in predf]).unique() for predb in preds]
                track_unused_since += 1
                for b in range(track_unused_since.shape[0]):
                    track_unused_since[b,used_tracks[b]] = 0

            weight_dict = criterion.weight_dict
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()

            loss_dict_track_reduced = reduce_dict(loss_dict_track)

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled, **loss_dict_track_reduced)

            if tensorboard is not None:
                current_step = (epoch - 1) * len(self.val_loader) + iter_step
                tensorboard.add_scalar('eval/loss_eval', loss_value, current_step)
                for loss in ["loss_pw_unscaled", "loss_pwc_unscaled","loss_ce_unscaled",
                             "loss_bbox_unscaled", "loss_giou_unscaled", 
                             "loss_track_bbox_unscaled", "loss_track_giou_unscaled",
                             "loss_track_bbox_prev_unscaled", "loss_track_giou_prev_unscaled",
                             "loss_track_bbox_next_unscaled", "loss_track_giou_next_unscaled"]:
                    if loss in loss_dict_reduced_unscaled:
                        tensorboard.add_scalar(f'eval/{loss}', loss_dict_reduced_unscaled[loss], current_step)
                for loss in ["precision", "recall", "f1", 
                             "track_pwc_accuracy", "correct_pwc_tracks",
                             "track_pwf_accuracy", "correct_pwf_tracks",
                             "full_track_accuracy"]:
                    if loss in loss_dict_track_reduced:
                        tensorboard.add_scalar(f'eval/{loss}', loss_dict_track_reduced[loss], current_step)
                        

                # tensorboard.add_scalar('eval/loss_pw_unscaled', loss_dict_reduced_unscaled["loss_pw_unscaled"], current_step)
                # tensorboard.add_scalar('eval/loss_pwc_unscaled', loss_dict_reduced_unscaled["loss_pwc_unscaled"], current_step)
                # tensorboard.add_scalar('eval/loss_pwf_unscaled', loss_dict_reduced_unscaled["loss_pwf_unscaled"], current_step)
                # tensorboard.add_scalar('eval/loss_ce_unscaled', loss_dict_reduced_unscaled["loss_ce_unscaled"], current_step)
                # tensorboard.add_scalar('eval/loss_bbox_unscaled', loss_dict_reduced_unscaled["loss_bbox_unscaled"], current_step)
                # tensorboard.add_scalar('eval/loss_giou_unscaled', loss_dict_reduced_unscaled["loss_giou_unscaled"], current_step)
                # tensorboard.add_scalar('eval/loss_track_bbox_unscaled', loss_dict_reduced_unscaled["loss_track_bbox_unscaled"], current_step)
                # tensorboard.add_scalar('eval/loss_track_giou_unscaled', loss_dict_reduced_unscaled["loss_track_giou_unscaled"], current_step)
                # tensorboard.add_scalar('eval/loss_track_bbox_prev_unscaled', loss_dict_reduced_unscaled["loss_track_bbox_prev_unscaled"], current_step)
                # tensorboard.add_scalar('eval/loss_track_giou_prev_unscaled', loss_dict_reduced_unscaled["loss_track_giou_prev_unscaled"], current_step)
                # tensorboard.add_scalar('eval/loss_track_bbox_next_unscaled', loss_dict_reduced_unscaled["loss_track_bbox_next_unscaled"], current_step)
                # tensorboard.add_scalar('eval/loss_track_giou_next_unscaled', loss_dict_reduced_unscaled["loss_track_giou_next_unscaled"], current_step)
                # tensorboard.add_scalar('eval/precision', loss_dict_track_reduced["precision"], current_step)
                # tensorboard.add_scalar('eval/recall', loss_dict_track_reduced["recall"], current_step)
                # tensorboard.add_scalar('eval/f1', loss_dict_track_reduced["f1"], current_step)
                # tensorboard.add_scalar('eval/track_pwc_accuracy', loss_dict_track_reduced["track_pwc_accuracy"], current_step)
                # tensorboard.add_scalar('eval/correct_pwc_tracks', loss_dict_track_reduced["correct_pwc_tracks"], current_step)
                # tensorboard.add_scalar('eval/track_pwf_accuracy', loss_dict_track_reduced["track_pwf_accuracy"], current_step)
                # tensorboard.add_scalar('eval/correct_pwf_tracks', loss_dict_track_reduced["correct_pwf_tracks"], current_step)
                # tensorboard.add_scalar('eval/full_track_accuracy', loss_dict_track_reduced["full_track_accuracy"], current_step)

        # gather stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

        loss_avg = metric_logger.loss.global_avg
        # loss_pw_unscaled_avg = metric_logger.loss_pw_unscaled.global_avg
        # loss_pwc_unscaled_avg = metric_logger.loss_pwc_unscaled.global_avg
        # loss_pwf_unscaled_avg = metric_logger.loss_pwf_unscaled.global_avg
        # loss_ce_unscaled_avg = metric_logger.loss_ce_unscaled.global_avg
        # loss_bbox_unscaled_avg = metric_logger.loss_bbox_unscaled.global_avg
        # loss_giou_unscaled_avg = metric_logger.loss_giou_unscaled.global_avg
        # loss_track_bbox_unscaled_avg = metric_logger.loss_track_bbox_unscaled.global_avg
        # loss_track_giou_unscaled_avg = metric_logger.loss_track_giou_unscaled.global_avg
        # loss_track_bbox_prev_unscaled_avg = metric_logger.loss_track_bbox_prev_unscaled.global_avg
        # loss_track_giou_prev_unscaled_avg = metric_logger.loss_track_giou_prev_unscaled.global_avg
        # loss_track_bbox_next_unscaled_avg = metric_logger.loss_track_bbox_next_unscaled.global_avg
        # loss_track_giou_next_unscaled_avg = metric_logger.loss_track_giou_next_unscaled.global_avg
        # precision_avg = metric_logger.precision.global_avg
        # recall_avg = metric_logger.recall.global_avg
        # f1_avg = metric_logger.f1.global_avg
        # tacc_avg = metric_logger.full_track_accuracy.global_avg
        # taccpwc_avg = metric_logger.track_pwc_accuracy.global_avg  
        # tACCpwc_avg = metric_logger.correct_pwc_tracks.global_avg
        # taccpwf_avg = metric_logger.track_pwf_accuracy.global_avg  
        # tACCpwf_avg = metric_logger.correct_pwf_tracks.global_avg


        if tensorboard is not None:
            tensorboard.add_scalar('epoch/loss_avg_eval', loss_avg, epoch)

            for loss in ["loss_pw_unscaled", "loss_pwc_unscaled","loss_ce_unscaled",
                         "loss_bbox_unscaled", "loss_giou_unscaled", 
                         "loss_track_bbox_unscaled", "loss_track_giou_unscaled",
                         "loss_track_bbox_prev_unscaled", "loss_track_giou_prev_unscaled",
                         "loss_track_bbox_next_unscaled", "loss_track_giou_next_unscaled"]:
                if loss in loss_dict_reduced_unscaled:
                    tensorboard.add_scalar(f'epoch/{loss}_avg_eval', metric_logger[loss].global_avg, epoch)
            for loss in ["precision", "recall", "f1", 
                         "track_pwc_accuracy", "correct_pwc_tracks",
                         "track_pwf_accuracy", "correct_pwf_tracks",
                         "full_track_accuracy"]:
                if loss in loss_dict_track_reduced:
                    tensorboard.add_scalar(f'epoch/{loss}_avg_eval', metric_logger[loss].global_avg, epoch)

            # tensorboard.add_scalar('epoch/loss_pw_avg_eval', loss_pw_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/loss_pwc_avg_eval', loss_pwc_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/loss_pwf_avg_eval', loss_pwf_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/loss_ce_avg_eval', loss_ce_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/loss_bbox_avg_eval', loss_bbox_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/loss_giou_avg_eval', loss_giou_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/loss_track_bbox_avg_eval', loss_track_bbox_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/loss_track_giou_avg_eval', loss_track_giou_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/loss_track_bbox_prev_avg_eval', loss_track_bbox_prev_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/loss_track_giou_prev_avg_eval', loss_track_giou_prev_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/loss_track_bbox_next_avg_eval', loss_track_bbox_next_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/loss_track_giou_next_avg_eval', loss_track_giou_next_unscaled_avg, epoch)
            # tensorboard.add_scalar('epoch/precision_eval', precision_avg, epoch)
            # tensorboard.add_scalar('epoch/recall_eval', recall_avg, epoch)
            # tensorboard.add_scalar('epoch/f1_eval', f1_avg, epoch)
            # tensorboard.add_scalar('epoch/full_track_acc_eval', tacc_avg, epoch)
            # tensorboard.add_scalar('epoch/track_pwc_acc_eval', taccpwc_avg, epoch)
            # tensorboard.add_scalar('epoch/corect_pwc_tracks_eval', tACCpwc_avg, epoch)
            # tensorboard.add_scalar('epoch/track_pwf_acc_eval', taccpwf_avg, epoch)
            # tensorboard.add_scalar('epoch/corect_pwf_tracks_eval', tACCpwf_avg, epoch)

        return loss_avg
