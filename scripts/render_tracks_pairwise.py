import os
import sys
import inspect
import argparse
import cv2

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.dirname(f"{parentdir}/mcmot"))

import matplotlib.pyplot as plt
from ml import av
from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes
import numpy as np
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from models.detr.util.box_ops import box_cxcywh_to_xyxy

from main_pairwise import collate_fn, _build_dataset, _build_model, _process_data
from trackeval_mmptrack import post_process

from utils.args import parse_args
from data.mmptracking import *

from mmp_cameraview_eval import compare_dataframes
import pandas as pd
import motmetrics as mm
from collections import defaultdict, OrderedDict
import warnings

from icecream import install
install()

"""
python scripts/sanity_multiview_dataloader.py eval --CFG configs/train/multiview_match/multiview_match.yaml 
"""

FONT_SIZE = 40
FONT_PTH = '/zdata/users/dpatel/Phetsarath_OT.ttf'

# for output bounding box post-processing
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
    return b

def show(img,i):
    npimg = img.numpy()    
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.imsave(f"savedimval{i}.jpg", np.transpose(npimg, (1, 2, 0)))
    plt.show()


@torch.no_grad()
def plot_tracks(exp_dir):   
    ic.disable()
    torch.set_printoptions(linewidth=500,  precision=2, sci_mode=False)

    # load checkpoint 
    chkpt_pth = f'{exp_dir}/checkpoints/PAIRWISE.pth'
    chkpt = torch.load(chkpt_pth, map_location='cpu')
    cfg = chkpt["cfg"]
    cfg.CMD="eval"

    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    reset_unused = cfg.MODELS.PAIRWISE.RESET_UNUSED or 0
    reset_unused = reset_unused * cfg.DATASET.FRAMES_PER_SAMPLE 

    reset_bad = False
    use_postprocessing = False

    if "MMPTracking" in cfg.DATASET.ROOT:
        cfg.DATASET.ROOT = "/net/mlfs02/data/projects/shared/datasets/MMPTracking/"
        if "industry" in cfg.DATASET.LOCATIONS["validation"]["64pm"][0]:
            cfg.DATASET.LOCATIONS = {"validation": {"64pm":['industry_safety_0', 'industry_safety_2', 'industry_safety_3', 'industry_safety_4']}}
        if "lobby" in cfg.DATASET.LOCATIONS["validation"]["64pm"][0]:
            cfg.DATASET.LOCATIONS = {"validation": {"64pm":['lobby_0', 'lobby_1', 'lobby_2', 'lobby_3']}}
        if "office" in cfg.DATASET.LOCATIONS["validation"]["64pm"][0]:
            cfg.DATASET.LOCATIONS = {"validation": {"64pm":['office_0', 'office_1', 'office_2']}}
    elif  "MTMC_Tracking_AIC23_Track1" in cfg.DATASET.ROOT:
        cfg.DATASET.ROOT = "/zdata/projects/shared/datasets/MTMC_Tracking_AIC23_Track1"
    elif "DanceTrack" in cfg.DATASET.ROOT:
        pass
    else:
        raise ValueError(f'Invalid dataset {cfg.DATASET.ROOT}')

    cfg.DATASET.FRAMES_PER_SAMPLE = 1
    cfg.DATASET.AUGUMENTATIONS=False
    cfg.DATASET.SUBSAMPLE_RATE = cfg.DATASET.FRAME_STRIDE

#    cfg.DATASET.LOCATIONS = {"validation": {"64pm":['industry_safety_0']}}

    dev = 'cuda'
    batch_size = 1  # keep batch size one to get single sample prediction
    val_dataset, train_dataset = _build_dataset(cfg, build_train=False)
    loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=1,
            collate_fn=collate_fn,
            shuffle=False)

    model, criterion, postprocessor,pp_loss = _build_model(cfg, val_dataset.ncams)
    postprocessor = postprocessor["tracks"]
    pp_loss = pp_loss["tracks"]
    
    state_dict = chkpt['state_dict']
    consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')
    res = model.load_state_dict(state_dict, strict=True)
    print(f'Loaded {chkpt_pth=} with {res=} {chkpt["loss"]} {chkpt["epoch"]}')
    model.to(dev)
    model.eval()
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

    fps = 15
    h, w = [360, 640]
    output_video_file = 'realtime_pred.mp4'

    if model.num_views <= 3:
        nrows = 1
    else:   
        nrows = int(math.sqrt(model.num_views))


    vh = h * nrows
    vw = w * math.ceil(model.num_views/nrows)

    # init video stream
    media = av.open(output_video_file, 'w')
    video = media.add_stream('h264', fps)   
    video.height, video.width = vh, vw    
    #video.height, video.width = 720, 1280
#    video.height, video.width = 1080, 820
    video.bit_rate = 1000000
    video = media.streams[0]

    tgt = None
    bad = torch.zeros(model.num_queries, dtype=torch.bool, device = dev)
    prev_outputs_batch = None
    prev_targets_batch = None

    pred_list = defaultdict(list)
    gt_list = defaultdict(list)

    loss_avg = defaultdict(lambda:0)


    n_frames = 0
    max_track = 0
    tracks_inuse = torch.tensor([],device=dev, dtype=torch.int)
    num_queries = model.num_queries

    if use_postprocessing:
        track_map = -1 * torch.ones((model.num_views, num_queries), dtype=torch.int, device=dev)
    else:
        track_map = torch.arange(num_queries).repeat(model.num_views,1)

    track_unused_since = torch.zeros((batch_size, num_queries), dtype=torch.int, device=dev)
    new_track_penalty = torch.ones(num_queries, device = dev)
    for i, data in enumerate(loader):
        ic(i)
        if i < 0:
            continue
        clips_batch, targets_batch, new_segment, frames, _ = _process_data(data, dev)
        assert clips_batch.shape[0]==1

        H, W = clips_batch.shape[-2:]
        
        reset_tgt = new_segment.unsqueeze(1).repeat(1,num_queries)
        if reset_unused > 0:                
            track_unused_since[reset_tgt] = 0
            reset_tgt[track_unused_since >= reset_unused] = True

        if reset_bad:
            reset_tgt = reset_tgt.logical_or(bad)

        out, costs, tgt = model(clips_batch, tgt, reset_tgt)

       
        # new_track_penalty[track_unused_since[0,:]==0] += 0.05
        # new_track_penalty[track_unused_since[0,:]>0] -= 0.05
        # new_track_penalty = new_track_penalty.clamp(0.5,1)
        # costs *= new_track_penalty

        outputs_batch, raw_outputs, keep = postprocessor(out, costs, keep_prob=0.9)

        bad = torch.ones(costs.shape[4], dtype=torch.bool, device = dev)
        for v in range(costs.shape[2]):
            bad = bad.logical_and(costs[0,0,v,keep[0][0][v],:].sum(0) > 1.2)
        # if bad.any():
        #    for v in range(costs.shape[2]):
        #         print(i, v, costs[0,0,v,keep[0][0][v],:])

     
        for v in range(costs.shape[2]):
            ic(i, v, costs[0,0,v,keep[0][0][v],:])

        used_tracks = [torch.cat([predv["tracks"].long() for predf in predb for predv in predf]).unique() for predb in outputs_batch]
        new_tracks = [t for t in used_tracks[0] if track_unused_since[0,t] > 0]

        track_unused_since += 1
        for b in range(track_unused_since.shape[0]):
            track_unused_since[b,used_tracks[b]] = 0

        
        if prev_outputs_batch is not None:
            outputs_2frames = [[prev_outputs_batch[0][0], outputs_batch[0][0]]]
            targets_2frames = [[prev_targets_batch[0][0], targets_batch[0][0]]]
            loss_dict_track = pp_loss(outputs_2frames, targets_2frames)

            for k in loss_dict_track.keys():
                loss_avg[k] += loss_dict_track[k]
            n_frames+=1
        else:
            prev_outputs_batch = [[[{"tracks":torch.tensor([],dtype=torch.long),"boxes":torch.tensor([])}]*model.num_views]]


        tracks = pp_loss.get_tracks(targets_batch[0])

        out_img = []
        clipf = clips_batch[0][0]
        targetf = targets_batch[0][0]
        predf = outputs_batch[0][0]
        prev_predf = prev_outputs_batch[0][0]

        gtm=[]
        dm=[]

        if use_postprocessing:
            track_map = post_process(predf, prev_predf, track_map, tracks_inuse, pp_loss)        
            tracks_inuse = track_map.unique()
            tracks_inuse = tracks_inuse[tracks_inuse >= 0]        
    

        for v, (clip, target, pred, prev_pred) in enumerate(zip(clipf, targetf, predf, prev_predf)):

            d, g = pp_loss.match(pred["boxes"], target["boxes"], 0.5)
            gtm.append(g)
            dm.append(d)

            
            pred_boxes = rescale_bboxes(pred["boxes"], (W,H)) if pred["boxes"].numel() else pred["boxes"]
            gt_boxes = rescale_bboxes(target["boxes"], (W,H)) if target["boxes"].numel() else target["boxes"]

            for trk,box in zip(pred["tracks"], pred_boxes):
                trk = track_map[v,trk].item()
                box = box.tolist()
                pred_list[v].append({"FrameId":i,"Id":trk,'X':int(float(box[0])), 'Y':int(float(box[1])), 'Width':int(float(box[2]-box[0])), 'Height':int(float(box[3]-box[1])), 'Confidence':1.0})
            for trk,box in zip(target["tracks"],gt_boxes):
                trk = trk.item()
                box = box.tolist()
                gt_list[v].append({"FrameId":i,"Id":trk,'X':int(float(box[0])), 'Y':int(float(box[1])), 'Width':int(float(box[2]-box[0])), 'Height':int(float(box[3]-box[1])), 'Confidence':1.0})


            pred_tracks = [str(track_map[v,t].item()) for t in pred["tracks"]]
#            pred_labels = ["{}\n{:.2f}".format(track_map[v,t].item(),c.item()) for t,c in zip(pred["tracks"], pred["costs"])]        
            pred_labels = ["{}".format(track_map[v,t].item()) for t in pred["tracks"]]        
            clip = ((clip.cpu() * std[:, None, None] + mean[:, None, None]) * 255.0).to(torch.uint8)
            colors_pred = [av.utils.rgb(int(tid), integral=True) for tid in pred_tracks]
            clip_pred = draw_bounding_boxes(clip, pred_boxes, colors=colors_pred, labels=pred_labels, fill=False, width=4, font=FONT_PTH, font_size=FONT_SIZE)
            clip_pred = np.ascontiguousarray(TORCH_2_CV(clip_pred), dtype=np.uint8)
            clip_pred = cv2.putText(clip_pred, f"{i}", (20, 20),cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 3)
            clip_pred = CV_2_TORCH(clip_pred)
            out_img.append(clip_pred)
        frame = make_grid(out_img, nrow=len(predf)//nrows)
        frame = av.VideoFrame.from_ndarray(frame.permute(1, 2, 0).numpy(), format='rgb24')
        packets = video.encode(frame)
        media.mux(packets)

        if False:
            print(reset_tgt)
            for t in range(tracks.shape[0]):
                for v in range(tracks.shape[2]):
                    g = tracks[t,0,v]
                    if g != -1:
                        d = gtm[v][g]
                        if d >= 0:
                            c = costs[0,0,v,keep[0][0][v],:]
                            print(i, t, 0, v, predf[v]["tracks"][d].item(),predf[v]["detr_det_id"][d].item(), c[d,:])

            for v in range(tracks.shape[2]):
                c = costs[0,0,v,keep[0][0][v].logical_not(),:]
                for j in range(c.shape[0]):
                    print(i,0,0,v,0,0,c[j,:])

        prev_outputs_batch = outputs_batch
        prev_targets_batch = targets_batch

        if i == 3000:
            break

    # output video
    packets = video.encode(None)   
    if packets:
        media.mux(packets)
    media.close()


    pred_dfs = OrderedDict([(cam_id, pd.DataFrame(rows).set_index(['FrameId', 'Id']).sort_index()) for cam_id, rows in pred_list.items()])
    gt_dfs = OrderedDict([(cam_id, pd.DataFrame(rows).set_index(['FrameId', 'Id']).sort_index()) for cam_id, rows in gt_list.items()])
    
    accs, names = compare_dataframes(gt_dfs, pred_dfs)

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics + ['num_frames'], generate_overall=True)

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


    for k in loss_avg.keys():
        loss_avg[k] = loss_avg[k]/n_frames

    torch.set_printoptions(linewidth=500,  precision=5, sci_mode=False)
    print(dict(loss_avg))
    torch.set_printoptions(linewidth=500,  precision=2, sci_mode=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('EXP_DIR', help='path to run directory')
    parser.add_argument('-s','--SEED', default='1204', type=int, help='Random seed')

    exp_dir = parser.parse_args().EXP_DIR

    plot_tracks(exp_dir)

