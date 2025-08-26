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
from models.detr.util import box_ops

from main_pairwise import collate_fn, _build_dataset, _build_model, _process_data

from utils.args import parse_args
from data.mmptracking import *



"""
python scripts/sanity_multiview_dataloader.py eval --CFG configs/train/multiview_match/multiview_match.yaml 
"""

torch.set_printoptions(linewidth = 200, precision=2, sci_mode=False)

FONT_SIZE = 30
FONT_PTH = '/zdata/users/dpatel/Phetsarath_OT.ttf'

# for output bounding box post-processing
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_ops.box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
    return b

def show(img,i):
    npimg = img.numpy()    
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.imsave(f"savedimval{i}.jpg", np.transpose(npimg, (1, 2, 0)))
    plt.show()


@torch.no_grad() 
def match(pred, gt, threshold):
    # matches predictions with ground truth based on iou. 
    # iou must be larger than  threshold to match.
    gtm = -1*torch.ones(gt["boxes"].shape[0],dtype=torch.int)
    dm = -1*torch.ones(pred["boxes"].shape[0],dtype=torch.int)
    pred = box_ops.box_cxcywh_to_xyxy(pred["boxes"])
    gt = box_ops.box_cxcywh_to_xyxy(gt["boxes"])
    iou,_ = box_ops.box_iou(pred, gt)
    iou[iou<threshold] = 0
    pred_idx, gt_idx = linear_sum_assignment(iou.cpu(), maximize=True)
    idx1 = []
    idx2 = []
    for p,g in zip(pred_idx, gt_idx):
        if iou[p,g] > 0:
            idx1.append(p)
            idx2.append(g)
    return idx1, idx2

@torch.no_grad()
def plot_tracks(exp_dir):   

    # random.seed(cfg.SEED)
    # torch.manual_seed(cfg.SEED)
    # np.random.seed(cfg.SEED)

    # load checkpoint 
    # exp_dir ='runs/train/exp282-PAIRWISE-train-R50-cipr-gpu05_Aug20_141750'
    #chkpt_pth = f'{exp_dir}/checkpoints/PAIRWISE-best.pth'
    chkpt_pth = f'{exp_dir}/checkpoints/PAIRWISE.pth'
    chkpt = torch.load(chkpt_pth, map_location='cpu')
    cfg = chkpt["cfg"]

    cfg.DATASET.FRAMES_PER_SAMPLE = 2
    cfg.DATASET.AUGUMENTATIONS=False
    cfg.DATASET.SUBSAMPLE_RATE = cfg.DATASET.FRAME_STRIDE


    dev = 'cuda'
    val_dataset, train_dataset = _build_dataset(cfg, build_train=True)
    loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1, # keep batch size one to get single sample prediction
            num_workers=1,
            collate_fn=collate_fn,
            shuffle=False)

    

    model, criterion, postprocessor,pp_loss = _build_model(cfg, val_dataset.ncams)
    postprocessor = postprocessor["tracks"]
    pp_loss = pp_loss["tracks"]

    
    #chkpt_pth = f'{exp_dir}/checkpoints/PAIRWISE-best.pth'
    chkpt_pth = f'{exp_dir}/checkpoints/PAIRWISE.pth'
    chkpt = torch.load(chkpt_pth, map_location='cpu')
    state_dict = chkpt['state_dict']
    consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')
    res = model.load_state_dict(state_dict, strict=True)
    print(f'Loaded {chkpt_pth=} with {res=} {chkpt["loss"]} {chkpt["epoch"]}')
    model.to(dev)

    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

    fps = 15
    h, w = [360, 640]
    output_video_file = 'realtime_pred.mp4'

    # init video stream
    media = av.open(output_video_file, 'w')
    video = media.add_stream('h264', fps)   
    video.height, video.width = 720, 1280
    video.bit_rate = 1000000
    video = media.streams[0]

    tgt = None
    prev_costs = None
    prev_outputs = None
    for i, data in enumerate(loader):
        print(i)

        clips_batch, targets_batch, new_segment, frames = _process_data(data, dev)
        assert clips_batch.shape[0]==1

        H, W = clips_batch.shape[-2:]
  
        out, costs, tgt = model(clips_batch, None)

        # if prev_costs is not None:
        #     track_match_logprob = torch.matmul(prev_costs[0,-1,:,:,:].transpose(1,2),costs[0,0,:,:,:]).log().sum(dim=0)
        #     ptx,ctx = linear_sum_assignment(track_match_logprob.cpu(), maximize=True)
        #     ctx = torch.from_numpy(ctx)
        #     costs = costs[:,:,:,:,ctx]
        # prev_costs = costs

        outputs_batch, raw_outputs, keep = postprocessor(out, costs, keep_prob=0.9)
        num_queries = model.num_queries

        if prev_outputs is not None:
            track_match_logprob = torch.zeros(num_queries, num_queries, device = costs.device)
            nviews = torch.zeros_like(track_match_logprob)
            for v,(view1, view2) in enumerate(zip(prev_outputs[0][-1],outputs_batch[0][0])):
                idx1, idx2 = match(view1, view2, 0.5)
                idx1 = torch.tensor(idx1,dtype=torch.long)
                idx2 = torch.tensor(idx2,dtype=torch.long)
                idx1 = view1["detr_det_id"][idx1].to(costs.device)
                idx2 = view2["detr_det_id"][idx2].to(costs.device)
                tracks1 = view1["tracks"]
                tracks1 = tracks1[tracks1>=0].type(torch.long).unsqueeze(1)
                tracks2 = view2["tracks"]            
                tracks2 = tracks2[tracks2>=0].type(torch.long).unsqueeze(0)
                track_match_logprob[tracks1,tracks2] += torch.matmul(prev_costs[0,-1,v,idx1,:].transpose(0,1), costs[0,0,v,idx2,:]).log()[tracks1,tracks2]
                nviews[tracks1,tracks2] += 1
            track_match_logprob[nviews>0.5] = track_match_logprob[nviews>0.5]/nviews[nviews>0.5]
            track_match_logprob[nviews<0.5] = -1e5
            ptx,ctx = linear_sum_assignment(track_match_logprob.cpu(), maximize=True)
            ctx = torch.from_numpy(ctx)
            costs = costs[:,:,:,:,ctx]

        # this is wastefull as we have already done it. 
        outputs_batch, raw_outputs, keep = postprocessor(out, costs, keep_prob=0.9)

        prev_outputs = outputs_batch
        prev_costs = costs

        # loss_dict = criterion((outputs, out0, out1), targets)
        # weight_dict = criterion.weight_dict
        # for k in loss_dict.keys(): 
        #     if k not in weight_dict:
        #         print(k)
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

#        used_queries = set()
        for f, (clipf, targetf, predf) in enumerate(zip(clips_batch[0], targets_batch[0], outputs_batch[0])):
            out_img = []
            for v, (clip, target, pred) in enumerate(zip(clipf, targetf, predf)):
                pred_boxes = rescale_bboxes(pred["boxes"], (W,H)) if pred["boxes"].numel() else pred["boxes"]
                pred_tracks = [str(t.item()) for t in pred["tracks"]]
#               used_queries.union(set(pred["tracks"].tolist()))
                pred_labels = ["{}\n{:.2f}".format(t.item(),c.item()) for t,c in zip(pred["tracks"],pred["costs"])]        
                clip = ((clip.cpu() * std[:, None, None] + mean[:, None, None]) * 255.0).to(torch.uint8)
                colors_pred = [av.utils.rgb(int(tid), integral=True) for tid in pred_tracks]
                clip_pred = draw_bounding_boxes(clip, pred_boxes, colors=colors_pred, labels=pred_labels, fill=True, width=2, font=FONT_PTH, font_size=FONT_SIZE)
                clip_pred = np.ascontiguousarray(TORCH_2_CV(clip_pred), dtype=np.uint8)
                clip_pred = cv2.putText(clip_pred, f"{i}", (20, 20),cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 3)
                clip_pred = CV_2_TORCH(clip_pred)
                out_img.append(clip_pred)
            frame = make_grid(out_img, nrow=len(predf)//2)
            frame = av.VideoFrame.from_ndarray(frame.permute(1, 2, 0).numpy(), format='rgb24')
            packets = video.encode(frame)
            media.mux(packets)

#        unused_queries = torch.tensor(list(set(range(tgt.shape[1])).difference(used_queries)), dtype=torch.long)
    #    tgt[0,unused_queries,:] = torch.zeros_like(tgt[0,unused_queries,:])

        if i == 500:
            break
    
    # output video
    packets = video.encode(None)   
    if packets:
        media.mux(packets)
    media.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('EXP_DIR', help='path to run directory')
    parser.add_argument('-s','--SEED', default='1204', type=int, help='Random seed')

    exp_dir = parser.parse_args().EXP_DIR

    plot_tracks(exp_dir)

