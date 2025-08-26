import os
import sys
import inspect

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

from main_pairwise import collate_fn, _build_dataset, _build_model

from utils.args import parse_args
from data.mmptracking import *

"""
python scripts/sanity_multiview_dataloader.py eval --CFG configs/train/multiview_match/multiview_match.yaml 
"""

FONT_SIZE = 30
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
def plot_tracks(cfg):   

    # random.seed(cfg.SEED)
    # torch.manual_seed(cfg.SEED)
    # np.random.seed(cfg.SEED)

    dev = 'cuda'
    val_dataset, train_dataset = _build_dataset(cfg, build_train=True)
    loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1, # keep batch size one to get single sample prediction
            num_workers=1,
            collate_fn=collate_fn,
            shuffle=True)

    

    model, criterion, postprocessor,pp_loss = _build_model(cfg, val_dataset.ncams)
    postprocessor = postprocessor["tracks"]
    pp_loss = pp_loss["tracks"]

    
    # load checkpoint 
    exp_dir ='runs/train/exp239-PAIRWISE-train-R50-ml-gpu05_Aug12_132159'
#    chkpt_pth = f'{exp_dir}/checkpoints/PAIRWISE-best.pth'
    chkpt_pth = f'{exp_dir}/checkpoints/PAIRWISE.pth'
    chkpt = torch.load(chkpt_pth, map_location='cpu')
    state_dict = chkpt['state_dict']
    consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')
    res = model.load_state_dict(state_dict, strict=True)
    print(f'Loaded {chkpt_pth=} with {res=} {chkpt["loss"]} {chkpt["epoch"]}')
    model.to(dev)

    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])

    tgt = None
    reset_target = True

    for i, data in enumerate(loader):
        clips, boxes, tracks = data

        # TODO: multi frame: use single frame for now
        clips_batch = clips.to(dev, non_blocking=True) # B, F,Cm, C, H, W
        gt_boxes_batch = [[[{k: v.to(dev, non_blocking=True) for k, v in d.items()} for d in b] for b in bf] for bf in boxes]
        gt_tracks_batch = [[[{k: v.to(dev, non_blocking=True) for k, v in d.items()} for d in t] for t in tf] for tf in tracks]
        for bx, tk in zip(gt_boxes_batch, gt_tracks_batch):
            for bf, tf in zip(bx,tk):
                for b,t in zip(bf,tf): 
                    b.update({'labels': torch.zeros((len(b['boxes']),), dtype=torch.int64, device=dev)})
                    b.update(t)
        targets_batch = gt_boxes_batch

        H, W = clips_batch.shape[-2:]

        if reset_target:
            tgt = None    
        out, costs, tgt = model(clips_batch, tgt)

        outputs_batch, raw_outputs, keep = postprocessor(out, costs, keep_prob=0.9)


        num_queries = model.num_queries

        # loss_dict = criterion((outputs, out0, out1), targets)
        # weight_dict = criterion.weight_dict
        # for k in loss_dict.keys(): 
        #     if k not in weight_dict:
        #         print(k)
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        out_img = []
        torch.set_printoptions(linewidth=500)
        for j, (clips, targets, preds) in enumerate(zip(clips_batch, targets_batch, outputs_batch)):
            tracks = pp_loss.get_tracks(targets)
            for f, (clipf, targetf, predf) in enumerate(zip(clips, targets, preds)):
                gtm=[]
                dm=[]
                for v, (clip, target, pred) in enumerate(zip(clipf, targetf, predf)):
                    d, g = pp_loss.match(pred, target, 0.5)
                    gtm.append(g)
                    dm.append(d)

                    pred_boxes = rescale_bboxes(pred["boxes"], (W,H)) if pred["boxes"].numel() else pred["boxes"]
                    gt_boxes = rescale_bboxes(target["boxes"], (W,H))
                    pred_tracks = [str(t.item()) for t in pred["tracks"]]
                    pred_labels = ["{}\n{:.2f}".format(t.item(),c.item()) for t,c in zip(pred["tracks"],pred["costs"])]
                    gt_tracks = [str(t.item()) for t in target["tracks"].int()]          
                    clip = ((clip.cpu() * std[:, None, None] + mean[:, None, None]) * 255.0).to(torch.uint8)
                    colors_pred = [av.utils.rgb(int(tid), integral=True) for tid in pred_tracks]
                    colors_gt = [av.utils.rgb(int(tid), integral=True) for tid in gt_tracks]
                    clip_pred = draw_bounding_boxes(clip, pred_boxes, colors=colors_pred, labels=pred_labels, fill=True, width=2, font=FONT_PTH, font_size=FONT_SIZE)
                    clip_gt = draw_bounding_boxes(clip, gt_boxes, colors=colors_gt, labels=gt_tracks, fill=True, width=2, font=FONT_PTH, font_size=FONT_SIZE)
                    clip_list = [clip_gt, clip_pred]

                    if pred["track_boxes"] is not None:
                        track_boxes = rescale_bboxes(pred["track_boxes"], (W,H)) if pred["track_boxes"].numel() else pred["track_boxes"]
                        clip_track_boxes = draw_bounding_boxes(clip, track_boxes, colors=colors_pred, labels=pred_labels, fill=True, width=2, font=FONT_PTH, font_size=FONT_SIZE)
                        clip_list += [clip_track_boxes]

                    clip = make_grid(clip_list, nrow=len(clip_list))
                    out_img.append(clip)

                for t in range(tracks.shape[0]):
                    matched_dets=[]
                    for v in range(tracks.shape[2]):
                        g = tracks[t,f,v]
                        if g != -1:
                            matched_dets.append(gtm[v][g])
                        else:
                            matched_dets.append(-1)    
                    if all([d >= 0 for d in matched_dets]):
                        for v in range(tracks.shape[2]):
                            d = matched_dets
                            c = costs[j,f,v,keep[j][f][v],:]
                            print(i, t, f, v, predf[v]["tracks"][d[v]].item(),c[d[v],:])

        out_grid = make_grid(out_img, nrow=1)
        show(out_grid,i)
        if i > 0: 
            break




if __name__ == '__main__':
    cfg = parse_args()
    plot_tracks(cfg)

