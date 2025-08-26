import os
import sys
import shlex
import inspect
import argparse
import subprocess
from collections import defaultdict, OrderedDict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.dirname(f"{parentdir}/mcmot"))

import numpy as np
from tqdm import tqdm

from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from data.mmptracking import *
from models.detr.util.box_ops import box_cxcywh_to_xyxy
from main_pairwise import collate_fn, _build_dataset, _build_model, _process_data
import warnings

#from icecream import install
#install()

"""
python scripts/sanity_multiview_dataloader.py eval --CFG configs/train/multiview_match/multiview_match.yaml 
"""

SEED = 1024
FONT_SIZE = 30
FONT_PTH = '/zdata/users/dpatel/Phetsarath_OT.ttf'

def post_process(predf, prev_predf, track_map, tracks_inuse, pp_loss):
    max_track = track_map.max() + 1
    for v, (pred, prev_pred) in enumerate(zip(predf, prev_predf)):
        new_tracks = set.difference(set(pred["tracks"].tolist()), set(prev_pred["tracks"].tolist())) 
        lost_tracks = set.difference(set(prev_pred["tracks"].tolist()), set(pred["tracks"].tolist()))
        new_tracks_idx = torch.tensor([t in new_tracks for t in pred["tracks"].tolist()], dtype=torch.bool)
        lost_tracks_idx = torch.tensor([t in lost_tracks for t in prev_pred["tracks"].tolist()],dtype=torch.bool)
        new_track_boxes = pred["boxes"][new_tracks_idx]
        lost_track_boxes = prev_pred["boxes"][lost_tracks_idx]
        ntm, ltm = pp_loss.match(new_track_boxes, lost_track_boxes, 0.5)
        new_tracks = pred["tracks"].long()[new_tracks_idx]
        lost_tracks = prev_pred["tracks"].long()[lost_tracks_idx]
        for nt, pt in enumerate(ntm):
            if pt != -1:
                track_map[v, new_tracks[nt]] = track_map[v,lost_tracks[pt]]
            else:
                # this is a new track
                track_map[v, new_tracks[nt]] = max_track
                max_track += 1        
        track_map[v, lost_tracks] = -1

    conflict = True
    majority_track = -1 * torch.ones(track_map.shape[1], device=track_map.device, dtype = track_map.dtype)
    while conflict:
        for t in range(track_map.shape[1]):
            tm = track_map[:,t]
            unique_tracks, counts = tm.unique(return_counts=True) 
            counts = counts[unique_tracks >= 0]
            unique_tracks = unique_tracks[unique_tracks >=0]
            if unique_tracks.numel() > 0:
                #if unique_tracks.numel() > 1:
                    #warnign_str = f"Track {t} has been mapped to different tracks in different views: {tm}" 
                    #warnings.warn(warnign_str)
                inuse = torch.isin(unique_tracks, tracks_inuse, assume_unique = True)
                if torch.any(inuse):
                    #prioritize tracks in use
                    counts = counts[inuse]
                    unique_tracks = unique_tracks[inuse]
                # select the track that appears the most. 
                _, mt = torch.max(counts,0)
                majority_track[t] = unique_tracks[mt]
        unique_majority, inverse, counts = majority_track.unique(return_inverse=True, return_counts=True)
        conflict = False
        potential_conflict_idx = torch.arange(counts.shape[0])[counts>1]
        for idx in potential_conflict_idx:
            conflicting_track = unique_majority[idx]
            if conflicting_track == -1:
                continue
            conflicting_queries = torch.arange(inverse.shape[0])[inverse==idx]
            conflicts = (track_map[:,conflicting_queries] >= 0)                
            conflicting_views = conflicts.sum(dim=1) > 1
            conflicts = conflicts[conflicting_views,:]
            conflicting_queries = conflicting_queries[conflicts.sum(dim=0)>0]               
            if (conflicting_queries.shape[0] > 0):
                conflict = True
                conflicts = (track_map[:,conflicting_queries] == conflicting_track)
                _, mt = torch.max(conflicts.sum(dim=0),0)
                tm = track_map[:,conflicting_queries[mt]]
                track_map[:,conflicting_queries[mt]][tm>0] = conflicting_track
                conflicting_queries = torch.cat((conflicting_queries[:mt],conflicting_queries[mt+1:]))
                conflicts = track_map[:,conflicting_queries]
                mask = torch.zeros_like(track_map)
                mask = mask.bool()
                mask[:,conflicting_queries] = True
                mask = mask.logical_and(track_map==conflicting_track)
                track_map[mask] = max_track
                max_track += 1

    for t in range(track_map.shape[1]):
        tm = track_map[:,t]
        track_map[tm>=0,t] = majority_track[t]                 
    return track_map


# for output bounding box post-processing
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
    return b

@torch.no_grad()
def plot_tracks(exp_dir):   
    torch.set_printoptions(linewidth=500,  precision=2, sci_mode=False)

    # load checkpoint 
    chkpt_pth = exp_dir
    assert Path(exp_dir).exists(), f'{exp_dir=} does not exist'
    chkpt = torch.load(chkpt_pth, map_location='cpu')
    cfg = chkpt["cfg"]
    cfg.CMD="eval"

    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    reset_unused = 1 # cfg.MODELS.PAIRWISE.RESET_UNUSED or 0
    # NOTE: trained with 1 meaning reset unused track embed if it is not used for upto 1 frame
    reset_unused = reset_unused * cfg.DATASET.FRAMES_PER_SAMPLE 

    reset_bad = False
    use_postprocessing = False # was True

    reset_every = 0 # was -1 

    cfg.DATASET.ROOT = "/net/mlfs02/data/projects/shared/datasets/MMPTracking/"
#    cfg.DATASET.ROOT = "/mnt/ml_data/MMPTracking/"

    cfg.DATASET.FRAMES_PER_SAMPLE = 1
    cfg.DATASET.AUGUMENTATIONS = False
    cfg.DATASET.SUBSAMPLE_RATE = cfg.DATASET.FRAME_STRIDE

    if "industry" in cfg.DATASET.LOCATIONS["validation"]["64pm"][0]:
        cfg.DATASET.LOCATIONS = {"validation": {"64pm":['industry_safety_0', 'industry_safety_2', 'industry_safety_3', 'industry_safety_4']}}
#    cfg.DATASET.LOCATIONS = {"validation": {"64pm":['lobby_0', 'lobby_1', 'lobby_2', 'lobby_3']}}
#    cfg.DATASET.LOCATIONS = {"validation": {"64pm":['office_0', 'office_1', 'office_2']}}

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
    model.eval().to(dev)

    tgt = None

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

    false_neg, false_pos, n_gts, mismatched_pwc = 0, 0, 0, 0
    pwc_prec = []
    pwc_rec = []
    for i, data in enumerate(tqdm(loader)):
        if i < 0:
            continue
        clips_batch, targets_batch, new_segment, frames, metadata = _process_data(data, dev)
        assert clips_batch.shape[0]==1

        H, W = clips_batch.shape[-2:]

        reset_tgt = new_segment.unsqueeze(1).repeat(1,num_queries)
        if reset_unused > 0:                
            track_unused_since[reset_tgt] = 0
            reset_tgt[track_unused_since >= reset_unused] = True

        if reset_bad:
            tgt_prev = tgt 

        if reset_every > 0 and i >= reset_every:
            for t in range(num_queries):
                if i%reset_every == t*50:
                    reset_tgt[0,t] = True 

        out, costs, tgt = model(clips_batch, tgt, reset_tgt)

        outputs_batch, output_tracks, keep = postprocessor(out, costs, keep_prob=0.9)


        if reset_bad:
            bad = torch.ones(costs.shape[4], dtype=torch.bool, device = dev)
            for v in range(costs.shape[2]):
                bad = bad.logical_and(costs[0,0,v,keep[0][0][v],:].sum(0) > 1.5)

            if bad.any():
                reset_tgt = reset_tgt.logical_or(bad)

                out, costs, tgt = model(clips_batch, tgt_prev, reset_tgt)

                outputs_batch, output_tracks, keep = postprocessor(out, costs, keep_prob=0.9)

     
        used_tracks = [torch.cat([predv["tracks"].long() for predf in predb for predv in predf]).unique() for predb in outputs_batch]
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

        targetf = targets_batch[0][0]
        predf = outputs_batch[0][0]
        prev_predf = prev_outputs_batch[0][0]
        metadata_f = metadata[0]['data'][0]
        trackf = output_tracks[0][0]

        dm, gtm = [], []
        assert len(targetf) == len(predf) == len(metadata_f)

        if use_postprocessing:
            track_map = post_process(predf, prev_predf, track_map, tracks_inuse, pp_loss)        
            tracks_inuse = track_map.unique()
            tracks_inuse = tracks_inuse[tracks_inuse >= 0]    

        for v, (target, pred, meta) in enumerate(zip(targetf, predf, metadata_f)):

            pred_boxes = rescale_bboxes(pred["boxes"], (W,H)) if pred["boxes"].numel() else pred["boxes"]
            pred_track_boxes = rescale_bboxes(trackf["track_boxes"][:,v,:], (W,H)) if trackf["track_boxes"].numel() else trackf["track_boxes"][:,v,:]
            gt_boxes = rescale_bboxes(target["boxes"], (W,H)) if target["boxes"].numel() else target["boxes"]

            d, g = pp_loss.match(trackf["track_boxes"][:,v,:], target["boxes"], 0.5)
            gtm.append(g)
            dm.append(d)

            false_neg += (g==-1).sum()
            n_gts += g.shape[0]
            false_pos += (d==-1).sum()

            scene_name = meta["scene_id"]
            scene_id = f'{meta["scene_id"]}_{meta["cam_no"]}'
            frame = meta["frame_no"]
            ut = []

            for trk, box in zip(trackf["tracks"], pred_track_boxes):
                # print(track_map[v,trk].item() < 0)
                if track_map[v,trk].item() < 0:
                    trk = track_map[:,trk].max().item()
                    if trk in ut:
                        continue
                    ut.append(trk)
                # else:
                #     continue
                box = box.tolist()
                pred_list[scene_id].append({
                    'FrameId':frame + 1, # NOTE: trackeval MOT17 evaluation expects 1-indexed frame_ids
                    'Id':trk.item(),
                    'X':int(float(box[0])), 
                    'Y':int(float(box[1])), 
                    'Width':int(float(box[2]-box[0])), 
                    'Height':int(float(box[3]-box[1])), 
                    'Confidence':1.0, 
                    'wx': -1, 
                    'wy': -1,
                    'wz': -1,
                })

            for trk, box in zip(target["tracks"], gt_boxes):
                trk = trk.item()
                box = box.tolist()
                gt_list[scene_id].append({
                    'FrameId':frame + 1, # NOTE: trackeval MOT17 evaluation expects 1-indexed frame_ids
                    'Id':trk,
                    'X':int(float(box[0])), 
                    'Y':int(float(box[1])), 
                    'Width':int(float(box[2]-box[0])), 
                    'Height':int(float(box[3]-box[1])), 
                    'Confidence':1.0, 
                    'wx': -1, 
                    'wy': -1,
                    'wz': -1,
                })

        correct_pwc = 0
        n_pred_tracks_pwc = 0
        n_gt_tracks_pwc = 0
        for v1 in range(len(gtm)):
            for v2 in range(v1+1,len(gtm)):
                # get the number of predicted pairs of objects
                for d,pt in enumerate(trackf["tracks"]):
                    n_pred_tracks_pwc += 1
                    g1 = dm[v1][d]
                    g2 = dm[v2][d]
                    if g1 != -1 and g2 != -1: 
                        if targetf[v1]["tracks"][g1] == targetf[v2]["tracks"][g2]:
                            correct_pwc += 1
                        else: 
                            mismatched_pwc += 1 

                for gt in targetf[v1]["tracks"]:
                    if gt in targetf[v2]["tracks"]:
                        n_gt_tracks_pwc += 1

        pwc_prec.append(correct_pwc/n_pred_tracks_pwc if n_pred_tracks_pwc > 0 else 1)
        pwc_rec.append(correct_pwc/n_gt_tracks_pwc if n_gt_tracks_pwc > 0 else 1)

        prev_outputs_batch = outputs_batch
        prev_targets_batch = targets_batch

        # if i == 1000:
        #     break

    assert len(pred_list) != 0, f'Found {len(pred_list)=}'

    import uuid
    unique_id = uuid.uuid4().hex
    root = Path('scripts/eval_outputs/')
    gt_parent = root / 'gt/mmptracking/MOT17-train'
    pred_parent = root / 'trackers/mmptracking/MOT17-train'
    seqmaps_path = root / 'gt/mmptracking/seqmaps/MOT17-train.txt'
    # tracker_name = Path(exp_dir).stem
    tracker_name = f'{scene_name}_{Path(exp_dir).stem}_{unique_id}'

    # generate gt files
    scene_ids = []
    for scene_id, rows in gt_list.items():
        gt_out_path = gt_parent / f'{scene_id}_{unique_id}' / 'gt' / f'gt.txt'
        seqinfo_out_path = gt_parent / f'{scene_id }_{unique_id}'/ 'seqinfo.ini'
        gt_out_path.parent.mkdir(exist_ok=True, parents=True)
        with open(str(gt_out_path), 'w') as f:
            f.writelines([', '.join(map(str, list(r.values()))) + '\n' for r in rows])

        # get max of both gt and predicted FrameID
        pred_rows = pred_list[scene_id]
        seq_len = max([max([r["FrameId"] for r in rows]), max([r["FrameId"] for r in pred_rows])])
        seqinfo = f'[Sequence]\nname={scene_id}_{unique_id}\nseqLength={seq_len}'

        with open(str(seqinfo_out_path), 'w') as f:
            f.write(seqinfo)
        
        scene_ids.append(f'{scene_id}_{unique_id}')

    # write seqmap: stupid thing needed by the `trackeval` 
    seqmaps_path.parent.mkdir(exist_ok=True, parents=True)
    with open(seqmaps_path, 'w') as f:
        f.writelines(''.join([f'{s}\n' for s in ['name'] + scene_ids]))
    
    # generate preds file 
    for scene_id, rows in pred_list.items():
        pred_out_path = pred_parent / tracker_name / 'data' / f'{scene_id}_{unique_id}.txt'
        pred_out_path.parent.mkdir(exist_ok=True, parents=True)
        with open(str(pred_out_path), 'w') as f:
            f.writelines([', '.join(map(str, list(r.values()))) + '\n' for r in rows])

    print(f'Saved GT and predictions to {root}')
    
    eval_cmd = f'python submodules/trackeval/scripts/run_mot_challenge.py \
    --USE_PARALLEL True \
    --NUM_PARALLEL_CORES 8 \
    --METRICS HOTA CLEAR Identity \
    --TRACKERS_TO_EVAL {tracker_name} \
    --GT_FOLDER {str(gt_parent.parent)} \
    --TRACKERS_FOLDER {str(pred_parent.parent)} \
    --DO_PREPROC False'
 
    try:
        subprocess.run(shlex.split(eval_cmd), check=True)
    except subprocess.CalledProcessError as e:
        return eval_cmd, e

    for k in loss_avg.keys():
        loss_avg[k] = loss_avg[k]/n_frames

    AIDP = torch.tensor(pwc_prec).mean().item()
    AIDR = torch.tensor(pwc_rec).mean().item()
    AIDF1 = 2*AIDP*AIDR/(AIDP + AIDR)
    CVAA = 1 - (2*mismatched_pwc + false_neg + false_pos)/n_gts
    cross_camera_association_losses = {"AIDP":AIDP, "AIDR":AIDR, "AIDF1": AIDF1, "CVAA":CVAA, "mismatched_pwc":mismatched_pwc, "false_neg":false_neg, "false_pos":false_pos, "n_gts":n_gts}

    torch.set_printoptions(linewidth=500,  precision=5, sci_mode=False)
    print(dict(loss_avg))
    print(cross_camera_association_losses)
    torch.set_printoptions(linewidth=500,  precision=2, sci_mode=False)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('EXP_DIR', help='path to run directory')
    parser.add_argument('-s','--SEED', default='1204', type=int, help='Random seed')

    args = parser.parse_args()
    exp_dir = args.EXP_DIR

    plot_tracks(exp_dir)

