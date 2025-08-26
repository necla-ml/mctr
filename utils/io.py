import os
import json
from glob import glob
from pathlib import Path
from collections import defaultdict
import yaml

# third party
import torch

def write_cfg(dir, cfg):
    with open(f"{dir}/config.yaml", "wt") as f:
        yaml.dump(cfg, f)


def write_scores(metas_list, dir):

    pair_score_dict = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    all_dets_dict = defaultdict(dict)
    for meta in metas_list:
        scene_cam_ij = f'{meta["scene"]}_{meta["cam_i"]}_{meta["cam_j"]}'
        uuid_i = meta['uuid_i']
        uuid_j = meta['uuid_j']
        # add score both ways i -> j and j -> i
        pair_score_dict[scene_cam_ij][uuid_i][uuid_j] = meta['pred_score']
        pair_score_dict[scene_cam_ij][uuid_j][uuid_i] = meta['pred_score']

        all_dets_dict[uuid_i] = {
            'scene': meta['scene'],
            'scene_no': meta['scene_no'],
            'cam': meta['cam_i'],
            'frame': meta['frame_i'],
            'det': meta['det_i'],
            'loc': meta['loc_i'],
            'gt_det': meta['gt_det_i'],
            'gt_track_id': meta['gt_track_id_i']
        }

        all_dets_dict[uuid_j] = {
            'scene': meta['scene'],
            'scene_no': meta['scene_no'],
            'cam': meta['cam_j'],
            'frame': meta['frame_j'],
            'det': meta['det_j'],
            'loc': meta['loc_j'],
            'gt_det': meta['gt_det_j'],
            'gt_track_id': meta['gt_track_id_j']
        }

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    for scene_cam_ij, pair_scores in pair_score_dict.items():
        with open(os.path.join(str(dir), f'{scene_cam_ij}_scores.json'), 'w+') as f:
            json.dump(pair_scores, f)

    with open(os.path.join(str(dir), f'dets.json'), 'w+') as f:
        json.dump(all_dets_dict, f)

def write_stats(stats_dict, dir):
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(str(dir), 'stats.log'), 'a') as f:
        f.write(json.dumps(stats_dict) + '\n')

def save_checkpoint(chkpt, is_best, dir='output', name='model'):
    os.makedirs(os.path.join(dir, 'checkpoints'), exist_ok=True)
    if is_best:
        torch.save(chkpt, os.path.join(dir, 'checkpoints', f'{name}-best.pth'))
    torch.save(chkpt, os.path.join(dir, 'checkpoints', f'{name}.pth'))

def logdir(logdir='runs/train', log_prefix='exp', comment='', timestamped=True):
    """Increment logdir in the fromat runs/{prefix}{seq}_{comment=arch}_{timestamp}
    """
    dir = str(Path(logdir, log_prefix))
    # others = sorted(glob(dir + '*'))

    import socket
    from datetime import datetime
    timestamp = f"{socket.gethostname()}_{datetime.now().strftime('%b%d_%H%M%S')}"
    
    # if others:
    #     seq = max([int(x[len(dir):x.find('-') if '-' in x else None]) for x in others]) + 1
    return f"{dir}-{comment}{f'-{timestamp}' if timestamped else ''}"
