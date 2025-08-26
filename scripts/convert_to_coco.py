import os
import sys
import json
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import json
from pathlib import Path
from collections import defaultdict

import tqdm

import torch
import torch.nn.functional as F
from ml.vision.ops import clip_boxes_to_image

from data.mmptracking import MMPTrackPairs

def xyxy2xywh(x):
    y = x.clone()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

root = Path('/zdata/projects/shared/datasets/MMPTracking')
splits = ['train', 'validation']
image_path = 'images'
feats_path = 'dets-yolo7-cls_thresh_0.4-nms_thresh_0.5-feats_DukeMTMC_SBS_S50'
labels_path = 'labels'
H, W = [360, 640]
stride = 5

coco_port = root / 'coco_type_yolo_filtered'
mmp_labels_json = 'mmp_labels.json'

if not Path(mmp_labels_json).exists():
    mmp_labels = {}
    for split in splits:
        print(f'Working on {split}')
        split_path = root / split
        split_img_path = split_path / image_path
        images = list(split_img_path.glob('*/*/*.jpg'))
        labels = [str(im).replace(image_path, labels_path).replace('.jpg', '.json') for im in images]

        for l in labels:
            with open(l, 'r') as f:
                annot = json.load(f)

            key = list(Path(l).parts)
            key[-1] = key[-1].replace('.json', '')
            key = '-'.join(key[-5:])
            mmp_labels[key] = annot

    with open(mmp_labels_json, 'w+') as f:
        json.dump(mmp_labels, f)
else:
    print(f'Found {mmp_labels_json}', flush=True)

def convert_mmp_labels_to_coco():
    with open(mmp_labels_json, 'rb') as f:
        mmp_labels = json.load(f)
        
    split_txt = defaultdict(list)
    for i, (key, annot) in enumerate(mmp_labels.items()):
        split, _, scene_time, scene, frame = key.split('-')
        frame_no = int(frame.split('_')[1])
        if frame_no % stride != 0:
            continue

        # frame_pth = root / split / image_path / scene_time / scene / f'{frame}.jpg'
        # assert frame_pth.exists()

        split = 'val' if split == 'validation' else split # validation -> val

        # # create symlink to frame
        # dest_dir = coco_port / 'images' / split
        # dest_dir.mkdir(parents=True, exist_ok=True)
        # dest_vid = dest_dir / f'{key}.jpg'
        # if dest_vid.is_symlink():
        #     dest_vid.unlink()
        # dest_vid.symlink_to(frame_pth.resolve(), target_is_directory=False)

        """
        cls, x, y, w, h (normalized by image size)
        0 0.262891 0.506944 0.111719 0.336111
        0 0.632422 0.277083 0.036719 0.215278
        0 0.6125 0.25 0.040625 0.205556
        0 0.592969 0.248611 0.039062 0.225
        0 0.616406 0.236111 0.045312 0.225
        """
        """
        annot: {"track_id": xyxy}
        """
        boxes = [torch.tensor(bbox) for tid, bbox in annot.items()]
        if boxes:
            boxes = torch.stack(boxes)
            boxes = clip_boxes_to_image(boxes, (H, W))
            boxes = xyxy2xywh(boxes.float())
            # norm
            boxes[:, [0, 2]] /= W
            boxes[:, [1, 3]] /= H

            dets_label = coco_port / 'labels' / split / f'{key}.txt'
            boxes = torch.cat([torch.zeros(len(boxes), 1), boxes], dim=1).tolist()
                
            boxes = [' '.join([str(i) for i in b]) + '\n' for b in boxes]
            with open(dets_label, 'w+') as f:
                f.writelines(boxes)

            split_txt[split].append(f'./images/{split}/{key}.jpg\n')

    for split, split_lst in split_txt.items():
        with open(str(coco_port / f'{split}.txt'), 'w+') as f:
            f.writelines(split_lst)


def convert_feat_to_coco():
    with open(mmp_labels_json, 'rb') as f:
        mmp_labels = json.load(f)
    
    split_txt = defaultdict(list)
    for i, (key, annot) in tqdm.tqdm(enumerate(mmp_labels.items()), total=len(mmp_labels.items())):
        split, _, scene_time, scene, frame = key.split('-')
        frame_no = int(frame.split('_')[1])
        if frame_no % stride != 0:
            continue

        feat_pth = root / split / feats_path / scene_time / scene / f'{frame}.pth'
        sample = torch.load(str(feat_pth), map_location='cpu')
        det_people = sample['people']
        num_people = sample['num_people']

        # match boxes with iou and only take gt_people boxes that match with det_people

        frame_pth = root / split / image_path / scene_time / scene / f'{frame}.jpg'
        assert frame_pth.exists()

        split = 'val' if split == 'validation' else split # validation -> val

        # create symlink to frame
        dest_dir = coco_port / 'images' / split
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_vid = dest_dir / f'{key}.jpg'
        if dest_vid.is_symlink():
            dest_vid.unlink()
        dest_vid.symlink_to(frame_pth.resolve(), target_is_directory=False)

        """
        cls, x, y, w, h (normalized by image size)
        0 0.262891 0.506944 0.111719 0.336111
        0 0.632422 0.277083 0.036719 0.215278
        0 0.6125 0.25 0.040625 0.205556
        0 0.592969 0.248611 0.039062 0.225
        0 0.616406 0.236111 0.045312 0.225
        """
        """
        annot: {"track_id": xyxy}
        """
        boxes = [torch.tensor(bbox) for tid, bbox in annot.items()]
        tids = [tid for tid in annot]
        
        if boxes:
            boxes = torch.stack(boxes)
            boxes = clip_boxes_to_image(boxes, (H, W))
            boxes_x = boxes[:, 0::2]
            boxes_y = boxes[:, 1::2]

            # remove gt with zero length in either x, y dim
            len_x = boxes_x[:, 1] - boxes_x[:, 0]
            len_y = boxes_y[:, 1] - boxes_y[:, 0]
            valid_len = torch.logical_and(len_x != 0, len_y != 0)
  
            boxes = boxes[valid_len.view(-1)]

            # pad to make the cost matrix square
            pad = len(boxes) - len(det_people)
            if pad > 0:
                import torch.nn.functional as F
                det_people_pad = F.pad(det_people, [0, 0, 0, pad])
            else:
                det_people_pad = det_people

            idx, cost = MMPTrackPairs.match_dets(boxes, det_people_pad, metric='iou')

            # remove pads 
            if pad > 0:
                # keep only unpad dets, feats & ids
                valid_idx = [i for i, idx_val in enumerate(idx) if idx_val < num_people]
                boxes = boxes[valid_idx]

            boxes = xyxy2xywh(boxes.float())
            # norm
            boxes[:, [0, 2]] /= W
            boxes[:, [1, 3]] /= H

            dets_label = coco_port / 'labels' / split / f'{key}.txt'
            dets_label.parent.mkdir(exist_ok=True, parents=True)
            boxes = torch.cat([torch.zeros(len(boxes), 1), boxes], dim=1).tolist()
            
            boxes = [' '.join([str(i) for i in b]) + '\n' for b in boxes]
            with open(dets_label, 'w+') as f:
                f.writelines(boxes)

            split_txt[split].append(f'./images/{split}/{key}.jpg\n')

    for split, split_lst in split_txt.items():
        with open(str(coco_port / f'{split}.txt'), 'w+') as f:
            f.writelines(split_lst)

convert_feat_to_coco()