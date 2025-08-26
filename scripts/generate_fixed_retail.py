"""
Script to generate groundtruth labels for retail scene in MMPtracking dataset with labels from visible people only (remove people boxes behind shelfs)
This is still not fully accurate but much better than trying to overfit model for bounding boxes and tracks without any person in them. 
The fix relies on yolo pretrained model to filter out boxes where people are behind shelf or fully occluded. Note: this may result in removal of 
good ground truth boxes if yolo cannot detect them. 
"""

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

import numpy as np
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_iou

import matplotlib.pyplot as plt
from ml import av
from tqdm import tqdm

import torch
import torch.nn.functional as F
from ml.vision.ops import clip_boxes_to_image
from scipy.optimize import linear_sum_assignment

from data.mmptracking import MMPTrackPairs

root = Path('/zdata/projects/shared/datasets/MMPTracking')
splits = ['train', 'validation']
feats_path = 'dets-yolo7-cls_thresh_0.4-nms_thresh_0.5-feats_DukeMTMC_SBS_S50'
labels_path = 'labels'
H, W = [360, 640]
iou_thresh = 0.01

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


def match_dets(gt_boxes, det_boxes, metric='iou'):
    """
    Match GT and detection boxes and return indices with linear sum assignment
    """
    cost_mat = None
    if metric == 'l1':
        cost_mat = torch.cdist(gt_boxes[:, :4].float(), det_boxes[:, :4].float(), p=1)
    elif metric == 'iou':
        cost_mat = -1 * box_iou(gt_boxes[:, :4], det_boxes[:, :4]) # [N, M]
    else:
        raise ValueError(f'Invalid value for {metric=}')

    row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_mat)
    return row_ind, col_ind, cost_mat


def fix_retail_labels(l):

    with open(l, 'r') as f:
        annot = json.load(f)

    feat_pth = l.replace(labels_path, feats_path).replace('json', 'pth')
    assert Path(feat_pth).exists()
    sample = torch.load(str(feat_pth), map_location='cpu')
    det_people = sample['people'][:, :4]

    """
    annot: {"track_id": xyxy}
    """
    boxes = [torch.tensor(bbox) for bbox in annot.values()]
    tids = torch.tensor([int(tid) for tid in annot])

    fixed_labels = {}
        
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
        tids = tids[valid_len.view(-1)]

        row_idx, col_idx, cost = match_dets(boxes, det_people, metric='iou')

        tids = tids[row_idx]
        boxes = boxes[row_idx]
        filtered_iou = (-1 * cost[row_idx, col_idx]) > iou_thresh
        # filter if iou of gt and det is less than iou_thresh
        tids = tids[filtered_iou]
        boxes = boxes[filtered_iou]

        assert len(boxes) == len(tids)
        fixed_labels = {str(tid.item()): box.tolist() for tid, box in zip(tids, boxes)}

    return fixed_labels, det_people

for split in splits:
    print(f'Working on {split}')
    split_path = root / split
    split_img_path = split_path /labels_path
    labels = list(split_img_path.glob('*/*/*.json'))
    labels = list(filter(lambda x: 'retail' in x, [str(l) for l in labels]))
    print(f'Found {len(labels)=} in retail')
    for l in tqdm(labels):
        fixed_labels, det_people = fix_retail_labels(l)
        frame_path = l.replace(labels_path, 'images').replace('json', 'jpg')
        image = read_image(frame_path)
        ids = list(fixed_labels.keys())
        boxes = torch.tensor(list(fixed_labels.values()))
        boxes = torch.cat([boxes, det_people[:, :4]], dim=0)
        colors = [av.utils.rgb(int(tid), integral=True) for tid in ids] + [(255, 255, 255) for _ in range(len(det_people))]

        labels = ids + ['det' for _ in range(len(det_people))]

        # for i, box in enumerate(boxes):
        #     x1, y1, x2, y2 = box.int().tolist()
        #     write_jpeg(image[:, y1:y2, x1:x2], f'scripts/outputs/person-{cam_id}-{i}.jpg')

        from torchvision.utils import draw_bounding_boxes
        image = draw_bounding_boxes(image, boxes, colors=colors, labels=labels, fill=True)
        show(image)


