import os
from glob import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, _generate_color_palette

def show(i, img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    # plt.show()
    plt.imsave('test.jpg', np.transpose(npimg, (1, 2, 0)))

all_videos = glob('/zdata/projects/shared/datasets/MTMC_Tracking_AIC23_Track1/*/*/*/*.mp4')

for i in range(0, 50):
    video_pth = Path(all_videos[i])
    print(video_pth)
    if 'c009' not in str(video_pth):
        continue 
    frames = sorted(list((video_pth.parent / 'frames').glob('*.jpg')))
    label_pth = video_pth.parent / 'label.txt'

    labels = defaultdict(list)
    with open(label_pth, 'r') as f:
        lines = f.readlines()
        for line in lines:
            fid, tid, x, y, w, h, *_ = line.strip().split(',')
            fid, tid, x, y, w, h = int(fid), int(tid), int(x), int(y), int(w), int(h)

            labels[fid].append(torch.tensor([tid, x, y, x+w, y+h]))

    train = False
    train_upto_idx = int(len(frames) * 0.7)
    if train:
        frames = frames[:train_upto_idx]
    else:
        frames = frames[train_upto_idx:]

    # print(list(labels.keys())[:5])
    # print(max(labels), min(labels), len(labels))
    # print(print(label_pth.parent), len(labels), len(frames))
    for frame_id, frame_pth in enumerate(frames):
        label_frame_id = frame_id if train else train_upto_idx + frame_id
        if frame_id % 1 == 0:
            frame = read_image(str(frame_pth))
            if labels[fid]:
                boxes = labels[label_frame_id]
                boxes = torch.stack(boxes) if len(boxes) > 0 else torch.empty(0, 5)
                print(frame_pth, boxes)
                tids = [str(tid.item()) for tid in boxes[:, 0]]
                frame = draw_bounding_boxes(frame, boxes=boxes[:, 1:], labels=tids, colors=_generate_color_palette(len(tids)), width=4)
                show(frame_id, frame)
                print('annot')
            else:
                print('No labels')
    break