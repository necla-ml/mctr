import os
import re
import json
import operator
import itertools
from pathlib import Path
from collections import defaultdict

# third party
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, Sampler
from torchvision.io import read_image, write_jpeg, ImageReadMode

import numpy as np
from ml.vision.ops import clip_boxes_to_image

from .box_ops import box_xyxy_to_cxcywh

numbers = re.compile(r'(\d+)')
OPERATORS = [operator.add, operator.sub]
TORCH_2_CV = lambda img: img.permute(1, 2, 0).numpy()[:, :, ::-1] # rgb to bgr, chw to hwc
CV_2_TORCH = lambda img: torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1) # bgr to rgb, hwc to chw

def numerical_sort(value):
    parts = numbers.split(str(value))
    parts[1::2] = map(int, parts[1::2])
    return parts

class MTMCClipsFull(Dataset):
    def __init__(self, cfg_dataset, split='train'):
        super().__init__()

        self.root = Path(cfg_dataset.ROOT)
        assert self.root.exists(), f'Provided ROOT: {self.root} does not exist'
        self.split = split
        self.train = split == 'train'
        self.train_percent = cfg_dataset.TRAIN_PERCENT
   
        self.frames_path = cfg_dataset.FRAMES_PATH
        self.frames_ext = cfg_dataset.FRAMES_EXT

        self.labels_path = cfg_dataset.LABELS_PATH

        self.cams = cfg_dataset.CAMS
        self.locations = cfg_dataset.LOCATIONS
        self.num_frames = cfg_dataset.NUM_FRAMES # number of frames for 3D Deter models. Should be 1. 
        self.frames_per_sample = cfg_dataset.FRAMES_PER_SAMPLE or 1 # how many frames to include in one sample 
        self.frame_stride = cfg_dataset.FRAME_STRIDE or 1
        self.frame_shape = cfg_dataset.FRAME_SHAPE
        self.subsample_rate = cfg_dataset.SUBSAMPLE_RATE or 1 # create new samples every X frames. Similar to frame stride, but this is used when creating new samples. 
                                                              # e.g. if frame stride = 5 and and subsample rate = 1 then 
                                                              # an instance will contain frames 0 5 10 etc. but the next instance will
                                                              # contain frames 1 6 11 etc.


        normalize = T.Compose([
            T.Resize(size=self.frame_shape),
            T.Lambda(lambda x: x.float().div(255.0)),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform = T.Compose([normalize])

        self.augumentations = [T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2, saturation=0.2),
                               T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                               T.RandomAdjustSharpness(sharpness_factor=2,p=0.5),
                               T.RandomGrayscale(p=1),
                               T.RandomAutocontrast(p=0.5)]

        if split == 'train' and cfg_dataset.AUGUMENTATIONS:
            self.augument = T.RandomChoice(self.augumentations)
        else:
            self.augument = lambda x : x

        self.samples, self.frames_per_scene_per_cam, self.ncams = self._get_samples_path()

    def _get_samples_path(self):
        print(f'\nCollecting samples from {self.split} set', flush=True)
        frames_per_scene_per_cam = defaultdict(lambda: defaultdict(int))
        frames_per_scene = defaultdict(lambda: set())
        cams_per_scene = defaultdict(lambda: set())
        samples_per_scene = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        # FIXME: hardcoded split to train. 
        # XXX: use training data for both train and val
        split_path = self.root / 'train'
        # always use train set
        # XXX: added list of self.locations['train] so that only single scene sequence can be specified from the config
        locations = self.locations['train']
        locations = isinstance(locations, str) and [locations] or locations
        for scene_name in locations:
            scene_id = scene_name
            for cam_path in (split_path / scene_name).glob('*'): 
                if cam_path.is_file():
                    print(f'SKIPPING: {cam_path}')
                    continue

                frames_path_lst = list((cam_path / self.frames_path).glob(f'*.{self.frames_ext}'))
                frames_path_lst.sort(key=numerical_sort)

                # c020, c021 -> 21, 22
                cam_id = int(cam_path.name.replace('c', ''))

                cam_labels = defaultdict(list)
                labels_pth = cam_path / self.labels_path
                with open(str(labels_pth), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        fid, tid, x, y, w, h, *_ = line.strip().split(',')
                        fid, tid, x, y, w, h = int(fid), int(tid), int(x), int(y), int(w), int(h)
                        # xywh to xyxy
                        cam_labels[fid].append(torch.tensor([tid, x, y, x+w, y+h]))

                # XXX: workaround to split dynamically from train set 
                train_upto_idx = int(len(frames_path_lst) * self.train_percent)
                if self.train:
                    frames_path_lst = frames_path_lst[:train_upto_idx]
                else:
                    frames_path_lst = frames_path_lst[train_upto_idx:]

                for frame_id, frame_path in enumerate(frames_path_lst):
                    label_frame_id = frame_id if self.train else train_upto_idx + frame_id
                    if self.cams and self.cams[scene_name] is not None and cam_id not in self.cams[scene_name]:
                        # skip if cam not in specified
                        # print(f'Skipping scene: {scene_name}-cam: {cam_id} not specified in config')
                        continue
                    
                    frame_id = label_frame_id
                    cam_sample = {
                            'scene_id': scene_id, 
                            'cam_no': cam_id, 
                            'frame_no': frame_id, 
                            'frame_path': frame_path,
                            'frame_labels': cam_labels[label_frame_id]
                    } 
                    frames_per_scene_per_cam[scene_id][cam_id] += 1
                    frames_per_scene[scene_id].add(frame_id)
                    cams_per_scene[scene_id].add(cam_id)
                    samples_per_scene[scene_id][frame_id][cam_id] = cam_sample    

            frames_per_scene[scene_id] = list(frames_per_scene[scene_id])
            frames_per_scene[scene_id].sort()
            cams_per_scene[scene_id] = list(cams_per_scene[scene_id])
            cams_per_scene[scene_id].sort()

        ncams = -1

        samples = []
        for scene, scene_frames in samples_per_scene.items():
            cams = cams_per_scene[scene]
            print(f'For {scene=} found {cams=}')
            if ncams == -1:
                ncams = len(cams)
            assert len(cams) == ncams, "There must be the same number of cameras in each scene"
            for start_frame in range(0, len(scene_frames)- self.frame_stride * self.frames_per_sample + 1, self.subsample_rate):
                frames = frames_per_scene[scene][slice(start_frame, start_frame + self.frame_stride * self.frames_per_sample, self.frame_stride)]
                # TODO: deal with the possibility that the data is missing for a particular frame or cam
                samples.append({"data":[[scene_frames[f][c] for c in cams] for f in frames],
                                "new_segment": start_frame == 0,
                                "frames": frames})

        print(f'Found {len(samples)=}')

        return samples, frames_per_scene_per_cam, ncams

    def get_frame_idx(self, latest_idx, sample_length, sample_rate, num_frames):
        """
        Sample frame indexes given current index, sample length and rate
        """
        # 14005, 1, 1 6000, 
        seq = list(range(latest_idx, latest_idx - sample_length, -sample_rate))
        seq.reverse()
        for seq_idx in range(len(seq)):
            if seq[seq_idx] < 0:
                seq[seq_idx] = 0
            elif seq[seq_idx] >= num_frames:
                seq[seq_idx] = num_frames - 1

        return seq

    def load_clip(self, sample):
        
        frame_path, cam_no, frame_no = sample['frame_path'], sample['cam_no'], sample['frame_no']
        scene_id = sample['scene_id']
        total_frames = self.frames_per_scene_per_cam[scene_id][cam_no]

        # NOTE: nvidia frames starts with idx 1
        fp = frame_path.parent / f'video_{str(frame_no + 1).zfill(6)}.{self.frames_ext}'
        assert fp.exists(), f'Missing {fp}'
        frame = read_image(str(fp), mode=ImageReadMode.RGB)
        clip = [frame]

        clip = torch.stack(clip)
        H, W = clip.shape[-2:]

        tracks = sample['frame_labels']
        track_ids = [t[0] for t in tracks]
        tracks = [t[1:] for t in tracks]

        if tracks:
            tracks = torch.stack(tracks).float() # N, 4
            track_ids = torch.stack(track_ids).float() # N, 1
        else:
            tracks = torch.empty(0, 4)
            track_ids = torch.empty(0)

        # fix negative and zero length box coords
        tracks = clip_boxes_to_image(tracks, size=[H,W])
        boxes_x = tracks[:, 0::2]
        boxes_y = tracks[:, 1::2]

        # remove gt with zero length in either x, y dim
        len_x = boxes_x[:, 1] - boxes_x[:, 0]
        len_y = boxes_y[:, 1] - boxes_y[:, 0]
        valid_len = torch.logical_and(len_x != 0, len_y != 0)

        tracks = tracks[valid_len]
        track_ids = track_ids[valid_len]

        # xyxy to cxcywh
        # normalize by H, W
        tracks[:, [0, 2]] /= W 
        tracks[:, [1, 3]] /= H
        tracks = box_xyxy_to_cxcywh(tracks)
        tracks = torch.cat([track_ids[:, None], tracks], dim=-1)

        # TODO: transform clip/tracks
        clip = self.transform(clip)

        return clip, tracks

    def __getitem__(self, index):
        sample = self.samples[index]
        data = [[self.load_clip(s) for s in smpl] for smpl in sample["data"]]
        boxes = [[{'boxes':d[1][:,1:]} for d in dta] for dta in data]
        tracks = [[{'tracks':d[1][:,0]} for d in dta] for dta in data]
        clips = torch.stack([torch.stack([d[0] for d in dta]) for dta in data])
        # apply the same augumentations to all clips 
        F, Cm, T, C, H, W = clips.shape
        clips = self.augument(clips.view(-1, C,H,W)).view(F,Cm,T,C,H,W)
        return clips, boxes, tracks, sample["new_segment"], sample["frames"], sample

    def __len__(self):
        return len(self.samples)
