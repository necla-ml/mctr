import os
import re
import csv
import json
import uuid
import math
import random
import operator
import itertools
from pathlib import Path
from collections import defaultdict
import copy

# third party
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, Sampler
from torchvision.io import read_image, write_jpeg, ImageReadMode

import numpy as np
from ml.vision.ops import box_iou, clip_boxes_to_image
from scipy.optimize import linear_sum_assignment

from utils.misc import is_main_process
from .box_ops import box_xyxy_to_cxcywh

numbers = re.compile(r'(\d+)')
OPERATORS = [operator.add, operator.sub]
TORCH_2_CV = lambda img: img.permute(1, 2, 0).numpy()[:, :, ::-1] # rgb to bgr, chw to hwc
CV_2_TORCH = lambda img: torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1) # bgr to rgb, hwc to chw

def numerical_sort(value):
    parts = numbers.split(str(value))
    parts[1::2] = map(int, parts[1::2])
    return parts

def class_weights(samples):
    """
    Args:
        samples: Tensor(N)
            Tensor of len(samples) with labels as the value
    Returns:
        weights for each sample
    """
    class_sample_count = torch.tensor([len(torch.where(samples == t)[0]) for t in torch.unique(samples)])
    class_weight = 1. / class_sample_count
    samples_weight = torch.tensor([class_weight[t] for t in samples])
    return samples_weight, class_weight

def box_center_ground_plane(box):
    """
    Center of the bounding box ([x1 + x2] / 2) at the lowest point (y2)

    Args:
        box: Tensor(4)
            x1, y1, x2, y2
    Returns:
        np.array(1, 3)
    """
    return {'X': 0.5 * (box[2] + box[0]), 'Y': box[3]} # bottom-center


class MMPTrackClipsFullBatchSampler(Sampler):
    def __init__(self, batch_size: int, p_reset, data_size, iterations_per_epoch = 1000):
        self.batch_size = batch_size
        self.p_reset = p_reset
        self.data_size=data_size
        self.iterations_per_epoch = iterations_per_epoch
        self.idx = torch.randint(self.data_size,(self.batch_size,))
    def set_p_reset(self, p_reset):
        self.p_reset = p_reset
    def __iter__(self):
        it = 0
        while it < self.iterations_per_epoch:
            self.idx += 1
            reset = torch.logical_or((self.idx == self.data_size),(torch.rand(self.batch_size) < self.p_reset))
            self.idx[reset] = torch.randint(self.data_size, (reset.sum(),))
            it += 1
            # this deepcopy is needed here to be compatible with multiprocessing. 
            # Othwerwise the batch indices would be overwritten before
            # the worker processes would have teh chance to fetch the data
            ret = copy.deepcopy(self.idx)
            yield ret
    def __len__(self):
        return self.iterations_per_epoch #self.data_size//self.batch_size


class MMPTrackClipsFull(Dataset):
    def __init__(self, cfg_dataset, split='train'):
        super().__init__()

        self.root = Path(cfg_dataset.ROOT)
        assert self.root.exists(), f'Provided ROOT: {self.root} does not exist'
        self.split = split
        self.train = split == 'train'
   
        self.frames_path = cfg_dataset.FRAMES_PATH
        self.frames_ext = cfg_dataset.FRAMES_EXT

        self.labels_path = cfg_dataset.LABELS_PATH
        self.labels_ext = cfg_dataset.LABELS_EXT

        if cfg_dataset.TOPDOWN_PATH:
            self.topdown_path = cfg_dataset.TOPDOWN_PATH
            self.topdown_ext = cfg_dataset.TOPDOWN_EXT
            self.topdown_max = cfg_dataset.TOPDOWN_MAX
        else:
            self.topdown_path = None

        self.cams = cfg_dataset.CAMS
        self.frame_ranges = cfg_dataset.FRAME_RANGES
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
            # NOTE: changing here for greyscale vs rgb comparison for ablations
            self.augument = lambda x: x # T.Grayscale(num_output_channels=3) # 

        self.samples, self.frames_per_scene_per_cam, self.ncams = self._get_samples_path()

    def _get_samples_path(self):
        print(f'\nCollecting samples from {self.split} set', flush=True)
        frames_per_scene_per_cam = defaultdict(lambda: defaultdict(int))
        frames_per_scene = defaultdict(lambda: set())
        cams_per_scene = defaultdict(lambda: set())
        samples_per_scene = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
        split_path = self.root / self.split / self.frames_path
        for name, values in self.locations[self.split].items():
            # name: 64am, 63pm, etc.
            if values == None:
                continue
            if values == '*':
                # all subdirs
                # filter out .zip or other files that are not sub directories
                valid_dirs = [v.name for v in filter(lambda x: not x.is_file(), list((split_path / name).glob('*')))]
            elif isinstance(values, str) and len(values) > 1:
                # NOTE: avoid eval on wrong gt
                # XXX: hard coded to avoid validation path with wrong gt
                valid_dirs = [v.name for v in filter(lambda x: not x.is_file(), list((split_path / name).glob('*'))) if values in str(v) and '64pm/industry_safety_1' not in str(v)]
                # if self.train:
                #     print('Using only 1 clip')
                #     valid_dirs = valid_dirs[:2]
                print(f'Working with {valid_dirs=}')
            else:
                # subset 
                valid_dirs = values                
    
            for v in valid_dirs:
                # v: scene_name_0 e.g lobby_0, industry_safety_3
                scene_name = v.split('_') # lobby, 0
                scene_name, scene_no = '_'.join(scene_name[:-1]), scene_name[-1]
                pth = split_path / name / v
                assert pth.exists(), f'Provided {pth} does not exist'

                frames_path_lst = list(pth.glob(f'*.{self.frames_ext}'))
                frames_path_lst.sort(key=numerical_sort)

                for frame_path in frames_path_lst:
                    cam_id = int(os.path.splitext(frame_path.name)[0].split('_')[-1])
                    frame_id = int(frame_path.name.split('_')[1])
                    if self.cams[scene_name] is not None and cam_id not in self.cams[scene_name]:
                        # skip if cam not in specified
                        continue
                    if self.frame_ranges[scene_name] is not None and ( frame_id < self.frame_ranges[scene_name][0] or frame_id > self.frame_ranges[scene_name][1] ):
                        # skip if frame not in specified range
                        continue
                    scene_id = f'{name}_{scene_name}_{scene_no}'
                    cam_sample = {
                            'scene_id': scene_id, 
                            'cam_no': cam_id, 
                            'frame_no': frame_id, 
                            'frame_path': frame_path
                    } 
                    frames_per_scene_per_cam[scene_id][cam_id] += 1
                    frames_per_scene[scene_id].add(frame_id)
                    cams_per_scene[scene_id].add(cam_id)
                    samples_per_scene[scene_id][frame_id][cam_id]=cam_sample                        
                frames_per_scene[scene_id] = list(frames_per_scene[scene_id])
                frames_per_scene[scene_id].sort()
                cams_per_scene[scene_id] = list(cams_per_scene[scene_id])
                cams_per_scene[scene_id].sort()

        ncams = -1

        samples = []
        for scene, scene_frames in samples_per_scene.items():
            # cameras numbers should match between scenes. 
            # this assumes that a simple sort will get matching cameras in the same positions. 
            cams = cams_per_scene[scene]
            if ncams == -1:
                ncams = len(cams)
            assert len(cams) == ncams, "There must be the same nubmer of cameras in each scene"
            for start_frame in range(0, len(scene_frames)- self.frame_stride*self.frames_per_sample + 1, self.subsample_rate):
                frames = frames_per_scene[scene][slice(start_frame, start_frame+self.frame_stride*self.frames_per_sample, self.frame_stride)]
                # TO DO: deal with the possibility that the data is missing for a particular frame or cam
                samples.append({"data":[[scene_frames[f][c] for c in cams] for f in frames],
                                "new_segment": start_frame == 0,
                                "frames": frames})

        print(f'Found {len(samples)=}')

        return samples, frames_per_scene_per_cam, ncams

    def get_frame_idx(self, latest_idx, sample_length, sample_rate, num_frames):
        """
        Sample frame indexes given current index, sample length and rate
        """
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

        #frame_seq = self.get_frame_idx(frame_no, self.frame_stride * self.num_frames, self.frame_stride, total_frames)
        frame_seq = self.get_frame_idx(frame_no, 1 * self.num_frames, 1, total_frames)

        clip = []
        for frame_no in frame_seq:
            # rgb_00955_2: rgb + frame_id + cam_id
            fp = frame_path.parent / f'rgb_{str(frame_no).zfill(5)}_{cam_no}.{self.frames_ext}'
            assert fp.exists(), f'Missing {fp}'
            frame = read_image(str(fp), mode=ImageReadMode.RGB)
            clip.append(frame)

        # get track_ids 
        label_path = str(frame_path).replace(self.frames_path, self.labels_path).replace(self.frames_ext, self.labels_ext)

        if self.topdown_path is not None:
            topdown_path = str(frame_path).replace(self.frames_path, self.topdown_path).replace(self.frames_ext, self.topdown_ext).replace("rgb_","topdown_")
            topdown_path = re.sub(f"_[0-9]*\.{self.topdown_ext}$", f".{self.topdown_ext}", topdown_path)
        
        # FIXME: hard coded stuff 
        # NOTE: hack to avoid invisible people (behind the shelf) from messing up training 
        if self.train and 'retail' in label_path:
            label_path = label_path.replace(self.labels_path, 'fixed_labels')

        with open(label_path, 'rb') as f:
            annots = json.load(f)

        
        if self.topdown_path is not None:
            with open(topdown_path, 'r') as f:
                topdown_labels = {}
                csv_reader = csv.reader(f, delimiter=',')
                for row in csv_reader:
                    topdown_labels[int(row[0])] = torch.tensor([int(float(row[1])),int(float(row[2]))],dtype=torch.float)

        tracks = []
        track_ids = []
        track_topdown = []
        for tid, bbox in annots.items():
            tracks.append(torch.tensor(bbox, dtype=torch.float))
            track_ids.append(torch.tensor(float(tid), dtype=torch.float))
            if self.topdown_path is not None:
                track_topdown.append(topdown_labels[int(tid)])

        if tracks:
            tracks = torch.stack(tracks) # N, 4
            track_ids = torch.stack(track_ids) # N, 1
            if self.topdown_path is not None:
                track_topdown = torch.stack(track_topdown) # N, 2
        else:
            tracks = torch.empty(0, 4)
            track_ids = torch.empty(0)
            track_topdown = torch.empty(0, 2)

        # fix negative and zero length box coords
        tracks = clip_boxes_to_image(tracks, size=self.frame_shape)
        boxes_x = tracks[:, 0::2]
        boxes_y = tracks[:, 1::2]

        # remove gt with zero length in either x, y dim
        len_x = boxes_x[:, 1] - boxes_x[:, 0]
        len_y = boxes_y[:, 1] - boxes_y[:, 0]
        valid_len = torch.logical_and(len_x != 0, len_y != 0)

        tracks = tracks[valid_len, :]
        track_ids = track_ids[valid_len]
        if self.topdown_path:
            track_topdown = track_topdown[valid_len,:]
        
        clip = torch.stack(clip)
        H, W = clip.shape[-2:]

        # xyxy to cxcywh
        # normalize by H, W
        tracks[:, [0, 2]] /= W 
        tracks[:, [1, 3]] /= H
        tracks = box_xyxy_to_cxcywh(tracks)

        if self.topdown_path is not None:
            # topdown labels to [0,1] by dividing by topdown_max
            track_topdown /= self.topdown_max
            tracks = torch.cat([track_ids[:, None], tracks, track_topdown], dim=-1)
        else:
            tracks = torch.cat([track_ids[:, None], tracks], dim=-1)

        # TODO: transform clip/tracks
        clip = self.transform(clip)

        return clip, tracks

    def __getitem__(self, index):

        sample = self.samples[index]
        data = [[self.load_clip(s) for s in smpl] for smpl in sample["data"]]

        targets = [[{'tracks':d[1][:,0], 'boxes':d[1][:,1:5]} for d in dta] for dta in data]
        if self.topdown_path is not None:
            [[t.update({'topdown':d[1][:,5:7]}) for t,d in zip(tgt,dta)] for tgt, dta in zip(targets, data)]

        clips = torch.stack([torch.stack([d[0] for d in dta]) for dta in data])
        # apply the same augumentations to all clips 
        F,Cm,T,C,H,W = clips.shape
        clips = self.augument(clips.view(-1, C,H,W)).view(F,Cm,T,C,H,W)
        return clips, targets, sample["new_segment"], sample["frames"], sample

    def __len__(self):
        return len(self.samples)

