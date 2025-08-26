import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.dirname(f"{parentdir}/mcmot"))

import torch
from tqdm import tqdm

from utils.args import parse_args
from data.dancetrack import DanceTrackClipsFull

def collate_fn(batch):

    batch_clips, batch_boxes, batch_tracks, batch_new_segment, batch_frames, batch_metadata = [], [], [], [], [], []

    for clips, boxes, tracks, new_segment, frames, metadata in batch:
        batch_clips.append(clips)
        batch_boxes.append(boxes)
        batch_tracks.append(tracks)
        batch_new_segment.append(new_segment)
        batch_frames.append(frames)
        batch_metadata.append(metadata)

    batch_clips = torch.stack(batch_clips) # B, F, Cm, T, C, H, W
    assert batch_clips.shape[3] == 1
    batch_clips = batch_clips.squeeze(dim=3)  # B, F, Cm, C, H, W
    batch_frames = torch.tensor(batch_frames, dtype=torch.long)

    return batch_clips, batch_boxes, batch_tracks, batch_new_segment, batch_frames, batch_metadata


def _build_dataset(cfg, build_train = False):
    cfg_dataset = cfg.DATASET
    val_dataset = DanceTrackClipsFull(cfg_dataset, split='validation')
    train_dataset = None
    if cfg.CMD == 'train' or build_train:
        train_dataset = DanceTrackClipsFull(cfg_dataset, split='train')
    return val_dataset, train_dataset

def sanity_dataset(cfg):
    val_dataset, train_dataset = _build_dataset(cfg)
    for data in tqdm(val_dataset):
        clips, boxes, tracks, *_ = data
        # print(_)
        # print(clips.shape)
        # print(len(boxes))
        # print(len(tracks))
        # break

if __name__ == '__main__':
    cfg = parse_args()
    sanity_dataset(cfg)
