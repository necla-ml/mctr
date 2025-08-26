import os
import sys
import inspect
import logging
import subprocess
from time import time
from glob import glob
from pathlib import Path
from argparse import ArgumentParser

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# third party
import torch as th

from ml.ws.common import ParallelExecutor

logging.getLogger(__file__).setLevel('INFO')

CONTAINERS = ['mp4', 'avi', 'MOV']

def process_v2f(video, out_fmt='jpg'):
    video = str(video)
    video_pth = Path(video)
    name = os.path.basename(os.path.splitext(video_pth)[0])
    output = f'{video_pth.parent}/frames'
    Path(output).mkdir(parents=True, exist_ok=True)
    ffmpeg = ['ffmpeg', '-y', '-i', video, f"{output}/{name}_%06d.{out_fmt}"]
    print(' '.join(ffmpeg), flush=True)
    with open(os.devnull, 'w') as devnull:
        try:
            subprocess.run(ffmpeg, stdout=devnull, stderr=devnull, check=True)
        except subprocess.CalledProcessError as e:
            return video, e

def v2f():
    r"""
    Extract video frames at a specified FPS using FFmpeg in parallel.
    """
    video_pths = glob('/zdata/projects/shared/datasets/MTMC_Tracking_AIC23_Track1/*/*/*/*.mp4')
    total = len(video_pths)
    failed = []

    t = time()
    pe = ParallelExecutor(cpu_bound=True, max_workers=th.get_num_threads())
    tasks = [(process_v2f, video_pth) for video_pth in video_pths]
    results = pe.run(tasks)
    for res in results:
        if res is not None:
            video, e = res
            failed.append(res)
            logging.error(f"Failed to extract frames from {video}: {e.stderr}{e.stdout}")
    elapse = time() - t

    print(f"Done processing {total-len(failed)}/{total} videos for {elapse:.3f}s, {elapse/len(args.path):.3f}s/video")

if __name__ == '__main__':
    v2f()
