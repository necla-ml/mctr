import json
from pathlib import Path
from collections import defaultdict, OrderedDict

import av
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, make_grid

from ml.vision.utils import rgb

FONT_SIZE = 30
FONT_PTH = '/zdata/users/dpatel/Phetsarath_OT.ttf'

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()

def plot(frame, bboxes, labels, colors):
    frame = draw_bounding_boxes(frame, bboxes, labels=labels, colors=colors, fill=False, width=4, font=FONT_PTH, font_size=FONT_SIZE)
    return frame

results = '/net/mlfs01/export/users/alex/Research/mcmot/cafe_shop_0_test/results/predicted_tracks.json'
frames_root = f'/zdata/projects/shared/datasets/MMPTracking/validation/images/64pm/cafe_shop_0'
cams = [1, 2, 3, 4]
frame_stride = 1

fps = 15
h, w = [360, 640]
output_video_file = 'cafe_shop_0.mp4'

# init video stream
media = av.open(output_video_file, 'w')
video = media.add_stream('h264', fps)
video.height, video.width = 720, 1280
video.bit_rate = 1000000
video = media.streams[0]

with open(results, 'rb') as f:
    results = json.load(f)

dets_per_frame_per_cam = defaultdict(lambda: defaultdict(dict))
for det, det_data in results.items():
    # get all dets per cam per frame
    cam_id = det_data['camera']
    frame_id = f'rgb_{str(det_data["frame"] * frame_stride).zfill(5)}_{cam_id}.jpg'
    dets_per_frame_per_cam[cam_id][frame_id][det] = det_data

# verify equal frames in all cams
assert all([len(dets_per_frame_per_cam[c]) == len(dets_per_frame_per_cam[cams[0]]) for c in cams])

for c in cams:
    # order by frame
    dets_per_frame_per_cam[c] = OrderedDict(sorted(dets_per_frame_per_cam[c].items()))

cam_dets = list(zip(*[dets_per_frame_per_cam[c].items() for c in cams]))
for idx, frame_dets in tqdm(enumerate(cam_dets), total=len(cam_dets)):
    # frame_dets: [(cam1_frame1, dets), (cam2_frame1, dets), ....]

    across_cams = []
    for f in frame_dets:
        f_id = f[0]
        f_dets = f[1]
        
        frame_path = Path(frames_root) / f_id
        assert frame_path.exists()
        frame = read_image(str(frame_path))
        
        bboxes = []
        tts = []
        pts = []
        for det in f_dets.values():
            bboxes.append(torch.tensor(det['bb']))
            tts.append(det['true_track'])
            pts.append(det['pred_track'])

        if bboxes:
            # plot track on frame
            bboxes = torch.stack(bboxes)[:, :4]
            colors = [rgb(int(t_id), integral=True) for t_id in pts]
            # tts = [f'{t_id}' for t_id in tts]
            pts = [f'{t_id}' for t_id in pts]

            frame_pt = plot(frame, bboxes, labels=pts, colors=colors)
            frame_grid = frame_pt
        else:
            frame_grid = frame
        
        across_cams.append(frame_grid)

    across_cams = make_grid(across_cams, nrow=len(cams) // 2)

    # encode/mux rendered frame
    frame = av.VideoFrame.from_ndarray(across_cams.permute(1, 2, 0).numpy(), format='rgb24')
    packets = video.encode(frame)
    media.mux(packets)
    # show(across_cams)

# output video
packets = video.encode(None)
if packets:
    media.mux(packets)
media.close()


