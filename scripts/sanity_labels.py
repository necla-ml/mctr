import json
from pathlib import Path

import numpy as np

import torch
from torchvision.io import read_image, write_jpeg
from torchvision.utils import draw_bounding_boxes, make_grid

import matplotlib.pyplot as plt
from ml import av

FONT_SIZE = 30
FONT_PTH = '/zdata/users/dpatel/Phetsarath_OT.ttf'

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.imsave(f"savedimval.png", np.transpose(npimg, (1, 2, 0)))
    plt.show()

cams = [1, 2, 3, 4, 5, 6]
stride = 500
total_frames = 5 * stride
images = Path('/zdata/projects/shared/datasets/MMPTracking/validation/images/64pm/retail_0')

for frame_no in range(10, total_frames, stride):
    out_img = []
    for cam_id in cams:
        frame_path = str(images / f'rgb_{str(frame_no).zfill(5)}_{cam_id}.jpg')
        # frame_path = str(images / f'rgb_06520_1.jpg')
        label_path = frame_path.replace('images', 'labels').replace('jpg', 'json')
        image = read_image(frame_path)
        with open(label_path, 'r') as f:
            labels = json.load(f)
        ids = list(labels.keys())
        boxes = torch.tensor(list(labels.values()))
        colors = [av.utils.rgb(int(tid), integral=True) for tid in ids]

        # for i, box in enumerate(boxes):
        #     x1, y1, x2, y2 = box.int().tolist()
        #     write_jpeg(image[:, y1:y2, x1:x2], f'scripts/outputs/person-{cam_id}-{i}.jpg')

        from torchvision.utils import draw_bounding_boxes
        image = draw_bounding_boxes(image, boxes, colors=colors, labels=ids, width=4, font=FONT_PTH, font_size=FONT_SIZE)
        out_img.append(image)

    out_grid = make_grid(out_img, nrow=3)
    show(out_grid)
    break

