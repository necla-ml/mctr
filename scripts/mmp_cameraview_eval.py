import os
import json
import argparse
from collections import defaultdict, OrderedDict

import pandas as pd
import motmetrics as mm

def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
        names.append(k)

    return accs, names


class LabelReader:
    def __init__(self, label_dir, num_cam, stride=1):
        self._label_dir = label_dir
        self._num_frames = len([name for name in os.listdir(self._label_dir) if os.path.isfile(os.path.join(self._label_dir, name))]) // num_cam
        self._num_cam = num_cam
        self.stride = stride
    
    def read_single_frame(self, frame_id):
        labels = {}
        for cam_id in range(1, self._num_cam+1):
            raw_labels = json.load(open(os.path.join(self._label_dir, 'rgb_'+str(frame_id).zfill(5)+'_'+str(cam_id)+'.json')))
            raw_list = []
            for gt_id, (x_min, y_min, x_max, y_max) in raw_labels.items():
                raw_list.append({'FrameId':frame_id, 'Id':int(gt_id), 'X':int(float(x_min)), 'Y':int(float(y_min)), 'Width':int(float(x_max-x_min)), 'Height':int(float(y_max-y_min)), 'Confidence':1.0})
            labels[cam_id] = raw_list
        return labels
    
    def read(self):
        rows_list = defaultdict(list)
        for i in range(0, self._num_frames, self.stride):
            single_frame_labels = self.read_single_frame(i)
            for cam_id in range(1, self._num_cam+1):
                rows_list[cam_id].extend(single_frame_labels[cam_id])
        return OrderedDict([(cam_id, pd.DataFrame(rows).set_index(['FrameId', 'Id']).sort_index()) for cam_id, rows in rows_list.items()]) 


class PredReader:
    def __init__(self, filename, gt_json, stride=1):
        self.stride = stride
        self.filename = filename
        self.gt_json = gt_json

    def read_json(self):
        with open(self.filename, 'rb') as f:
            results = json.load(f)

        with open(self.gt_json, 'rb') as f:
            dets = json.load(f)

        all_dets = {}
        for pth, det in dets.items():
            for det_id, det_meta in det['dets_meta'].items():
                all_dets[det_id] = det_meta

        dets_per_frame_per_cam = defaultdict(list)
        for det_id, det_data in results.items():
            # get all dets per cam per frame
            cam_id = det_data['camera']
            frame_id = det_data["frame"]
            if frame_id % self.stride != 0:
                continue
            x_min, y_min, x_max, y_max = all_dets[det_id.split('_')[-1]]['gt_bbox'][:4] # XXX: use gt bbox
            # x_min, y_min, x_max, y_max = det_data['bb'][:4] # XXX: use yolo bbox
            score = det_data['bb'][-2]
            pt = det_data['pred_track'] # XXX: use pred tracks
            # pt = det_data['true_track'] # XXX: use gt tracks
            if int(pt) == 0:
                # XXX: skip detection estimated as false positive
                continue
            dets_per_frame_per_cam[cam_id].append({'FrameId':frame_id, 'Id':int(pt), 'X':int(float(x_min)), 'Y':int(float(y_min)), 'Width':int(float(x_max-x_min)), 'Height':int(float(y_max-y_min)), 'Confidence':1.0})

        return dets_per_frame_per_cam
    
    def read(self):
        rows_list = self.read_json()
        return OrderedDict([(cam_id, pd.DataFrame(rows).set_index(['FrameId', 'Id']).sort_index()) for cam_id, rows in rows_list.items()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path', default=f'/net/mlfs01/export/users/alex/Research/mcmot/cafe_shop_0_test/results/predicted_tracks.json', help='Path to input json file')
    args = parser.parse_args()

    site = 'cafe_shop'
    site_no = '0'
    site_dir = f'{site}_{site_no}'
    num_cam = 4
    gt_json = '/zdata/users/dpatel/mcmot/validation_cafe_dets.json'

    pred_path = args.pred_path
    gt_dfs = LabelReader(f'/zdata/projects/shared/datasets/MMPTracking/validation/labels/64pm/{site_dir}', num_cam=num_cam, stride=5).read()
    pred_dfs = PredReader(pred_path, gt_json=gt_json, stride=1).read()
    
    # print(gt_dfs[1].tail(n=5))
    # print(pred_dfs[1].tail(n=5))
    # for i in range(1, 5):
    #     gt_dfs[i] = gt_dfs[i].drop(index=[7520, 7515, 7510, 7505, 7500])
        # print(gt_dfs[i].last_valid_index(), pred_dfs[i].last_valid_index())
        # print(pred_dfs[i].index.difference(gt_dfs[i].index))
        # print(pred_dfs[i].last_valid_index(), gt_dfs[i].last_valid_index())
        
    accs, names = compare_dataframes(gt_dfs, pred_dfs)

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics + ['num_frames'], generate_overall=True)

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)