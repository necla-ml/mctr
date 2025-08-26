import os
import sys
import shlex
import inspect
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

"""
The column of csv file is: cid(camera id), fid(frame id), pid(person id), tl_x, tl_y, w, h
How to run?
python evalMCTA.py --gt_path PATH/TO/GT --pred_path PATH/TO/PRED
"""

pred_dir = 'scripts/eval_outputs/trackers/mmptracking/MOT17-train/exp415-PAIRWISE-train-R50-cipr-gpu11_Oct04_201221/data'
pred_files = Path(pred_dir).glob('*.txt')

gt_dir = Path('scripts/eval_outputs/gt/mmptracking/MOT17-train/')
gt_files = Path(gt_dir).glob('*')

cam_preds = defaultdict(lambda: defaultdict())
cam_gt = defaultdict(lambda: defaultdict())

for pf in pred_files:
    # pf: 64pm_industry_safety_0_1.txt
    pf_file = os.path.splitext(pf.name)[0]
    sc = pf_file.split('_')
    clip_name, cam_id = '_'.join(sc[:-1]), sc[-1]
    with open(str(pf), 'r') as f:
        # frame_id, tid, xl, yl, h, w, conf, -1, -1, -1
        file_content = [','.join([str(cam_id)] + [str(int(float(v))) for v in f.strip().split(',')[:6]])  + '\n' for f in f.readlines()]
    cam_preds[clip_name][cam_id] = file_content

    gf = gt_dir / f'{clip_name}_{cam_id}' / 'gt/gt.txt'
    with open(str(gf), 'r') as f:
        # frame_id, tid, xl, yl, h, w, conf, -1, -1, -1
        file_content = [','.join([str(cam_id)] + [str(int(float(v))) for v in f.strip().split(',')[:6]]) + '\n' for f in f.readlines()]
    cam_gt[clip_name][cam_id] = file_content

with tempfile.TemporaryDirectory() as tmp:
    for clip_name, cam_dict in cam_preds.items():
        gt_path = Path(tmp) / clip_name / 'gts'
        preds_path = Path(tmp) / clip_name / 'preds'
        gt_path.mkdir(parents=True, exist_ok=True)
        preds_path.mkdir(parents=True, exist_ok=True)
        for cam_id, preds in cam_dict.items():
            gts = cam_gt[clip_name][cam_id]
            with open(str(gt_path / f'{cam_id}.csv'), 'w') as f:
                f.writelines(['cid,fid,pid,x1,y1,w,h\n'] + gts)
            with open(str(preds_path / f'{cam_id}.csv'), 'w') as f:
                f.writelines(['cid,fid,pid,x1,y1,w,h\n'] + gts)
            print(f'Wrote gts and preds for {clip_name=}:{cam_id=}')
            if int(cam_id) == 3:
                break
        
        # run mcta for this clip and gather statistics
        eval_cmd = f'python submodules/mcta/evalMCTA.py \
        --gt_path {str(gt_path)} \
        --pred_path {str(preds_path)}'
 
        try:
            read_process = subprocess.Popen(shlex.split(eval_cmd), stdout=subprocess.PIPE)
            out_dict = eval(read_process.stdout.readlines()[-1])
            print(out_dict)
            mcta = 2*(out_dict['precision']*out_dict['recall']/(out_dict['precision']+out_dict['recall']))*(1-out_dict['mismatch_c']/out_dict['truepos_c'])*(1-out_dict['mismatch_s']/out_dict['truepos_s'])
            print(mcta)
        except subprocess.CalledProcessError as e:
            raise e




