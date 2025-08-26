"""
Extract yolo boxes and features for MMPtrack data
"""

import os
import sys
import inspect
from time import time
from pathlib import Path

# third party
import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.io import write_jpeg

from ml.data.sampler import DistributedNoExtraSampler
from ml.vision.ops import box_iou, dets_select

from utils.misc import init_distributed_mode
from data.mmptracking import MMPTrackFrames
from reid import FeatureExtractor

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

@torch.no_grad()
def extract(cfg):
    # setup dist 
    init_distributed_mode(cfg)
    print(f"Working with cfg: \n{cfg}")

    dataset_cfg = cfg.DATASET
    extract_cfg = cfg.EXTRACT

    # setup dataset and dataloader
    def collate(batch):
        meta = []
        images = []
        for image, boxes, ids, img_path in batch:
            images.append(image)
            meta.append((boxes, ids, img_path))
        images = torch.stack(images)
        return images, meta

    start_ext = time()
    dataset = MMPTrackFrames(dataset_cfg) 
    if extract_cfg.SANITY:
        dataset.sanity_check()
        exit()
    sampler = DistributedNoExtraSampler(dataset) if cfg.DIST else None
    dataloader = DataLoader(
        dataset,
        batch_size=extract_cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=torch.get_num_threads()-1,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate
    )

    class Detector:
        def __init__(self, dev, cfg_detector, amp=True):
            self.module = None
            self.dev = dev
            self.amp = amp

            arch = cfg_detector.ARCH.lower()
            cfg = cfg_detector[cfg_detector.ARCH]
            if arch == 'yolo7':
                from ml.vision.models import yolo7
            elif arch == 'yolo5':
                from ml.vision.models import yolo5
            
            pretrained = cfg.PRETRAINED
            s3 = pretrained.BUCKET and pretrained.KEY and dict(bucket=pretrained.BUCKET, key=pretrained.KEY) or None
            # FIXME: workaround to avoid conflict between torchub and local modules 
            import sys
            utils_mod = sys.modules.pop('utils')
            models_mod = sys.modules.pop('models')
            det = eval(arch)(classes=cfg.CLASSES.N,
                        pretrained=True, 
                        chkpt=pretrained.CHKPT, 
                        tag=cfg.TAG, 
                        model_dir=cfg.MODEL_DIR, 
                        s3=s3, 
                        fuse=cfg.FUSE, 
                        pooling=cfg.POOLING,
                        force_reload=pretrained.RELOAD)
            sys.modules['utils'] = utils_mod
            sys.modules['models'] = models_mod
            det.eval()
            for param in det.parameters():
                param.requires_grad = False
            self.det = det
            self.det.to(dev)
            self.det_cfg = cfg

        @torch.no_grad()
        def infer(self, frames):
            features = [None] * len(frames)
            with torch.cuda.amp.autocast(enabled=self.amp):
                # detector handles pre and post processing
                dets, features = self.det.detect(frames, batch_preprocess=True, size=self.det_cfg.SCALES, cls_thres=self.det_cfg.CLS_THRESH, nms_thres=self.det_cfg.NMS_THRESH)
                # fp16 to fp32
                features = [feats.float() for feats in features]
                dets = [ds.float() for ds in dets]

            return dets, features

    cfg.DEV = torch.cuda.default_stream().device
    # setup detector
    detector = Detector(cfg.DEV, extract_cfg.DETECTOR)
    # setup reid feature extractor
    reid_extractor = FeatureExtractor(cfg.DEV, extract_cfg.REID)
    reid_model = extract_cfg.REID.MODEL
    
    for i, data in enumerate(dataloader):
        start = time()
        # data: images: Tensor(len(metas), 3, H, W), metas: [[gt_boxes, gt_ids, img_path], ...]
        images, metas = data
        # batch inference
        dets, feats = detector.infer(images)

        assert len(images) == len(dets) == len(feats) == len(metas), f'Mismatch between metas, dets and feats size'

        for image, det, feat, meta in zip(images, dets, feats, metas):
            # det: [N, xyxy-score-class]
            # feat: [N, feat_size]
 
            # filter out non person objects
            # person class is [0] in both COCO & OBJ365
            persons_idx = dets_select(det, [0])
            people = det[persons_idx]
            people_feat = feat[persons_idx]

            assert len(people_feat) == len(people)

            # extract reid features using person box crops
            def extract_reid_feats(ppl):
                crops = []
                for _, xyxysc in enumerate(ppl):
                    x1, y1, x2, y2 = xyxysc[:4].int().tolist()
                    crop = image[:, y1:y2, x1:x2]
                    crop = reid_extractor.preprocess(crop)
                    # write_jpeg(crop.to(torch.uint8), f'test-{i}.jpg')
                    crops.append(crop)
                if crops:
                    crops = torch.stack(crops)
                    reid_feats = reid_extractor.infer(crops)
                else:
                    reid_feats = torch.zeros(0, reid_extractor.feat_size)
                assert len(reid_feats) == len(crops)
                
                return reid_feats
            
            gt_people, gt_ids, img_path = meta 
            reid_people_feats = extract_reid_feats(people)
            # gt_reid_people_feats = extract_reid_feats(gt_people)
            
            # write_jpeg(draw_bounding_boxes(image, people[:, :4]), 'test.jpg')

            # create out path
            out_dir = f'dets-{extract_cfg.DETECTOR.ARCH.lower()}-cls_thresh_{detector.det_cfg.CLS_THRESH}-nms_thresh_{detector.det_cfg.NMS_THRESH}-feats_{reid_model}'
            out_path = str(img_path).replace(dataset_cfg.IMAGE_EXT, extract_cfg.OUT_EXT).replace(dataset_cfg.IMAGE_PATH, out_dir)
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            output = {
                # 'image': image.cpu(),
                'img_size': image.shape, # [C, H, W]
                'img_path': img_path.name, # 000_rgb000.jpg
                'num_people': len(people),
                'people': people.cpu(), # N, xyxysc
                'detector_people_feats': people_feat.flatten(1).cpu(), # N, feat_size
                'reid_people_feats': reid_people_feats.cpu(), # N, feat_size
                'gt_people': gt_people, # xyxy
                #'reid_gt_people_feats': gt_reid_people_feats.cpu(), # N_gt, feat_size
                'gt_ids': gt_ids
            }

            torch.save(output, str(out_path))
        
        end = time() - start
        print(f'[{i+1}/{len(dataloader)}] Extracted dets for batch size: {len(images)} in {end:.2f}s')
    
    print(f'Completed extraction in {time() - start_ext:.2f}sec')

if __name__ == '__main__':
    from utils.args import parse_args
    cfg = parse_args()
    extract(cfg)
