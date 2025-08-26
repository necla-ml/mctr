import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from .util import box_ops

class PostProcessTracks(nn.Module):

    @torch.no_grad()
    def forward(self, outputs, costs, keep_prob=0.9, prev_per_track_bbox = None):

        bs, nframes, nviews, nd, nt = costs.shape
        out_frames = outputs['pred_boxes'].shape[0]//(bs*nviews)
        assert out_frames*bs*nviews == outputs['pred_boxes'].shape[0]
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        nclass = out_logits.shape[-1]
        out_logits = out_logits.view((bs, out_frames, nviews, nd, nclass))
        out_bbox = out_bbox.view((bs,out_frames,nviews,nd,4))

        context = out_frames - nframes
        start = context//2
        end = out_frames - (context - start)
        out_logits = out_logits[:,start:end,:,:,:]
        out_bbox = out_bbox[:,start:end,:,:,:]
        if "track_pred_boxes" in outputs:
            out_track_bbox = outputs['track_pred_boxes']


        probas = out_logits.softmax(-1)[:,:,:,:,:-1]
        out_tracks = -torch.ones((bs,nframes,nviews,nd), dtype=torch.int)
        out_costs = -torch.ones((bs,nframes,nviews,nd))
        out = []
        out_trk = []
        keeps = []

        per_track_bbox = torch.zeros((bs,out_frames,nviews,nt,4), dtype=costs.dtype, device = costs.device)

        for b in range(bs):
            of = []
            kf = []
            tf = []
            for f in range(nframes):
                o = []
                k = []
                tracks = []
                track_boxes = []
                for v in range(nviews):
                    keep = probas[b,f,v,:,:].max(-1).values > keep_prob
                    idx = torch.arange(nd, dtype=torch.long)[keep]
                    c = costs[b,f,v,keep,:].cpu()

                    if prev_per_track_bbox is not None:
                        cur_box = box_ops.box_cxcywh_to_xyxy(out_bbox[b,f,v,keep,:])
                        prev_box = box_ops.box_cxcywh_to_xyxy(prev_per_track_bbox[b,f,v,:,:])
                        iou,_ = box_ops.box_iou(cur_box, prev_box)
                        c = c + 0.01*iou.cpu()

                    didx, tidx = linear_sum_assignment(c, maximize=True)
                    didx = idx[torch.from_numpy(didx)]
                    tidx = torch.from_numpy(tidx)
                    out_tracks[b,f,v,didx] = tidx.int()
                    out_costs[b,f,v,didx] = costs[b,f,v, didx, tidx].cpu()

                    per_track_bbox[b,f,v,tidx,:] = out_bbox[b,f,v,didx,:]

                    if "track_pred_boxes" in outputs:
                        out_pred_track_bbox = out_track_bbox[b,f,v,tidx,:]
                    else:
                        out_pred_track_bbox = None

                    k.append(keep)
                    o.append({"boxes": out_bbox[b,f,v,keep,:],
                            "detr_det_id": idx,
                            "tracks":out_tracks[b,f,v,keep],
                            "costs":out_costs[b,f,v,keep],
                            "all_costs":c,
                            "track_boxes":out_pred_track_bbox})

                    for trk in tidx:
                        if not (trk in tracks):
                            tracks.append(trk)
                            track_boxes.append(out_track_bbox[b,f,:,trk,:])
            
                of.append(o)
                kf.append(k)
                if len(tracks) > 0:
                    tf.append({"tracks":torch.tensor(tracks,dtype=torch.long),"track_boxes":torch.stack(track_boxes, dim=0)})
            out.append(of)
            out_trk.append(tf)
            keeps.append(kf)
        

        raw_outputs = {"pred_logits":out_logits, "pred_boxes": out_bbox, "pred_tracks": out_tracks}


        return out, out_trk, keeps, per_track_bbox


def build_postprocessor(cfg):
    return PostProcessTracks()