import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

class PostProcessTracks(nn.Module):

    @torch.no_grad()
    def forward(self, outputs, costs, keep_prob=0.9):

        bs, nframes, nviews, nd, nt = costs.shape
        out_frames = outputs['pred_boxes'].shape[0]//(bs*nviews)
        assert out_frames*bs*nviews == outputs['pred_boxes'].shape[0]
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        nclass = out_logits.shape[-1]
        out_logits = out_logits.view((bs, out_frames, nviews, nd, nclass))
        out_bbox = out_bbox.view((bs,out_frames,nviews,nd,4))

        context = out_frames - nframes
        start = context // 2
        end = out_frames - (context - start)
        out_logits = out_logits[:,start:end,:,:,:]
        out_bbox = out_bbox[:,start:end,:,:,:]
        if "track_pred_boxes" in outputs:
            out_track_bbox = outputs['track_pred_boxes']

        if 'topdown' in outputs:
            out_topdown = outputs['topdown'].detach().cpu()

        probas = out_logits.softmax(-1)[:,:,:,:,:-1]
        out_tracks = -torch.ones((bs,nframes,nviews,nd), dtype=torch.int)
        out_costs = -torch.ones((bs,nframes,nviews,nd))
        out = []
        out_trk = []
        keeps = []
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
                    didx, tidx = linear_sum_assignment(c, maximize=True)
                    didx = idx[torch.from_numpy(didx)]
                    tidx = torch.from_numpy(tidx)
                    out_tracks[b,f,v,didx] = tidx.int()
                    out_costs[b,f,v,didx] = costs[b,f,v, didx, tidx].cpu()

                    k.append(keep)
                    o.append({
                        "boxes": out_bbox[b,f,v,keep,:],
                        "detr_det_id": idx,
                        "tracks":out_tracks[b,f,v,keep],
                        "costs":out_costs[b,f,v,keep],
                        "all_costs":c,
                        "track_boxes": out_track_bbox[b,f,v,tidx,:] if "track_pred_boxes" in outputs else None,
                        "topdown": out_topdown[b,f,tidx,:] if 'topdown' in outputs else None
                    })

                    for trk in tidx:
                        if not (trk in tracks):
                            tracks.append(trk)
                            if "track_pred_boxes" in outputs:
                                track_boxes.append(out_track_bbox[b,f,:,trk,:])
            
                of.append(o)
                kf.append(k)
                if len(tracks) > 0:
                    tf.append({"tracks":torch.tensor(tracks,dtype=torch.long),"track_boxes":torch.stack(track_boxes, dim=0) if len(track_boxes) > 0 else None})
            out.append(of)
            out_trk.append(tf)
            keeps.append(kf)
        

        raw_outputs = {"pred_logits":out_logits, "pred_boxes": out_bbox, "pred_tracks": out_tracks}

        return out, out_trk,keeps

def build_postprocessor(cfg):
    return PostProcessTracks()
