import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
from .util import box_ops

class PostProcessLoss(nn.Module):

    @torch.no_grad() 
    def match(self, pred, gt, threshold):
        # matches predictions with ground truth based on iou. 
        # iou must be larger than  threshold to match.
        gtm = -1*torch.ones(gt.shape[0],dtype=torch.int)
        dm = -1*torch.ones(pred.shape[0],dtype=torch.int)
        if gt.shape[0] == 0 or pred.shape[0] == 0:
            return dm,gtm
            
        pred = box_ops.box_cxcywh_to_xyxy(pred)
        gt = box_ops.box_cxcywh_to_xyxy(gt)
        iou,_ = box_ops.box_iou(pred, gt)
        iou[iou<threshold] = 0
        pred_idx, gt_idx = linear_sum_assignment(iou.cpu(), maximize=True)
        for p,g in zip(pred_idx, gt_idx):
            if iou[p,g] > 0:
                gtm[g] = p
                dm[p] = g
        return dm, gtm

    @torch.no_grad()
    def get_tracks(self, tgt):
        # return track tensor \tracks where first dimiension represents one track,
        # second dimension represents the frame and third dimension represents a view. 
        #  tracks(i,f,j) = d if detection d in frame f and view j belongs to track i
        #  tracks(i,f,j) = -1 if track i is not visible in frame f and view j
        frames = len(tgt)
        views = len(tgt[0])
        max_track = [t["tracks"].max().item() if t["tracks"].numel()>0 else -1 for tgtf in tgt for t in tgtf]
        max_track = int(max(max_track))
        if max_track < 0:
            print("No ground truth detections in this clip")
            print(tgt)
            max_track = 0
            
        tracks = -1*(torch.ones((max_track,frames,views),dtype=torch.int))
        for f, tgtf in enumerate(tgt):
            for v, trk in enumerate(tgtf):
                ndets = trk["tracks"].shape[0]
                tracks[trk["tracks"].long()-1,f,v] = torch.arange(ndets, dtype=torch.int)
        return tracks


    @torch.no_grad()
    def postprocess_loss(self, outputs, targets, threshold=0.5):
        prec = rec = tacc = taccpwc = tACCpwc = taccpwf = tACCpwf = prf1 = 0
        dev = targets[0][0][0]["boxes"].device
        
        out_frames = len(outputs[0])
        target_frames = len(targets[0])
        context = target_frames - out_frames +1
        start = context//2
        end = target_frames - (context - 1 - start)
        targets = [t[slice(start,end)] for t in targets]

        for b, (outf, tgtf) in enumerate(zip(outputs, targets)):
            gtmf = []
            dmf = []

            tracks = self.get_tracks(tgtf)
            fn=fp=ngt=nd=ct=n_tracks=ctpwc=wtpwc=n_tracks_pwc=ctpwf=wtpwf=n_tracks_pwf=0
            for _, (out, tgt) in enumerate(zip(outf, tgtf)):
                gtm = []
                dm = []
                for _,(o, t) in enumerate(zip(out,tgt)):
                    d, g = self.match(o["boxes"],t["boxes"], threshold)
                    gtm.append(g)
                    dm.append(d)
                    fn += (g==-1).sum()
                    ngt += g.shape[0]
                    fp += (d==-1).sum()
                    nd += d.shape[0]
                gtmf.append(gtm)
                dmf.append(dm)

            for t in range(tracks.shape[0]):
                pt = -2
                for f in range(tracks.shape[1]):
                    for v in range(tracks.shape[2]):
                        i = tracks[t,f,v]
                        if i == -1:
                            continue
                        if gtmf[f][v][i] == -1:
                            pt = -3
                            break                    
                        pti = outf[f][v]["tracks"][gtmf[f][v][i]]
                        if pti == -1:
                            pt = -1
                            break
                        if pt == -2:
                            pt = pti
                        if pt != pti:
                            pt = -4
                            break
                if pt >= 0:
                    # entire track is correct
                    ct += 1
                if pt != -2:
                    # this track was visible in at least one view
                    n_tracks += 1
                for f in range(tracks.shape[1]):
                    for v1 in range(tracks.shape[2]):
                        i = tracks[t,f,v1]
                        if i == -1: # or gtm[v1][i] == -1:
                            continue
                        if gtmf[f][v1][i] == -1:
                            pti = -2
                        else:
                            pti = outf[f][v1]["tracks"][gtmf[f][v1][i]]
                        for v2 in range(v1+1,tracks.shape[2]):
                            j = tracks[t,f,v2]
                            if j == -1: # or gtm[v2][j] == -1:
                                continue
                            n_tracks_pwc += 1
                            if gtmf[f][v2][j] == -1:
                                ptj = -2
                            else:
                                ptj = outf[f][v2]["tracks"][gtmf[f][v2][j]]
                            if pti == -2 or ptj == -2:
                                continue
                            if pti == ptj and pti != -1:
                                ctpwc += 1
                            else:
                                wtpwc += 1
                for v in range(tracks.shape[2]):
                    for f1 in range(tracks.shape[1]):
                        i = tracks[t,f1,v]
                        if i == -1: # or gtm[v1][i] == -1:
                            continue
                        if gtmf[f1][v][i] == -1:
                            pti = -2
                        else:
                            pti = outf[f1][v]["tracks"][gtmf[f1][v][i]]
                        for f2 in range(f1+1,tracks.shape[1]):
                            j = tracks[t,f2,v]
                            if j == -1: # or gtm[v2][j] == -1:
                                continue
                            n_tracks_pwf += 1
                            if gtmf[f2][v][j] == -1:
                                ptj = -2
                            else:
                                ptj = outf[f2][v]["tracks"][gtmf[f2][v][j]]
                            if pti == -2 or ptj == -2:
                                continue
                            if pti == ptj and pti != -1:
                                ctpwf += 1
                            else:
                                wtpwf += 1
            p = 1 if nd == 0 else (nd-fp)/nd
            r = 1 if ngt == 0 else (ngt-fn)/ngt
            prec += p
            rec += r
            prf1 += 0 if (p + r) == 0 else (2*p*r)/(p + r)
            tacc += 1 if n_tracks == 0 else ct/n_tracks 
            taccpwc += 1 if (ctpwc+wtpwc) == 0 else ctpwc/(ctpwc+wtpwc)
            tACCpwc += 1 if n_tracks_pwc == 0 else ctpwc/n_tracks_pwc
            taccpwf += 1 if (ctpwf+wtpwf) == 0 else ctpwf/(ctpwf+wtpwf)
            tACCpwf += 1 if n_tracks_pwf == 0 else ctpwf/n_tracks_pwf
        prec = prec/len(outputs)
        rec = rec/len(outputs)
        prf1 = prf1/len(outputs)
        tacc = tacc/len(outputs)        
        taccpwc = taccpwc/len(outputs)
        tACCpwc = tACCpwc/len(outputs)
        taccpwf = taccpwf/len(outputs)
        tACCpwf = tACCpwf/len(outputs)
        return {"precision":torch.tensor([prec], device=dev), 
                "recall":torch.tensor([rec], device=dev),
                "f1": torch.tensor([prf1], device=dev), 
                "full_track_accuracy": torch.tensor([tacc], device=dev),
                "track_pwc_accuracy":torch.tensor([taccpwc], device=dev), 
                "correct_pwc_tracks":torch.tensor([tACCpwc], device=dev),
                "track_pwf_accuracy":torch.tensor([taccpwf], device=dev), 
                "correct_pwf_tracks":torch.tensor([tACCpwf], device=dev)}

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.postprocess_loss(outputs, targets)


def build_postprocess_loss(arg):
    return PostProcessLoss()