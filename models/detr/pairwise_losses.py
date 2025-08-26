import torch
import torch.nn.functional as F
from torch import nn

from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .matcher import build_matcher
from .util import box_ops


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_track_boxes(self, costs, outputs, targets, indices, num_boxes):
        # if "track_pred_boxes" not in outputs:
        #     return {"loss_track_bbox": torch.tensor(0, device=costs.device),
        #             "loss_track_giou": torch.tensor(0,device=costs.device),
        #             "loss_track_bbox_prev": torch.tensor(0, device=costs.device),
        #             "loss_track_giou_prev": torch.tensor(0,device=costs.device),
        #             "loss_track_bbox_next": torch.tensor(0, device=costs.device),
        #             "loss_track_giou_next": torch.tensor(0,device=costs.device)}
        bs, nf, nv, nd, nt = costs.shape
        out_frames = outputs['pred_boxes'].shape[0]//(bs*nv)
        assert out_frames*bs*nv == outputs['pred_boxes'].shape[0]
        context = out_frames - nf + 1
        start = context//2
        # end = out_frames - (context - 1 - start)
        # idx = [b*out_frames*nv+f*nv+v for b in range(bs) for f in range(start, end) for v in range(nv)]
        # tgt = [targets[i] for i in idx]
        # ind = [indices[i] for i in idx]
        loss_giou = 0
        loss_bbox = 0
        loss_giou_prev = 0
        loss_bbox_prev = 0
        loss_giou_next = 0
        loss_bbox_next = 0
        for b in range(bs):
            for f in range(nf):
                for v in range(nv):
                    idx = b*out_frames*nv + (f+start)*nv + v
#                    idx_assert = b*out_frames*nv + f*nv + v                    
                    match_idx = indices[idx]
#                    assert match_idx == ind[idx_assert]
                    target_boxes = targets[idx]['boxes'][match_idx[1],:]
#                    assert target_boxes.equal(tgt[idx_assert]['boxes'][match_idx[1],:])
                    track_boxes = outputs['track_pred_boxes'][b,f,v,:,:]                
                    c = costs[b,f,v,match_idx[0],:]

                    loss_giou += (c*(1 - box_ops.generalized_box_iou(
                                        box_ops.box_cxcywh_to_xyxy(target_boxes),
                                        box_ops.box_cxcywh_to_xyxy(track_boxes)))).sum()
                    loss_bbox += (c*torch.abs(target_boxes[:,None,:]-track_boxes[None,:,:]).sum(-1)).sum()

                    if "track_pred_boxes_prev" in outputs and f < nf-1:
                        track_boxes = outputs['track_pred_boxes_prev'][b,f+1,v,:,:]                
                        loss_giou_prev += (c*(1 - box_ops.generalized_box_iou(
                                            box_ops.box_cxcywh_to_xyxy(target_boxes),
                                            box_ops.box_cxcywh_to_xyxy(track_boxes)))).sum()
                        loss_bbox_prev += (c*torch.abs(target_boxes[:,None,:]-track_boxes[None,:,:]).sum(-1)).sum()
                    if "track_pred_boxes_next" in outputs and f > 0:
                        track_boxes = outputs['track_pred_boxes_next'][b,f-1,v,:,:]                
                        loss_giou_next += (c*(1 - box_ops.generalized_box_iou(
                                            box_ops.box_cxcywh_to_xyxy(target_boxes),
                                            box_ops.box_cxcywh_to_xyxy(track_boxes)))).sum()
                        loss_bbox_next += (c*torch.abs(target_boxes[:,None,:]-track_boxes[None,:,:]).sum(-1)).sum()

        # num_boxes should be different for _prev and _next
        # but we leave it like this for simplicity
        # it is simialar to having a slightly lower weight               
        losses = {
            "loss_track_bbox": loss_bbox/num_boxes,
            "loss_track_giou": loss_giou/num_boxes,
            "loss_track_bbox_prev": loss_bbox_prev/num_boxes,
            "loss_track_giou_prev": loss_giou_prev/num_boxes,
            "loss_track_bbox_next": loss_bbox_next/num_boxes,
            "loss_track_giou_next": loss_giou_next/num_boxes
        }
        return losses


    def loss_topdown(self, costs, outputs, targets, indices, num_boxes):
        bs, nf, nv, nd, nt = costs.shape
        out_frames = outputs['topdown'].shape[0]//bs
        assert out_frames*bs == outputs['topdown'].shape[0]
        context = out_frames - nf + 1
        start = context//2
        loss_topdown = 0
        for b in range(bs):
            for f in range(nf):
                for v in range(nv):
                    idx = b*out_frames*nv + (f+start)*nv + v
                    match_idx = indices[idx]
                    target_topdown = targets[idx]['topdown'][match_idx[1],:]
                    pred_topdown = outputs['topdown'][b,f,:,:]                
                    c = costs[b,f,v,match_idx[0],:]
                    loss_topdown += (c*torch.abs(target_topdown[:,None,:]-pred_topdown[None,:,:]).sum(-1)).sum()           
        losses = {
            "loss_topdown": loss_topdown/num_boxes
        }

        return losses


    def loss_pairwise(self, costs, outputs, targets, indices):
        bs , cost_frames, views, _, _ = costs.shape
        out_frames = outputs['pred_boxes'].shape[0]//(bs*views)
        assert out_frames*bs*views == outputs['pred_boxes'].shape[0]
        context = out_frames - cost_frames + 1
        start = context//2
        end = out_frames - (context - 1 - start)
        idx = [b*out_frames*views+f*views+v for b in range(bs) for f in range(start, end) for v in range(views)]
        tgt = [targets[i] for i in idx]
        ind = [indices[i] for i in idx]
        assert len(tgt) == bs*cost_frames*views
        lossc = self.loss_pairwise_between_cams(costs, tgt, ind)
        lossf = self.loss_pairwise_between_frames(costs, tgt, ind)
                
        losses = {'loss_pwc': lossc, 'loss_pwf': lossf, 'loss_pw': (lossc+lossf)}
        return losses 


    def loss_pairwise_between_cams(self, costs, targets, indices):
        bs, frames, views, _, _ = costs.shape
        loss_bce = 0
        for f in range(frames):
            c = costs[:,f,:,:,:]
            idx = [b*frames*views+f*views+v for b in range(bs) for v in range(views)]
            t = [targets[ind] for ind in idx]
            i = [indices[ind] for ind in idx]
            loss_bce += self.loss_pairwise_bce(c,t,i)
        
        return loss_bce/frames


    def loss_pairwise_between_frames(self, costs, targets, indices):
        bs, frames, views, _, _ = costs.shape
        loss_bce = 0.0
        for v in range(views):
            c = costs[:,:,v,:,:]
            idx = [b*frames*views+f*views+v for b in range(bs) for f in range(frames)]
            t = [targets[ind] for ind in idx]
            i = [indices[ind] for ind in idx]
            loss_bce += self.loss_pairwise_bce(c,t,i)
        
        return loss_bce/views      


    def loss_pairwise_bce(self, costs, targets, indices):
        bs, v, _, _ = costs.shape
        if v < 2:
            return torch.tensor(0, device=costs.device)
        x = []
        y = []
        for v1 in range(v):
            idx1  = slice(v1,bs*v,v)
            indices1 = indices[idx1]
            targets1 = targets[idx1]
            for v2 in range(v1+1,v):
                idx2  = slice(v2,bs*v,v)
                indices2 = indices[idx2]
                targets2 = targets[idx2]
                pairwise_cost = torch.matmul(costs[:,v1,:,:], costs[:,v2,:,:].transpose(1,2))
                for b in range(bs):
                    # indices of the matched detections
                    sidx = tuple(torch.cartesian_prod(indices1[b][0], indices2[b][0]).transpose(1,0))
                    # indices of corresponding targets
                    tidx = tuple(torch.cartesian_prod(indices1[b][1], indices2[b][1]).transpose(1,0))

                    x.append(pairwise_cost[b][sidx])
                    y.append(torch.eq(targets1[b]["tracks"][tidx[0]], targets2[b]["tracks"][tidx[1]]).long())


                    # for i1,j1 in zip(indices1[b]):
                    #     trk1 = targets1[b]["tracks"][j1]
                    #     for i2,j2 in zip(indices2[b]):
                    #         trk2 = targets2[b]["tracks"][j2]
                    #         y.append(torch.eq(trk1,trk2).long())
                    #         x.append(pairwise_cost[i1,i2])

        x = torch.cat(x)
        y = torch.cat(y)
        # avoid nan when the cam scene is empty (no person)
        if len(x) == 0:
            loss_pairwise = torch.tensor(0.0, device=costs.device)
        else:
            loss_pairwise = F.binary_cross_entropy(x,y.float()) 

        return loss_pairwise

    def loss_embedding(self, embeddings):
        # loss to discurage the embedding to be very similar
        # max(0,cos(embi,embj)-0.5)
        norm_emb = F.normalize(embeddings, dim=-1)
        nemb = embeddings.shape[-2]
        cos_mat = torch.triu(torch.maximum(torch.matmul(norm_emb,norm_emb.transpose(-2,-1))-0.5, torch.zeros(nemb, nemb, device=embeddings.device)),diagonal=1)
        loss = 2*torch.sum(cos_mat,dim=(-2,-1))/(nemb*(nemb-1))
        losses = {"loss_emb": torch.mean(loss)}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, costs, track_embeddings, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             costs: pairwise costs. need to fine a better name :) 
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        #flatten the targets
        targets = [tgt_v for tgt_b in targets for tgt_f in tgt_b for tgt_v in tgt_f]

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # need to compute num_boxes without the context
        # and witout first and last frames if predicting track boxes
        # for now leave it like this. it is similar to having a differnt 
        # weight for the loss

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if "pairwise" in loss:
                losses.update(self.loss_pairwise(costs, outputs, targets, indices))
            elif "track" in loss:
                losses.update(self.loss_track_boxes(costs, outputs, targets, indices, num_boxes))
            elif "embedding" in loss:
                losses.update(self.loss_embedding(track_embeddings))
            elif "topdown" in loss:
                losses.update(self.loss_topdown(costs, outputs, targets, indices, num_boxes))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "pairwise":
                        continue # no pairwise loss on the auxiliary outputs
                    if "track" in loss:
                        continue # no track bb loss on auxiliary outputs
                    if "topdown" in loss:
                        continue # no topdown losses on auxiliary outputs
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == "embedding":
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_criterion(args):

    matcher = build_matcher(args)
    num_classes = args.num_classes
    device = args.device
    weight_dict = {
        'loss_ce': 1, 
        'loss_bbox': args.bbox_loss_coef, 
        'loss_giou': args.giou_loss_coef,
    }
    if args.pairwise_loss:
        weight_dict.update({'loss_pw': 5})
    if args.pred_topdown:
        weight_dict.update({"loss_topdown":4})
    if args.pred_track_boxes:
        weight_dict.update({'loss_track_bbox': 2,
                            'loss_track_giou': 1,
                            'loss_track_bbox_prev': 2,
                            'loss_track_giou_prev': 1,
                            'loss_track_bbox_next': 2,
                            'loss_track_giou_next': 1,})
    if args.embedding_loss:
        weight_dict.update({"loss_emb": 1})
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_pw" and "track" not in k})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes'] #, 'track'] # 'cardinality'
    if args.pairwise_loss:
        losses+= ["pairwise"]
    if args.pred_track_boxes:
        losses += ['track']
    if args.pred_topdown:
        losses += ['topdown']
    if args.embedding_loss:
        losses += ['embedding']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes,
                             matcher=matcher, 
                             weight_dict=weight_dict,
                             eos_coef=args.eos_coef, 
                             losses=losses)
    criterion.to(device)
    return criterion

