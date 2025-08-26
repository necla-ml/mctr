# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .util import box_ops
import copy

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.class_embed_1 = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed_1 = MLP(hidden_dim, hidden_dim, 4, 3)
        self.decoder_1 = copy.deepcopy(self.transformer.decoder)

        self.cam_embedding = nn.Embedding(2, hidden_dim)

    def forward_decoder(self, memory, query_embed, mask, pos_embed):
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2)
    
    def forward_decoder_1(self, memory, query_embed, mask, pos_embed):
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder_1(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2)

    def forward_encoder(self, src, mask, query_embed, pos_embed, targets):
        # flatten NxCxHxW to HWxNxC
        (bs, c, h, w), dev = src.shape, src.device
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        # add learned cam id embedding for cam0 and cam1
        src[:, ::2] += self.cam_embedding(torch.tensor(0, device=dev))
        src[:, 1::2] += self.cam_embedding(torch.tensor(1, device=dev))

        # FIXME: workaround multiview: concat features from corresponding views
        src = src.reshape(src.shape[0] * 2, src.shape[1] // 2, -1) # [(feat_hw_cam0 + feat_hw_cam1), bs // 2, dim]
        pos_embed = pos_embed.reshape(pos_embed.shape[0] * 2, pos_embed.shape[1] // 2, -1)
        mask = mask.reshape(mask.shape[0] // 2, mask.shape[1] * 2)
        # -----------------------
        
        memory = self.transformer.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # separate per view
        mask = mask.reshape(mask.shape[0] * 2, mask.shape[1] // 2) # [bs, n]
        pos_embed = pos_embed.reshape(pos_embed.shape[0] // 2, pos_embed.shape[1] * 2, -1) # [n, bs, dim]
        memory = memory.reshape(memory.shape[0] // 2, memory.shape[1] * 2, -1) # [n, bs, dim]

        hs = self.forward_decoder(memory, query_embed, mask, pos_embed) # [num_dec_layers, bs, num_queries, dim]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        # DETR like per view outputs
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)


        # per view 
        idxes_0, idxes_1 = slice(0, bs, 2), slice(1, bs, 2)
        mask0, mask1 = mask[idxes_0], mask[idxes_1]
        memory0, memory1 = memory[:, idxes_0], memory[:, idxes_1] 
        pos_embed0, pos_embed1 = pos_embed[:, idxes_0], pos_embed[:, idxes_1]
        query_embed0, query_embed1 = query_embed[:, idxes_0], query_embed[:, idxes_1]
        track_embed0, track_embed1 = hs[-1][idxes_0], hs[-1][idxes_1]

        # track embed: detected people in view_1
        # query_embed: people visible in view_2 but not in view_1 
        # query_embed0 = torch.cat([track_embed1, query_embed0], dim=0) # [t+q, bs, dim]
        query_embed1 = torch.cat([track_embed0.permute(1, 0, 2), query_embed1], dim=0) # [t+q, bs, dim]

        # hs0 = self.forward_decoder(memory0, query_embed0, mask0, pos_embed0)
        hs1 = self.forward_decoder_1(memory1, query_embed1, mask1, pos_embed1)

        # outputs_class0 = self.class_embed(hs0)
        # outputs_coord0 = self.bbox_embed(hs0).sigmoid()

        outputs_class1 = self.class_embed_1(hs1)
        outputs_coord1 = self.bbox_embed_1(hs1).sigmoid()

        # track query outputs view0
        # out0 = {'pred_logits': outputs_class0[-1], 'pred_boxes': outputs_coord0[-1]}
        # if self.aux_loss:
        #     out0['aux_outputs'] = self._set_aux_loss(outputs_class0, outputs_coord0)
        out0 = None

        # track query outputs view1
        out1 = {'pred_logits': outputs_class1[-1], 'pred_boxes': outputs_coord1[-1]}
        if self.aux_loss:
            out1['aux_outputs'] = self._set_aux_loss(outputs_class1, outputs_coord1)

        return out, out0, out1

    def forward(self, samples: NestedTensor, targets):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()

        assert mask is not None
        out, out0, out1 = self.forward_encoder(self.input_proj(src), mask, self.query_embed.weight, pos[-1], targets)

        return out, out0, out1

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_queries, matcher, weight_dict, eos_coef, losses):
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
        self.num_queries = num_queries
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
        #if len(targets) < 1:
        #    return {'loss_ce': torch.tensor(0, device=self.empty_weight.device)}
        
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

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
        if len(targets) < 1:
            return {'loss_bbox': torch.tensor(0, device=self.empty_weight.device), 'loss_giou': torch.tensor(0, device=self.empty_weight.device)}
        
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
    
    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward_view(self, outputs, targets, indices, view, aux_i=None):

        tracks = [t['tracks'] for t in targets]
        boxes = [t['boxes'] for t in targets]
        labels = [t['labels'] for t in targets]

        assert sum([len(b) for b in boxes]) == sum([len(t['tracks']) for t in targets])

        if view == 1:
            tracks_0 = [t.cpu().int().tolist() for t in tracks[::2]]
            indices_0 = indices[::2]
            
            boxes_1 = boxes[1::2]
            labels_1 = labels[1::2]
            tracks_1 = [t.cpu().int().tolist() for t in tracks[1::2]]
            
        else:
            tracks_0 = [t.cpu().int().tolist() for t in tracks[1::2]]
            indices_0 = indices[1::2]
            
            boxes_1 = boxes[::2]
            labels_1 = labels[::2]
            tracks_1 = [t.cpu().int().tolist() for t in tracks[::2]]

        pred_logits = outputs['pred_logits'] # [bs, t + q, 2]
        pred_boxes = outputs['pred_boxes'] # [bs, t + q, 4]

        new = self.get_new_people(pred_logits, pred_boxes, tracks_0, tracks_1, boxes_1)
        matched = self.get_matched_people(pred_logits, pred_boxes, tracks_0, indices_0, tracks_1, boxes_1, labels_1)

        losses = {}
        for (out, tar, idx), loss_type in zip([new, matched], ['new', 'matched']):
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in tar)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_logits.device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.losses:
                l_dict = self.get_loss(loss, out, tar, idx, num_boxes)
                l_dict = {k + (aux_i and f'_{view}_{loss_type}_{aux_i}' or f'_{view}_{loss_type}'): v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def forward_view_x(self, outputs, targets, indices, view):

        tracks = [t['tracks'] for t in targets]
        boxes = [t['boxes'] for t in targets]
        labels = [t['labels'] for t in targets]

        assert sum([len(b) for b in boxes]) == sum([len(t['tracks']) for t in targets])

        if view == 1:
            tracks_0 = [t.cpu().int().tolist() for t in tracks[::2]]
            indices_0 = indices[::2]
            
            boxes_1 = boxes[1::2]
            labels_1 = labels[1::2]
            tracks_1 = [t.cpu().int().tolist() for t in tracks[1::2]]
            
        else:
            tracks_0 = [t.cpu().int().tolist() for t in tracks[1::2]]
            indices_0 = indices[1::2]
            
            boxes_1 = boxes[::2]
            labels_1 = labels[::2]
            tracks_1 = [t.cpu().int().tolist() for t in tracks[::2]]

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        pred_logits = outputs_without_aux['pred_logits'] # [bs, t + q, 2]
        pred_boxes = outputs_without_aux['pred_boxes'] # [bs, t + q, 4]

        new = self.get_new_people(pred_logits, pred_boxes, tracks_0, tracks_1, boxes_1)
        matched = self.get_matched_people(pred_logits, pred_boxes, tracks_0, indices_0, tracks_1, boxes_1, labels_1)

        losses = {}
        for (out, targets, indices), loss_type in zip([new, matched], ['new', 'matched']):
            # if not len(targets) > 0:
            #     # no new dets in the other view
            #     continue

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.losses:
                l_dict = self.get_loss(loss, out, targets, indices, num_boxes)
                l_dict = {k + f'_{view}_{loss_type}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in out:
            for i, aux_outputs in enumerate(out['aux_outputs']):
                pred_logits = aux_outputs['pred_logits'] # [bs, t + q, 2]
                pred_boxes = aux_outputs['pred_boxes'] # [bs, t + q, 4]

                new = self.get_new_people(pred_logits, pred_boxes, tracks_0, tracks_1, boxes_1)
                matched = self.get_matched_people(pred_logits, pred_boxes, tracks_0, indices_0, tracks_1, boxes_1, labels_1)

                for (out, targets, indices), loss_type in zip([new, matched], ['new', 'matched']):
                    # if not len(targets) > 0:
                    #     # no new dets in the other view
                    #     continue
         
                    # Compute the average number of target boxes accross all nodes, for normalization purposes
                    num_boxes = sum(len(t["labels"]) for t in targets)
                    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
                    if is_dist_avail_and_initialized():
                        torch.distributed.all_reduce(num_boxes)
                    num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

                    # Compute all the requested losses
                    for loss in self.losses:
                        l_dict = self.get_loss(loss, out, targets, indices, num_boxes)
                        l_dict = {k + f'_{view}_{loss_type}_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses

    def get_new_people(self, pred_logits, pred_boxes, tracks_0, tracks_1, boxes_1):
        dev = pred_boxes.device

        new_logits = pred_logits[:, -self.num_queries:, :]
        new_boxes = pred_boxes[:, -self.num_queries:, :]

        targets_new = []
        for trk0, trk1, gt_bbox in zip(tracks_0, tracks_1, boxes_1):
            new_boxes_gt=torch.empty(0,4, device=dev)
            new_label_gt = torch.empty(0, dtype=torch.int64, device=dev)
            new_people = set(trk1) - set(trk0)
            if len(new_people) > 0:
                new_people_idx = [trk1.index(p) for p in new_people]
                new_boxes_gt = torch.stack([gt_bbox[pi] for pi in new_people_idx])
                new_label_gt = torch.tensor([0 for _ in range(len(new_boxes_gt))], device=dev)
            targets_new.append({
                    'boxes': new_boxes_gt,
                    'labels': new_label_gt
            })

        outputs_new = {
            'pred_logits': new_logits,
            'pred_boxes': new_boxes
        }

        indices_new = self.matcher(outputs_new, targets_new)

        return outputs_new, targets_new, indices_new

    def get_matched_people(self, pred_logits, pred_boxes, tracks_0, indices_0, tracks_1, boxes_1, labels_1):

        matched_logits = pred_logits[:, :-self.num_queries]
        matched_boxes = pred_boxes[:, :-self.num_queries]

        assert len(indices_0) == len(tracks_0) == len(tracks_1)

        matched_indices = []
        for idx, trk0, trk1 in zip(indices_0, tracks_0, tracks_1):
            indices_i = []
            indices_j = []
            for (i, j) in zip(*idx):
                # trk0 = [1, 2, 3, 4, 5, 6] => [1, 3, 6] are only visible in this view but not in other and are considered no object in the other view
                # trk1 = [2, 4, 5, 8, 9] => [8, 9] are considered new objects
                tid = trk0[j]
                if tid in trk1:
                    indices_i.append(i)
                    indices_j.append(trk1.index(tid))

            matched_indices.append((torch.tensor(indices_i, dtype=torch.int64), torch.tensor(indices_j, dtype=torch.int64)))

        outputs_matches = {
            'pred_logits': matched_logits,
            'pred_boxes': matched_boxes
        }
        targets_matches = [{'labels': l, 'boxes': b} for l, b in zip(labels_1, boxes_1)]

        return outputs_matches, targets_matches, matched_indices

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        out, out0, out1 = outputs

        outputs_without_aux = {k: v for k, v in out.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(out.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, out, targets, indices, num_boxes))

        # XXX: multiview: compute across cam matching loss: cam_0 to cam_1
        out1_without_aux = {k: v for k, v in out1.items() if k != 'aux_outputs'}
        losses_view_0_1 = self.forward_view(out1_without_aux, targets, indices, view=1)
        losses.update(losses_view_0_1)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in out:
            assert len(out['aux_outputs']) == len(out1['aux_outputs'])
            for i, (aux_outputs, aux_outputs1) in enumerate(zip(out['aux_outputs'], out1['aux_outputs'])):
                indices = self.matcher(aux_outputs, targets)
                # XXX: multiview: compute across cam matching loss: cam_0 to cam_1
                l_dict1 = self.forward_view(aux_outputs1, targets, indices, view=1, aux_i=i)
                losses.update(l_dict1)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    num_classes = args.num_classes
    device = args.device

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    matcher = build_matcher(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss
    )
    
    weight_dict = {
        'loss_ce': 1, 
        'loss_bbox': args.bbox_loss_coef, 
        'loss_giou': args.giou_loss_coef,
        'loss_ce_0_new':5, 
        'loss_ce_1_new':5,
        'loss_ce_0_matched':5, 
        'loss_ce_1_matched':5,
        'loss_bbox_0_new':args.bbox_loss_coef, 
        'loss_bbox_1_new':args.bbox_loss_coef,
        'loss_bbox_0_matched':args.bbox_loss_coef, 
        'loss_bbox_1_matched':args.bbox_loss_coef,
        'loss_giou_0_new':args.giou_loss_coef, 
        'loss_giou_1_new':args.giou_loss_coef,
        'loss_giou_0_matched':args.giou_loss_coef, 
        'loss_giou_1_matched':args.giou_loss_coef,
    }
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes'] # 'cardinality'
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, num_queries=args.num_queries, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
