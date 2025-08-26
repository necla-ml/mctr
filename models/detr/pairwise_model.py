import copy
import torch
import torch.nn.functional as F
from torch import nn
import math
from .detrClean import build_detr
from .pairwise_decoder import build_pairwise_decoder

class PairwiseModel(nn.Module):
    def __init__(self, detr, pairwise_decoder, num_queries, num_views, proj_dim_scale, no_proj = False, 
                 pred_track_boxes = False, pred_track_boxes_time = False, pred_topdown=False, add_bboxes = True, scaled_costs=True,
                 detached_until = 0, tgt_as_pos = False, zero_qe = False, context_size = 1,
                 return_intermediate=False):
        super().__init__()
        self.detr = detr
        self.pairwise = pairwise_decoder
        self.num_queries = num_queries
        self.num_views = num_views
        self.detached_until = detached_until
        self.add_bboxes = add_bboxes
        self.context_size = context_size
        hidden_dim = detr.transformer.d_model
        if self.add_bboxes:
            hidden_dim = hidden_dim + 4*self.pairwise.nheads 
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.scaled_costs = scaled_costs # whether to scale the costs by sqrt(dim) before taking the softmax 
        self.tgt_as_pos = tgt_as_pos # if true query from previous frames become positional embedings for the next frame
        self.zero_qe = zero_qe # if true, the track query positional embeddings are only used when there are no track queries coming from the previous frame. Otherwise they are set to zero. 


        # projection layers for getting the cost matrix.
        # two for each view. One for the track queries and one for
        # the DETR object queries. 

        if no_proj:
            lin_layer = myIdentity()
        else:
            proj_dim = int(hidden_dim*proj_dim_scale)
            lin_layer = myLinear(hidden_dim, proj_dim)
        self.tgt_proj = _get_clones(lin_layer, num_views)
        self.src_proj = _get_clones(lin_layer, num_views)

        # mlp heads for predicting the position of the bounding box 
        # in each view
        self.pred_track_boxes = pred_track_boxes
        self.pred_track_boxes_time = pred_track_boxes_time
        if self.pred_track_boxes:
            bbox_embed = MLP1(hidden_dim, hidden_dim, 4, 3)
            self.bbox_embed = _get_clones(bbox_embed, num_views)
            if self.pred_track_boxes_time:
                self.bbox_embed_prev = _get_clones(bbox_embed, num_views)
                self.bbox_embed_next = _get_clones(bbox_embed, num_views)

        self.pred_topdown = pred_topdown
        if self.pred_topdown:
            self.topdown_embed = MLP1(hidden_dim, hidden_dim, 2, 3)

    def forward(self, samples, tgt = None, reset_tgt=None):
        bs, num_frames, num_views, C, H, W = samples.shape

        samples = samples.view(bs * num_frames * num_views, C, H, W)
        out, hs = self.detr(samples) 
 
        if self.detached_until > 0:
            hs = hs.detach()
            self.detached_until -= 1

        src = hs[-1] # take only the last output from deter.    hs: bs*nviews x numq x d  

        if self.add_bboxes:
            src = torch.cat((src, out['pred_boxes'].repeat(1,1,self.pairwise.nheads)), dim = -1) 


        batch_size, src_queries, hidden_dim = src.shape
        assert(batch_size == bs*num_frames*num_views)

        src = src.view((bs, num_frames, num_views, src_queries, hidden_dim))

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        if tgt is None:
            reset_tgt=torch.ones(bs,dtype=torch.bool,device=query_embed.device)
            if self.tgt_as_pos:
                tgt = query_embed
            else:
                tgt = torch.zeros_like(query_embed)
        else:
            tgt=tgt.detach() # we don't want gradients backpropagating

        if reset_tgt is None:
            reset_tgt = torch.zeros(bs,dtype=torch.bool,device=query_embed.device)

        if reset_tgt.dim() == 1:
            reset_tgt = reset_tgt.unsqueeze(1).repeat(1,self.num_queries)

        if (self.tgt_as_pos):
            tgt[reset_tgt,:] = query_embed[reset_tgt,:]
        else:
            tgt[reset_tgt,:] = torch.zeros_like(query_embed)[reset_tgt,:]
            if self.zero_qe:
                query_embed[reset_tgt.logical_not(),:] = torch.zeros_like(query_embed)[reset_tgt.logical_not(),:]

        costs = []
        track_pred_boxes = []  
        track_pred_boxes_prev = []  
        track_pred_boxes_next = []   
        pred_topdown = []
        for f in range(0, num_frames - self.context_size + 1):
            s = src[:,f:(f+self.context_size),:,:]
            if (self.tgt_as_pos):
                qe = tgt
                tgt = torch.zeros_like(qe)
            else:
                qe = query_embed

            c, tgt, tpb, tpb_prev, tpb_next, topdown = self.forward_frame(s, tgt, qe)
            
            if self.zero_qe:
                # we are resetting all the query embedings to zero after the first frame'
                # this will have no effect if self.tgt_as_pos is true
                query_embed = torch.zeros_like(query_embed)

            costs.append(c)
            track_pred_boxes.append(tpb)
            track_pred_boxes_prev.append(tpb_prev)
            track_pred_boxes_next.append(tpb_next)
            pred_topdown.append(topdown)

        costs = torch.stack(costs, dim=1)
        if self.pred_track_boxes: 
            out["track_pred_boxes"] = torch.stack(track_pred_boxes, dim=1)
            if self.pred_track_boxes_time:
                out["track_pred_boxes_prev"] = torch.stack(track_pred_boxes_prev, dim=1)
                out["track_pred_boxes_next"] = torch.stack(track_pred_boxes_next, dim=1)
        if self.pred_topdown:
            out["topdown"] = torch.stack(pred_topdown, dim=1)

        #out["sizes"] = (bs, num_frames, num_views, src_queries)
        return out, costs, tgt

    def forward_frame(self, src, tgt, query_embed):
        # bs, num_views, C, H, W = samples.shape
        # samples = samples.view(bs * num_views, C, H, W)
        # out, hs = self.detr(samples) 

        # # don't propagate gradients back to deter for a number of steps, until there is 
        # # something learned in the pariwise decoder step. 

        # if self.detached_until > 0:
        #     hs = hs.detach()
        #     self.detached_until -= 1

        # src = hs[-1] # take only the last output from deter.    hs: bs*nviews x numq x d  

        # if self.add_bboxes:
        #     src = torch.cat((src, out['pred_boxes'].repeat(1,1,self.pairwise.nheads)), dim = -1) 


        # batch_size, src_queries, hidden_dim = src.shape
        # assert(batch_size == bs*num_views)

        # src = src.view((bs, num_views, src_queries, hidden_dim))
        bs, num_frames, num_views, src_queries, hidden_dim = src.shape
        assert (num_views == self.num_views)
        #assert (src_queries == self.num_queries)
        

        tgt, src = self.pairwise(tgt, src, query_pos = query_embed) #tgt: dec_layers x bs x numq x d, src: dec_layers x bs x num_frames x nviews x numq x d

        projected_targets = [self.tgt_proj[v](tgt[-1]) for v in range(self.num_views)]
        projected_targets = torch.stack(projected_targets, dim=1)

        projected_sources = [self.src_proj[v](src[-1,:, num_frames//2, v, :,:]) for v in range(self.num_views)]
        projected_sources = torch.stack(projected_sources, dim=1)
        costs = torch.matmul(projected_sources, projected_targets.transpose(2,3)) #  bs x nviews x numq x numq
        #take softmax over columns     
        if self.scaled_costs: 
            costs = costs/math.sqrt(projected_targets.shape[3])
        costs = costs.softmax(3)

        track_coord = None
        track_coord_prev = None
        track_coord_next = None
        # predict the bb in each view from the track heads
        if self.pred_track_boxes: 
            track_coord = [self.bbox_embed[v](tgt[-1]).sigmoid() for v in range(self.num_views)] # bs x numq x 4
            track_coord = torch.stack(track_coord, dim = 1) # bs x nviews x numq x 4
            if self.pred_track_boxes_time:
                track_coord_prev = [self.bbox_embed_prev[v](tgt[-1]).sigmoid() for v in range(self.num_views)] # bs x numq x 4
                track_coord_prev = torch.stack(track_coord_prev, dim = 1) # bs x nviews x numq x 4
                track_coord_next = [self.bbox_embed_next[v](tgt[-1]).sigmoid() for v in range(self.num_views)] # bs x numq x 4
                track_coord_next = torch.stack(track_coord_next, dim = 1) # bs x nviews x numq x 4
                
        pred_topdown = None
        if self.pred_topdown:
            pred_topdown = self.topdown_embed(tgt[-1]).sigmoid() 

        return costs, tgt[-1], track_coord, track_coord_prev, track_coord_next, pred_topdown

class myLinear(nn.Linear):
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.constant_(self.bias.data,0)

class myIdentity(nn.Identity):
    def _reset_parameters(self):
        pass

class MLP1(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def _reset_parameters(self):
        for i,l in enumerate(self.layers):
            nn.init.constant_(l.bias.data, 0)
            if i < self.num_layers:
                nn.init.kaiming_normal_(l.weight.data, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.xavier_uniform_(l.weight.data)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    ret = nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    for m in ret:
        m._reset_parameters()
    return ret


def build_pairwise(args):

    detr = build_detr(args)
    pairwise_decoder = build_pairwise_decoder(args)

    model = PairwiseModel(
        detr = detr,
        pairwise_decoder=pairwise_decoder,
        num_queries=args.pw_queries,
        num_views = pairwise_decoder.nviews,
        proj_dim_scale = args.proj_dim_scale,
        no_proj = args.no_proj,
        pred_track_boxes = args.pred_track_boxes,
        pred_track_boxes_time = args.pred_track_boxes_time,
        pred_topdown = args.pred_topdown,
        detached_until = args.detached_until,
        add_bboxes=args.add_bboxes,
        scaled_costs=args.scaled_costs,
        tgt_as_pos=args.tgt_as_pos,
        zero_qe=args.zero_qe,
        context_size=args.context_size,
        return_intermediate=False
    )
    return model

