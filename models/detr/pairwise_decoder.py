import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class PairwiseDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, nviews = 2, weighted_views = False,
                 self_attention_before= True, self_attention_after=False,
                 num_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 pass_through_src = False,
                 return_intermediate=False):
        super().__init__()
        pairwise_layer = PairwiseDecoderLayer(d_model, nhead, nviews, weighted_views, self_attention_before, self_attention_after, dim_feedforward, dropout,
                                              activation, pass_through_src)
        self.layers = _get_clones(pairwise_layer, num_layers)
        self.num_layers = num_layers
        self.nheads = nhead
        self.nviews = nviews
        self.return_intermediate = return_intermediate  

    def forward(self, tgt, sources,
                tgt_mask: Optional[Tensor] = None,
                src_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        assert(sources.shape[2]==self.nviews)

        out_tgt = tgt
        out_src = sources

        intermediate_tgt = []
        intermediate_src = []

        for layer in self.layers:
            out_tgt, out_src = layer(out_tgt, out_src, tgt_mask=tgt_mask,
                               src_mask=src_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate_tgt.append(out_tgt)
                intermediate_src.append(out_src)

        if self.return_intermediate:
            return torch.stack(intermediate_tgt), torch.stack(intermediate_src)

        return out_tgt.unsqueeze(0), out_src.unsqueeze(0)


class PairwiseDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nviews, weighted_views = False, self_attention_before= True, self_attention_after=False, dim_feedforward=2048, dropout=0.1,
                 activation="relu", pass_through_src=False):
        super().__init__()
        self.tgt_module = TgtDecoderModule(d_model, nhead, nviews, weighted_views, self_attention_before, self_attention_after, dim_feedforward, dropout,
                                           activation)
        self.pass_through_src = pass_through_src
        if not pass_through_src:
            self.src_modules = _get_clones(SrcDecoderModule(d_model, nhead, dim_feedforward, dropout,
                                                            activation), nviews)        
        self.nviews = nviews
        self._reset_parameters()

    def _reset_parameters(self):
        self.tgt_module._reset_parameters()
        if not self.pass_through_src:
            for m in self.src_modules:
                m._reset_parameters()

    def forward(self, tgt, sources,
                     tgt_mask: Optional[Tensor] = None,
                     src_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        tgt = self.tgt_module(tgt, sources, tgt_mask, src_mask, tgt_key_padding_mask, 
                              src_key_padding_mask, pos, query_pos)

        if not self.pass_through_src:
            sources = [[self.src_modules[s](tgt, src.squeeze(dim=1),
                                           tgt_mask, src_mask,
                                           tgt_key_padding_mask,
                                           src_key_padding_mask,
                                           pos, query_pos)   
                        for 
                            s, src in enumerate(frm_src.squeeze(dim=1).split(1, dim=1)) ]
                        for f, frm_src in enumerate(sources.split(1,dim=1))]
            sources = torch.stack([torch.stack(src, dim=1) for src in sources], dim=1) 
        

        return tgt, sources 

class TgtDecoderModule(nn.Module):

    def __init__(self, d_model, nhead, nviews, weighted_views = False,
                 self_attention_before= True, self_attention_after=False,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attention_before = self_attention_before
        self.self_attention_after = self_attention_after
        if self.self_attention_before:
            self.self_attn_before = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        self.cross_attn = _get_clones(nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True), nviews)
        if self.self_attention_after:
            self.self_attn_after = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)

        self.weighted_views = weighted_views        
        if self.weighted_views:
            self.view_weighting = nn.Linear(d_model,1)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_fn = activation
        self.activation = _get_activation_fn(activation)
        self.nviews = nviews
        self._reset_parameters()

    def _reset_parameters(self):
        if self.self_attention_before:
            self.self_attn_before._reset_parameters()
        if self.self_attention_after:
            self.self_attn_after._reset_parameters()
    
        for m in self.cross_attn:
            m._reset_parameters()

        nn.init.xavier_uniform_(self.linear2.weight.data)
        nn.init.constant_(self.linear2.bias.data,0)
        nn.init.kaiming_normal_(self.linear1.weight.data, mode='fan_out', nonlinearity=self.activation_fn)
        nn.init.constant_(self.linear1.bias.data,0)
        if self.weighted_views:
            nn.init.kaiming_normal_(self.view_weighting.weight.data, mode='fan_out', nonlinearity=self.activation_fn)
            nn.init.constant_(self.view_weighting.bias.data,0)
                    
        for nl in (self.norm1, self.norm2, self.norm3):
            nn.init.constant_(nl.weight.data, 1)
            nn.init.constant_(nl.bias.data, 0)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, sources,
                     tgt_mask: Optional[Tensor] = None,
                     src_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        assert(sources.shape[2]==self.nviews)
        if (self.self_attention_before):
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn_before(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)    
        tgt2 = [[self.cross_attn[s](query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(src.squeeze(dim=1), pos),
                                   value=src.squeeze(dim=1), attn_mask= src_mask,
                                   key_padding_mask=src_key_padding_mask)[0]  
                    for 
                        s, src in enumerate(frame_src.squeeze(dim=1).split(1, dim=1)) ] 
                for f, frame_src in enumerate(sources.split(1, dim=1))]

        # flatten 
        tgt2 = [f for frm in tgt2 for f in frm]

        if self.weighted_views: 
            tgt2 = torch.stack(tgt2, dim=0)
            weights = self.view_weighting(tgt2)
            weights = weights/math.sqrt(tgt2.shape[-1])
            weights = weights.softmax(dim=0)
            tgt2 = tgt2 * weights
            tgt2 = tgt2.sum(dim=0)
        else:
            tgt2 = torch.stack(tgt2, dim=0).mean(dim=0)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if self.self_attention_after:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn_after(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)    

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



class SrcDecoderModule(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation_fn = activation
        self.activation = _get_activation_fn(activation)
        self._reset_parameters()
        
    def _reset_parameters(self):
        self.self_attn._reset_parameters()
        self.cross_attn._reset_parameters()

        nn.init.xavier_uniform_(self.linear2.weight.data)
        nn.init.constant_(self.linear2.bias.data,0)
        nn.init.kaiming_normal_(self.linear1.weight.data, mode='fan_out', nonlinearity=self.activation_fn)
        nn.init.constant_(self.linear1.bias.data,0)

        for nl in (self.norm1, self.norm2, self.norm3):
            nn.init.constant_(nl.weight.data, 1)
            nn.init.constant_(nl.bias.data, 0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, src,
                     tgt_mask: Optional[Tensor] = None,
                     src_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.cross_attn(query=self.with_pos_embed(src, pos),
                                   key=self.with_pos_embed(tgt, query_pos),
                                   value=tgt, attn_mask= tgt_mask,
                                   key_padding_mask=tgt_key_padding_mask)[0]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)
        return src


def _get_clones(module, N):
    ret = nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    for m in ret:
        m._reset_parameters()
    return ret


def build_pairwise_decoder(args):
    hidden_dim = args.hidden_dim
    if args.add_bboxes:
        hidden_dim += 4*args.nheads

    return PairwiseDecoder(
        d_model=hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        nviews = args.nviews,
        weighted_views = args.weighted_views,
        self_attention_before = args.self_attention_before,
        self_attention_after = args.self_attention_after,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.pw_layers,
        pass_through_src=args.pw_passthrough_src,
        return_intermediate=False,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")