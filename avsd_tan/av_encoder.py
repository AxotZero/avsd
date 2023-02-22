import itertools
from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.blocks import (BridgeConnection, PositionwiseFeedForward, PositionalEncoder)
from .utils import get_seg_feats


def build_mlp(dims, dout_p):
    return nn.Sequential(
        *list(itertools.chain(*[
            [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(dout_p)] 
            for i in range(len(dims)-1)
        ]))
    )


class VisualEncoder(nn.Module):
    def __init__(self, cfg, dims, dout_p, pre_dout=0.5):
        super().__init__()

        hidden_dim = dims[-1]
        self.pre_dropout = nn.Dropout(pre_dout)
        self.encode_rgb = build_mlp(dims, dout_p)
        self.encode_flow = build_mlp(dims, dout_p)
        self.combine_rgb_flow = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), 
            nn.ReLU()
        )
        self.pos_enc = PositionalEncoder(cfg.d_model, cfg.dout_p)

        attn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2,
            dropout=cfg.dout_p, batch_first=True
        )
        self.self_attn = nn.TransformerEncoder(
            encoder_layer=attn_layer,
            num_layers=cfg.num_encoder_layers,
        )
        
    
    def forward(self, rgb, flow, mask=None):
        rgb = F.normalize(rgb, dim=-1)
        rgb = self.pre_dropout(rgb)
        rgb = self.encode_rgb(rgb)
        
        flow = F.normalize(flow, dim=-1)
        flow = self.pre_dropout(flow)
        flow = self.encode_flow(flow)

        feat = torch.cat([rgb, flow], dim=-1)
        feat = self.combine_rgb_flow(feat)

        feat = self.pos_enc(feat)

        feat = self.self_attn(feat, src_key_padding_mask=mask)
        return feat


class AudioEncoder(nn.Module):
    def __init__(self, cfg, dims, dout_p, pre_dout=0.5):
        super().__init__()

        hidden_dim = dims[-1]
        self.pre_dropout = nn.Dropout(pre_dout)
        self.encode_aud = build_mlp(dims, dout_p)

        self.pos_enc = PositionalEncoder(cfg.d_model, cfg.dout_p)

        attn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2,
            dropout=cfg.dout_p, batch_first=True
        )
        self.self_attn = nn.TransformerEncoder(
            encoder_layer=attn_layer,
            num_layers=cfg.num_encoder_layers,
        )
        
    def forward(self, aud, mask=None):
        aud = F.normalize(aud, dim=-1)
        aud = self.pre_dropout(aud)
        aud = self.encode_aud(aud)
        aud = self.pos_enc(aud)
        aud = self.self_attn(aud, src_key_padding_mask=mask)
        return aud


class BottleneckTransformerLayer(nn.Module):

    def __init__(self, cfg):
        super(BottleneckTransformerLayer, self).__init__()

        heads = 4
        d_model = cfg.d_model
        dout = cfg.dout_p

        self.att1 = nn.MultiheadAttention(d_model, num_heads=heads, dropout=dout, batch_first=True)
        self.att3 = nn.MultiheadAttention(d_model, num_heads=heads, dropout=dout, batch_first=True)
        self.att4 = nn.MultiheadAttention(d_model, num_heads=heads, dropout=dout, batch_first=True)
        self.att2 = nn.MultiheadAttention(d_model, num_heads=heads, dropout=dout, batch_first=True)

        self.ffn1 = PositionwiseFeedForward(d_model, d_model*2, dout_p=dout)
        self.ffn2 = PositionwiseFeedForward(d_model, d_model*2, dout_p=dout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

    def forward(self, a, b, t, pe=None, a_mask=None, b_mask=None):
        da = self.norm1(a)
        db = self.norm2(b)
        dt = self.norm3(t)

        ka = da if pe is None else da + pe
        kb = db if pe is None else db + pe

        at = self.att1(dt, ka, da, key_padding_mask=a_mask, need_weights=False)[0]
        bt = self.att2(dt, kb, db, key_padding_mask=b_mask, need_weights=False)[0]

        t = t + at + bt
        dt = self.norm4(t)

        qa = da if pe is None else da + pe
        qb = db if pe is None else db + pe

        a = a + self.att3(qa, dt, dt, need_weights=False)[0]
        b = b + self.att4(qb, dt, dt, need_weights=False)[0]

        da = self.norm5(a)
        db = self.norm6(b)

        a = a + self.ffn1(da)
        b = b + self.ffn2(db)

        return a, b, t


class BottleneckTransformer(nn.Module):

    def __init__(self, cfg):
        super(BottleneckTransformer, self).__init__()

        d_model = cfg.d_model
        num_tokens = 4
        num_layers = cfg.num_encoder_layers

        self.token = nn.Parameter(F.normalize(torch.randn(num_tokens, d_model)))
        self.encoder = nn.ModuleList([
            BottleneckTransformerLayer(cfg)
            for _ in range(num_layers)
        ])

    def forward(self, a, b, a_mask=None, b_mask=None):
        t = self.token.expand(a.size(0), -1, -1)
        for enc in self.encoder:
            a, b, t = enc(a, b, t, a_mask=a_mask, b_mask=b_mask)
        return a, b


class CrossTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        num_tokens = 4
        num_layers = cfg.num_encoder_layers

        self.token_type_embedding = nn.Parameter(F.normalize(torch.randn(2, d_model)))
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=d_model*4,
                dropout=cfg.dout_p, batch_first=True
            ),
            num_layers=cfg.num_encoder_layers,
        )

    def forward(self, a, b, a_mask=None, b_mask=None):
        bs, seq_len, d_model = a.size()
        a = a + self.token_type_embedding[0]
        b = b + self.token_type_embedding[1]

        ab = torch.cat((a,b), dim=1)
        mask = torch.cat((a_mask, b_mask), dim=1)
        ab = self.encoder(ab, src_key_padding_mask=mask)
        a, b = torch.split(ab, seq_len, dim=1)
        return a, b


class AVEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_seg = cfg.num_seg
        self.seg_method = cfg.seg_method
        self.visual_encoder = VisualEncoder(
            cfg, 
            dims=[2048, 512, cfg.d_model],
            dout_p=cfg.dout_p,
            pre_dout=0.5
        )
        self.audio_encoder = AudioEncoder(
            cfg, 
            dims=[128, cfg.d_model],
            dout_p=cfg.dout_p,
            pre_dout=0.2
        )
        self.cross_encoder = BottleneckTransformer(cfg)
        self.visual_weight = 2
        self.audio_weight = 0.5

    
    def forward(self, rgb, flow, aud, vis_mask=None, aud_mask=None):
        v = self.visual_encoder(rgb, flow, vis_mask)
        a = self.audio_encoder(aud, aud_mask)

        v, a = self.cross_encoder(v, a, a_mask=vis_mask, b_mask=aud_mask)

        v = get_seg_feats(v, self.num_seg, vis_mask, method=self.seg_method) * self.visual_weight
        a = get_seg_feats(a, self.num_seg, aud_mask, method=self.seg_method) * self.audio_weight
        return v, a


class AVMapping(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        self.mapping = BridgeConnection(d_model*2, d_model, dout_p=cfg.dout_p)
    
    def forward(self, A, V, S, **kwargs):
        num_sen = S.size()[1]
        AV = torch.cat([A,V], dim=-1)
        AV = self.mapping(AV) # bs, num_seg, d_model
        AV = AV.unsqueeze(1).expand(-1, num_sen, -1, -1)
        return AV


class AVFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        assert self.d_model % self.num_head == 0

        self.d_k = self.d_model // self.num_head

        self.to_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_v = nn.Linear(cfg.d_model, cfg.d_model)

        # self.av_norm = nn.LayerNorm(cfg.d_model)
        self.s_norm = nn.LayerNorm(cfg.d_model)

        self.ff = PositionwiseFeedForward(cfg.d_model, cfg.d_model*2, dout_p=cfg.dout_p)
    
    def forward(self, A, V, S=None):
        """
        input:
            A: bs, num_seg, d_model
            V: bs, num_seg, d_model
            S: bs, num_sen, d_model
        output:
            ret: bs, num_sen, num_seg, d_model
        """
        
        bs, num_seg, d_model = A.size()
        num_sen = 1
        av = torch.stack([A, V], dim=2) # bs, num_seg, 2, d_model
        # av = self.av_norm(av)
        v = self.to_v(av).view(bs, 1, num_seg, 2, self.num_head, self.d_k)

        if S is None:
            # no text embedding, simply average the av embedding
            out = v.mean(dim=-3)
        else:
            # av fusion by text embedding
            num_sen = S.size()[1]
            # use each sentence feature as query to fuse each AV segment
            S = self.s_norm(S)
            q = self.to_q(S ).view(bs, num_sen, 1, 1, self.num_head, self.d_k)
            k = self.to_k(av).view(bs, 1, num_seg, 2, self.num_head, self.d_k)
            # compute attention
            attn = (q*k).sum(-1, keepdim=True) # bs, num_sen, num_seg, 2, num_head, d_k
            attn = attn / np.sqrt(self.d_k)
            attn = F.softmax(attn, dim=-3)
            
            # weighted sum by value
            out = (attn*v).sum(dim=-3) # bs, num_sen, num_seg, num_head, d_k

        out = out.view(bs, num_sen, num_seg, -1) # bs, num_sen, num_seg, d_model

        # fully connected
        out = self.ff(out)
        return out