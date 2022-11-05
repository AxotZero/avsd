import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.blocks import (BridgeConnection, FeatureEmbedder, Identity,
                          PositionwiseFeedForward, VocabularyEmbedder)
from model.generators import Generator



class AVFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        assert self.d_model % self.num_head == 0

        self.d_k = self.d_model // self.num_head

        self.encode_A = nn.Linear(cfg.d_model, cfg.d_model)
        self.encode_V = nn.Linear(cfg.d_model, cfg.d_model)
        self.encode_S = nn.Linear(cfg.d_model, cfg.d_model)

        self.to_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_v = nn.Linear(cfg.d_model, cfg.d_model)

        self.ff = PositionwiseFeedForward(cfg.d_model, cfg.d_model, dout_p=0.0)
    
    def forward(self, A, V, S):
        """
        input:
            A: bs, num_seg, d_model
            V: bs, num_seg, d_model
            S: bs, num_sen, d_model
        output:
            ret: bs, num_sen, num_seg, d_model
        """
        
        # map to same dim
        A = self.encode_A(A)
        V = self.encode_V(V)
        S = self.encode_S(S)
        
        bs, num_seg, d_model = A.size()
        num_sen = S.size()[1]
        
        # use each sentence feature as query to fuse each AV segment
        av = torch.stack([A, V], dim=2) # bs, num_seg, 2, d_model
        q = self.to_q(S ).view(bs, num_sen, 1, 1, self.num_head, self.d_k)
        k = self.to_k(av).view(bs, 1, num_seg, 2, self.num_head, self.d_k)
        v = self.to_v(av).view(bs, 1, num_seg, 2, self.num_head, self.d_k)
        
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