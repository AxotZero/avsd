from pdb import set_trace as bp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks import (PositionalEncoder, VocabularyEmbedder, 
                          PositionwiseFeedForward, ResidualConnection)
from model.multihead_attention import MultiheadedAttention


class CrossAttentionLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_head = cfg.num_head
        # pre norm
        self.text_norm = nn.LayerNorm(cfg.d_model)
        self.av_norm = nn.LayerNorm(cfg.d_model)

        # qkv
        self.to_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_v = nn.Linear(cfg.d_model, cfg.d_model)
        
        # output
        # self.norm = nn.LayerNorm(cfg.d_model)
        # self.ff = PositionwiseFeedForward(cfg.d_model, cfg.d_model*2, cfg.dout_p)

    
    def forward(self, text, av_feat):
        """
        input:
            text: bs, num_word, d_model
            av_feat: bs, num_valid, d_model
        output:
            out: bs, num_word, d_model,
            attn: bs, num_word, num_valid
        """
        
        bs, num_valid, d_model = av_feat.size()
        num_word = text.size(1)
        d_k = d_model // self.num_head

        text = self.text_norm(text)
        av_feat = self.av_norm(av_feat)
        
        q = self.to_q(text).view(bs, num_word, 1, self.num_head, d_k)
        k = self.to_k(av_feat).view(bs, 1, num_valid, self.num_head, d_k)
        v = self.to_v(av_feat).view(bs, 1, num_valid, self.num_head, d_k)

        attn = (q*k).sum(-1, keepdim=True)
        attn = attn / np.sqrt(d_k) # bs, num_word, num_valid, num_head, 1
        attn = F.sigmoid(attn)

        # model output
        out = (attn*v).sum(dim=-3)
        out = out.view(bs, num_word, d_model)
        # out = self.norm(out)
        # out = self.ff(out)
        
        return out, attn.mean(-2).squeeze(-1) # mean attn weight of each head and squeeze 


class UniDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiheadedAttention(cfg.d_model, cfg.d_model, cfg.d_model, cfg.num_head, cfg.dout_p, cfg.d_model)
        self.res1 = ResidualConnection(cfg.d_model, cfg.dout_p)
    
        self.ff = PositionwiseFeedForward(cfg.d_model, cfg.d_model*2, dout_p=cfg.dout_p)
        self.res2 = ResidualConnection(cfg.d_model, cfg.dout_p)
    
    def forward(self, x, mask):
        x = self.res1(x, lambda y: self.attn(y,y,y, mask))
        x = self.res2(x, self.ff)
        return x


class CrossDecoderLayer(UniDecoderLayer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cross_attn = CrossAttentionLayer(cfg)
        self.dropout = nn.Dropout(cfg.dout_p)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
    
    def forward(self, x, y, mask):
        x = self.res1(x, lambda x: self.attn(x,x,x, mask))

        res, attn = self.cross_attn(x, y)
        x = self.norm1(x) + self.dropout(self.norm2(res))

        x = self.res2(x, self.ff)
        return x, attn


class UniDecoder(nn.Module):
    def __init__(self, cfg, vocab_size, cls_idx):
        super().__init__()
        self.cls_idx = cls_idx
        self.emb_C = VocabularyEmbedder(vocab_size, cfg.d_model)
        self.pos_C = PositionalEncoder(cfg.d_model, dout_p=cfg.dout_p)
        self.pre_dropout = nn.Dropout(0.3)
        self.layers = nn.ModuleList([
            UniDecoderLayer(cfg)
            for _ in range(cfg.num_encoder_layers)
        ])

    def forward(self, text_ids, text_mask, get_caption_emb=False):
        text = self.emb_C(text_ids)
        text = self.pre_dropout(text)
        text = self.pos_C(text)
        for layer in self.layers:
            text = layer(text, text_mask)
        
        if get_caption_emb:
            # bp()
            cls_mask = text_ids == self.cls_idx
            bs, seq_len, emb_dim = text.size()
            caption_emb = text[cls_mask].view(bs, emb_dim)
            return text, caption_emb

        return text


class CrossDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossDecoderLayer(cfg)
            for _ in range(cfg.num_decoder_layers)
        ])

    def forward(self, text, av_feat, text_mask):
        cross_attn_ws = []
        for layer in self.layers:
            text, attn_w = layer(text, av_feat, text_mask)

            cross_attn_ws.append(attn_w)
        return text, cross_attn_ws


class TextEncoder(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()

        self.pre_dropout = nn.Dropout(0.3)
        self.emb_C = VocabularyEmbedder(vocab_size, cfg.d_model)
        self.pos_enc_C = PositionalEncoder(cfg.d_model, cfg.dout_p)

        self.sent_start_idx = cfg.sent_start_idx
        self.sent_end_idx = cfg.sent_end_idx
        # self.gru = GRU(cfg)
    
    def forward(self, text):
        text = self.emb_C(text)
        text = self.pre_dropout(text)
        text = self.pos_enc_C(text)
        return text