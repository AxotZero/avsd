from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.multihead_attention import MultiheadedAttention
from model.blocks import ResidualConnection, PositionwiseFeedForward, PositionalEncoder, VocabularyEmbedder
from .rnn import GRU


class UniDecoder(nn.Module):
    def __init__(self, cfg, vocab_size, cls_idx):
        super().__init__()
        self.cls_idx = cls_idx
        self.pre_dropout = nn.Dropout(0.3)
        self.emb_C = VocabularyEmbedder(vocab_size, cfg.d_model)
        self.pos_enc_C = PositionalEncoder(cfg.d_model, cfg.dout_p)
        self.gru = GRU(cfg)
    
    def forward(self, text_indices, text_mask=None, get_caption_emb=False):
        text = self.emb_C(text_indices)
        text = self.pre_dropout(text)
        text = self.pos_enc_C(text)
        text = self.gru(text)

        if get_caption_emb:
            cls_mask = text_indices == self.cls_idx
            bs, seq_len, emb_dim = text.size()
            caption_emb = text[cls_mask].view(bs, emb_dim)
            return text, caption_emb

        return text


class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_head = cfg.num_head
        self.to_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_v = nn.Linear(cfg.d_model, cfg.d_model)
        # self.norm = nn.LayerNorm(cfg.d_model)

        self.text_norm = nn.LayerNorm(cfg.d_model)
        self.av_norm = nn.LayerNorm(cfg.d_model)

        self.dropout = nn.Dropout(cfg.dout_p)
    
    def forward(self, text, av_feat, attn_sent_index):
        """
        input:
            text: bs, num_word, d_model
            av_feat: bs, num_sent, num_valid, d_model
            attn_sent_index: bs, num_word
        output:
            out: bs, num_word, d_model,
            attn: bs, num_word, num_valid
        """
        
        bs, num_sen, num_valid, d_model = av_feat.size()
        num_word = text.size()[1]
        d_k = d_model // self.num_head

        text = self.text_norm(text)
        av_feat = self.av_norm(av_feat)
        
        q = self.to_q(text).view(bs, num_word, 1, self.num_head, d_k)
        k = self.to_k(av_feat).view(bs, num_sen, num_valid, self.num_head, d_k)
        v = self.to_v(av_feat).view(bs, num_sen, num_valid, self.num_head, d_k)
        
        ## only attn to given attn_sent_index av_feat 
        # for k and v, get the corresponding feature map of each word for text
        # bp()
        batch_indices = [[i] for i in range(bs)]
        k = k[[batch_indices, attn_sent_index]] # bs, num_word, num_valid, num_head, d_k
        v = v[[batch_indices, attn_sent_index]] # bs, num_word, num_valid, num_head, d_k

        attn = (q*k).sum(-1, keepdim=True)
        attn = attn / np.sqrt(d_k) # bs, num_word, num_valid, num_head, 1
        attn = F.sigmoid(attn)
        # attn = F.softmax(attn, dim=-3)

        # model output
        out = (attn*v).sum(dim=-3)
        out = out.view(bs, num_word, d_model)
        
        return out, attn.mean(-2).squeeze(-1) # mean attn weight of each head and squeeze 


class CrossDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dropout = nn.Dropout(cfg.dout_p)

        # text self attn
        self.text_att = MultiheadedAttention(cfg.d_model, cfg.d_model, cfg.d_model, cfg.num_head, cfg.dout_p, cfg.d_model)
        self.res1 = ResidualConnection(cfg.d_model, cfg.dout_p)

        # cross attn
        # self.av_weight = nn.Parameter(torch.Tensor([1.]))
        self.av_weight = 1
        self.cross_attn = CrossAttention(cfg)
        self.dropout = nn.Dropout(cfg.dout_p)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)


        # ff
        self.ff = PositionwiseFeedForward(cfg.d_model, cfg.d_model*2, dout_p=cfg.dout_p)
        self.res2 = ResidualConnection(cfg.d_model, cfg.dout_p)

    def forward(self, text, av_feat, text_mask, attn_sent_index):
        """
        input:
            text: bs, num_word, d_model
            av_feat: bs, num_sent, num_valid, d_model
            text_mask: bs, num_word, d_model
            attn_sent_index: bs, num_word
        output:
            text: bs, num_word, d_model
        """
        
        # text self attention + residual
        text = self.res1(text, lambda x: self.text_att(x,x,x, text_mask))

        # cross_attn + res
        res, attn = self.cross_attn(text, av_feat, attn_sent_index)
        text = self.norm1(text) + self.dropout(self.norm2(res) * self.av_weight)

        # ff + res
        text = self.res2(text, self.ff)

        return text, attn


class CrossDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([CrossDecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])
    
    def compute_sentence_attn_w(self, attn, padding_mask, attn_sent_index):
        """
        input:
            attn: num_decoder_layer, bs, num_word, num_valid
            attn_sent_index: bs, num_word
            text_mask: bs, num_word
        output:
            attn: bs, num_sent, num_valid # average of attn_weight of given sentence
        """
        attn = torch.stack(attn, dim=0).mean(dim=0) # bs, num_word, num_valid
        
        bs, _, num_valid = attn.size()
        num_sent = torch.max(attn_sent_index) + 1
        
        sent_attn = []    
        for sent_idx in range(num_sent):
            sent_mask = (attn_sent_index == sent_idx) & padding_mask
            attn2 = attn.clone()
            attn2[~sent_mask] = 0
            num_word_of_sent = sent_mask.float().sum(dim=-1, keepdim=True) # bs, 1
            attn2 = attn2.sum(1) / num_word_of_sent # bs, num_valid
            sent_attn.append(attn2)
        sent_attn = torch.stack(sent_attn, dim=1) # bs, num_sent, num_valid
        return sent_attn
    
    def forward(self, text, av_feat, padding_mask=None, text_mask=None, attn_sent_index=None):
        attns = []
        for decoder in self.layers:
            text, attn = decoder(text, av_feat, text_mask, attn_sent_index)
            attns.append(attn)
            
        attn = self.compute_sentence_attn_w(attns, padding_mask, attn_sent_index)
        return text, attn
        

