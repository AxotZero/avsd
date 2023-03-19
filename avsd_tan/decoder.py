from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.multihead_attention import MultiheadedAttention
from model.blocks import ResidualConnection, PositionwiseFeedForward, PositionalEncoder, VocabularyEmbedder, BridgeConnection
from .rnn import GRU


# class UniDecoderLayer(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()

#         # text self attn
#         self.text_att = MultiheadedAttention(cfg.d_model, cfg.d_model, cfg.d_model, cfg.num_head, cfg.dout_p)
#         self.res1 = ResidualConnection(cfg.d_model, cfg.dout_p)

#         # ff
#         self.ff = PositionwiseFeedForward(cfg.d_model, cfg.d_model*2, dout_p=cfg.dout_p)
#         self.res2 = ResidualConnection(cfg.d_model, cfg.dout_p)

#     def forward(self, text, text_mask):
#         """
#         input:
#             text: bs, num_word, d_model
#             av_feat: bs, num_sent, num_valid, d_model
#             text_mask: bs, num_word, d_model
#             attn_sent_index: bs, num_word
#         output:
#             text: bs, num_word, d_model
#         """
        
#         # text self attention + residual
#         text = self.res1(text, lambda x: self.text_att(x,x,x, text_mask))
#         # ff + res
#         text = self.res2(text, self.ff)
#         return text


class UniDecoder(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.gru = GRU(cfg)
    
    def forward(self, text, text_mask=None):
        text = self.gru(text)
        return text


class WordEmb(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.pre_dropout = nn.Dropout(0.3)
        self.emb_C = VocabularyEmbedder(vocab_size, cfg.d_model)
        self.pos_enc_C = PositionalEncoder(cfg.d_model, cfg.dout_p)
    
    def forward(self, text_ids):
        text = self.emb_C(text_ids)
        text = self.pre_dropout(text)
        text = self.pos_enc_C(text)
        return text


class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_head = cfg.num_head
        self.to_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_v = nn.Linear(cfg.d_model, cfg.d_model)

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


class Fusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        assert self.d_model % self.num_head == 0

        self.d_k = self.d_model // self.num_head

        self.to_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.to_v = nn.Linear(cfg.d_model, cfg.d_model)

        self.mod_norm = nn.LayerNorm(cfg.d_model)
        self.q_norm = nn.LayerNorm(cfg.d_model)

    
    def forward(self, mod1, mod2, query):
        """
        input:
            mod1, mod2: bs, num_word, d_model
            query: bs, num_word, d_model
        output:
            ret: bs, num_word, d_model
        """
        
        bs, num_word, d_model = query.size()
        mod = torch.stack([mod1, mod2], dim=2) # bs, num_word, 2, d_model
        mod = self.mod_norm(mod)
        query = self.q_norm(query)

        # use each sentence feature as query to fuse each AV segment
        q = self.to_q(query).view(bs, num_word, 1, self.num_head, self.d_k)
        v = self.to_v(mod  ).view(bs, num_word, 2, self.num_head, self.d_k)
        k = self.to_k(mod  ).view(bs, num_word, 2, self.num_head, self.d_k)
        # compute attention
        attn = (q*k).sum(-1, keepdim=True) # bs, num_word, 2, num_head, d_k
        attn = attn / np.sqrt(self.d_k)
        attn = F.softmax(attn, dim=-3)
        
        # weighted sum by value
        out = (attn*v).sum(dim=-3) # bs, num_word, num_head, d_k

        out = out.view(bs, num_word, -1) # bs, num_sen, d_model

        return out


class CrossDecoderLayer(nn.Module):
    def __init__(self, cfg, is_teacher=True):
        super().__init__()
        self.is_teacher = is_teacher
        self.dropout = nn.Dropout(cfg.dout_p)

        # text self attn
        self.text_att = MultiheadedAttention(cfg.d_model, cfg.d_model, cfg.d_model, cfg.num_head, cfg.dout_p)
        self.res1 = ResidualConnection(cfg.d_model, cfg.dout_p)
        # self.gru = GRU(cfg, num_layers=1)

        # cross attn
        # self.av_weight = nn.Parameter(torch.Tensor([1.]))
        self.cross_attn = CrossAttention(cfg)
        self.dropout = nn.Dropout(cfg.dout_p)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)

        if is_teacher:
            self.cap_norm = nn.LayerNorm(cfg.d_model)
            self.caption_cross_attn = MultiheadedAttention(cfg.d_model, cfg.d_model, cfg.d_model, cfg.num_head, cfg.dout_p)
            self.fusion = Fusion(cfg)

        # ff
        self.ff = PositionwiseFeedForward(cfg.d_model, cfg.d_model*2, dout_p=cfg.dout_p)
        self.res2 = ResidualConnection(cfg.d_model, cfg.dout_p)

    def forward(self, dialog_embs, map2d, caption_embs=None, 
                      dialog_text_mask=None, caption_pad_mask=None, 
                      attn_sent_index=None):
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
        dialog_embs = self.res1(dialog_embs, lambda x: self.text_att(x,x,x, dialog_text_mask))

        # cross_attn + res
        res, attn = self.cross_attn(dialog_embs, map2d, attn_sent_index)
        if self.is_teacher:
            caption_embs = self.cap_norm(caption_embs)
            res2 = self.caption_cross_attn(dialog_embs, caption_embs, caption_embs, caption_pad_mask)
            res = self.fusion(res, res2, dialog_embs)

        dialog_embs = self.norm1(dialog_embs) + self.dropout(self.norm2(res))

        # ff + res
        dialog_embs = self.res2(dialog_embs, self.ff)
        return dialog_embs, attn


class CrossDecoder(nn.Module):
    def __init__(self, cfg, is_teacher):
        super().__init__()
        self.layers = nn.ModuleList([CrossDecoderLayer(cfg, is_teacher) for _ in range(cfg.num_decoder_layers)])
    
    def forward(self, 
                dialog_embs, map2d, caption_embs = None, 
                dialog_pad_mask = None, dialog_text_mask = None, 
                caption_pad_mask= None, caption_text_mask = None,
                attn_sent_index = None):

        attns = []
        hidden_embs = []
        for layer in self.layers:
            dialog_embs, attn = layer(dialog_embs, map2d, caption_embs, 
                                      dialog_text_mask, caption_pad_mask, 
                                      attn_sent_index)
            attns.append(attn)
            hidden_embs.append(dialog_embs)
        
        hidden_embs = hidden_embs[len(hidden_embs)//2:]
        hidden_embs = torch.stack(hidden_embs, dim=1)
        attn = self.compute_sentence_attn_w(attns, dialog_pad_mask.squeeze(), attn_sent_index)

        return dialog_embs, hidden_embs, attn
    
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
        

