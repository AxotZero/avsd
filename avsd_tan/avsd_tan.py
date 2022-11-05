
# import sys
# import os
# # print(os.getcwd())
# # sys.path.append(os.path.abspath("/media/hd03/axot_data/AVSD-DSTC10_baseline/model"))

import torch
import torch.nn as nn

from model.blocks import (BridgeConnection, FeatureEmbedder, Identity,
                          PositionalEncoder, VocabularyEmbedder)
from model.generators import Generator
from .av_fusion import AVFusion
from .tan import TAN
from .decoder import Decoder 
from .rnn import GRU


class MainModel(nn.Module):
    def __init__(self, cfg, train_dataset):
        super().__init__()
        self.d_model = cfg.d_model
        self.last_only = cfg.last_only
        # margin of sentence
        self.context_start_idx = train_dataset.context_start_idx
        self.context_end_idx = train_dataset.context_end_idx

        # encode features
        self.emb_A = FeatureEmbedder(cfg.d_aud, cfg.d_model)
        self.emb_V = FeatureEmbedder(cfg.d_vid, cfg.d_model)
        self.emb_C = VocabularyEmbedder(train_dataset.trg_voc_size, cfg.d_model)
        self.pos_enc_A = PositionalEncoder(cfg.d_model, cfg.dout_p)
        self.pos_enc_V = PositionalEncoder(cfg.d_model, cfg.dout_p)
        self.pos_enc_C = PositionalEncoder(cfg.d_model, cfg.dout_p)

        # encode word embedding
        self.gru = GRU(cfg)

        self.av_fusion = AVFusion(cfg)
        self.tan = TAN(cfg)

        self.decoder = Decoder(cfg)

        self.generator = Generator(cfg.d_model, train_dataset.trg_voc_size)


    def forward(self, feats, text, padding_mask=None, text_mask=None, last_only=False):
        if padding_mask is None:
            padding_mask = text.new_ones(text.size())
        if text_mask is None:
            bs, num_word = text.size()
            text_mask = text.new_ones((bs, num_word, num_word))


        # get feature
        V = feats['rgb'] + feats['flow']
        A = feats['audio']
        C = text
        
        # get mask for sentence_feature
        sent_mask = (text == self.context_end_idx)
        
        # extract embedding
        A = self.pos_enc_A(self.emb_A(A)) # bs, num_seg, d_audio
        V = self.pos_enc_V(self.emb_V(V)) # bs, num_seg, d_video
        C = self.pos_enc_C(self.emb_C(C)) # bs, num_word, d_cap
        C = self.gru(C) # bs, num_word, d_cap
        
        # get sentence embedding
        # Warning: if it is test mode, batch size should be 1 or last_only should be True
        S = C[sent_mask.bool()].view(C.size()[0], -1, self.d_model) # bs, num_sent, d_dim
        if self.last_only:
            S = S[:, [-1]]
        
        # get 2d_tan feature
        AV = self.av_fusion(A, V, S) # bs, num_sen, num_seg, d_model
        AV = self.tan(AV, S) # bs, num_sent, num_valid, d_model
        
        # specify the index of map2d the word need to attend
        attn_sent_index = (text == self.context_start_idx).long() 
        for i in range(1, text.size()[1]):
            attn_sent_index[:, i] += attn_sent_index[:, i-1]
        attn_sent_index -= 1
        
        # decode and return attention weight of each sentence
        C, attn = self.decoder(C, AV, padding_mask, text_mask, attn_sent_index)
            
        C = self.generator(C)
        return C, attn
