
# import sys
# import os
# # print(os.getcwd())
# # sys.path.append(os.path.abspath("/media/hd03/axot_data/AVSD-DSTC10_baseline/model"))

import torch
import torch.nn as nn

from model.blocks import (FeatureEmbedder, Identity,
                          PositionalEncoder, VocabularyEmbedder)
from model.generators import Generator, GruGenerator
from .av_fusion import AVFusion
from .tan import TAN
from .decoder import Decoder 
from .rnn import GRU


class AVSDTan(nn.Module):
    def __init__(self, cfg, train_dataset):
        super().__init__()
        self.d_model = cfg.d_model
        self.last_only = cfg.last_only
        # margin of sentence
        self.pad_idx = train_dataset.pad_idx
        self.context_start_idx = train_dataset.context_start_idx
        self.context_end_idx = train_dataset.context_end_idx

        # encode features
        self.emb_A = FeatureEmbedder(cfg.d_aud, cfg.d_model)
        self.emb_V = FeatureEmbedder(cfg.d_vid, cfg.d_model)
        self.emb_C = VocabularyEmbedder(train_dataset.trg_voc_size, cfg.d_model)
        self.pos_enc_A = PositionalEncoder(cfg.d_model, cfg.dout_p)
        self.pos_enc_V = PositionalEncoder(cfg.d_model, cfg.dout_p)
        self.pos_enc_C = PositionalEncoder(cfg.d_model, cfg.dout_p)
        # self.drop_A = nn.Dropout(0.5)
        # self.drop_V = nn.Dropout(0.5)
        # self.drop_T = nn.Dropout(0.3)

        self.drop_A = nn.Dropout(0)
        self.drop_V = nn.Dropout(0)
        self.drop_T = nn.Dropout(0)

        # encode word embedding
        self.gru = GRU(cfg)

        self.av_fusion = AVFusion(cfg)
        self.tan = TAN(cfg)

        self.decoder = Decoder(cfg)

        self.generator = Generator(cfg.d_model, train_dataset.trg_voc_size)
        # self.generator = GruGenerator(cfg, voc_size=train_dataset.trg_voc_size)


    def forward(self, feats, text, padding_mask=None, text_mask=None, map2d=None, ret_map2d=False):
        if padding_mask is None:
            padding_mask = (text != self.pad_idx).bool()
        if text_mask is None:
            bs, num_word = text.size()
            text_mask = text.new_ones((bs, num_word, num_word))

        # get sent feat
        C = text
        C = self.pos_enc_C(self.drop_T(self.emb_C(C))) # bs, num_word, d_cap
        C = self.gru(C) # bs, num_word, d_cap

        if map2d is not None:
            AV = map2d
        else:
            # get av feature
            V = feats['rgb'] + feats['flow']
            A = feats['audio']
            V = self.drop_V(V)
            A = self.drop_A(A)
            
            # get mask for sentence_feature
            sent_mask = (text == self.context_end_idx)
            
            # extract embedding
            A = self.pos_enc_A(self.emb_A(A)) # bs, num_seg, d_audio
            V = self.pos_enc_V(self.emb_V(V)) # bs, num_seg, d_video
            
            # get sentence embedding
            # Warning: if it is test mode, batch size should be 1 or last_only should be True
            S = C[sent_mask.bool()].view(C.size()[0], -1, self.d_model) # bs, num_sent, d_dim
            # if self.last_only:
            #     S = S[:, [-1]]
            
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


        if ret_map2d:
            return C, attn, AV
        return C, attn
