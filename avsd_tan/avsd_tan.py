
# import sys
# import os
# # print(os.getcwd())
# # sys.path.append(os.path.abspath("/media/hd03/axot_data/AVSD-DSTC10_baseline/model"))
from pdb import set_trace as bp
import torch
import torch.nn as nn

from model.generators import Generator, GruGenerator
from .av_encoder import AVEncoder, AVFusion, AVMapping
from .tan import TAN
from .decoder import Decoder 
from .rnn import GRU
from .text_encoder import TextEncoder


class AVSDTan(nn.Module):
    def __init__(self, cfg, train_dataset):
        super().__init__()
        vocab_size = train_dataset.trg_voc_size
        self.d_model = cfg.d_model
        self.last_only = cfg.last_only
        # margin of sentence
        self.pad_idx = train_dataset.pad_idx
        self.context_start_idx = train_dataset.context_start_idx
        self.context_end_idx = train_dataset.context_end_idx

        # encode features
        self.text_encoder = TextEncoder(cfg, vocab_size)

        # encode word embedding
        self.av_encoder = AVEncoder(cfg)
        self.av_fusion = AVFusion(cfg)
        self.tan = TAN(cfg)

        self.decoder = Decoder(cfg)

        # self.generator = Generator(cfg.d_model, vocab_size)
        self.generator = GruGenerator(cfg, voc_size=train_dataset.trg_voc_size)


    def forward(self, 
                feats, text, visual_mask=None, audio_mask=None,
                padding_mask=None, text_mask=None, map2d=None, ret_map2d=False):
        if padding_mask is None:
            padding_mask = (text != self.pad_idx).bool()
        if text_mask is None:
            bs, num_word = text.size()
            text_mask = text.new_ones((bs, num_word, num_word))

        # get sent feat
        C = self.text_encoder(text) # bs, num_word, d_model
        if map2d is not None:
            AV = map2d
        else:
            # sentence mask
            sent_mask = (text == self.context_end_idx)

            # get av feature
            V, A = self.av_encoder(
                feats['rgb'], feats['flow'], feats['audio'], 
                vis_mask=~visual_mask, aud_mask=~audio_mask
            ) # bs, num_seg, d_video for A and V

            
            # get sentence embedding, if it is test mode, batch size should be 1
            S = C[sent_mask.bool()].view(C.size()[0], -1, self.d_model) # bs, num_sent, d_dim
            
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
