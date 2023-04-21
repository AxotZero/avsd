
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
from .decoder import UniDecoder, CrossDecoder 
from .rnn import GRU

from loss import TanLoss, TanIouMeanLoss, LabelSmoothing, ContrasitiveLoss


class AVSDTan(nn.Module):
    def __init__(self, cfg, train_dataset):
        super().__init__()
        vocab_size = train_dataset.vocab_size
        self.d_model = cfg.d_model
        self.last_only = cfg.last_only
        # margin of sentence
        self.pad_idx = train_dataset.pad_idx
        self.cls_idx = train_dataset.cls_idx
        self.sent_start_idx = train_dataset.sent_start_idx
        self.sent_end_idx = train_dataset.sent_end_idx
        self.cap_idx = train_dataset.cap_idx

        # encode features
        self.uni_decoder = UniDecoder(cfg, vocab_size, self.cls_idx)
        self.uni_decoder.emb_C.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)

        # encode word embedding
        self.av_encoder = AVEncoder(cfg)
        self.av_fusion = AVFusion(cfg)
        self.tan = TAN(cfg)

        self.cross_decoder = CrossDecoder(cfg)

        self.generator = Generator(cfg.d_model, vocab_size)
        # self.generator = GruGenerator(cfg, vocab_size)


        self.sim_loss = ContrasitiveLoss()
        self.tan_loss = TanLoss(cfg.min_iou, cfg.max_iou)
        self.gen_loss = LabelSmoothing(cfg.smoothing, self.pad_idx, self.cls_idx)


    def get_sent_indices(self, text):
        # specify the index of map2d the word need to attend
        sent_indices = ((text == self.sent_start_idx) | (text == self.cap_idx)).long() 
        sent_indices = torch.cumsum(sent_indices, dim=-1) - 1
        sent_indices = torch.clamp(sent_indices, min=0)
        return sent_indices

    def get_mask(self, text):
        bs, num_word = text.size()
        padding_mask = (text != self.pad_idx)
        mask = torch.ones(1, num_word, num_word)
        mask = torch.tril(mask, 0).bool().to(text.get_device())
        text_mask = padding_mask.unsqueeze(-2) & mask
        # text_mask = torch.ones(bs, num_word, num_word).triu(1).bool().to(text.get_device())
        return padding_mask, text_mask

    def embed_map2d(self, rgb, flow, audio, sent_feats=None, visual_mask=None, audio_mask=None):
        V, A = self.av_encoder(
            rgb, flow, audio, 
            vis_mask=visual_mask, aud_mask=audio_mask
        ) # bs, num_seg, d_video for A and V

        AV = self.av_fusion(A, V, sent_feats) # bs, num_sen, num_seg, d_model
        map2d, video_emb = self.tan(AV) # bs, num_sent, num_valid, d_model
        return map2d, video_emb
        # return AV, None

    def forward(self, 
                feats=None, visual_mask=None, audio_mask=None,  # video, audio feature
                dialog_x=None, dialog_y=None,                   # dialog
                caption_x=None, caption_y=None,                 # caption
                tan_target=None, tan_mask=None,                 # tan
                map2d=None, compute_loss=True, ret_map2d=False):# return something

        rgb, flow, audio = feats['rgb'], feats['flow'], feats['audio']
        bs = rgb.size(0)

        # if caption_x is not None:
        #     pad_mask, text_mask = self.get_mask(caption_x)
        #     embs, caption_emb = self.uni_decoder(caption_x, text_mask, get_caption_emb=True)
        #     cap_map2d, video_emb = self.embed_map2d(rgb, flow, audio, None, visual_mask, audio_mask)            
        #     sent_indices = self.get_sent_indices(caption_x)
        #     embs, _ = self.cross_decoder(embs, 
        #                                  cap_map2d, 
        #                                  pad_mask, 
        #                                  text_mask,
        #                                  sent_indices)
        #     gen_caption = self.generator(embs)
        

        if dialog_x is not None:
            pad_mask, text_mask = self.get_mask(dialog_x)
            embs = self.uni_decoder(dialog_x, text_mask, get_caption_emb=False)

            if map2d is None:
                sent_mask = dialog_x == self.sent_end_idx
                sent_feats = embs[sent_mask].view(bs, -1, self.d_model) # bs, num_sent, d_dim
                map2d, _ = self.embed_map2d(rgb, flow, audio, sent_feats, visual_mask, audio_mask)

            sent_indices = self.get_sent_indices(dialog_x)
            embs, attn_w = self.cross_decoder(embs, 
                                              map2d, 
                                              pad_mask, 
                                              text_mask,
                                              sent_indices)
            gen_dialog = self.generator(embs)

        ### write loss func
        if compute_loss:
            # sim_loss = self.sim_loss(video_emb, caption_emb)
            tan_loss = self.tan_loss(attn_w, tan_target, tan_mask)
            dialog_loss = self.gen_loss(gen_dialog, dialog_y)
            # caption_loss = self.gen_loss(gen_caption, caption_y)
            caption_loss = torch.tensor(0.0).to(dialog_loss.get_device())
            sim_loss = torch.tensor(0.0).to(dialog_loss.get_device())
            return sim_loss, tan_loss, dialog_loss, caption_loss
            # return None, None, dialog_loss, caption_loss

        if ret_map2d:
            return gen_dialog, attn_w, map2d
        return gen_dialog, attn_w
