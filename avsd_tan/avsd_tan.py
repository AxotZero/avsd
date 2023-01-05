
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
from .text_decoder import TextEncoder, UniDecoder, CrossDecoder
from loss import TanLoss, TanIouMeanLoss, LabelSmoothing, ContrasitiveLoss

def exist(x):
    return x is not None

class AVSDTan(nn.Module):
    def __init__(self, cfg, train_dataset):
        super().__init__()
        vocab_size = train_dataset.vocab_size
        cls_idx = train_dataset.cls_idx
        self.d_model = cfg.d_model
        self.last_only = cfg.last_only

        # margin of sentence
        self.pad_idx = train_dataset.pad_idx
        self.sent_start_idx = train_dataset.sent_start_idx
        self.sent_end_idx = train_dataset.sent_end_idx
        self.cap_idx = train_dataset.cap_idx

        # encode text
        self.text_uni_decoder = UniDecoder(cfg, vocab_size, cls_idx)
        self.text_cross_decoder = CrossDecoder(cfg)

        # encode av
        self.av_encoder = AVEncoder(cfg)
        self.av_fusion = AVFusion(cfg)
        self.tan = TAN(cfg)

        self.generator = Generator(cfg.d_model, train_dataset.vocab_size)

        self.sim_loss = ContrasitiveLoss()
        self.tan_loss = TanLoss(cfg.min_iou, cfg.max_iou)
        self.gen_loss = LabelSmoothing(cfg.smoothing, self.pad_idx, cls_idx)


    def get_sent_indices(self, text):
        sent_indices = ((text == self.sent_start_idx) | (text == self.cap_idx)).long() 
        for i in range(1, text.size()[1]):
            sent_indices[:, i] += sent_indices[:, i-1]
        sent_indices -= 1
        return sent_indices

    def compute_sentence_attn_w(self, attn, padding_mask, sent_index):
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
        num_sent = torch.max(sent_index) + 1
        
        sent_attn = []    
        for sent_idx in range(num_sent):
            sent_mask = (sent_index == sent_idx) & padding_mask
            attn2 = attn.clone()
            attn2[~sent_mask] = 0
            num_word_of_sent = sent_mask.float().sum(dim=-1, keepdim=True) # bs, 1
            attn2 = attn2.sum(1) / num_word_of_sent # bs, num_valid
            sent_attn.append(attn2)
        sent_attn = torch.stack(sent_attn, dim=1) # bs, num_sent, num_valid
        return sent_attn

    def embed_map2d(self, rgb, flow, audio, vis_mask, aud_mask, sent_feats=None):
        V, A = self.av_encoder(
            rgb, flow, audio, 
            vis_mask=vis_mask, aud_mask=aud_mask
        ) # bs, num_seg, d_video for A and V

        AV = self.av_fusion(A, V, sent_feats) # bs, num_sen, num_seg, d_model
        AV = self.tan(AV) # bs, num_sent, num_valid, d_model
        return AV

    def get_mask(self, text):
        padding_mask = (text != self.pad_idx)
        num_word = text.size(-1)
        mask = torch.ones(1, num_word, num_word)
        mask = torch.tril(mask, 0).bool().to(text.get_device())
        text_mask = padding_mask.unsqueeze(-2) & mask
        # text_mask = torch.ones(num_word, num_word).triu(1).bool().to(text.get_device())
        return padding_mask, text_mask

    def forward(self, 
                feats=None, visual_mask=None, audio_mask=None,  # video, audio feature
                dialog_x=None, dialog_y=None,                   # dialog
                caption_x=None, caption_y=None,                 # caption
                tan_target=None, tan_mask=None,                 # tan
                map2d=None, compute_loss=True, ret_map2d=False):# return something
        
        # gen dialog
        if dialog_x is not None:
            sent_indices = self.get_sent_indices(dialog_x)

            dialog_pad_mask, dialog_text_mask = self.get_mask(dialog_x)
            pred_dialog = self.text_uni_decoder(dialog_x, dialog_text_mask, get_caption_emb=False)
            sent_mask = (dialog_x == self.sent_end_idx)
            sent_feats = pred_dialog[sent_mask].view(pred_dialog.size(0) -1, self.d_model)

            if map2d is None:
                map2d, video_emb = self.embed_map2d(
                    feats['rgb'], feats['flow'], feats['audio'], 
                    vis_mask=visual_mask, aud_mask=audio_mask,
                    sent_feats=sent_feats
                )

            pred_dialog, attn_w = self.text_cross_decoder(pred_dialog, map2d, dialog_text_mask)
            pred_dialog = self.generator(pred_dialog)
            
            # process attn_w
            
            attn_w = self.compute_sentence_attn_w(attn_w, dialog_pad_mask, sent_indices)
        
        # gen caption
        if caption_x is not None:
            _, caption_text_mask = self.get_mask(caption_x)
            pred_caption, caption_emb = self.text_uni_decoder(caption_x, caption_text_mask, get_caption_emb=True)
            pred_caption, _ = self.text_cross_decoder(pred_caption, map2d, caption_text_mask)
            pred_caption = self.generator(pred_caption)

        # compute_loss
        if compute_loss:
            sim_loss = self.sim_loss(video_emb, caption_emb)
            tan_loss = self.tan_loss(attn_w, tan_target, tan_mask)
            dialog_loss = self.gen_loss(pred_dialog, dialog_y)
            caption_loss = self.gen_loss(pred_caption, caption_y)
            return sim_loss, tan_loss, dialog_loss, caption_loss
        
        # return
        if ret_map2d:
            return pred_dialog, attn_w, map2d
        return pred_dialog, attn_w
        
        