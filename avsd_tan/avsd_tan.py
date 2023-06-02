
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
from .decoder import WordEmb, UniDecoder, CrossDecoder 
from .rnn import GRU

from loss import TanLoss, TanIouMeanLoss, LabelSmoothing, ContrasitiveLoss, SoftCrossEntropy


class AVSDTan(nn.Module):
    def __init__(self, cfg, train_dataset, is_teacher=True):
        super().__init__()
        vocab_size = train_dataset.vocab_size
        self.d_model = cfg.d_model
        self.last_only = cfg.last_only
        self.is_teacher = is_teacher

        # text idx
        self.pad_idx = train_dataset.pad_idx
        self.cls_idx = train_dataset.cls_idx
        self.sent_start_idx = train_dataset.sent_start_idx
        self.sent_end_idx = train_dataset.sent_end_idx
        self.cap_idx = train_dataset.cap_idx
        self.sum_idx = train_dataset.sum_idx

        # loss weight
        self.sim_weight = cfg.sim_weight
        self.tan_weight = cfg.tan_weight
        self.teacher_weight  = cfg.teacher_weight
        self.student_weight  = cfg.student_weight

        # encode features
        self.word_emb = WordEmb(cfg, vocab_size)
        self.word_emb.emb_C.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)
        self.dialog_uni_decoder = UniDecoder(cfg, vocab_size)
        self.caption_uni_decoder = UniDecoder(cfg, vocab_size)

        # encode word embedding
        self.av_encoder = AVEncoder(cfg, self.pad_idx)
        if cfg.av_mapping:
            self.av_fusion = AVMapping(cfg)
        else:
            self.av_fusion = AVFusion(cfg)
        self.tan = TAN(cfg)

        self.cross_decoder = CrossDecoder(cfg, self.is_teacher)

        self.generator = Generator(cfg.d_model, vocab_size)

    def forward(self, 
                feats=None, visual_mask=None, audio_mask=None,  # video, audio feature
                dialog_x=None, caption_x=None, 
                map2d=None):

        rgb, flow, audio = feats['rgb'], feats['flow'], feats['audio']
        bs = rgb.size(0)
        
        # encode dialog
        dialog_pad_mask, dialog_text_mask = self.get_mask(dialog_x)
        dialog_embs = self.word_emb(dialog_x)
        dialog_embs = self.dialog_uni_decoder(dialog_embs)

        
        # encode video and audio feature and generate fused-map2d
        if map2d is None:
            sent_mask = dialog_x == self.sent_end_idx
            sent_feats = dialog_embs[sent_mask].view(bs, -1, self.d_model) # bs, num_sent, d_dim
            map2d, _ = self.embed_map2d(rgb, flow, audio, sent_feats, visual_mask, audio_mask)

        # indices which represents each word belongs to which sentence,
        # to assign which map2d the word need to attend
        sent_indices = self.get_sent_indices(dialog_x)

        # embs of each word and the averaged attention weight of each sentence
        if self.is_teacher:
            caption_pad_mask, caption_text_mask = self.get_mask(caption_x)
            caption_embs = self.word_emb(caption_x)
            caption_embs = self.caption_uni_decoder(caption_embs)
        else:
            caption_pad_mask, caption_text_mask, caption_embs = [None]*3


        dialog_embs, hidden_embs, attn_w = self.cross_decoder(dialog_embs, 
                                                                map2d, 
                                                                caption_embs,
                                                                dialog_pad_mask, dialog_text_mask,
                                                                caption_pad_mask, caption_text_mask,
                                                                sent_indices)


        # convert to the probability of predicted word
        gen_dialog = self.generator(dialog_embs)
        return gen_dialog, hidden_embs, map2d, attn_w

    def get_sent_indices(self, text):
        sent_indices = ((text == self.sent_start_idx) | (text == self.cap_idx)).long()
        sent_indices = torch.cumsum(sent_indices, dim=-1) - 1
        sent_indices = torch.clamp(sent_indices, min=0)
        return sent_indices

    def get_mask(self, text):
        bs, num_word = text.size()
        padding_mask = (text != self.pad_idx)
        mask = torch.ones(bs, num_word, num_word)
        mask = torch.tril(mask, 0).bool().to(text.get_device())

        text_mask = padding_mask.unsqueeze(-2) & mask
        text_mask = text_mask.unsqueeze(1)
        padding_mask = padding_mask.view(bs, 1, 1, num_word)
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


class JST(nn.Module):
    def __init__(self, cfg, train_dataset):
        super().__init__()
        self.jst = cfg.jst
        if cfg.jst:
            self.teacher = AVSDTan(cfg, train_dataset, is_teacher=True)
        self.student = AVSDTan(cfg, train_dataset, is_teacher=False)

        self.pad_idx = train_dataset.pad_idx
        self.cls_idx = train_dataset.cls_idx
        self.sent_start_idx = train_dataset.sent_start_idx
        self.sent_end_idx = train_dataset.sent_end_idx
        self.cap_idx = train_dataset.cap_idx

        # loss
        self.sim_loss = nn.MSELoss()
        self.tan_loss = TanLoss(cfg.min_iou, cfg.max_iou)
        self.gen_loss = SoftCrossEntropy(self.pad_idx, self.cls_idx, train_dataset.vocab_size)
        # self.gen_loss = LabelSmoothing(0, self.pad_idx, self.cls_idx)

    def forward(self,
                feats=None, visual_mask=None, audio_mask=None,  # video, audio feature
                dialog_x=None, dialog_y=None,                   # dialog
                tan_target=None, tan_mask=None,                 # tan
                map2d=None, ret_map2d=False, compute_loss=True):

        student_gen_dialog, student_hidden_embs, map2d, attn_w = self.student(
            feats, visual_mask, audio_mask, dialog_x, caption_x=None, map2d=map2d)
        
        if not compute_loss:
            if ret_map2d:
                return student_gen_dialog, attn_w, map2d
            return student_gen_dialog, attn_w
        
        if self.jst:
            teacher_gen_dialog, teacher_hidden_embs, _, _ = self.teacher(
                feats, visual_mask, audio_mask,  dialog_x, caption_x=caption_x, map2d=None)
        
            teacher_gen_loss = self.gen_loss(teacher_gen_dialog, dialog_y)
            student_gen_loss = self.gen_loss(student_gen_dialog, dialog_y, teacher_gen_dialog.detach())
        
            sim_loss = self.sim_loss(student_hidden_embs, teacher_hidden_embs)
        else:
            student_gen_loss = self.gen_loss(student_gen_dialog, dialog_y)
            teacher_gen_loss = torch.tensor(0.0).to(dialog_x.get_device())
            sim_loss = torch.tensor(0.0).to(dialog_x.get_device())
        tan_loss = self.tan_loss(attn_w, tan_target, tan_mask)

        return teacher_gen_loss, student_gen_loss, sim_loss, tan_loss
