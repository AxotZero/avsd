import torch
import torch.nn as nn

from .rnn import GRU
from model.blocks import (PositionalEncoder, VocabularyEmbedder)


class TextEncoder(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()

        self.pre_dropout = nn.Dropout(0.3)
        self.emb_C = VocabularyEmbedder(vocab_size, cfg.d_model)
        self.pos_enc_C = PositionalEncoder(cfg.d_model, cfg.dout_p)
        self.gru = GRU(cfg)
    
    def forward(self, text):
        text = self.emb_C(text)
        text = self.pre_dropout(text)
        text = self.pos_enc_C(text)
        text = self.gru(text)
        return text