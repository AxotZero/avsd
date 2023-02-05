import torch.nn as nn
import torch.nn.functional as F

from avsd_tan.rnn import GRU


class Generator(nn.Module):

    def __init__(self, d_model, voc_size):
        super(Generator, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, voc_size)
        print('Using vanilla Generator')

    def forward(self, x):
        '''
        Inputs:
            x: (B, Sc, Dc)
        Outputs:
            (B, seq_len, voc_size)
        '''
        x = self.norm(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)



class GruGenerator(nn.Module):

    def __init__(self, cfg, voc_size):
        super().__init__()
        self.gru = GRU(cfg)
        self.linear = nn.Linear(cfg.d_model, voc_size)
        
        print('Using GRU Generator')

    def forward(self, x):
        '''
        Inputs:
            x: (B, Sc, Dc)
        Outputs:
            (B, seq_len, voc_size)
        '''
        x = self.gru(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)