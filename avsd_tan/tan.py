from pdb import set_trace as bp

import torch
import torch.nn as nn
from torch.functional import F

from .utils import get_valid_position, get_2d_position, get_pooling_counts


class Feat2D(nn.Module):
    def __init__(self, cfg, poolers_position):
        super().__init__()
        self.poolers_position = poolers_position

        pooling_counts = get_pooling_counts(cfg.num_seg)
        self.poolers = [nn.MaxPool1d(2,1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            self.poolers.extend(
                [nn.MaxPool1d(3,2)] + [nn.MaxPool1d(2,1) for _ in range(c - 1)]
            )
        self.conv = nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=1, stride=1)

    def forward(self, x):
        """
        input:
            x: bs*num_sent, num_seg, d_model
        output:
            map2d: bs*num_sent, num_seg, num_seg, d_model
        """
        B, D, N = x.size()
        map2d = x.new_zeros(B, D, N, N)
        
        map2d[:, :, range(N), range(N)] = x
        for pooler, (i, j) in zip(self.poolers, self.poolers_position):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return self.conv(map2d)


class Convs(nn.Module):
    def __init__(self, cfg, mask2d, kernel_size=9, num_stack_layers=4): 
        super().__init__()
        k = kernel_size
        input_size = cfg.d_model
        hidden_size = cfg.d_model*2

        # Padding to ensure the dimension of the output map2d
        mask_kernel = torch.ones(1,1,k,k).to(mask2d.device) 
        first_padding = (k - 1) * num_stack_layers // 2

        self.weights = [
            self.mask2weight(mask2d, mask_kernel, padding=first_padding) 
        ]
        self.convs = nn.ModuleList(
            [nn.Conv2d(input_size, hidden_size, kernel_size, padding=first_padding)]
        )
 
        for _ in range(num_stack_layers - 1):
            self.weights.append(self.mask2weight(self.weights[-1] > 0, mask_kernel))
            self.convs.append(nn.Conv2d(hidden_size, hidden_size, k))
        self.fc = nn.Linear(hidden_size, input_size)
        
    def mask2weight(self, mask2d, mask_kernel, padding=0):
        # from the feat2d.py,we can know the mask2d is 4-d
        weight = torch.conv2d(mask2d[None,None,:,:].float(),
            mask_kernel, padding=padding)[0, 0]
        weight[weight > 0] = 1 / weight[weight > 0]
        return weight
    
    def forward(self, x):
        for conv, weight in zip(self.convs, self.weights):
            x = conv(x).relu() * weight
        x = x.permute(0, 2, 3, 1) # bs*sent, d_model, N, N -> # bs*sent,N, N, d_model
        x = self.fc(x)
        return x


class TAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.valid_position = get_valid_position(cfg.num_seg)
        poolers_position, self.mask2d = get_2d_position(cfg.num_seg)
        self.mask2d = self.mask2d.to('cpu')

        self.encode_S = nn.Linear(cfg.d_model, cfg.d_model)
        self.feat2d = Feat2D(cfg, poolers_position)
        self.convs = Convs(cfg, self.mask2d)
        
    def forward(self, AV, S):
        """
        input:
            F: bs, num_sent, num_seg, d_model
            S: bs, num_sent, d_model
        output:
            map2d: bs, num_sent, num_valid, d_model
        """
        
        bs, num_sent, num_seg, d_model = AV.size()
        # let batch processing more convinience
        AV = AV.view(-1, num_seg, d_model)
        S = S.view(-1, d_model)
        
        # build map2d
        AV = AV.transpose(-1, -2) # for cnn and pooling
        map2d = self.feat2d(AV) # bs*num_sent, d_model, num_seg, num_seg, 
        S = self.encode_S(S)
        
        # Fuse sentence and feature by Hamard Product
        map2d = map2d * S[:, :, None, None]
        map2d = F.normalize(map2d)
        
        # conv
        map2d = self.convs(map2d) # bs*sent, N, N, d_model
        
        # get num_sent back
        map2d = map2d.view(bs, num_sent, num_seg, num_seg, d_model)
        
        # return valid_position
        va = self.valid_position
        map2d = map2d[:, :, va[:, 0], va[:, 1]] # bs, num_sent, num_valid, d_model
        
        return map2d 