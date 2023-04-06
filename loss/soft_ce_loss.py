import torch
import torch.nn as nn
import torch.nn.functional as F
# import pdb.set_trace as bp
from pdb import set_trace as bp

class SoftCrossEntropy(nn.Module):
    
    def __init__(self, pad_idx, cls_idx, num_vocab):
        super().__init__()
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx
        self.num_vocab = num_vocab
        
    def forward(self, pred, target, soft_target=None):  # pred (B, S, V), target (B, S)
        # soft_target = target
        if soft_target is None:
            soft_target = F.one_hot(target, num_classes=self.num_vocab)
        else:
            soft_target = F.softmax(soft_target, dim=-1)
        
        target = target.view(-1)
        mask = ~((target == self.pad_idx) | (target == self.cls_idx))
        pred = pred.view(-1, self.num_vocab)[mask]
        soft_target = soft_target.view(-1, self.num_vocab)[mask]
        return torch.sum(torch.mul(-F.log_softmax(pred, dim=-1), soft_target)) / pred.size(0)