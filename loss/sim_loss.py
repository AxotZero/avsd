import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import einsum

class ContrasitiveLoss(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__() 
        self.temp = nn.Parameter(torch.tensor([temp]).float())
    
    def forward(self, video_embs, text_embs):
        ce = F.cross_entropy

        bs = video_embs.size(0)

        # l2 norm
        video_embs = F.normalize(video_embs, dim=-1)
        text_embs = F.normalize(text_embs, dim=-1)

        sim = einsum('i d, j d -> i j', text_embs, video_embs)
        sim = sim * self.temp.exp()
        contrastive_labels = torch.arange(bs, device=video_embs.get_device())

        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5

        return contrastive_loss