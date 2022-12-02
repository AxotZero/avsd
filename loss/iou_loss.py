import torch
from torch.functional import F 


class TanLoss(object):
    def __init__(self, min_iou=0.5, max_iou=1.0, iou_mean=None, device=None):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.epsilon = 1e-7
        self.iou_mean = torch.tensor(iou_mean).to(device).view(1, 1, -1) # 1, 1, num_valid
        

    def scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores, ious, train_mask):
        """
        scores: bs, num_sent, num_valid
        ious: bs, num_sent, num_valid
        """
        bs, num_sent, num_valid = scores.size()
        
        train_mask = train_mask.view(bs*num_sent)
        scores = scores.view(bs*num_sent, num_valid)[train_mask]
        ious = ious.view(bs*num_sent, num_valid)[train_mask]

        ious = self.scale(ious).clamp(0, 1)

        loss_pos = -1 * torch.mean(ious * torch.log(scores + self.epsilon) * (1-self.iou_mean))
        loss_neg = -1 * torch.mean((1-ious) * torch.log((1-scores) + self.epsilon) * self.iou_mean)
        loss = loss_pos + loss_neg
        return loss

        # return F.binary_cross_entropy(scores, ious)


