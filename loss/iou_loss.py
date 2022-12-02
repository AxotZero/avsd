from torch.functional import F 


class TanLoss(object):
    def __init__(self, min_iou=0.5, max_iou=1.0, ):
        self.min_iou, self.max_iou = min_iou, max_iou

    def scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores, ious):
        """
        scores: bs, num_sent, num_valid
        ious: bs, num_sent, num_valid
        """
        ious = self.scale(ious).clamp(0, 1)
        return F.binary_cross_entropy(scores, ious)


