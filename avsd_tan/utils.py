from pdb import set_trace as bp
from functools import lru_cache
import random

import torch
import numpy as np


def get_seg_feats(feats, num_seg=64, mask=None, method='mean'):
        """
        feat: torch.tensor, it can be audio/visual
        method: 'mean', 'max'
        """
        
        if method == 'mean':
            func = torch.mean
        elif method == 'max':
            def foo(x, dim):
                return torch.max(x, dim=dim)[0]
            func = foo
        elif method == 'sample':
            def foo(x, dim=0):
                return random.choice(x)

            func = foo
        else:
            raise Exception('method should be one of ["mean", "max"]')
        bs, seq_len, hidden_size = feats.shape
        if mask is not None:
            seqs_len = (~mask).sum(dim=-1)
        
        rets = torch.zeros((bs, num_seg, hidden_size)).to(feats.get_device())
        # rets = torch.zeros((bs, num_seg, hidden_size))
        for b in range(bs):
            feat = feats[b]
            ret = rets[b]
            if mask is not None:
                seq_len = int(seqs_len[b])
            if seq_len < num_seg:
                ret_idx = 0
                ret_float_idx = 0
                ret_step = num_seg / seq_len
                for seq_idx in range(seq_len):
                    ret[ret_idx] = feat[seq_idx]
                    ret_float_idx += ret_step
                    ret_idx = round(ret_float_idx)
            else:
                seq_idx = 0
                seq_float_idx = 0
                seq_step = seq_len / num_seg
                for ret_idx in range(num_seg):
                    seq_float_idx += seq_step
                    f = feat[seq_idx: round(seq_float_idx)]
                    ret[ret_idx] = func(f, dim=0)
                    seq_idx = round(seq_float_idx)
        return rets


def compute_iou(interval_1, interval_2):
    start_i, end_i = interval_1[0], interval_1[1]
    start, end = interval_2[0], interval_2[1]
    intersection = max(0, min(end, end_i) - max(start, start_i))
    union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
    iou = float(intersection) / (union + 1e-8)
    return iou


def get_pooling_counts(N=64):
    assert N >= 16 and (N & (N-1) == 0)
    pooling_counts = [15]
    while N > 16:
        N /= 2
        pooling_counts.append(8)
    return pooling_counts 
    

@lru_cache(None)
def get_2d_position(N=64):
    """
    output:
        poolers_range: range of each poolers
        mask2d: mask of valid position
    """
    pooling_counts = get_pooling_counts(N)

    stride, offset = 1, 0
    mask2d = torch.zeros(N, N, dtype=torch.bool)
    mask2d[range(N), range(N)] = 1

    poolers_range = []
    for c in pooling_counts:
        for _ in range(c): 
            # fill a diagonal line 
            offset += stride
            i, j = range(0, N - offset, stride), range(offset, N, stride)
            
            mask2d[i, j] = 1
            poolers_range.append((i, j))
        stride *= 2
    return poolers_range, mask2d

@lru_cache(None)
def get_valid_position(N=64):
    poolers_range, _ = get_2d_position(N)
    valid_position = [(i,i) for i in range(N)]
    for i_range, j_range in poolers_range:
        for i, j in zip(i_range, j_range):
            valid_position.append((i, j))
    valid_position = np.array(valid_position)
    return valid_position


def get_valid_position_norm(N=64):
    vp = get_valid_position(N).copy()
    vp[:, 1] += 1
    vp = vp / float(N)
    return vp 


if __name__ == '__main__':
    get_valid_position(64)