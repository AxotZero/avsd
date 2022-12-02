from pdb import set_trace as bp
import argparse
import ast
import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from tqdm import tqdm

from avsd_tan.utils import get_valid_position_norm, compute_iou
from utils.combine_files import save_pickle


args = argparse.ArgumentParser()
args.add_argument('--csv_path', type=str, default='data/dstc10_train.csv')
args.add_argument('--num_seg', type=int, default=32)
args.add_argument('--min_iou', type=float, default=0.5)
args.add_argument('--max_iou', type=float, default=1.0)
args = args.parse_args()

save_path = f'data/iou_mean_{args.min_iou}-{args.max_iou}_{args.num_seg}.pkl'
if os.path.exists(save_path):
    exit()

print(f'generate iou_mean {save_path}')

# read data
"""
data - element:
    all seq_start and seq_end of this sentence
    seq_start
    seq_end
"""
data = [] 
df = pd.read_csv(args.csv_path, sep='\t')
for i, row in df.iterrows():
    if not row['train_mask']:
        continue
    seq_start = row['seq_start']
    seq_end = row['seq_end']
    duration = row['duration']
    # parse seq_start, seq_end
    seq_start = ast.literal_eval(seq_start)
    seq_end = ast.literal_eval(seq_end)

    seq_start = [i/duration for s in seq_start for i in s]
    seq_end = [i/duration for s in seq_end for i in s]

    data.append((seq_start, seq_end))

# get valid position / num_seg
vps = get_valid_position_norm(args.num_seg)

# compute ious_mean
ious_mean = []
for vs, ve in tqdm(vps):
    ious = []
    for starts, ends in data:
        max_iou = 0
        for s, e in zip(starts, ends):
            iou = compute_iou((s,e), (vs, ve))
            # scale
            iou = (iou - args.min_iou) / (args.max_iou - args.min_iou)
            # clamp
            iou = max(0, min(iou, 1))

            max_iou = max(iou, max_iou)
        ious.append(max_iou)
    ious_mean.append( sum(ious) / len(ious))

save_pickle(ious_mean, save_path)


