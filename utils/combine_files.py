import pickle
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = './data'
FEAT_DIR = f'./data/features'
random.seed(2626)

def save_pickle(dic, save_path, protocol=4):
    print('save pkl to ' + save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f, protocol=protocol)


def load_pickle(load_path):
    print('load pkl from ' + load_path)
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


def get_seg_feats(feat, num_seg=64, method='sample'):
        """
        feat: torch.tensor, it can be audio/visual
        method: 'mean', 'max'
        """
        
        # if method == 'mean':
        #     func = torch.mean
        # elif method == 'max':
        #     def foo(x, dim):
        #         return torch.max(x, dim=dim)[0]
        #     func = foo
        # elif method == 'sample':
        def func(x, dim=0):
            return random.choice(x)

        # func = foo
        # else:
        #     raise Exception('method should be one of ["mean", "max"]')
        
        seq_len, hidden_size = feat.shape
        if seq_len == num_seg:
            return feat
        
        # mask = torch.zeros(num_seg).bool()
        ret = np.zeros((num_seg, hidden_size))
        if seq_len < num_seg:
            ret_idx = 0
            ret_float_idx = 0
            ret_step = num_seg / seq_len
            for seq_idx in range(seq_len):
                ret[ret_idx] = feat[seq_idx]
                # mask[ret_idx] = True
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
                # mask[ret_idx] = True
                seq_idx = round(seq_float_idx)
        # return ret, mask
        return ret


def combine(phase='train'):
    
    if phase == 'train':
        AUDIO_DIR, VIDEO_DIR = 'vggish', 'video_feats'
    elif phase == 'test':
        AUDIO_DIR, VIDEO_DIR = 'vggish_testset', 'video_feats_testset'
    else:
        raise
    
    AUDIO_DIR = f'{FEAT_DIR}/{AUDIO_DIR}'
    VIDEO_DIR = f'{FEAT_DIR}/{VIDEO_DIR}'
    
    ids = [path[:-4] for path in os.listdir(AUDIO_DIR)]

    d = {}
    for _id in tqdm(ids, ncols=60):
        d[_id] = {
            # 'audio': get_seg_feats(np.load(f'{AUDIO_DIR}/{_id}.npy').astype(np.float32), 64),
            # 'rgb': get_seg_feats(np.load(f'{VIDEO_DIR}/{_id}_rgb.npy').astype(np.float32), 64),
            # 'flow': get_seg_feats(np.load(f'{VIDEO_DIR}/{_id}_flow.npy').astype(np.float32), 64),
            'audio': np.load(f'{AUDIO_DIR}/{_id}.npy').astype(np.float32),
            'rgb': np.load(f'{VIDEO_DIR}/{_id}_rgb.npy').astype(np.float32),
            'flow': np.load(f'{VIDEO_DIR}/{_id}_flow.npy').astype(np.float32),
        }

    save_pickle(d, f'{FEAT_DIR}/{phase}2.pkl', protocol=4)

    if phase == 'train':
        
        TRAIN_DF_PATH = f'{DATA_DIR}/dstc10_train.csv'
        VALID_DF_PATH = f'{DATA_DIR}/dstc10_val.csv'

        train_df = pd.read_csv(TRAIN_DF_PATH, sep='\t').iloc[:100]
        valid_df = pd.read_csv(VALID_DF_PATH, sep='\t').iloc[:20]
        train_df.to_csv(f'{DATA_DIR}/dstc10_train_debug.csv', sep='\t', index=None)
        valid_df.to_csv(f'{DATA_DIR}/dstc10_val_debug.csv', sep='\t', index=None)

        debug_ids = list(train_df.video_id) + list(valid_df.video_id)
        debug_dict = {video_id:d[video_id] for video_id in debug_ids}
        save_pickle(debug_dict, f'{FEAT_DIR}/train_debug.pkl')

if __name__ == '__main__':
    combine('train')
    combine('test')
