import pickle
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

FEAT_DIR = '/media/hd03/axot_data/AVSD-DSTC10_baseline/data/features'


def save_pickle(dic, save_path, protocol=4):
    print('save pkl to ' + save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f, protocol=protocol)


def load_pickle(load_path):
    print('load pkl from ' + load_path)
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


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
            'audio': np.load(f'{AUDIO_DIR}/{_id}.npy').astype(np.float32),
            'rgb': np.load(f'{VIDEO_DIR}/{_id}_rgb.npy').astype(np.float32),
            'flow': np.load(f'{VIDEO_DIR}/{_id}_flow.npy').astype(np.float32),
        }

    save_pickle(d, f'{FEAT_DIR}/{phase}.pkl', protocol=4)

    if phase == 'train':
        DATA_DIR = '/media/hd03/axot_data/AVSD-DSTC10_baseline/data'
        TRAIN_DF_PATH = '/media/hd03/axot_data/AVSD-DSTC10_baseline/data/dstc10_train.csv'
        VALID_DF_PATH = '/media/hd03/axot_data/AVSD-DSTC10_baseline/data/dstc10_val.csv'

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
