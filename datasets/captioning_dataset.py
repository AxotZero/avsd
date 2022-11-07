import ast

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torchtext import data

from datasets.load_features import fill_missing_features, load_features_from_npy
from avsd_tan.utils import get_valid_position

def caption_iterator(cfg, batch_size, phase):
    print(f'Contructing caption_iterator for "{phase}" phase')
    
    CAPTION = data.ReversibleField(
        tokenize=str.split, init_token=cfg.start_token, eos_token=cfg.end_token,
        pad_token=cfg.pad_token, lower=False, batch_first=True, is_target=True
    )
    INDEX = data.Field(
        sequential=False, use_vocab=False, batch_first=True
    )
    
    # the order has to be the same as in the table
    fields = [
        ('video_id', None),
        ('caption', CAPTION),
        ('start', None),
        ('end', None),
        ('duration', None),
        ('seq_start', None),
        ('seq_end', None),
        ('phase', None),
        ('idx', INDEX),
    ]

    dataset = data.TabularDataset(
        path=cfg.train_meta_path, format='tsv', skip_header=True, fields=fields,
    )
    if cfg.word_emb_caps:
        CAPTION.build_vocab(dataset.caption, min_freq=cfg.min_freq_caps, vectors=cfg.word_emb_caps)
    else:
        CAPTION.build_vocab(dataset.caption, min_freq=cfg.min_freq_caps)
    train_vocab = CAPTION.vocab

    if phase == 'val':
        dataset = data.TabularDataset(path=cfg.val_meta_path, format='tsv', skip_header=True, fields=fields)
    elif phase == 'test':
        dataset = data.TabularDataset(path=cfg.test_meta_path, format='tsv', skip_header=True, fields=fields)

    if cfg.debug:
        if phase == 'train':
            dataset = data.TabularDataset(
                path=f'{cfg.train_meta_path[:-4]}_debug.csv', format='tsv', skip_header=True, fields=fields,
            )
        elif phase =='val':
            dataset = data.TabularDataset(
                path=f'{cfg.val_meta_path[:-4]}_debug.csv', format='tsv', skip_header=True, fields=fields,
            )

    # sort_key = lambda x: data.interleave_keys(len(x.caption), len(y.caption))
    datasetloader = data.BucketIterator(dataset, batch_size, sort_key=lambda x: 0, 
                                        # device=torch.device(cfg.device), 
                                        device=None,
                                        repeat=False, shuffle=True)
    return train_vocab, datasetloader


class I3DFeaturesDataset(Dataset):
    
    def __init__(self, features_path, feature_name, meta_path, device, pad_idx, get_full_feat, cfg):
        self.cfg = cfg
        self.features_path = features_path
        self.feature_name = f'{feature_name}_features'
        self.feature_names_list = [self.feature_name]
        self.device = device
        self.dataset = pd.read_csv(meta_path, sep='\t')
        self.pad_idx = pad_idx
        self.get_full_feat = get_full_feat
        
        if self.feature_name == 'i3d_features':
            self.feature_size = 2048
        else:
            raise Exception(f'Inspect: "{self.feature_name}"')
    
    def __getitem__(self, indices):
        video_ids, captions, starts, ends, vid_stacks_rgb, vid_stacks_flow = [], [], [], [], [], []

        for idx in indices:
            idx = idx.item()
            video_id, caption, start, end, duration, seq_start, seq_end, _, _ = self.dataset.iloc[idx]
            
            stack = load_features_from_npy(
                self.cfg, self.feature_names_list, video_id, start, end, duration, 
                self.pad_idx, self.get_full_feat
            )

            vid_stack_rgb, vid_stack_flow = stack['rgb'], stack['flow']
            
            # either both None or both are not None (Boolean Equivalence)
            both_are_None = vid_stack_rgb is None and vid_stack_flow is None
            none_is_None = vid_stack_rgb is not None and vid_stack_flow is not None
            assert both_are_None or none_is_None
            
            # # sometimes stack is empty after the filtering. we replace it with noise
            if both_are_None:
                # print(f'RGB and FLOW are None. Zero (1, D) @: {video_id}')
                vid_stack_rgb = fill_missing_features('zero', self.feature_size)
                vid_stack_flow = fill_missing_features('zero', self.feature_size)
    
            # append info for this index to the lists
            video_ids.append(video_id)
            captions.append(caption)
            starts.append(start)
            ends.append(end)
            vid_stacks_rgb.append(vid_stack_rgb)
            vid_stacks_flow.append(vid_stack_flow)
            
        vid_stacks_rgb = pad_sequence(vid_stacks_rgb, batch_first=True, padding_value=self.pad_idx)
        vid_stacks_flow = pad_sequence(vid_stacks_flow, batch_first=True, padding_value=0)
                
        starts = torch.tensor(starts).unsqueeze(1)
        ends = torch.tensor(ends).unsqueeze(1)

        batch_dict = {
            'video_ids': video_ids,
            'captions': captions,
            'starts': starts.to(self.device),
            'ends': ends.to(self.device),
            'feature_stacks': {
                'rgb': vid_stacks_rgb.to(self.device),
                'flow': vid_stacks_flow.to(self.device),
            }
        }
        
        return batch_dict

    def __len__(self):
        return len(self.dataset)
    
class VGGishFeaturesDataset(Dataset):
    
    def __init__(self, features_path, feature_name, meta_path, device, pad_idx, get_full_feat, cfg):
        self.cfg = cfg
        self.features_path = features_path
        self.feature_name = 'vggish_features'
        self.feature_names_list = [self.feature_name]
        self.device = device
        self.dataset = pd.read_csv(meta_path, sep='\t')
        self.pad_idx = pad_idx
        self.get_full_feat = get_full_feat
        self.feature_size = 128
            
    def __getitem__(self, indices):
        video_ids, captions, starts, ends, aud_stacks = [], [], [], [], []

        # [3]
        for idx in indices:
            idx = idx.item()
            video_id, caption, start, end, duration, seq_start, seq_end, _, _ = self.dataset.iloc[idx]
            
            stack = load_features_from_npy(
                self.cfg, self.feature_names_list, video_id, start, end, duration,
                self.pad_idx, self.get_full_feat
            )
            aud_stack = stack['audio']
            
            # sometimes stack is empty after the filtering. we replace it with noise
            if aud_stack is None:
                # print(f'VGGish is None. Zero (1, D) @: {video_id}')
                aud_stack = fill_missing_features('zero', self.feature_size)
    
            # append info for this index to the lists
            video_ids.append(video_id)
            captions.append(caption)
            starts.append(start)
            ends.append(end)
            aud_stacks.append(aud_stack)
            
        # [4] see ActivityNetCaptionsDataset.__getitem__ documentation
        aud_stacks = pad_sequence(aud_stacks, batch_first=True, padding_value=self.pad_idx)
                
        starts = torch.tensor(starts).unsqueeze(1)
        ends = torch.tensor(ends).unsqueeze(1)

        batch_dict = {
            'video_ids': video_ids,
            'captions': captions,
            'starts': starts.to(self.device),
            'ends': ends.to(self.device),
            'feature_stacks': {
                'audio': aud_stacks.to(self.device),
            }
        }

        return batch_dict

    def __len__(self):
        return len(self.dataset)
    
class AudioVideoFeaturesDataset(Dataset):
    
    def __init__(self, feature_pkl, meta_path, device, pad_idx, get_full_feat, cfg):
        self.feature_pkl = feature_pkl
        self.cfg = cfg
        self.num_workers = self.cfg.num_workers
        self.device = device
        self.dataset = pd.read_csv(meta_path, sep='\t')
        self.pad_idx = pad_idx
        self.get_full_feat = get_full_feat
        
        self.video_feature_size = 2048
        self.audio_feature_size = 128

        # self.tan = cfg.tan
        # if self.tan:
        self.num_seg = cfg.num_seg

    def get_seg_feats(self, feat, num_seg=64, method='mean'):
        """
        feat: torch.tensor, it can be audio/visual
        method: 'mean', 'max'
        """
        
        if method == 'mean':
            func = torch.mean
        elif method == 'max':
            func = torch.max
        else:
            raise Exception('method should be one of ["mean", "max"]')
        
        seq_len, hidden_size = feat.shape
        if seq_len == num_seg:
            return feat
        
        ret = torch.zeros((num_seg, hidden_size))
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
        return ret

    def iou(self, interval_1, interval_2):
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    def __getitem__(self, indices):
        video_ids, captions, starts, ends = [], [], [], []
        vid_stacks_rgb, vid_stacks_flow, aud_stacks = [], [], []
        sents_iou_target_stacks = []
        
        # [3]
        for idx in indices:
            idx = idx.item()
            video_id, caption, start, end, duration, seq_start, seq_end, _, _ = self.dataset.iloc[idx]
            
            stack = load_features_from_npy(
                self.feature_pkl, self.cfg, 
                video_id, start, end, duration, self.pad_idx, self.get_full_feat
            )
            vid_stack_rgb, vid_stack_flow, aud_stack = stack['rgb'], stack['flow'], stack['audio']

            # either both None or both are not None (Boolean Equivalence)
            both_are_None = vid_stack_rgb is None and vid_stack_flow is None
            none_is_None = vid_stack_rgb is not None and vid_stack_flow is not None
            assert both_are_None or none_is_None

            # sometimes vid_stack and aud_stack are empty after the filtering. 
            # we replace it with noise.
            # tied with assertion above
            if (vid_stack_rgb is None) and (vid_stack_flow is None):
                # print(f'RGB and FLOW are None. Zero (1, D) @: {video_id}')
                vid_stack_rgb = fill_missing_features('zero', self.video_feature_size)
                vid_stack_flow = fill_missing_features('zero', self.video_feature_size)
            if aud_stack is None:
                # print(f'Audio is None. Zero (1, D) @: {video_id}')
                aud_stack = fill_missing_features('zero', self.audio_feature_size)
            
            # if self.tan:
                # build clip features
            vid_stack_rgb = self.get_seg_feats(vid_stack_rgb, self.num_seg)
            vid_stack_flow = self.get_seg_feats(vid_stack_flow, self.num_seg)
            aud_stack = self.get_seg_feats(aud_stack, self.num_seg)

            valid_position = get_valid_position(self.num_seg)
            seq_start = ast.literal_eval(seq_start)
            seq_end = ast.literal_eval(seq_end)
            sents_iou_target = []
            for s, e in zip(seq_start, seq_end):
                s_frame = round(s / duration * self.num_seg)
                e_frame = round(e / duration * self.num_seg)
                iou_target = []
                for va in valid_position:
                    iou_target.append(self.iou((s_frame, e_frame), va))
                sents_iou_target.append(iou_target)
                

            # append info for this index to the lists
            video_ids.append(video_id)
            captions.append(caption)
            starts.append(start)
            ends.append(end)
            vid_stacks_rgb.append(vid_stack_rgb)
            vid_stacks_flow.append(vid_stack_flow)
            aud_stacks.append(aud_stack)

            # if self.tan:
            sents_iou_target_stacks.append(sents_iou_target)
            
        # [4] see ActivityNetCaptionsDataset.__getitem__ documentation
        # rgb is padded with pad_idx; flow is padded with 0s: expected to be summed later
        # if not self.tan:
        # vid_stacks_rgb = pad_sequence(vid_stacks_rgb, batch_first=True, padding_value=self.pad_idx)
        # vid_stacks_flow = pad_sequence(vid_stacks_flow, batch_first=True, padding_value=0)
        # aud_stacks = pad_sequence(aud_stacks, batch_first=True, padding_value=self.pad_idx)
        vid_stacks_rgb = torch.stack(vid_stacks_rgb, dim=0)
        vid_stacks_flow = torch.stack(vid_stacks_flow, dim=0)
        aud_stacks = torch.stack(aud_stacks, dim=0)

        starts = torch.tensor(starts).unsqueeze(1)
        ends = torch.tensor(ends).unsqueeze(1)

        batch_dict = {
            'video_ids': video_ids,
            'captions': captions,
            'starts': starts,
            'ends': ends,
            'feature_stacks': {
                'rgb': vid_stacks_rgb,
                'flow': vid_stacks_flow,
                'audio': aud_stacks,
            },
            'tan_label': torch.tensor(sents_iou_target_stacks) # bs, num_sent, num_valid
        }

        return batch_dict
        
    def __len__(self):
        return len(self.dataset)


class AVSD10Dataset(Dataset):
    
    def __init__(self, cfg, phase, feature_pkl, get_full_feat):

        '''
            For the doc see the __getitem__.
        '''
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.phase = phase
        self.get_full_feat = get_full_feat

        if phase == 'train':
            self.meta_path = cfg.train_meta_path
            self.batch_size = cfg.train_batch_size
        elif phase == 'val':
            self.meta_path = cfg.val_meta_path
            self.batch_size = cfg.inference_batch_size
        elif phase == 'test':
            self.meta_path = cfg.test_meta_path
            self.batch_size = cfg.inference_batch_size
        else:
            raise NotImplementedError

        # caption dataset *iterator*
        self.train_vocab, self.caption_loader = caption_iterator(cfg, self.batch_size, self.phase)
        
        self.trg_voc_size = len(self.train_vocab)
        self.pad_idx = self.train_vocab.stoi[cfg.pad_token]
        self.start_idx = self.train_vocab.stoi[cfg.start_token]
        self.end_idx = self.train_vocab.stoi[cfg.end_token]
        self.context_start_idx = self.train_vocab.stoi[cfg.context_start_token]
        self.context_end_idx = self.train_vocab.stoi[cfg.context_end_token]

        if cfg.modality == 'video':
            self.features_dataset = I3DFeaturesDataset(
                self.meta_path, torch.device(cfg.device), 
                self.pad_idx, self.get_full_feat, cfg
            )
        elif cfg.modality == 'audio':
            self.features_dataset = VGGishFeaturesDataset(
                self.meta_path, torch.device(cfg.device), 
                self.pad_idx, self.get_full_feat, cfg
            )
        elif cfg.modality == 'audio_video':
            self.features_dataset = AudioVideoFeaturesDataset(
                feature_pkl, self.meta_path, torch.device(cfg.device), 
                self.pad_idx, self.get_full_feat, cfg
            )
        else:
            raise Exception(f'it is not implemented for modality: {cfg.modality}')
            
        # initialize the caption loader iterator
        self.update_iterator() 
        
    def __getitem__(self, index):
        # caption_data = next(self.caption_loader_iter)
        # to_return = self.features_dataset[caption_data.idx]
        # to_return['caption'] = caption_data

        # return to_return

        caption_data = self.caption_datas[index]
        ret = self.features_dataset[caption_data.idx]
        ret['caption'] = caption_data.caption
        return ret


    def __len__(self):
        return len(self.caption_loader)
    
    def update_iterator(self):
        '''This should be called after every epoch'''
        self.caption_datas = list(iter(self.caption_loader))

        
    def dont_collate(self, batch):
        return batch[0]