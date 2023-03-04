import ast
import random
from pdb import set_trace as bp

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
from torch.utils.data.dataset import Dataset
from torchtext import data
import torchtext

from datasets.load_features import fill_missing_features, load_features_from_npy
from avsd_tan.utils import compute_iou, get_valid_position_norm, get_seg_feats

def caption_iterator(cfg, batch_size, phase):
    print(f'Contructing caption_iterator for "{phase}" phase')
    
    CAPTION = data.ReversibleField(
        tokenize=str.split, init_token=cfg.start_token, eos_token=cfg.end_token,
        pad_token=cfg.pad_token, lower=False, batch_first=True, is_target=True
    )
    SUMMARY = data.ReversibleField(
        tokenize=str.split, init_token=cfg.start_token, eos_token=cfg.end_token,
        pad_token=cfg.pad_token, lower=False, batch_first=True, is_target=True
    )
    DIALOG = data.ReversibleField(
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
        ('summary', SUMMARY),
        ('dialog', DIALOG),
        ('start', None),
        ('end', None),
        ('duration', None),
        ('seq_start', None),
        ('seq_end', None),
        ('tan_mask', None),
        ('phase', None),
        ('idx', INDEX),
    ]

    dataset = data.TabularDataset(
        path=cfg.train_meta_path, format='tsv', skip_header=True, fields=fields,
    )

    vectors = torchtext.vocab.GloVe(name='840B', dim=300, cache='./.vector_cache')
    text_fields = [dataset.dialog, dataset.caption, dataset.summary]
    # text_fields = [dataset.dialog]
    # build vocab
    DIALOG.build_vocab(*text_fields, min_freq=cfg.min_freq_caps, vectors=vectors)
    setattr(CAPTION, 'vocab', DIALOG.vocab)
    setattr(SUMMARY, 'vocab', DIALOG.vocab)

    train_vocab = DIALOG.vocab

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
        self.max_feature_seq_len = 512

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
            def foo(x, dim):
                return torch.max(x, dim=dim)[0]
            func = foo
        elif method == 'sample':
            def foo(x, dim=0):
                return random.choice(x)

            func = foo
        else:
            raise Exception('method should be one of ["mean", "max"]')
        
        seq_len, hidden_size = feat.shape
        if seq_len == num_seg:
            return feat
        
        # mask = torch.zeros(num_seg).bool()
        ret = torch.zeros((num_seg, hidden_size))
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

    
    def generate_feature_mask(self, feats_len):
        max_feat_len = max(feats_len)
        masks = []
        for feat_len in feats_len:
            mask = torch.tensor(
                [0]*feat_len + [1]*(max_feat_len-feat_len)
            ).bool()
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks
    
    def __getitem__(self, indices):
        video_ids, captions, summarys, dialogs, starts, ends =[], [], [], [], [], []
        vid_stacks_rgb, vid_stacks_flow, aud_stacks = [], [], []
        sents_iou_target_stacks = []
        tan_masks = []
        
        # [3]
        for idx in indices:
            idx = idx.item()
            video_id, caption, summary, dialog, start, end, duration, seq_starts, seq_ends, tan_mask, _, _ = self.dataset.iloc[idx]
            
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
            
            # get clip feature
            feat_len = max(min(len(vid_stack_rgb), len(vid_stack_flow), len(aud_stack)), self.num_seg)
            # feat_len = self.num_seg
            vid_stack_rgb = self.get_seg_feats(vid_stack_rgb, feat_len, method=self.cfg.seg_method)
            vid_stack_flow = self.get_seg_feats(vid_stack_flow, feat_len, method=self.cfg.seg_method)
            aud_stack = self.get_seg_feats(aud_stack, feat_len, method=self.cfg.seg_method)

            # generate target
            valid_position_norm = get_valid_position_norm(self.num_seg)
            if type(seq_starts) == str and seq_ends.startswith('['):
                seq_starts = ast.literal_eval(seq_starts)
                seq_ends = ast.literal_eval(seq_ends)
                tan_mask = ast.literal_eval(tan_mask)
            else:
                seq_starts = [[-1]]
                seq_ends = [[-1]]
                tan_mask = [-1]

            sents_iou_target = []

            for seq_start, seq_end in zip(seq_starts, seq_ends):
                ious_target = []
                for vs, ve in valid_position_norm:
                    # get max iou of given valid_position
                    max_iou_of_vp = 0
                    for s, e in zip(seq_start, seq_end):
                        s = s / duration
                        e = e / duration
                        iou = compute_iou((s, e), (vs, ve))
                        max_iou_of_vp = max(max_iou_of_vp, iou)
                    ious_target.append(max_iou_of_vp)
                sents_iou_target.append(ious_target)
                

            # append info for this index to the lists
            video_ids.append(video_id)
            summarys.append(summary)
            dialogs.append(dialog)
            captions.append(caption)
            starts.append(int(start))
            ends.append(int(end))
            vid_stacks_rgb.append(vid_stack_rgb)
            vid_stacks_flow.append(vid_stack_flow)
            aud_stacks.append(aud_stack)
            tan_masks.append(tan_mask)

            # if self.tan:
            sents_iou_target_stacks.append(sents_iou_target)
            
        # [4] see ActivityNetCaptionsDataset.__getitem__ documentation
        # rgb is padded with pad_idx; flow is padded with 0s: expected to be summed later
        vids_len = [len(v) for v in vid_stacks_rgb]
        auds_len = [len(v) for v in aud_stacks]
        vid_stacks_rgb = pad_sequence(vid_stacks_rgb, batch_first=True, padding_value=self.pad_idx)
        vid_stacks_flow = pad_sequence(vid_stacks_flow, batch_first=True, padding_value=0)
        aud_stacks = pad_sequence(aud_stacks, batch_first=True, padding_value=self.pad_idx)

        # generate visual and audio mask
        vids_mask = self.generate_feature_mask(vids_len)
        auds_mask = self.generate_feature_mask(auds_len)

        starts = torch.tensor(list(starts)).unsqueeze(1)
        ends = torch.tensor(list(ends)).unsqueeze(1)
        tan_masks = torch.tensor(tan_masks).bool()
        
        batch_dict = {
            'video_ids': video_ids,
            'captions': captions,
            'summarys': summarys,
            'dialogs': dialogs,
            'starts': starts,
            'ends': ends,
            'feature_stacks': {
                'rgb': vid_stacks_rgb,
                'flow': vid_stacks_flow,
                'audio': aud_stacks,
            },
            'visual_mask': vids_mask,
            'audio_mask': auds_mask,
            'tan_label': torch.tensor(sents_iou_target_stacks), # bs, num_sent, num_valid
            'tan_mask': tan_masks # bs, num_sent
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
        self.shrank = cfg.shrank
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
        print('num_vocab', len(self.train_vocab))
        
        self.vocab_size = len(self.train_vocab)
        self.pad_idx = self.train_vocab.stoi[cfg.pad_token]
        self.start_idx = self.train_vocab.stoi[cfg.start_token]
        self.end_idx = self.train_vocab.stoi[cfg.end_token]
        self.sent_start_idx = self.train_vocab.stoi[cfg.sent_start_token]
        self.sent_end_idx = self.train_vocab.stoi[cfg.sent_end_token]
        self.cls_idx = self.train_vocab.stoi['CLS']
        self.cap_idx = self.train_vocab.stoi['C:']
        self.sum_idx = self.train_vocab.stoi['S:']
        # bp()
        self.features_dataset = AudioVideoFeaturesDataset(
            feature_pkl, self.meta_path, torch.device(cfg.device), 
            self.pad_idx, self.get_full_feat, cfg
        )
            
        # initialize the caption loader iterator
        self.update_iterator() 
        
    def __getitem__(self, index):
        caption_data = self.caption_datas[index]
        ret = self.features_dataset[caption_data.idx]
        ret['caption'] = caption_data.caption
        ret['summary'] = caption_data.summary
        ret['dialog'] = caption_data.dialog

        if self.phase in ('train', 'val') and self.shrank:
            shrank_ids = sorted(random.sample(range(10), random.randint(1, 10)))
            shranked_dialogs = []
            for dialog in caption_data.dialog:
                shranked_dialog = []
                sent_count = -2
                start_idx = 1
                for i, token in enumerate(dialog):
                    if token in (self.sent_start_idx, self.end_idx):
                        sent_count += 1
                        if sent_count in shrank_ids:
                            end_idx = i+1 if token == self.end_idx else i
                            shranked_dialog.append(dialog[start_idx:end_idx])
                        start_idx = i
                shranked_dialog = torch.cat(shranked_dialog, dim=0)
                shranked_dialog = torch.cat([torch.tensor([self.start_idx]).long(), shranked_dialog], dim=0)
                shranked_dialogs.append(shranked_dialog)
            
            shranked_dialogs = pad_sequence(shranked_dialogs, batch_first=True, padding_value=self.pad_idx)
            ret['dialog'] = shranked_dialogs
            
            # bp()
            ret['tan_label'] = ret['tan_label'][:, shrank_ids] # bs, num_sent, num_valid
            ret['tan_mask'] = ret['tan_mask'][:, shrank_ids] # bs, num_sent
            
        return ret


    def __len__(self):
        return len(self.caption_loader)
    
    def update_iterator(self):
        '''This should be called after every epoch'''
        self.caption_datas = list(iter(self.caption_loader))

        
    def dont_collate(self, batch):
        return batch[0]