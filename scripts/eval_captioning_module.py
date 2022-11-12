import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.captioning_dataset import AVSD10Dataset
from epoch_loops.captioning_epoch_loops import (teacher_forced_decoder, validation_1by1_loop)
from model.captioning_module import BiModalTransformer, Transformer
from utilities.captioning_utils import timer
from datasets.load_features import load_pickle
from avsd_tan.avsd_tan import AVSDTan

import wandb

def eval_cap(cfg):
    cfg.last_only=True
    cfg.pretrained_cap_model_path= f'{cfg.log_dir}/{cfg.exp_name}/best_cap_model.pt'
    cfg.reference_paths = ['./dstc10avsd_eval/data/test_set4DSTC10-AVSD_multiref+reason.json']
    cfg.unfreeze_word_emb = False
    cfg.inference_batch_size = 1
    cfg.device_ids = [cfg.device_ids[0]]
    cfg.num_workers = 0

    # cfg.inference_batch_size = cfg.train_batch_size // len(cfg.device_ids) * 2
    # cfg.device_ids = [cfg.device_ids[-1]]

    # doing our best to make it replicable
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(cfg.device_ids[0])

    test_pkl = load_pickle(f'{cfg.feature_dir}/test.pkl')
    test_dataset = AVSD10Dataset(cfg, 'test', test_pkl, get_full_feat=False)
    test_loader = DataLoader(test_dataset, collate_fn=test_dataset.dont_collate)

    cap_model_cpt = torch.load(cfg.pretrained_cap_model_path, map_location='cpu')
    model_cfg = cap_model_cpt['config']
    # if cfg.modality == 'audio_video':
    #     model = BiModalTransformer(model_cfg, test_dataset)
    # elif cfg.modality in ['video', 'audio']:
    #     model = Transformer(model_cfg, test_dataset)
    model = AVSDTan(cfg, test_dataset)

    model.to(torch.device(cfg.device))
    model = torch.nn.DataParallel(model, cfg.device_ids)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Number of Parameters: {param_num / 1000000} Mil.')
    model.load_state_dict(cap_model_cpt['model_state_dict'])

    # evaluation (1-by-1 word)
    metrics, duration = validation_1by1_loop(
        cfg, model, test_loader, teacher_forced_decoder, 0
    )
    print ('-' * 25)
    for metric, score in metrics.items():
        print ('| %s: %2.4f' % (metric, 100 * score))
    print ('-' * 25)

    if cfg.wandb:
        for metric, score in metrics.items():
            wandb.log({f'test/{metric}': score * 100})
                
