import numpy as np
import sys
import torch
from torch.utils.data import DataLoader

from datasets.captioning_dataset import AVSD10Dataset
from epoch_loops.captioning_epoch_loops import (greedy_decoder, teacher_forced_decoder, save_model, training_loop, validation_1by1_loop, validation_next_word_loop)
from loss.label_smoothing import LabelSmoothing
from loss.iou_loss import TanLoss
from model.captioning_module import BiModalTransformer, Transformer
from utilities.captioning_utils import timer
from datasets.load_features import load_pickle
from avsd_tan.avsd_tan import AVSDTan

import wandb

def train_cap(cfg):
    if cfg.wandb:
        wandb.init(
            project='avsd', 
            name=cfg.exp_name,
            config = {
                'epoch': cfg.epoch_num,
                'lr': cfg.lr,
                'bs': cfg.train_batch_size,
                'd_model': cfg.d_model,
                'num_layer': cfg.num_layer,
                'num_head': cfg.num_head,
                'num_seg': cfg.num_seg,
                'num_cnn_layer': cfg.num_cnn_layer,
                'dout_p': cfg.dout_p, 
                'weight_decay': cfg.weight_decay,
                'optimizer': cfg.optimizer,
            }
        )

    # doing our best to make it replicable
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # preventing PyTorch from allocating memory on the default device (cuda:0) when the desired 
    # cuda id for training is not 0.
    torch.cuda.set_device(cfg.device_ids[0])

    # data loader
    train_pkl = load_pickle(f'{cfg.feature_dir}/train{"_debug" if cfg.debug else ""}.pkl')
    train_dataset = AVSD10Dataset(cfg, 'train', train_pkl, get_full_feat=False)
    val_dataset = AVSD10Dataset(cfg, 'val', train_pkl, get_full_feat=False)
    train_loader = DataLoader(train_dataset, num_workers=cfg.num_workers, collate_fn=train_dataset.dont_collate)
    val_loader = DataLoader(val_dataset, num_workers=cfg.num_workers, collate_fn=val_dataset.dont_collate)

    if cfg.pretrained_cap_model_path is not None:
        cap_model_cpt = torch.load(cfg.pretrained_cap_model_path, map_location='cpu')
        model_cfg = cap_model_cpt['config']
    else:
        cap_model_cpt = None
        model_cfg = cfg

    # if cfg.modality == 'audio_video':
    #     model = BiModalTransformer(model_cfg, train_dataset)
    # elif cfg.modality in ['video', 'audio']:
    #     model = Transformer(model_cfg, train_dataset)

    model = AVSDTan(cfg, train_dataset)

    gen_criterion = LabelSmoothing(cfg.smoothing, train_dataset.pad_idx)
    tan_criterion = TanLoss()
    
    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), cfg.lr, (cfg.beta1, cfg.beta2), cfg.eps,
                                     weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), cfg.lr, cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    
    if cfg.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.lr_reduce_factor, patience=cfg.lr_patience
        )
    else:
        scheduler = None

    model.to(torch.device(cfg.device))
    model = torch.nn.DataParallel(model, cfg.device_ids)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Number of Trainable Parameters: {param_num / 1000000} Mil.')

    if cap_model_cpt is not None:
        model.load_state_dict(cap_model_cpt['model_state_dict'])

    # keeping track of the best model 
    best_metric = float('inf')
    # "early stopping" thing
    num_epoch_best_metric_unchanged = 0
    best_epoch = -1

    for epoch in range(cfg.epoch_num):
        print(f'The best metric was  for {num_epoch_best_metric_unchanged} epochs.')
        print(f'Expected early stop @ {epoch+cfg.early_stop_after-num_epoch_best_metric_unchanged}')
        print(f'Started @ {cfg.curr_time}; Current timer: {timer(cfg.curr_time)}')
        
        # stop training if metric hasn't been changed for cfg.early_stop_after epochs
        # if num_epoch_best_metric_unchanged == cfg.early_stop_after:
        #     break

        # train
        training_loop(cfg, model, train_loader, gen_criterion, tan_criterion, optimizer, epoch)
        # validation (next word)
        val_loss = validation_next_word_loop(
            cfg, model, val_loader, greedy_decoder, gen_criterion, tan_criterion, epoch
        )
        if scheduler is not None:
            scheduler.step(val_loss)


        # save_model
        if val_loss < best_metric:
            best_metric = val_loss
            save_model(cfg, epoch, model, optimizer, val_loss,
                        None, train_dataset.trg_voc_size)
            # reset the early stopping criterion
            num_epoch_best_metric_unchanged = 0
            best_epoch = epoch
        else:
            num_epoch_best_metric_unchanged += 1
        print(f'Validation scores at epoch {epoch}',
                f'(best val_loss: %2.4f)' % (best_metric * 100)
                if epoch==best_epoch else '')

        # validation (1-by-1 word)
        if epoch >= cfg.one_by_one_starts_at or (num_epoch_best_metric_unchanged == cfg.early_stop_after):
            val_metrics, duration = validation_1by1_loop(
                cfg, model, val_loader, teacher_forced_decoder, epoch)
            if cfg.wandb:
                wandb.log(
                    {
                        f'val_metric/{metric}': score * 100 
                        for metric, score in val_metrics.items()
                    },
                    step=epoch
                )
                
            # saving the model if it is better than the best so far
            # if best_metric < val_metrics[cfg.key_metric]:
            #     best_metric = val_metrics[cfg.key_metric]
            #     save_model(cfg, epoch, model, optimizer, val_loss,
            #                    val_metrics, train_dataset.trg_voc_size)
            #     # reset the early stopping criterion
            #     num_epoch_best_metric_unchanged = 0
            #     best_epoch = epoch
            # else:
            #     num_epoch_best_metric_unchanged += 1
            # Output the results
            print('-' * 25)
            for metric, score in val_metrics.items():
                print('| %s: %2.4f' % (metric, 100 * score))
            print('-' * 25)
            print('duration_of_1by1:', duration / 60, epoch)
            sys.stdout.flush()

            if (num_epoch_best_metric_unchanged == cfg.early_stop_after):
                break

    print(f'{cfg.curr_time}')
    print(f'best val_loss: %2.4f at epoch {best_epoch}' % (best_metric * 100))
