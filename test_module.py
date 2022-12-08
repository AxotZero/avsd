
if __name__ == '__main__':

    import pandas as pd
    wandb.init(
            project='avsd', 
            name=cfg.exp_name,
            config = {
                'epoch': cfg.epoch_num,
                'lr': cfg.lr,
                'bs': cfg.train_batch_size,
                'd_model': cfg.d_model,
                'num_encoder_layers': cfg.num_encoder_layers,
                'num_head': cfg.num_head,
                'num_seg': cfg.num_seg,
                'num_cnn_layer': cfg.num_cnn_layer,
                'weight_decay': cfg.weight_decay,
                'optimizer': cfg.optimizer,
            }
        )
    
