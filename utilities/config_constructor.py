import os
from time import localtime, strftime
from shutil import copytree, ignore_patterns

class Config(object):
    '''
    Note: don't change the methods of this class later in code.
    '''

    def __init__(self, args):
        '''
        Try not to create anything here: like new forders or something
        '''
        self.curr_time = strftime('%y%m%d%H%M%S', localtime())
        self.exp_name = args.exp_name

        self.debug = args.debug

        self.procedure = args.procedure
        # dataset
        self.train_meta_path = args.train_meta_path
        self.val_meta_path = args.val_meta_path
        self.test_meta_path = args.test_meta_path
        self.modality = args.modality
        self.feature_dir = args.feature_dir
        # make them d_video and d_audio
        self.d_vid = args.d_vid
        self.d_aud = args.d_aud
        self.start_token = args.start_token
        self.end_token = args.end_token
        self.pad_token = args.pad_token
        self.sent_start_token = args.sent_start_token
        self.sent_end_token = args.sent_end_token
        self.max_len = args.max_len
        self.min_freq_caps = args.min_freq_caps
        self.pretrained_cap_model_path = args.pretrained_cap_model_path
        self.word_emb_caps = args.word_emb_caps

        # model
        if 'train' in self.procedure:
            self.unfreeze_word_emb = args.unfreeze_word_emb
            self.model = args.model
            self.key_metric = args.key_metric

        self.dout_p = args.dout_p
        self.num_encoder_layers = args.num_encoder_layers
        self.num_decoder_layers = args.num_decoder_layers
        self.use_linear_embedder = args.use_linear_embedder
        if args.use_linear_embedder:
            self.d_model_video = args.d_model_video
            self.d_model_audio = args.d_model_audio
        else:
            self.d_model_video = self.d_vid
            self.d_model_audio = self.d_aud
        self.num_head = args.num_head
        self.d_model = args.d_model
        self.d_model_caps = args.d_model_caps
        if 'video' in self.modality:
            self.d_ff_video = 4*self.d_model_video if args.d_ff_video is None else args.d_ff_video
        if 'audio' in self.modality:
            self.d_ff_audio = 4*self.d_model_audio if args.d_ff_audio is None else args.d_ff_audio
        self.d_ff_caps = 4*self.d_model_caps if args.d_ff_caps is None else args.d_ff_caps
        # training
        self.device_ids = args.device_ids
        self.device = f'cuda:{self.device_ids[0]}'
        self.train_batch_size = args.batch_size * len(self.device_ids)
        self.inference_batch_size = self.train_batch_size
        self.num_workers = args.num_workers
        self.epoch_num = args.epoch_num
        self.one_by_one_starts_at = args.one_by_one_starts_at
        self.early_stop_after = args.early_stop_after
        # criterion
        self.smoothing = args.smoothing  # 0 == cross entropy
        self.grad_clip = args.grad_clip
        # optimizer
        self.optimizer = args.optimizer
        if self.optimizer == 'adam':
            self.beta1, self.beta2 = args.betas
            self.eps = args.eps
            self.weight_decay = args.weight_decay
        elif self.optimizer == 'sgd':
            self.momentum = args.momentum
            self.weight_decay = args.weight_decay
        else:
            raise Exception(f'Undefined optimizer: "{self.optimizer}"')
        # lr scheduler
        self.scheduler = args.scheduler
        if self.scheduler == 'constant':
            self.lr = args.lr
            self.weight_decay = args.weight_decay
        elif self.scheduler == 'reduce_on_plateau':
            self.lr = args.lr
            self.lr_reduce_factor = args.lr_reduce_factor
            self.lr_patience = args.lr_patience
        else:
            raise Exception(f'Undefined scheduler: "{self.scheduler}"')
        # evaluation
        self.reference_paths = args.reference_paths
        self.stopwords = args.stopwords
        self.last_only = args.last_only
        self.region_std_coeff = args.region_std_coeff

        self.num_seg = args.num_seg
        self.num_cnn_layer = args.num_cnn_layer
        self.cnn_kernel_size = args.cnn_kernel_size

        # logging
        # self.to_log = args.to_log if not args.debug else False
        self.to_log = args.to_log
        if args.to_log:
            self.log_dir = args.log_dir
            self.checkpoint_dir = args.log_dir  # the same yes
            self.log_path = os.path.join(self.log_dir, args.exp_name)
            self.model_checkpoint_path =os.path.join(self.checkpoint_dir, args.exp_name)
        else:
            self.log_dir = None
            self.log_path = None
        
        self.seg_method = args.seg_method
        self.wandb = args.wandb
        self.no_sen_fusion = args.no_sen_fusion
        self.min_iou = args.min_iou
        self.max_iou = args.max_iou
        self.num_gru_layers = args.num_gru_layers
        self.decoding_method = args.decoding_method
        self.topp = args.topp
        self.topk = args.topk

        self.sim_weight = args.sim_weight
        self.tan_weight = args.tan_weight
        self.dialog_weight = args.dialog_weight
        self.caption_weight = args.caption_weight
        self.shrank = args.shrank