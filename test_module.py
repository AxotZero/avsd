
if __name__ == '__main__':

    from easydict import EasyDict as edict
    from avsd_tan.avsd_tan import MainModel
    import torch

    cfg = edict({
        'd_model': 32,
        'd_aud': 128,
        'd_vid': 2048,
        'num_head': 4,
        'num_layer': 3,
        'dout_p': 0.2,
        'num_seg': 64,
        'last_only': False
    })
    train_dataset = edict({
        'trg_voc_size': 300,
        'context_start_idx': 1,
        'context_end_idx': 2,
    })

    model = MainModel(cfg, train_dataset)
    bs = 2
    num_seg = 64
    num_word = 21
    feats = {
        'rgb': torch.randn(bs, num_seg, 2048),
        'flow': torch.randn(bs, num_seg, 2048),
        'audio': torch.randn(bs, num_seg, 128),
    }
    text = torch.tensor([
        [1, 3, 2, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 3, 2, 3, 3, 3, 0, 0, 0],
        [1, 3, 2, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 2, 3, 3, 3, 0, 0, 0, 0],
    ]).long()
    # text_mask = (text != 0).bool()
    text_mask=None
    padding_mask = text != 0

    out = model(feats, text, padding_mask=padding_mask, text_mask=None)
    from pdb import set_trace; set_trace()