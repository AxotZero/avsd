import wandb

# upload baseline
run = wandb.init(
    project='avsd', 
    name='baseline',
    config = {},
    reinit=True
)

wandb.log({
    'test/IoU-1': 36.1,
    'test/IoU-2': 38.0,
    'test/Bleu_4': 24.7,
    'test/METEOR': 17.1,
    'test/ROUGE_L': 43.7,
    'test/CIDEr': 56.6,
})
run.finish()


# upload sota
run = wandb.init(
    project='avsd', 
    name='sota',
    config = {},
    reinit=True
)

wandb.log({
    'test/IoU-1': 52.1,
    'test/IoU-2': 55.0,
    'test/Bleu_4': 38.5,
    'test/METEOR': 24.7,
    'test/ROUGE_L': 53.9,
    'test/CIDEr': 88.8,
})
run.finish()

