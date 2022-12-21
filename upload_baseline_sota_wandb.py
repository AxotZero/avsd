import wandb

# upload layer_norm
run = wandb.init(
    project='avsd', 
    name='layer_norm_w2e-4_dout0.2',
    # config = {},
    # reinit=True
    resume=True
)
wandb.log({
    'test/IoU-1': 39.3260,
    'test/IoU-2': 41.3080,
    'test/Bleu_4': 29.5791,
    'test/METEOR': 21.0594,
    'test/ROUGE_L': 48.2873,
    'test/CIDEr': 68.7906,
})


# upload baseline
# run = wandb.init(
#     project='avsd', 
#     name='baseline',
#     config = {},
#     reinit=True
# )

# wandb.log({
#     'test/IoU-1': 36.1,
#     'test/IoU-2': 38.0,
#     'test/Bleu_4': 24.7,
#     'test/METEOR': 17.1,
#     'test/ROUGE_L': 43.7,
#     'test/CIDEr': 56.6,
# })
# run.finish()


# # upload sota
# run = wandb.init(
#     project='avsd', 
#     name='sota',
#     config = {},
#     reinit=True
# )

# wandb.log({
#     'test/IoU-1': 52.1,
#     'test/IoU-2': 55.0,
#     'test/Bleu_4': 38.5,
#     'test/METEOR': 24.7,
#     'test/ROUGE_L': 53.9,
#     'test/CIDEr': 88.8,
# })
# run.finish()

