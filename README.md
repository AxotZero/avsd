# 2D-MapFormer
![image](https://github.com/AxotZero/avsd/assets/41388159/39959d8c-e6f6-4a4b-8fb5-6b5ec3ed9c53)


Source Code for my master thesis **"2D-MapFormer: 2D-Map Transformer for Audio-Visual Scene-Aware Dialogue and Reasoning"** (Currently not published).

The Source Code is derived from 
* AVSD-DSTC10 Baseline: [Link](https://github.com/ankitshah009/AVSD-DSTC10_baseline)
* 2D-Tan module: [Link](https://github.com/chenjoya/2dtan)

## Usage
1. Requirments
    - conda
    - wandb
2. Environments Setting
    ```=shell
    . ./setup.sh
    ```
3. Download I3D and VGGish pretrained features
    ```=shell
    . ./download_data.sh
    python3 utils/combine_files.py # combine feature files into ./data/features/train.pkl and ./data/features/test.pkl
    ```
4. Train model
    1. Specify the `exp_name` in the `run.sh`. The trained model and model outputs will stored in `./log/{exp_name}/`. It will also be the experiment name of wandb
    2. Specify the `procedure='train_test'`
    3. Specify other hyperparameters. Please see `run.sh` and `main.py` for more details.
    4. run `. ./run.sh`.
        1. It will run training and testing automatically
        2. You will see the following procedure in the command line
            ```=shell
            train 15, tan:0.125, dig:2.272: 100%|█████| 4787/4787 [21:15<00:00,  3.75it/s]
            train 15, tan:0.112, dig:2.153
            val   15, tan:0.087, dig:1.985: 100%|█████| 1117/1117 [06:12<00:00,  3.00it/s]
            val   15, tan:0.109, dig:2.295
            The best metric was  for 0 epochs.
            Expected early stop @ 19
            train 16, tan:0.094, dig:2.097: 100%|█████| 4787/4787 [21:10<00:00,  3.77it/s]
            train 16, tan:0.112, dig:2.136
            val   16, tan:0.088, dig:2.005: 100%|█████| 1117/1117 [06:11<00:00,  3.01it/s]
            val   16, tan:0.109, dig:2.298
            ```
        3. You will see the following test result in the command line
            ```=shell
            DSTC10_beam_search result:
            | Bleu_1: 68.7000
            | Bleu_2: 55.5832
            | Bleu_3: 45.4938
            | Bleu_4: 37.5887
            | METEOR: 24.3038
            | ROUGE_L: 53.4955
            | CIDEr: 86.9928
            | IoU-1: 54.7007
            | IoU-2: 57.6148
            ```

## Model Architecture

| ![image](https://github.com/AxotZero/avsd/assets/41388159/39959d8c-e6f6-4a4b-8fb5-6b5ec3ed9c53) | 
|:--:| 
| Model Overview|

|![image](https://github.com/AxotZero/avsd/assets/41388159/5198839e-f96d-4d56-9fae-555c297b5eba)| 
|:--:| 
| Audio Visual Encoder|

| ![image](https://github.com/AxotZero/avsd/assets/41388159/efd7e99b-5d75-46fd-b2dc-a1d6a89d2a9e)| 
|:--:| 
| Sentence Cross Attention|


| ![image](https://github.com/AxotZero/avsd/assets/41388159/0ddbc65e-690a-488f-9e4e-f73a6f072eca)| 
|:--:| 
| Update Gate|

