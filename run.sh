#!/bin/bash

# . conda/bin/activate 

datapath=data/features

exp_name=bs4_d256_w1e-6

## procedure
procedure='train_test'
# procedure='train'
# procedure='test'

## model config
num_seg=32
cnn_kernel_size=5
num_cnn_layer=4
num_layers=2
d_model=256
weight_decay=0.000001

## training 
device_ids='0 1'
batch_size=2 # per device
num_workers=2
epoch_num=60
one_by_one_starts_at=50

## log and debug
# debug="--debug"
debug=""
# dont_log="--dont_log"
dont_log=""
# debug=""
# last_only="--last_only"
last_only=""
wandb="--wandb"
# wandb=""

train_set=./data/train_set4DSTC8-AVSD+reason.json
val_set=./data/valid_set4DSTC10-AVSD+reason.json
test_set=./dstc10avsd_eval/data/test_set4DSTC10-AVSD_multiref+reason.json
log_dir=./log

# check if the log directory exists
if [ -d "${log_dir}/${exp_name}/" ]; then
   echo \"${log_dir}/${exp_name}/\" already exists. Set a new exp_name different from \"${exp_name}\", or remove the directory
   return
fi
# convert data
echo "Coverting json files to csv for the tool"
python utils/generate_csv.py duration_info/duration_Charades_v1_480.csv $train_set train ./data/dstc10_train.csv
python utils/generate_csv.py duration_info/duration_Charades_v1_480.csv $val_set val ./data/dstc10_val.csv
python utils/generate_csv.py duration_info/duration_Charades_vu17_test_480.csv $test_set test $test_csv

# train
echo Start training
python main.py \
 --train_meta_path ./data/dstc10_train.csv \
 --val_meta_path ./data/dstc10_val.csv \
 --test_meta_path ./data/dstc10_test.csv \
 --reference_paths $val_set \
 --procedure $procedure \
 --batch_size $batch_size \
 --num_layer $num_layers \
 --unfreeze_word_emb \
 --d_vid 2048 --d_aud 128 \
 --d_model $d_model \
 --num_seg $num_seg \
 --cnn_kernel_size $cnn_kernel_size \
 --num_cnn_layer $num_cnn_layer \
 --use_linear_embedder \
 --device_ids $device_ids \
 --epoch_num $epoch_num \
 --one_by_one_starts_at $one_by_one_starts_at \
 --stopwords data/stopwords.txt \
 --exp_name $exp_name \
 --log_dir $log_dir \
 --num_workers $num_workers \
 --num_seg $num_seg \
 --weight_decay $weight_decay \
 $debug \
 $dont_log \
 $last_only \
 $wandb \