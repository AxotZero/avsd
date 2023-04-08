#!/bin/bash

# . conda/bin/activate 

datapath=data/features

exp_name=test_re
procedure='train_test'
# procedure='test'
device_ids='2'
num_layers=2

weight_decay=0.0001
lr=0.0001

num_workers=2
batch_size=8
epoch_num=100
one_by_one_starts_at=100
# wandb=""
# debug="--debug"
# dont_log="--dont_log"
dont_log=""
debug=""
wandb="--wandb"

decoder="greedy_decoder"


train_set=./data/train_set4DSTC8-AVSD+reason.json
val_set=./data/valid_set4DSTC10-AVSD+reason.json
test_set=./data/test_set4DSTC10-AVSD_multiref+reason.json
test_set2=mock_test_set4DSTC10-AVSD_from_DSTC7_multiref.json
log_dir=./log_re

# check if the log directory exists
# if [ -d "${log_dir}/${exp_name}/" ]; then
#    echo \"${log_dir}/${exp_name}/\" already exists. Set a new exp_name different from \"${exp_name}\", or remove the directory
#    return
# fi
# convert data
# echo "Coverting json files to csv for the tool"
# python utils/generate_csv.py duration_info/duration_Charades_v1_480.csv $train_set train ./data/dstc10_train.csv
# python utils/generate_csv.py duration_info/duration_Charades_v1_480.csv $val_set val ./data/dstc10_val.csv
# python utils/generate_csv.py duration_info/duration_Charades_vu17_test_480.csv $test_set test ./data/dstc10_test.csv
# python utils/generate_csv.py duration_info/duration_Charades_vu17_test_480.csv $test_set test ./data/dstc10_test2.csv
# return


# train
echo Start training
python main.py \
 --train_meta_path ./data/dstc10_train.csv \
 --val_meta_path ./data/dstc10_val.csv \
 --test_meta_path ./data/dstc10_test.csv \
 --reference_paths $val_set \
 --procedure $procedure \
 --B $batch_size \
 --N $num_layers \
 --unfreeze_word_emb \
 --d_vid 4096 --d_aud 128 \
 --d_model_video 128 \
 --d_ff_video 512 \
 --d_model_audio 64 \
 --d_ff_audio 128 \
 --d_model_caps 256 \
 --d_ff_caps 1024 \
 --d_model 200 \
 --smoothing 0 \
 --use_linear_embedder \
 --device_ids $device_ids \
 --epoch_num $epoch_num \
 --one_by_one_starts_at $one_by_one_starts_at \
 --stopwords data/stopwords.txt \
 --exp_name $exp_name \
 --log_dir $log_dir \
 --num_workers $num_workers \
 --decoder $decoder \
 --lr $lr \
 --weight_decay $weight_decay \
 $wandb \
 $debug \
 $dont_log


# echo Answer generation and evaluation for $test_set
# python main.py \
#  --train_meta_path ./data/dstc10_train.csv \
#  --test_meta_path $test_csv \
#  --reference_paths $test_set \
#  --video_features_path ${datapath}/video_feats$featpath_suffix/ \
#  --audio_features_path ${datapath}/vggish$featpath_suffix/ \
#  --procedure eval_cap \
#  --pretrained_cap_model_path ${log_dir}/${exp_name}/train_cap/best_cap_model.pt \
#  --B 12 \
#  --stopwords data/stopwords.txt \
#  --exp_name $exp_name \
#  --log_dir $log_dir \
#  --device_ids $device_ids \
#  --num_workers $num_workers \
#  $last_only