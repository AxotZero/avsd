#!/bin/bash

# . conda/bin/activate 

datapath=data/features
run=

exp_name=test_multi_process
device_ids='4 5'
num_workers=2

train_set=./data/train_set4DSTC8-AVSD+reason.json
val_set=./data/valid_set4DSTC10-AVSD+reason.json
log_dir=./log

# check if the log directory exists
if [ -d "${log_dir}/${exp_name}/train_cap" ]; then
   echo \"${log_dir}/${exp_name}/train_cap\" already exists. Set a new exp_name different from \"${exp_name}\", or remove the directory
   # exit
fi
# convert data
echo Coverting json files to csv for the tool
python utils/generate_csv.py duration_info/duration_Charades_v1_480.csv $train_set train ./data/dstc10_train.csv
python utils/generate_csv.py duration_info/duration_Charades_v1_480.csv $val_set val ./data/dstc10_val.csv

# train
echo Start training
$run python main.py \
 --train_meta_path ./data/dstc10_train.csv \
 --val_meta_path ./data/dstc10_val.csv \
 --reference_paths $val_set \
 --video_features_path ${datapath}/video_feats/ \
 --audio_features_path ${datapath}/vggish/ \
 --procedure train_cap \
 --B 10 \
 --N 2 \
 --unfreeze_word_emb \
 --d_vid 2048 --d_aud 128 \
 --d_model_video 128 \
 --d_model_audio 64 \
 --d_model_caps 256 \
 --use_linear_embedder \
 --device_ids $device_ids \
 --one_by_one_starts_at 60 \
 --stopwords data/stopwords.txt \
 --exp_name $exp_name \
 --log_dir $log_dir \
 --num_workers $num_workers \
#  --debug



# eval
run=

# use validation set instead of test set before the test set gets available
test_set=./dstc10avsd_eval/data/test_set4DSTC10-AVSD_multiref+reason.json
test_csv=./data/dstc10_test.csv
featpath_suffix=_testset
last_only=--last_only
log_dir=./log
python utils/generate_csv.py duration_info/duration_Charades_vu17_test_480.csv $test_set test $test_csv

echo Answer generation and evaluation for $test_set
$run python main.py \
 --train_meta_path ./data/dstc10_train.csv \
 --test_meta_path $test_csv \
 --reference_paths $test_set \
 --video_features_path ${datapath}/video_feats$featpath_suffix/ \
 --audio_features_path ${datapath}/vggish$featpath_suffix/ \
 --procedure eval_cap \
 --pretrained_cap_model_path ${log_dir}/${exp_name}/train_cap/best_cap_model.pt \
 --B 12 \
 --stopwords data/stopwords.txt \
 --exp_name $exp_name \
 --log_dir $log_dir \
 --device_ids $device_ids \
 --num_workers $num_workers \
 $last_only