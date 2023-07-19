# python3 ./evaluation/evaluate.py \
#     -r data/test_set4DSTC10-AVSD_multiref+reason.json \
#     -s log/cross_entropy2_freq4_smoothing/captioning_results_test.json \
#     -S evaluation/stopword_filter.py \
#     -l 

python3 ./evaluation/evaluate.py \
    -r data/mock_test_set4DSTC10-AVSD_from_DSTC7_multiref.json \
    -s log/transformer_uni_decoder/captioning_results_test.json \
    -S evaluation/stopword_filter.py \
    -l 

# python3 ./evaluation/evaluate.py \
#     -r data/mock_test_set4DSTC10-AVSD_from_DSTC7_multiref.json \
#     -s log/cross_entropy2/captioning_results_test_e0.json \
#     -S evaluation/stopword_filter.py \
#     -l 

# python3 ./evaluation/evaluate.py \
#     -r data/test_set4DSTC10-AVSD_multiref+reason.json \
#     -s log/cross_entropy2/captioning_results_test.json \
#     -S evaluation/stopword_filter.py \
#     -l 