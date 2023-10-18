export CUDA_VISIBLE_DEVICES=5

#infer goal
CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/unimind/infer_goal_unimind.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer facebook/bart-base \
    --plm_model facebook/bart-base \
    --per_device_eval_batch_size 32 \
    --max_sequence_length 512 \
    --max_target_length 80 \
    --output_dir ./unimind/ \
    --seed 12

#infer topic
CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/unimind/infer_topic_unimind.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer facebook/bart-base \
    --plm_model facebook/bart-base \
    --per_device_eval_batch_size 32 \
    --max_sequence_length 512 \
    --max_target_length 80 \
    --goal_outpath ./unimind/ \
    --output_dir ./unimind/ \
    --seed 12

#infer response
CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/unimind/infer_response_unimind.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer facebook/bart-base \
    --plm_model facebook/bart-base \
    --per_device_eval_batch_size 32 \
    --max_sequence_length 512 \
    --max_target_length 80 \
    --goal_outpath ./unimind/ \
    --topic_outpath ./unimind/ \
    --output_dir ./unimind/ \
    --seed 12

