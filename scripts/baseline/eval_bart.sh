export CUDA_VISIBLE_DEVICES=5

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/bart/infer_response_bart.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer facebook/bart-base \
    --plm_model facebook/bart-base \
    --per_device_eval_batch_size 32 \
    --max_sequence_length 512 \
    --max_target_length 80 \
    --output_dir ./bart/ \
    --seed 12