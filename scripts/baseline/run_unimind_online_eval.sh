export CUDA_VISIBLE_DEVICES=5

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/unimind/online_evaluation_unimind.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer facebook/bart-base \
    --plm_model facebook/bart-base \
    --num_items 1 \
    --target_set_path ./target_set/ \
    --horizon 5 \
    --max_sequence_length 512 \
    --max_gen_length 50 \
    --model_path ./unimind/ \
    --seed 22