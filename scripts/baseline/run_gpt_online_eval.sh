export CUDA_VISIBLE_DEVICES=5
seed=5

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/dialoggpt/online_evaluation_dialoggpt.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer microsoft/DialoGPT-small \
    --plm_model microsoft/DialoGPT-small \
    --num_items 100 \
    --target_set_path ./target_set_${seed}/ \
    --horizon 5 \
    --max_sequence_length 512 \
    --max_gen_length 50 \
    --model_path ./bart/ \
    --seed ${seed}