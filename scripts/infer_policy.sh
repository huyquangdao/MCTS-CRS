export CUDA_VISIBLE_DEVICES=5

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 infer_policy.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer roberta-base \
    --plm_model roberta-base \
    --num_train_epochs 5 \
    --hidden_size 128 \
    --lm_size 768 \
    --output_dir ./policy_model/ \
    --seed 22