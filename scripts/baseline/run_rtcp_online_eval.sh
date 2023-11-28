export CUDA_VISIBLE_DEVICES=5
seed=1

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/rtcp/online_evaluation_rtcp.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --policy_tokenizer bert-base-cased \
    --policy_plm_model bert-base-cased \
    --generation_tokenizer gpt2 \
    --generation_plm_model gpt2 \
    --lm_size 768 \
    --ffn_size 3072 \
    --n_layers 12 \
    --n_heads 8 \
    --fc_size 128 \
    --num_tokens 50 \
    --n_goal_toks 2 \
    --n_topic_toks 2 \
    --use_goal_topic \
    --freeze_plm \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 6345   \
    --max_sequence_length 512 \
    --learning_rate 5e-5 \
    --output_dir ./rtcp/ \
    --num_items 1 \
    --target_set_path ./target_set_${seed}/ \
    --horizon 5 \
    --seed ${seed}