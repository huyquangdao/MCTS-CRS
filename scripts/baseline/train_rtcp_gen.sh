export CUDA_VISIBLE_DEVICES=5

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/rtcp/train_gen_rtcp.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer gpt2 \
    --plm_model gpt2 \
    --num_train_epochs 5 \
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
    --goal_outpath ./rtcp/ \
    --seed 21