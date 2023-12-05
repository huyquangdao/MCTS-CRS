export CUDA_VISIBLE_DEVICES=5
seed=1

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/rtcp/online_evaluation_rtcp_bart.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --policy_tokenizer bert-base-cased \
    --plm_policy_model bert-base-cased \
    --generation_tokenizer facebook/bart-base \
    --plm_generation_model facebook/bart-base \
    --know_generation_tokenizer facebook/bart-base \
    --plm_know_generation_model facebook/bart-base \
    --lm_size 768 \
    --ffn_size 3072 \
    --n_layers 12 \
    --n_heads 8 \
    --fc_size 128 \
    --num_warmup_steps 6345   \
    --max_sequence_length 512 \
    --policy_model_path ./rtcp/ \
    --policy_model_path ./policy_model/ \
    --generation_model_path ./generation_model/ \
    --know_generation_model_path ./know_generation_model/ \
    --num_items 100 \
    --target_set_path ./target_set_${seed}/ \
    --horizon 5 \
    --seed ${seed}