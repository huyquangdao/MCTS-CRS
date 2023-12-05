export CUDA_VISIBLE_DEVICES=5
seed=1

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/rtcp/online_evaluation_rtcp_llama.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --policy_tokenizer bert-base-cased \
    --plm_policy_model bert-base-cased \
    --lm_size 768 \
    --ffn_size 3072 \
    --n_layers 12 \
    --n_heads 8 \
    --fc_size 128 \
    --max_sequence_length 512 \
    --policy_model_path ./rtcp/ \
    --num_items 100 \
    --target_set_path ./target_set_${seed}/ \
    --horizon 5 \
    --use_llm_score \
    --n 5 \
    --k 3 \
    --epsilon 1.0 \
    --seed ${seed}