export CUDA_VISIBLE_DEVICES=5

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/unimind/online_evaluation_unimind_bart.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --policy_tokenizer facebook/bart-base \
    --policy_plm_model facebook/bart-base \
    --generation_tokenizer facebook/bart-base \
    --plm_generation_model facebook/bart-base \
    --know_generation_tokenizer facebook/bart-base \
    --plm_know_generation_model facebook/bart-base \
    --policy_model_path ./unimind/ \
    --generation_model_path ./generation_model/ \
    --know_generation_model_path ./know_generation_model/ \
    --num_items 100 \
    --max_sequence_length 512 \
    --target_set_path ./target_set_${seed}/ \
    --horizon 5 \
    --use_llm_score \
    --n 5 \
    --k 5 \
    --epsilon 1.0 \
    --seed ${seed}