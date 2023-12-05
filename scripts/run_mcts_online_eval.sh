export CUDA_VISIBLE_DEVICES=5

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 online_evaluation.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --policy_tokenizer roberta-base \
    --plm_policy_model roberta-base \
    --generation_tokenizer facebook/bart-base \
    --plm_generation_model facebook/bart-base \
    --know_generation_tokenizer facebook/bart-base \
    --plm_know_generation_model facebook/bart-base \
    --rollouts 20 \
    --num_items 1 \
    --alg p_uct \
    --hidden_size 128 \
    --lm_size 768 \
    --target_set_path ./target_set/ \
    --memory_path simulated_results.txt \
    --horizon 5 \
    --max_sequence_length 512 \
    --max_gen_length 50 \
    --policy_model_path ./policy_model/ \
    --generation_model_path ./generation_model/ \
    --know_generation_model_path ./know_generation_model/ \
    --use_llm_score \
    --n 5 \
    --k 3 \
    --epsilon 1.0 \
    --seed 22