export CUDA_VISIBLE_DEVICES=5

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 examples/uct_dialogue_planning.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer roberta-base \
    --plm_model roberta-base \
    --rollouts 20 \
    --alg p_uct \
    --hidden_size 128 \
    --lm_size 768 \
    --horizon 5 \
    --max_sequence_length 512 \
    --model_path ./policy_model/ \
    --seed 22