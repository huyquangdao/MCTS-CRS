from collections import defaultdict

import torch

from config.config import GOAL_TOKEN, USER_TOKEN, SYSTEM_TOKEN, KNOW_TOKEN, PATH_TOKEN, SEP_TOKEN, PROFILE_TOKEN, \
    CONTEXT_TOKEN, TARGET, TOPIC_TOKEN, IGNORE_INDEX


def convert_example_to_feature_for_gpt_response_generation(tokenizer, instance, max_sequence_length=512,
                                                           max_target_length=50,
                                                           is_test=False):
    """
    function that convert an instance to input and labels for a response generation model.
    @param tokenizer: a huggingface tokenizer
    @param instance: an instance from the data.
    @param max_sequence_length: the maximum length of the input sequence.
    @param max_target_length: the maximum length of the target response
    @param is_test: True if inference or False if training.
    @return: an input sequence and its corresponding labels.
    """
    dialogue_context = instance['dialogue_context']
    dialogue_str = ""
    target = instance['task_background']['target_topic']
    for utt in dialogue_context:
        if utt['role'] == "user":
            dialogue_str += USER_TOKEN
        elif utt['role'] == 'assistant':
            dialogue_str += SYSTEM_TOKEN
        dialogue_str += utt['content']

    # construct the input sequence for response generation task
    input_str = f"{TARGET}: {target}  {CONTEXT_TOKEN}: {dialogue_str}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    # input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    # construct the label for response generation task

    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{SYSTEM_TOKEN}: " + instance['response']))
    label = label[:max_target_length]

    if not is_test:
        input_ids = input_ids + label + [tokenizer.eos_token_id]
        input_ids = input_ids[-(max_sequence_length):]
        label = [IGNORE_INDEX] * len(input_ids) + label + [tokenizer.eos_token_id]
    else:
        input_ids = input_ids + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{SYSTEM_TOKEN}: "))
        input_ids = input_ids[-(max_sequence_length):]

    return input_ids, label


def generate_response_gpt(generation_model, tokenizer, state, max_sequence_length, max_gen_length=50,
                          pad_to_multiple_of=True, padding='max_length', device=None):
    """
    function that generates a response with the BART model.
    @param generation_model: the finetuned huggingface pretrained PLM
    @param tokenizer: a huggingface tokenizer.
    @param state:  the current state of the env
    @param max_sequence_length: the maximum number of tokens in the input sequence.
    @param max_gen_length: the maximum number of tokens in the generated response.
    @param pad_to_multiple_of: True if we pad to multiple instances.
    @param padding: type of padding default = 'max length"
    @param device: device to allocate tensors
    @return: a generated knowledge utterance.
    """
    # convert state to input feature
    input_features = defaultdict(list)

    # convert state to input features
    input_ids, _ = convert_example_to_feature_for_gpt_response_generation(tokenizer=tokenizer, instance=state,
                                                                          max_sequence_length=max_sequence_length,
                                                                          is_test=True)
    input_features['input_ids'] = input_ids
    # padding the input features

    input_features = tokenizer.pad(
        input_features, padding=padding, pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_sequence_length
    )
    # convert features to torch tensors
    for k, v in input_features.items():
        if not isinstance(v, torch.Tensor):
            input_features[k] = torch.as_tensor(v, device=device).unsqueeze(0)

    # forward the input features through the model
    gen_seqs = generation_model.generate(
        **input_features,
        max_new_tokens=max_gen_length,
        no_repeat_ngram_size=3
    )

    # remove special tokens
    gen_resp_ids = []
    for gen_seq in gen_seqs:
        gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
        gen_resp_ids.append(gen_seq[len(input_ids):])

    decoded_preds = tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=True)
    decoded_preds = [decoded_pred.replace('<pad>', '').replace('<s>', '').replace('</s>', '') for decoded_pred in
                     decoded_preds]
    decoded_preds = [pred.strip() for pred in decoded_preds]

    return decoded_preds[0]
