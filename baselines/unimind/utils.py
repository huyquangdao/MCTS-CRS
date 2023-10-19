from collections import defaultdict
import torch
from config.config import GOAL_TOKEN, USER_TOKEN, SYSTEM_TOKEN, KNOW_TOKEN, PATH_TOKEN, SEP_TOKEN, PROFILE_TOKEN, \
    CONTEXT_TOKEN, TARGET, TOPIC_TOKEN


def convert_example_to_feature_for_unimind_goal_prediction(tokenizer, instance, max_sequence_length=512,
                                                           max_target_length=50, is_test=False,
                                                           is_gen=False):
    """
    function that creates the input example for unimind goal prediction task
    @param tokenizer: hugginface tokenizer
    @param instance: the input instance
    @param max_sequence_length: maximum number of tokens in the input sequence
    @param max_target_length: maximum number of tokens in the label
    @param is_gen: inference time.
    @return: input sequence for the unimind's goal prediction task.
    """
    dialogue_context = instance['dialogue_context']
    prev_goals = instance['pre_goals']

    # Example of the input of unimind goal prediction
    # “[user] Who is the star of the movie < stolen life >? [goal] QA [system] It is Xun Zhou.[user]
    # She is my goddess.[goal] Chit - chat about Star[system] You have a good taste.She is the most
    # popular actress in the Golden Eagle Award.[user] I like her very much.[goal]” dialogue contexts
    input_str = ""
    for utt, goal in list(zip(dialogue_context, prev_goals)):
        if utt['role'] == "user":
            input_str += USER_TOKEN + " "
        elif utt['role'] == 'assistant':
            input_str += GOAL_TOKEN + " "
            input_str += goal + " "
            input_str += SYSTEM_TOKEN + " "
        input_str += utt['content']

    input_str = f"{input_str} {GOAL_TOKEN}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # if not inference time.
    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{GOAL_TOKEN}: " + instance['goal']))
    label = label[:max_target_length]
    label = label + [tokenizer.eos_token_id]

    return input_ids, label


def convert_example_to_feature_for_unimind_topic_prediction(tokenizer, instance, max_sequence_length=512,
                                                            max_target_length=50, is_test=False,
                                                            is_gen=False):
    """
    function that creates the input sequence for unimind topic prediction task
    @param tokenizer:  hugginface tokenizer
    @param instance: the input instance
    @param max_sequence_length: maximum number of tokens in the input sequence
    @param max_target_length: maximum number of tokens in the target sequence
    @param is_test: if is in testing time
    @param is_gen:  if is in infrence time
    @return: the input sequence of the unimind topic prediction task
    """
    dialogue_context = instance['dialogue_context']
    prev_topics = instance['pre_topics']
    input_str = ""
    target_item = instance['task_background']['target_topic']
    for utt, topic in list(zip(dialogue_context, prev_topics)):
        if utt['role'] == "user":
            input_str += USER_TOKEN + " "
        elif utt['role'] == 'assistant':
            input_str += TOPIC_TOKEN + " "
            input_str += topic + " "
            input_str += SYSTEM_TOKEN + " "
        input_str += utt['content']

    # ground truth goal
    if not is_test:
        goal = instance['goal']
    else:
        goal = instance['pred_goal']

    input_str = f"{input_str} {GOAL_TOKEN} {goal} {TARGET} {target_item} {TOPIC_TOKEN}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # if not inference time.
    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instance['topic']))
    label = label[:max_target_length]
    label = label + [tokenizer.eos_token_id]

    return input_ids, label


def convert_example_to_feature_for_unimind_response_generation(tokenizer, instance, max_sequence_length=512,
                                                               max_target_length=50, is_test=False,
                                                               is_gen=False):
    """
    function that creates the input sequence for unimind response generation task
    @param tokenizer:  hugginface tokenizer
    @param instance: the input instance
    @param max_sequence_length: maximum number of tokens in the input sequence
    @param max_target_length: maximum number of tokens in the target sequence
    @param is_test: if is in testing time
    @param is_gen:  if is in infrence time
    @return: the input sequence of the unimind topic prediction task
    """
    dialogue_context = instance['dialogue_context']
    input_str = ""
    for utt in dialogue_context:
        if utt['role'] == "user":
            input_str += USER_TOKEN + " "
        elif utt['role'] == 'assistant':
            input_str += SYSTEM_TOKEN + " "
        input_str += utt['content']

    # ground truth goal and topic
    if not is_test:
        goal = instance['goal']
        topic = instance['topic']
    else:
        goal = instance['pred_goal']
        topic = instance['pred_topic']

    input_str = f"{input_str} {GOAL_TOKEN} {goal} {TOPIC_TOKEN} {topic} {SYSTEM_TOKEN}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # if not inference time.
    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{SYSTEM_TOKEN}: " + instance['response']))
    label = label[:max_target_length]
    label = label + [tokenizer.eos_token_id]

    return input_ids, label


def predict_action_unimind(generation_model, tokenizer, state, max_sequence_length, max_gen_length=50,
                           pad_to_multiple_of=True, padding='max_length', device=None):
    """
    function that generates a knowledge utterance with unimind
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
    input_ids, _ = convert_example_to_feature_for_unimind_goal_prediction(tokenizer=tokenizer, instance=state,
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
        gen_resp_ids.append(gen_seq)

    decoded_preds = tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=False)
    decoded_preds = [decoded_pred.replace('<pad>', '').replace('<s>', '').replace('</s>', '') for decoded_pred in
                     decoded_preds]
    decoded_preds = [pred.strip() for pred in decoded_preds]

    return decoded_preds[0]


def predict_topic_unimind(generation_model, tokenizer, action, state, max_sequence_length, max_gen_length=50,
                          pad_to_multiple_of=True, padding='max_length', device=None):
    """
    function that generates a topic using the unimind model
    @param generation_model: the finetuned huggingface pretrained PLM
    @param tokenizer: a huggingface tokenizer.
    @param action: the predicted action
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

    state['pred_goal'] = action

    # convert state to input features
    input_ids, _ = convert_example_to_feature_for_unimind_topic_prediction(tokenizer=tokenizer, instance=state,
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
        gen_resp_ids.append(gen_seq)

    decoded_preds = tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=False)
    decoded_preds = [decoded_pred.replace('<pad>', '').replace('<s>', '').replace('</s>', '') for decoded_pred in
                     decoded_preds]
    decoded_preds = [pred.strip() for pred in decoded_preds]

    return decoded_preds[0]


def generate_response_unimind(generation_model, tokenizer, action, topic, state, max_sequence_length, max_gen_length=50,
                              pad_to_multiple_of=True, padding='max_length', device=None):
    """
    function that generates a response with the UNIMIND model.
    @param generation_model: the finetuned huggingface pretrained PLM
    @param tokenizer: a huggingface tokenizer.
    @param action: the predicted action
    @param topic: the predicted topic
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

    state['pred_goal'] = action
    state['pred_topic'] = topic

    # convert state to input features
    input_ids, _ = convert_example_to_feature_for_unimind_response_generation(tokenizer=tokenizer, instance=state,
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
        gen_resp_ids.append(gen_seq)

    decoded_preds = tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=False)
    decoded_preds = [decoded_pred.replace('<pad>', '').replace('<s>', '').replace('</s>', '') for decoded_pred in
                     decoded_preds]
    decoded_preds = [pred.strip() for pred in decoded_preds]

    return decoded_preds[0]
