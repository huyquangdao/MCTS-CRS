import time
import os
from tqdm import tqdm
import math
from collections import defaultdict
import torch
import torch.nn.functional as F
from config.config import GOAL_TOKEN, USER_TOKEN, SYSTEM_TOKEN, KNOW_TOKEN, PATH_TOKEN, SEP_TOKEN, PROFILE_TOKEN, \
    CONTEXT_TOKEN, TARGET, TOPIC_TOKEN, IGNORE_INDEX

import numpy as np


def convert_example_to_feature_for_rtcp_goal_topic_prediction(tokenizer, instance, max_sequence_length=512,
                                                              goal2id=None, topic2id=None):
    """
    function that creates the input example for unimind goal prediction task
    @param tokenizer: hugginface tokenizer
    @param instance: the input instance
    @param max_sequence_length: maximum number of tokens in the input sequence
    @param goal2id: a dictionary that maps a goal or topic to an id.
    @return: input sequence for the unimind's goal prediction task.
    """
    dialogue_context = instance['dialogue_context']
    prev_goals = instance['pre_goals']
    prev_topics = instance['pre_topics']
    target_item = instance['task_background']['target_topic']

    # Context String
    input_str = ""
    for utt, goal in list(zip(dialogue_context, prev_goals)):
        if utt['role'] == "user":
            input_str += USER_TOKEN + " "
        elif utt['role'] == 'assistant':
            input_str += GOAL_TOKEN + " "
            input_str += goal + " "
            input_str += SYSTEM_TOKEN + " "
        input_str += utt['content']

    input_str = f"{input_str} {TARGET} {target_item} {GOAL_TOKEN}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # path string
    path_string = ""
    for goal, topic in list(zip(prev_goals, prev_topics)):
        path_string += F"{GOAL_TOKEN} {goal} {TOPIC_TOKEN} {topic} {SEP_TOKEN} "

    path_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(path_string))
    path_ids = path_ids[-(max_sequence_length - 2):]
    path_ids = [tokenizer.cls_token_id] + path_ids + [tokenizer.sep_token_id]

    # if not inference time.
    label_goal, label_topic = goal2id[instance['goal']], topic2id[instance['topic']]

    return input_ids, path_ids, label_goal, label_topic


def convert_example_to_feature_for_rtcp_response_generation(tokenizer, instance, max_sequence_length=512,
                                                            max_target_length=50, goal2id=None, topic2id=None,
                                                            is_test=False,
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
    dialogue_str = ""
    target = instance['task_background']['target_topic']
    for utt in dialogue_context:
        if utt['role'] == "user":
            dialogue_str += USER_TOKEN
        elif utt['role'] == 'assistant':
            dialogue_str += SYSTEM_TOKEN
        dialogue_str += utt['content']

    if not is_test:
        goal = instance['goal']
        topic = instance['topic']
    else:
        goal = instance['pred_goal']
        topic = instance['pred_topic']

    # construct the input sequence for response generation task
    input_str = f"{GOAL_TOKEN}: {goal} {TOPIC_TOKEN} {topic}  {CONTEXT_TOKEN}: {dialogue_str}"
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

    # convert goal and topic to indices.
    goal_id = goal2id[goal]
    topic_id = topic2id[topic]
    return input_ids, label, goal_id, topic_id


def predict_action_rtcp(generation_model, tokenizer, state, max_sequence_length, max_gen_length=50,
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
    input_ids, _ = convert_example_to_feature_for_rtcp_goal_topic_prediction(tokenizer=tokenizer, instance=state,
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


def generate_response_rtcp(generation_model, tokenizer, action, topic, state, max_sequence_length, max_gen_length=50,
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
    input_ids, _ = convert_example_to_feature_for_rtcp_response_generation(tokenizer=tokenizer, instance=state,
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


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(model, context, action_id, topic_id, tokenizer, args, contrained_vocab=None):
    special_tokens_ids = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    context = torch.tensor(context, dtype=torch.long, device=args.device).unsqueeze(0)
    generated = context
    n_ctx = model.plm.config.n_ctx
    output_ids = []
    action_tensor = torch.LongTensor([action_id]).unsqueeze(0).to(args.device)
    topic_tensor = torch.LongTensor([topic_id]).unsqueeze(0).to(args.device)

    for i in range(args.max_dec_len):
        input_ids = generated[0][-(n_ctx - 1):].unsqueeze(0)
        batch = {
            "input_ids": input_ids,
            "action_id": action_tensor,
            "topic_id": topic_tensor,
            "labels": None
        }
        lm_output = model(batch)
        #### we only consider token that belong to the contrained vocabulary.
        logits = lm_output["logits"]
        logits = logits[0, -1, :] / args.temperature

        if args.top_k > 0 or (args.top_p > 0 and args.top_p <= 1):
            filtered_logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.topk(probs, 1)[1]

        if i < args.min_dec_len and next_token.item() in special_tokens_ids:
            while next_token.item() in special_tokens_ids:
                next_token = torch.multinomial(probs, num_samples=1)
        output_ids.append(next_token.item())
        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

        if next_token.item() in special_tokens_ids:
            break

    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    # output_text = output_text.replace("<|endoftext|>","")
    # output_text = output_text.replace(" ", "")
    return output_text
