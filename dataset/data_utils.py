import random
from config.config import USER_TOKEN, SYSTEM_TOKEN, KNOW_TOKEN, PATH_TOKEN, SEP_TOKEN, PROFILE_TOKEN, CONTEXT_TOKEN


def convert_list_to_str(knowledge):
    """
    function that convert a list of 3 elements to a text string
    @param knowledge: a list of 3 element where each element is a string
    @return: a text string
    """
    if len(knowledge) == 0:
        return ""
    return f"{knowledge[0]} {knowledge[1]} {knowledge[2]}"


def convert_dict_to_str(profile):
    """
    function that convert a dictionary to a text string
    @param profile: a dictionary which contains information about the user.
    @return: a text string
    """
    out_str = ""
    for k, v in profile.items():
        if isinstance(v, list):
            value_str = ""
            for e in v:
                value_str += e
                value_str += " "
            out_str += f"{k}: {value_str}"
        elif isinstance(v, str):
            out_str += f"{k}: {v}"
        out_str += " "
    return out_str


def convert_example_to_feature(tokenizer, instance, max_sequence_length=512):
    """
    function that converts an instance (example ) to a sequence of various features
    @param tokenizer: a huggingface tokenizer.
    @param instance: a dictionary which contains dialogue context, knowledge, user profile and previous planned goals.
    @param max_sequence_length: the maximum number of tokens in the input sequence.
    @return: an input sequence which consists of knowledge , profile, path and dialogue context ids.
    """
    dialogue_context = instance['dialogue_context']
    knowledge = instance['knowledge']
    prev_paths = instance['pre_goals']
    profile = instance['task_background']['user_profile']

    dialogue_str = ""
    for utt in dialogue_context:
        if utt['role'] == "user":
            dialogue_str += USER_TOKEN
        elif utt['role'] == 'assistant':
            dialogue_str += SYSTEM_TOKEN
        dialogue_str += utt['content']

    path_str = ""
    for goal in prev_paths:
        path_str += goal
        path_str += " "
        path_str += SEP_TOKEN

    knowledge_str = convert_list_to_str(knowledge)
    profile_str = convert_dict_to_str(profile)

    input_str = f"{PROFILE_TOKEN}: {profile_str} {KNOW_TOKEN}: {knowledge_str} {PATH_TOKEN}: {path_str} {CONTEXT_TOKEN}: {dialogue_str}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    return input_ids


def randomly_sample_demonstrations(all_convs, instance, k=1):
    """
    function that randomly sample 1 demonstrations from the set of all training conversations.
    here we first filter out a set of examples that have the same target goal with the given one.
    Then we randomly choose 1 demonstration from the candidate set.
    @param all_convs: set of all training conversations
    @param instance: an instance which is a dictionary of dialogue context, task background.
    @param k: the number of sampled demonstrations, default = 1
    @return: a randomly chosen conversation.
    """
    candidate_instances = [x for x in all_convs if
                           x['target_topic'] == instance['task_background']['target_topic']]

    return random.choices(candidate_instances, k=k)
