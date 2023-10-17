import copy
import random
import pickle
from config.config import GOAL_TOKEN, USER_TOKEN, SYSTEM_TOKEN, KNOW_TOKEN, PATH_TOKEN, SEP_TOKEN, PROFILE_TOKEN, \
    CONTEXT_TOKEN, TARGET, TOPIC_TOKEN


def load_binary_file(file_path):
    """
    function thats load a binary file
    @param file_path: the path to the saved file
    @return: a pickle object
    """
    with open(file_path, 'rb') as f:
        object = pickle.load(f)
        return object


def save_binary_file(object, file_path):
    """
    function that save an object to a binary file
    @param object: an python object
    @param file_path: the path to the saved file
    @return: None
    """
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)


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


def convert_example_to_feature_for_goal_prediction(tokenizer, instance, max_sequence_length=512, goal2id=None):
    """
    function that converts an instance (example ) to a sequence of various features
    @param tokenizer: a huggingface tokenizer.
    @param instance: a dictionary which contains dialogue context, knowledge, user profile and previous planned goals.
    @param max_sequence_length: the maximum number of tokens in the input sequence.
    @param goal2id: a dictionary that map goal to index
    @return: an input sequence which consists of knowledge , profile, path and dialogue context ids.
    """
    dialogue_context = instance['dialogue_context']
    prev_paths = instance['pre_goals']

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

    input_str = f"{PATH_TOKEN}: {path_str} {CONTEXT_TOKEN}: {dialogue_str}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    label = goal2id[instance['goal']]

    return input_ids, label


def convert_example_to_feature_for_response_generation(tokenizer, instance, max_sequence_length=512,
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

    if not is_test:
        # ground truth goal for training the model
        goal = instance['goal']
        knowledge = instance['knowledge']
    else:
        # predicted goal for the inference step
        goal = instance['pred_goal']
        knowledge = instance['pred_know']

    if not isinstance(knowledge, str):
        knowledge_str = convert_list_to_str(knowledge)
    else:
        knowledge_str = knowledge
    dialogue_str = ""
    for utt in dialogue_context:
        if utt['role'] == "user":
            dialogue_str += USER_TOKEN
        elif utt['role'] == 'assistant':
            dialogue_str += SYSTEM_TOKEN
        dialogue_str += utt['content']

    # construct the input sequence for response generation task
    input_str = f"{KNOW_TOKEN}: {knowledge_str} {GOAL_TOKEN}: {goal} {CONTEXT_TOKEN}: {dialogue_str}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # construct the label for response generation task
    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{SYSTEM_TOKEN}: " + instance['response']))
    label = label[:max_target_length]
    label = label + [tokenizer.eos_token_id]

    return input_ids, label


def convert_example_to_feature_for_knowledge_generation(tokenizer, instance, max_sequence_length=512,
                                                        max_target_length=50,
                                                        is_test=False):
    """
    function that convert an instance to input and labels for a knowledge generation model.
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

    if not is_test:
        # ground truth goal for training the model
        goal = instance['goal']
    else:
        # predicted goal for the inference step
        goal = instance['pred_goal']
    dialogue_str = ""
    for utt in dialogue_context:
        if utt['role'] == "user":
            dialogue_str += USER_TOKEN
        elif utt['role'] == 'assistant':
            dialogue_str += SYSTEM_TOKEN
        dialogue_str += utt['content']

    # construct the input sequence for knowledge generation task
    input_str = f"{GOAL_TOKEN}: {goal} {TARGET}: {target} {CONTEXT_TOKEN}: {dialogue_str}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # construct the label for response generation task
    label = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(f"{KNOW_TOKEN}: " + convert_list_to_str(instance['knowledge'])))
    label = label[:max_target_length]
    label = label + [tokenizer.eos_token_id]

    return input_ids, label


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
                           x['target_goal'] == instance['task_background']['target_goal']]

    return random.choices(candidate_instances, k=k)


def save_policy_results(policy_preds, output_path, goal2id=None):
    """
    function that save the results of goal prediction task
    @param policy_preds: a list of predicted goals (in form of goal ids)
    @param output_path: the path to the saved file
    @param goal2id: the dictionary that convert goal to index
    @return: None
    """
    id2goal = {v: k for k, v in goal2id.items()}
    with open(output_path, 'w') as f:
        for pred in policy_preds:
            pred_text = id2goal[pred]
            f.write(pred_text + '\n')


def load_policy_results(output_path):
    """
    function that loads predicted goals from a file
    @param output_path: the file which contains the goal predictions
    @return: a list of predicted goals
    """
    pred_goals = []
    with open(output_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            pred_goals.append(line)
    return pred_goals


def save_knowledge_results(list_preds, output_path):
    """
    function that save the predicted knowledge
    @param list_preds: list of predicted knowledge
    @param output_path: the path to the saved file
    @return: None
    """
    with open(output_path, 'w') as f:
        for pred in list_preds:
            f.write(pred + "\n")


def load_knowledge_results(output_path):
    """
    function that load predicted knowledge
    @param output_path: the path to the saved file
    @return: a list of predicted knowledge
    """
    with open(output_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines


def merge_predictions(instances, policy_preds):
    """
    function that merge policy predictions with data instances.
    @param instances: a list of instances
    @param policy_preds: a list of predicted goals
    @return: a list of new instances.
    """
    for instance, pred_goal in list(zip(instances, policy_preds)):
        instance['pred_goal'] = pred_goal
    return instances


def merge_know_predictions(instances, know_preds):
    """
    function that merge knowledge predictions with data instances.
    @param instances: a list of instances
    @param know_preds: a list of predicted goals
    @return: a list of new instances.
    """
    for instance, pred_know in list(zip(instances, know_preds)):
        instance['pred_know'] = pred_know
    return instances


def create_target_set(train_convs, test_instances, num_items=10):
    """
    function that creates a target item set from the test set
    @param train_convs: a list of conversations from training set.
    @param test_instances: a list of test instances.
    @param num_items: the number of target item
    @return: a list of dictionary which contain information about the target item (name, goal and demonstration)
    """

    # copy instances before selecting target items
    copied_test_instances = copy.deepcopy(test_instances)
    random.shuffle(copied_test_instances)
    # get the set of items from the test set.
    i = 0
    selected_set = []
    selected_set_names = []
    while i < num_items:
        instance = copied_test_instances[i]
        if instance['task_background']['target_topic'] in selected_set_names:
            continue

        # sample a demonstration for user simulator:
        demonstrations = randomly_sample_demonstrations(
            all_convs=train_convs,
            instance=instance
        )
        # create the target
        target = {
            "topic": instance['task_background']['target_topic'],
            "goal": instance['task_background']['target_goal'],
            "demonstration": demonstrations[0]
        }
        selected_set.append(target)
        selected_set_names.append(target['topic'])
        i += 1

    return selected_set


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
    if not is_gen:
        label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{GOAL_TOKEN}: " + instance['goal']))
        label = label[:max_target_length]
        label = label + [tokenizer.eos_token_id]

        return input_ids, label

    return input_ids


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

    input_str = f"{input_str} {GOAL_TOKEN} {goal} {TOPIC_TOKEN}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # if not inference time.
    if not is_gen:
        label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{TOPIC_TOKEN}: " + instance['topic']))
        label = label[:max_target_length]
        label = label + [tokenizer.eos_token_id]

        return input_ids, label

    return input_ids


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
    if not is_gen:
        label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{SYSTEM_TOKEN}: " + instance['response']))
        label = label[:max_target_length]
        label = label + [tokenizer.eos_token_id]

        return input_ids, label

    return input_ids
