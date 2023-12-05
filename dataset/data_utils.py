import copy
import json
import math
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
    prev_topics = instance['pre_topics']
    target = instance['task_background']['target_topic']
    dialogue_str = ""
    for utt in dialogue_context:
        if utt['role'] == "user":
            dialogue_str += USER_TOKEN
        elif utt['role'] == 'assistant':
            dialogue_str += SYSTEM_TOKEN
        dialogue_str += utt['content']

    path_str = ""
    for goal, topic in list(zip(prev_paths, prev_topics)):
        path_str += GOAL_TOKEN
        path_str += goal
        path_str += TOPIC_TOKEN
        path_str += topic
        path_str += SEP_TOKEN

    input_str = f"{PATH_TOKEN}: {path_str} {TARGET}: {target} {CONTEXT_TOKEN}: {dialogue_str}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    label = goal2id[(instance['goal'], instance['topic'])]

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
    for utt in dialogue_context:
        if utt['role'] == "user":
            dialogue_str += USER_TOKEN
        elif utt['role'] == 'assistant':
            dialogue_str += SYSTEM_TOKEN
        dialogue_str += utt['content']

    if not is_test:
        # ground truth goal for training the model
        goal = instance['goal']
        topic = instance['topic']
        knowledge = instance['knowledge']
    else:
        # predicted goal for the inference step
        goal = instance['pred_goal']
        topic = instance['pred_topic']
        knowledge = instance['pred_know']

    if not isinstance(knowledge, str):
        knowledge_str = convert_list_to_str(knowledge)
    else:
        knowledge_str = knowledge

    # construct the input sequence for response generation task
    input_str = f"{GOAL_TOKEN}: {goal} {TOPIC_TOKEN}: {topic} {KNOW_TOKEN}: {knowledge_str}  {CONTEXT_TOKEN}: {dialogue_str}"
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
        # ground truth goal and topic for training the model
        goal = instance['goal']
        topic = instance['topic']
    else:
        # predicted goal and topic for the inference step
        goal = instance['pred_goal']
        topic = instance['pred_topic']
    dialogue_str = ""
    for utt in dialogue_context:
        if utt['role'] == "user":
            dialogue_str += USER_TOKEN
        elif utt['role'] == 'assistant':
            dialogue_str += SYSTEM_TOKEN
        dialogue_str += utt['content']

    # construct the input sequence for knowledge generation task
    input_str = f"{GOAL_TOKEN}: {goal} {TOPIC_TOKEN}: {topic} {TARGET}: {target} {CONTEXT_TOKEN}: {dialogue_str}"
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


def merge_topic_predictions(instances, policy_preds):
    """
    function that merge policy predictions with data instances.
    @param instances: a list of instances
    @param policy_preds: a list of predicted goals
    @return: a list of new instances.
    """
    for instance, pred_goal in list(zip(instances, policy_preds)):
        instance['pred_topic'] = pred_goal
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


def merge_topic_predictions(instances, topic_preds):
    """
    function that merge knowledge predictions with data instances.
    @param instances: a list of instances
    @param topic_preds: a list of predicted goals
    @return: a list of new instances.
    """
    for instance, pred_topic in list(zip(instances, topic_preds)):
        instance['pred_topic'] = pred_topic
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

    #
    while len(selected_set) < num_items:
        instance = copied_test_instances[i]
        if instance['task_background']['target_topic'] in selected_set_names:
            i += 1
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


def split_goal_topic(preds, goal2id):
    """
    function that splits policy predictions
    @param preds: a list of predictions
    @param goal2id: a dictionary that map goal to id
    @return: list of goals, list of topics
    """
    id2goal = {v: k for k, v in goal2id.items()}
    goals = []
    topics = []
    for pred in preds:
        goal, topic = id2goal[pred]
        goals.append(goal)
        topics.append(topic)
    return goals, topics


def save_simulated_results(f, state, dialogue_continuation):
    """
    function that saves dialogue state and its corresponding dialogue continuation
    @param state: an input dialogue state
    @param dialogue_continuation: the corresponding dialogue continuation
    @return: None
    """
    dic = {
        "state": state,
        "continuation": dialogue_continuation
    }
    f.write(json.dumps(dic) + "\n")


def save_generated_conversations(generated_conversations, file_path):
    """
    function that save generated conversations to a text file
    @param generated_conversations: list of generated conversations.
    @param file_path: the string that is the path to the file.
    @return: None
    """
    with open(file_path, 'w') as f:
        for generated_conv in generated_conversations:
            out_str = json.dumps(generated_conv)
            f.write(out_str + '\n')
