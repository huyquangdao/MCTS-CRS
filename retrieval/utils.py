import os
import json


def concatenate_sentences(list_of_sents):
    """
    function that concatenate sentences to form a dialogue context
    @param list_of_sents: list of sentences in the dialogue history
    @return: a single text string representing the dialogue context
    """
    output_str = ""
    for utt in list_of_sents:
        output_str += f"{utt['role']} : {utt['content']}"
        output_str += " "
    return output_str


def construct_mcts_memory(train_instances):
    """
    function that build a memory from using the training dataset.
    @param train_instances: a list of training instances.
    @return: a list of dialogue contexts
    """
    raw_memory = []
    for instance in train_instances:
        context = instance['dialogue_context']
        context = concatenate_sentences(context)
        raw_memory.append(context)
    return raw_memory


def load_memory_from_file(file_path):
    """
    function that loads historical instances into the memory
    @param file_path: the path to the file containing the memory.
    @return:
    """
    raw_memory = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            dic = json.loads(line)
            state = dic['state']
            continuation = dic['continuation']
            score = dic['score']
            raw_memory.append((state, continuation, score))
    return raw_memory


def construct_memory_loaded_from_file(raw_memory):
    """
    function that constructs a set of memory which is loaded from file
    @param raw_memory: list of pairs of historical instance, each contain a state and its continuation
    @return: a set of memory
    """
    raw_states = []
    raw_continations = []
    raw_scores = []
    for state, continuation, score in raw_memory:
        raw_state = concatenate_sentences(state['dialogue_context'])
        raw_states.append(raw_state)
        raw_continations.append(continuation)
        raw_scores.append(score)
    return raw_states, raw_continations, raw_scores
