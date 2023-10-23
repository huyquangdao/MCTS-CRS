import os


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
