import os


def construct_mcts_memory(train_instances):
    """
    function that build a memory from using the training dataset.
    @param train_instances: a list of training instances.
    @return: a list of dialogue contexts
    """
    raw_memory = []
    for instance in train_instances:
        context = instance['dialogue_context']
        raw_memory.append(context)
    return raw_memory
