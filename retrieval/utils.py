import os
import faiss


def build_indexes(embed_model, memory):

    sentence_embeds = embed_model.encode(memory)

def retrieve_candidates(query, indexes, k=10):
    pass


def build_mcts_memory(train_convs):
    pass
