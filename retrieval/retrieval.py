import os
import faiss
import numpy as np


class Memory:

    def __init__(self, embedding_model, raw_memory, instances, scores, d_model=384):
        """
        constructor for class me memory
        @param embedding_model: the sentence embedding model
        @param raw_memory: a set of raw dialogue contexts
        @param instances:
        @param d_model: the dimension of vector indexes
        """
        self.embedding_model = embedding_model
        self.raw_memory = raw_memory
        self.d_model = d_model
        self.instances = instances
        self.scores = scores
        self.index = self.build_index()

    def __len__(self):
        return len(self.raw_memory)

    def build_index(self):
        sentence_embeddings = self.embedding_model.encode(self.raw_memory)
        device = faiss.StandardGpuResources()
        assert sentence_embeddings.shape[0] == len(self.raw_memory)
        assert sentence_embeddings.shape[1] == self.d_model
        index = faiss.IndexFlatIP(self.d_model)  # build the index
        # add vectors to the index
        index.add(sentence_embeddings)
        index_gpu = faiss.index_cpu_to_gpu(device, 0, index)
        return index_gpu

    def search(self, queries, k=10):
        query_embed = self.embedding_model.encode([queries])
        D, I = self.index.search(query_embed, k)
        return D, I

    def update(self, new_memories):
        self.raw_memory.extend(new_memories)
        self.index = self.build_index()
