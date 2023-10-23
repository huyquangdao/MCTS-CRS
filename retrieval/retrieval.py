import os
import faiss
import numpy as np


class Memory:

    def __init__(self, embedding_model, raw_memory, d_model=384, ngpu=1):
        self.embedding_model = embedding_model
        self.raw_memory = raw_memory
        self.d_model = d_model
        self.ngpu = ngpu
        self.index = self.build_index()

    def build_index(self):
        sentence_embeddings = self.embedding_model.encode(self.raw_memory)
        resources = [faiss.StandardGpuResources() for i in range(self.ngpu)]
        assert sentence_embeddings.shape[0] == len(self.raw_memory)
        assert sentence_embeddings.shape[1] == self.d_model
        index = faiss.IndexFlatL2(self.d_model)  # build the index
        index.add(sentence_embeddings)  # add vectors to the index
        index_gpu = faiss.index_cpu_to_gpu_multiple_py(resources, index)
        return index_gpu

    def search(self, queries, k=10):
        D, I = self.index.search(queries, k)
        return D, I

    def update(self, new_memories):
        self.raw_memory.extend(new_memories)
        self.index = self.build_index()
