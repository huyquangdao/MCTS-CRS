import os
import faiss
import numpy as np


class Memory:

    def __init__(self, embedding_model, raw_memory, d_model):
        self.embedding_model = embedding_model
        self.raw_memory = raw_memory
        self.d_model = d_model

        self.index = self.build_index()

    def build_index(self):
        sentence_embeddings = self.embedding_model.encode(self.raw_memory)
        assert sentence_embeddings.shape[0] == len(self.raw_memory)
        assert sentence_embeddings.shape[1] == self.d_model
        sentence_embeddings = sentence_embeddings.detach().numpy()
        assert isinstance(sentence_embeddings, np.array)
        index = faiss.IndexFlatL2(self.d_model)  # build the index
        index.add(sentence_embeddings)  # add vectors to the index
        return index

    def search(self, queries, k=10):
        D, I = self.index.search(queries, k)
        return D, I
