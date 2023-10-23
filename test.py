import os

from retrieval.retrieval import Memory
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    raw_memory = ['This framework generates embeddings for each input sentence',
                  'Sentences are passed as a list of string.',
                  'The quick brown fox jumps over the lazy dog.']

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    memory = Memory(embedding_model=embedding_model, raw_memory=raw_memory, d_model=768)

    queries = [
        "Hello",
        "Hi I'm Huy"
    ]

    results = memory.search(queries, k=1)
    print(results)
