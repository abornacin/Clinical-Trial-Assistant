import numpy as np

class Retriever:
    def __init__(self, embeddings, texts):
        """
        Stores embeddings and corresponding texts.
        """
        self.embeddings = embeddings
        self.texts = texts

    def retrieve(self, query_embedding, top_k: int = 3):
        """
        Retrieves the top-k most similar text chunks.
        """
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.texts[i] for i in top_indices]
