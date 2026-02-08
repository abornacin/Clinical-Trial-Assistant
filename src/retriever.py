import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embedder import Embedder


class Retriever:
    def __init__(self, responses_path="/content/clinical-trial-assistant/data/predefined_responses.json"):
        with open(responses_path, "r", encoding="utf-8") as f:
            raw_responses = json.load(f)

        # Your JSON is a dict: {key: response_text}
        self.response_ids = list(raw_responses.keys())
        self.response_texts = list(raw_responses.values())

        self.embedder = Embedder()
        self.response_embeddings = self.embedder.embed(self.response_texts, batch_size=32)

    def retrieve(self, query_text, top_k=3):
        query_embedding = self.embedder.embed([query_text])[0]
        sims = cosine_similarity([query_embedding], self.response_embeddings)[0]

        top_indices = sims.argsort()[::-1][:top_k]

        top_responses = [self.response_texts[i] for i in top_indices]
        top_scores = [float(sims[i]) for i in top_indices]

        # Normalize scores to 0–1
        min_s, max_s = min(top_scores), max(top_scores)
        if max_s > min_s:
            norm_scores = [(s - min_s) / (max_s - min_s) for s in top_scores]
        else:
            norm_scores = [1.0 for _ in top_scores]

        return top_responses, norm_scores
