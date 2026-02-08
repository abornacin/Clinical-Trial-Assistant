import json
import numpy as np
from datetime import datetime, timezone
from sklearn.metrics.pairwise import cosine_similarity
from src.embedder import Embedder
from src.generator import Generator


class Chatbot:
    def __init__(self, responses_path="/content/clinical-trial-assistant/data/chatbot_responses.json", threshold=0.55):
        with open(responses_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # raw is a LIST of records → extract retrieved_response fields
        self.response_texts = [r["retrieved_response"] for r in raw if "retrieved_response" in r]

        if not self.response_texts:
            raise ValueError("No 'retrieved_response' fields found in chatbot_responses.json")

        self.embedder = Embedder()
        self.response_embeddings = self.embedder.embed(self.response_texts, batch_size=32)

        self.generator = Generator()
        self.threshold = threshold
        self.history = []

    def get_response(self, query_text):
        query_embedding = self.embedder.embed([query_text])[0]
        sims = cosine_similarity([query_embedding], self.response_embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= self.threshold:
            retrieved_response = self.response_texts[best_idx]
            confidence = best_score
        else:
            prompt = f"You are a helpful clinical trial assistant. Answer clearly:\n\n{query_text}"
            retrieved_response = self.generator.generate_safe(prompt)
            confidence = 0.30

        record = {
            "query_text": query_text,
            "retrieved_response": retrieved_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence_score": round(confidence, 3)
        }

        self.history.append(record)
        return record

    def save_history(self, output_path="/content/clinical-trial-assistant/data/sample_chatbot_responses.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
