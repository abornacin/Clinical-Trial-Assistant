import time
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class Generator:
    def __init__(self, model="llama-3.1-8b-instant"):
        self.model = model

    def generate_safe(self, prompt, retries=5, base_delay=2):
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a clinical trial assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                return response.choices[0].message.content

            except Exception as e:
                wait = base_delay * (2 ** attempt)
                print(f"⚠️ LLM call failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError("LLM request failed after retries.")
