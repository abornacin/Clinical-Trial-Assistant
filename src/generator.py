import os
from groq import Groq


class Generator:
    def __init__(self):
        """
        Initializes Groq client using API key from environment.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        self.client = Groq(api_key=api_key)

    def generate(self, context: str, question: str) -> str:
        """
        Generates a response using LLaMA 3 via Groq.
        """
        prompt = f"""
You are an automated Clinical Trials Assistant.

You help patients with:
• Visit schedules
• Study rules
• Side effect reporting
• Administrative questions

Rules:
• Be clear, calm, and empathetic
• Do NOT give medical diagnosis
• If symptoms are severe, encourage contacting the study team

Context:
{context}

Patient Question:
{question}

Answer:
"""
        completion = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )

        return completion.choices[0].message.content.strip()
