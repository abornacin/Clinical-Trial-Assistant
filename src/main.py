import os

# --- Load API key from .env (local / Codespaces) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# --- Load API key from Colab Secrets (if in Colab) ---
try:
    from google.colab import userdata
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
except:
    pass

import pandas as pd
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "clinical_trial_knowledge_base.csv")

EMERGENCY_KEYWORDS = ["chest pain", "difficulty breathing", "fainting", "severe bleeding", "emergency"]

def is_emergency(text):
    text = text.lower()
    return any(word in text for word in EMERGENCY_KEYWORDS)

def run(query):
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].tolist()

    embedder = Embedder()
    embeddings = embedder.embed(texts)

    retriever = Retriever(embeddings, texts)
    generator = Generator()

    if is_emergency(query):
        return "🚨 If this is urgent or severe, please contact your study doctor or emergency services immediately."

    q_embed = embedder.embed([query])[0]
    context = "\n".join(retriever.retrieve(q_embed))

    return generator.generate(context, query)

if __name__ == "__main__":
    print("🧪 Clinical Trial Assistant (type 'exit' to quit)\n")
    while True:
        q = input("Ask: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\n" + run(q) + "\n")