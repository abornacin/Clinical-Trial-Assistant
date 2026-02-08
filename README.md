# 🧪 Clinical Trial Assistant (RAG + LLM with Groq)

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about **clinical trials** using semantic search over a predefined knowledge base, with fallback to a Large Language Model for open-ended queries.

This project demonstrates how to adapt a generic LLM system into a **domain-specific healthcare assistant** with safety, structure, and modular design in mind.

---

## 🎯 Project Goals

- Sentence embeddings
- Cosine similarity for retrieval
- LLM fallback for unmatched queries
- Structured JSON outputs for integration

---

## 📌 Features

✔ Accepts natural language queries  
✔ Uses embeddings to compute semantic similarity  
✔ Retrieves the most relevant predefined responses  
✔ Falls back to LLM when no good match is found  
✔ Stores full conversation history  
✔ Outputs structured JSON for downstream systems  

---

## 🏗️ Project Structure

```text
clinical-trial-assistant/
├── data/
│   ├── knowledge_base.csv
│   ├── chatbot_responses.json
│   ├── processed_queries.csv
│   ├── query_responses.json
│   ├── predefined_responses.json
├── notebooks/
│   └── 02_clinical_trial_assistant_demo.ipynb
├── src/
│   ├── embedder.py
│   ├── retriever.py
│   ├── generator.py
│   ├── chatbot.py
├── .env.example                           
├── requirements.txt
├── README.md
└── LICENSE

```
---

## ⚠️ Medical Disclaimer

This project is for educational and informational purposes only.
It does NOT provide medical diagnosis or treatment advice.
Always consult a qualified healthcare professional.
