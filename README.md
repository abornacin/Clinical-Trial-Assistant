# 🧪 Clinical Trial Assistant (RAG + LLM with Groq)

An AI-powered assistant designed to support patients enrolled in clinical trials.  
It helps with visit schedules, study rules, side-effect reporting, and administrative questions using a Retrieval-Augmented Generation (RAG) pipeline and a large language model (LLM) via **Groq (LLaMA 3)**.

This project demonstrates how to adapt a generic LLM system into a **domain-specific healthcare assistant** with safety, structure, and modular design in mind.

---

## 🎯 Project Goals

- Build a realistic AI assistant for clinical trial participants  
- Apply RAG (Retrieval-Augmented Generation) for grounded answers  
- Use a free, fast LLM API (Groq + LLaMA 3)  
- Follow a professional, production-style project structure  
- Showcase modular, reusable Python components  

---

## 🧠 What the Assistant Can Do

✔ Answer questions about visit schedules  
✔ Explain study rules and participation guidelines  
✔ Help report side effects or symptoms  
✔ Handle administrative and logistical questions  
✔ Respond clearly and empathetically  
✔ Avoid medical diagnosis and escalate emergencies

---

## 🏗️ Project Structure

```text
clinical-trial-assistant/
├── data/
│   ├── clinical_trial_knowledge_base.csv   # Study rules, visits, meds, FAQs
│   ├── queries.json                        # Example user questions
├── notebooks/
│   └── 01_exploration.ipynb                # EDA + RAG + prompt experiments
├── src/
│   ├── embedder.py                        # Creates vector embeddings
│   ├── retriever.py                       # Retrieves relevant chunks
│   ├── generator.py                       # Calls LLM via Groq
│   └── main.py                            # Orchestrates the pipeline
├── .env.example                           # API key template
├── requirements.txt
├── README.md
└── LICENSE
