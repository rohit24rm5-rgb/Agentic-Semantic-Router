# Aegis Emotion Engine 🌀

An advanced, production-ready multi-agent AI system for Emotion Analysis. 
This project has been completely overhauled from a legacy monolithic TensorFlow/C++ architecture into a state-of-the-art **AI Engineering** pipeline.

## 🚀 Architectural Paradigm Shift

| Feature | 2025 Legacy Version | 2026 AI Engineering Version |
| :--- | :--- | :--- |
| **Inference Engine** | Local `.keras` Sequential Model | **LangGraph Swarm** + **Groq LLaMA 3 70B** |
| **Backend** | Python Monolith + C++ CMake | **FastAPI** (Async, Pydantic, REST) |
| **Context Memory** | None (Zero-Shot Only) | **ChromaDB RAG** (Kaggle Datasets) |
| **State Persistence** | None | **SQLite WAL** (LangGraph Checkpointer) |
| **Token Optimization** | N/A | **Zero-Routing Semantic Cache** (FastEmbed) |
| **High Availability** | C++ Client Crashes on errors | **Async Circuit Breaker** (Gemini 1.5 Fallback) |
| **UI Experience** | Basic Streamlit UI / Console | **Aesthetic Vanilla Glassmorphism UI** |

---

## 🧠 Multi-Agent Swarm Structure

We utilize a 3-agent orchestration pattern via LangGraph, strictly optimizing for the 6k TPM free-tier limit on Groq:

1. **Semantic Zero-Router**: Uses FastEmbed and ChromaDB to semantically match queries. If a query matches a cached response with >95% similarity, it returns instantly (0 tokens, <50ms latency).
2. **Lexical Emotion Agent**: Analyzes the explicit, surface-level emotional tone (Joy, Anger, Sadness).
3. **Contextual Nuance Agent**: Receives the lexical summary and cross-references RAG context (ingested from Kaggle Sarcasm datasets) to detect implicit irony and synthesize the *true* final emotion.

---

## 🛠️ Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Fill in GROQ_API_KEY, GEMINI_API_KEY, and Kaggle credentials
```

### 2. Data Engineering (RAG Ingestion)
Automatically download and index Kaggle datasets into ChromaDB for few-shot context:
```bash
python data/ingest_kaggle.py
```

### 3. Run the Inference Server
Launch the asynchronous FastAPI backend:
```bash
uvicorn src.emotion_analysis.app.main:app --reload
```

### 4. Open the Web App
Open `frontend/index.html` in your browser to experience the aesthetic, real-time emotion engine UI.

---

## 📊 Benchmarks & Latency Tracking

The system tracks latency, fallback rates, and cache hits autonomously. Run the stress-test suite to record benchmarks:

```bash
python tests/test_latency_ollama.py
```

*Note: The Zero-Routing cache regularly achieves **<50ms latency**, drastically outperforming the legacy TensorFlow Python load times while burning exactly zero LLM tokens.*
