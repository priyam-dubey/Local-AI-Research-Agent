# 🔬 Local AI Research Agent

> A fully **offline**, production-ready, stateful multi-agent research system powered by **LangGraph**, **LangChain**, **Ollama** (Mistral / Llama 3), and **FAISS** — with zero cloud dependencies.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-green)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-orange)](https://ollama.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)](https://streamlit.io)

---

## ✨ Feature Highlights

| Feature | Details |
|---|---|
| **Fully Offline** | All LLM inference via local Ollama; embeddings via HuggingFace sentence-transformers |
| **Stateful Multi-Agent** | 4-node LangGraph StateGraph with typed AgentState shared across all nodes |
| **Hybrid RAG** | FAISS vector store + DuckDuckGo web search with intelligent routing |
| **Self-Critique Loop** | Critic agent assigns quality scores and triggers re-research loops |
| **Multi-format Ingestion** | PDF, TXT, MD, and raw web URLs → chunked → FAISS |
| **Live Streaming UI** | Streamlit frontend with real-time SSE agent activity feed |
| **PDF Export** | One-click export of final Markdown reports to clean PDF via ReportLab |
| **Infinite Loop Guard** | `MAX_ITERATIONS` ceiling in Critic node prevents runaway graph cycles |

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     LOCAL AI RESEARCH AGENT                             ║
║                                                                          ║
║  ┌─────────────┐    ┌──────────────────────────────────────────────┐    ║
║  │  Streamlit  │◄──►│              FastAPI Backend                 │    ║
║  │   ui.py     │    │                 app.py                       │    ║
║  │             │    │  POST /research  GET /stream  POST /ingest   │    ║
║  └─────────────┘    └───────────────────┬──────────────────────────┘    ║
║                                         │                                ║
║                          ┌──────────────▼───────────────┐               ║
║                          │      LangGraph StateGraph     │               ║
║                          │          graph.py             │               ║
║                          │                               │               ║
║                          │  START                        │               ║
║                          │    │                          │               ║
║                          │    ▼                          │               ║
║                          │  ┌──────────┐                │               ║
║                          │  │ PLANNER  │  Decomposes     │               ║
║                          │  │ planner  │  query → plan   │               ║
║                          │  └────┬─────┘                │               ║
║                          │       │                       │               ║
║                          │       ▼                       │               ║
║              ┌───────────┤  ┌──────────────┐            │               ║
║              │  FAISS    │  │  RESEARCHER  │            │               ║
║              │  Index    │◄─┤  researcher  │            │               ║
║              │ (local)   │  │  (routes to  │            │               ║
║              └───────────┤  │  FAISS/DDG)  │            │               ║
║              ┌───────────┤  └──────┬───────┘            │               ║
║              │ DuckDuckGo│         │                     │               ║
║              │  Search   │         │                     │               ║
║              └───────────┤         ▼                     │               ║
║                          │  ┌──────────────┐            │               ║
║                          │  │    WRITER    │  Draft      │               ║
║                          │  │  (draft)     │  Markdown   │               ║
║                          │  └──────┬───────┘  Report     │               ║
║                          │         │                     │               ║
║                          │         ▼                     │               ║
║                          │  ┌──────────────┐            │               ║
║                          │  │    CRITIC    │            │               ║
║                          │  │   critic     │            │               ║
║                          │  │  score≥0.6?  │            │               ║
║                          │  └──────┬───────┘            │               ║
║                          │    No   │   Yes               │               ║
║                          │    ▼    │    ▼                │               ║
║                          │ [loop]  │  ┌──────────────┐  │               ║
║                          │         │  │    WRITER    │  │               ║
║                          │         └─►│   (final)    │  │               ║
║                          │            └──────┬───────┘  │               ║
║                          │                   │           │               ║
║                          │                  END          │               ║
║                          └───────────────────────────────┘               ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │                      RAG Subsystem                                │   ║
║  │  PDF/TXT/MD → PyPDFLoader/TextLoader → RecursiveCharacterSplitter│   ║
║  │  → HuggingFace Embeddings (all-MiniLM-L6-v2) → FAISS Index      │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
AI-Research-Agent/
├── requirements.txt          # Pinned dependencies
├── app.py                    # FastAPI gateway + SSE streaming + session mgmt
├── ui.py                     # Streamlit UI (sidebar upload, live feed, PDF export)
├── config.py                 # Centralized settings (Pydantic BaseSettings)
├── state.py                  # AgentState TypedDict with LangGraph reducers
├── graph.py                  # StateGraph topology: nodes, edges, conditions
├── agents/
│   ├── __init__.py
│   ├── planner.py            # Decomposes query → ordered task checklist
│   ├── researcher.py         # Routes FAISS/DDG, retrieves, synthesizes
│   ├── critic.py             # Scores report quality, triggers re-research
│   └── writer.py             # Generates polished Markdown + citations
├── rag/
│   ├── __init__.py
│   ├── ingestion.py          # PDF/TXT/MD/URL loading + chunking
│   └── vector_store.py       # FAISS index CRUD + persistence + metadata
├── data/
│   ├── uploads/              # Saved user file uploads
│   ├── faiss_index/          # Persisted FAISS index files
│   └── reports/              # Exported PDF reports
└── logs/                     # Loguru rotating log files
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- Ollama installed and running
- Windows, Linux, or macOS
- Internet connection required only for:
  - downloading Ollama models
  - downloading HuggingFace embedding models on first run
  - optional DuckDuckGo web search

### Recommended Hardware

| Configuration | Minimum | Recommended |
|--------------|----------|-------------|
| RAM | 8 GB | 16 GB+ |
| CPU | 4 cores | 6+ cores |
| Storage | 5 GB free | 10+ GB free |
| GPU | Not required | Optional |

### Ollama Model Recommendations

| Model | RAM Requirement | Recommendation |
|---------|----------------|---------------|
| mistral | ~8 GB | Recommended |
| llama3:8b | ~8–10 GB | Recommended |
| gemma:2b | ~4 GB | Best for low-end laptops |
| phi3:mini | ~4 GB | Lightweight alternative |

**Note:** Systems with only 8 GB RAM may need to close browsers and other applications before running larger models. If Ollama reports memory allocation errors, switch to a smaller model such as `gemma:2b` or `phi3:mini`.

### 1. Clone the repository

```bash
git clone https://github.com/youruser/AI-Research-Agent.git
cd AI-Research-Agent
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Pull the LLM via Ollama

```bash
# Start Ollama daemon (if not already running)
ollama serve

# Pull Mistral 7B (default)
ollama pull mistral

# OR pull Llama 3 8B
ollama pull llama3
```

To switch models, edit `config.py`:
```python
LLM_MODEL: str = "llama3"   # or "mistral"
```

### 4. (Optional) Configure environment

Copy `.env.example` to `.env` and override any defaults:

```bash
cp .env.example .env
```

Key variables:
```
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral
MAX_ITERATIONS=5
REFLECTION_THRESHOLD=0.6
```

---

## ▶️ Running the System

Open **two terminal windows**:

**Terminal 1 — Start the FastAPI backend:**
```bash
python app.py
# API docs: http://localhost:8000/docs
```

**Terminal 2 — Start the Streamlit UI:**
```bash
streamlit run ui.py
# Opens: http://localhost:8501
```

---

## ⚠️ Common Issues

### 1. Ollama Memory Allocation Error

Example:

```text
ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer
error loading model
---
Cause:
Insufficient available RAM for the selected model.

Solutions:
1. Close memory-intensive applications.
2. Restart Ollama.
3. Switch to a smaller model

##Bash : ollama pull gemma:2b

Then update:

LLM_MODEL = "gemma:2b"

4. Verify available memory:

Windows:
##Bash
systeminfo | findstr /C:"Total Physical Memory"

Linux:
free -h

## 🛠️ Usage Guide

### Ingest Documents (optional)
1. Open the **sidebar** in the Streamlit UI
2. Upload PDF/TXT/MD files, paste a URL, or type raw text
3. Click **Ingest** — chunks are embedded and added to FAISS

### Run Research
1. Type your research question in the main panel
2. Click **🚀 Research**
3. Watch the 4 agents work in real-time:
   - 🗺️ **Planner** breaks down the query
   - 🔬 **Researcher** retrieves evidence (local FAISS or web)
   - 🧐 **Critic** scores quality and may loop back
   - ✍️ **Writer** produces the final Markdown report
4. Download the report as a **PDF** with one click

### Direct API Usage

```bash
# Start a research session
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "How does RLHF work in modern LLMs?"}'

# Stream events (SSE)
curl -N http://localhost:8000/research/{session_id}/stream

# Get final result
curl http://localhost:8000/research/{session_id}/result
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `mistral` | Ollama model name |
| `LLM_TEMPERATURE` | `0.2` | Generation temperature |
| `MAX_ITERATIONS` | `5` | Hard ceiling on critic re-loop cycles |
| `REFLECTION_THRESHOLD` | `0.6` | Minimum critic score to proceed |
| `CHUNK_SIZE` | `1000` | RAG text chunk size (chars) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap for context continuity |
| `RETRIEVER_TOP_K` | `6` | FAISS results returned per query |
| `MAX_SEARCH_RESULTS` | `5` | DuckDuckGo results per sub-task |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |

---

## 🧠 Technical Design Decisions

### Preventing Infinite Loops in Cyclic Graphs
LangGraph's cyclic graphs can theoretically loop forever if the Critic never approves. Two complementary guards prevent this:
1. **`MAX_ITERATIONS` in `critic_node`**: If `state["iteration"] >= MAX_ITERATIONS`, the critic forcibly sets `needs_more_research=False` and `quality_score=1.0`, bypassing the normal threshold check.
2. **`_should_continue_research` conditional edge**: Independently checks `iteration >= MAX_ITERATIONS` before routing, ensuring the graph always terminates.

### Hybrid Retrieval Routing
The Researcher agent uses the local Ollama LLM to classify each sub-task as `"local"` (covered by uploaded documents) or `"web"` (requires live search). If local retrieval returns zero results, it automatically falls back to DuckDuckGo — ensuring no sub-task is left unanswered.

### Thread-Safe FAISS
FastAPI runs an async event loop with background tasks. The FAISS vector store uses a module-level `threading.Lock()` to serialize all index mutations, preventing race conditions when multiple documents are ingested concurrently.

### Context Window Management
All prompts are trimmed to safe lengths before being sent to Ollama (`evidence_text[:4000]`, `draft_report[:1500]`) to stay within the 8192-token context window of Mistral 7B.

### Persistent Index
After every `add_documents` call, `vector_store.py` calls `FAISS.save_local()`. On API startup, `load_persisted_index()` restores the previous session's knowledge base automatically.

---

## 📦 Dependencies (key packages)

| Package | Version | Purpose |
|---|---|---|
| `langgraph` | 0.2.28 | Stateful multi-agent graph orchestration |
| `langchain` | 0.3.7 | LLM chains, loaders, splitters |
| `langchain-ollama` | 0.2.0 | Ollama LLM integration |
| `faiss-cpu` | 1.9.0 | Dense vector similarity search |
| `sentence-transformers` | 3.2.1 | Local embeddings (all-MiniLM-L6-v2) |
| `duckduckgo-search` | 6.3.7 | Offline-friendly web search |
| `fastapi` | 0.115.4 | REST API + SSE streaming backend |
| `streamlit` | 1.40.1 | Interactive frontend UI |
| `reportlab` | 4.2.5 | PDF report generation |

---

## 🤝 Contributing

Pull requests welcome! Please open an issue first for major changes.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
