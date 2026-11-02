# ResearchRadar — Folder & Architecture Guide

This guide details the corporate-grade modular directory structure of ResearchRadar v3.

---

## 🗂️ Project Directory Structure

```
Researchradar/
├── app.py                      # Main Streamlit application and layout controller
├── requirements.txt            # Python production dependencies
├── .env.example                # Clean configuration template (safe to commit)
├── .gitignore                  # Git exclusions (API keys, caches, ChromaDB binaries)
├── README.md                   # Comprehensive repository overview and quickstart
│
├── core/                       # Core Architecture & RAG Engine
│   ├── __init__.py             # Clean package entrypoint and exported APIs
│   ├── config.py               # Centralized configuration & environment loader
│   ├── logger.py               # Structured logging & performance timer decorator
│   ├── security.py             # Magic-byte file validation & input sanitization
│   ├── metrics.py              # Operational health monitoring & latency tracker
│   ├── pdf_processor.py        # PDFPlumber page extractor & sliding-window chunker
│   ├── embedder.py             # SentenceTransformer all-MiniLM-L6-v2 singleton
│   ├── vector_store.py         # ChromaDB client & persistent collection manager
│   ├── retriever.py            # Hybrid RRF search (Dense ChromaDB + Sparse BM25Okapi)
│   ├── llm_handler.py          # Claude API caller & local grounded fallback engine
│   ├── query_router.py         # Slash command router (/debate, /compare, /predict, /workflow)
│   ├── plugins.py              # Specialized agent classes (Debate, Comparison, Prediction, Workflow)
│   ├── analytics.py            # Bias detector, cosine heatmap, stance & discourse tension
│   ├── engine.py               # High-level pipeline facade (ingest_paper & orchestration)
│   └── ui.py                   # Obsidian-slate design system & CSS injection
│
├── memory/                     # Conversation Memory
│   ├── __init__.py
│   └── conversation.py         # Sliding-window conversation history (5 turns)
│
├── docs/                       # System Documentation & Architecture Specs
│   ├── architecture.md         # Detailed system architecture blueprint & Mermaid flowcharts
│   ├── folder_guide.md         # Directory structure and module guide (this file)
│   ├── workflow.md             # End-to-end data flow and execution trace
│   └── powerbi_integration_guide.md # Analytical export guidelines
│
└── data/                       # Local Data & Persistent Stores
    ├── chroma_db/              # Persistent ChromaDB vector index (Git-ignored)
    ├── claims/                 # Extracted research claims
    └── exports/                # Analytical CSV datasets
```

---

## 🧩 Module Breakdown

### `core/config.py`
Centralizes all environment variables, file paths, embedding context window limits, and security boundaries. Eliminates scattered `os.getenv` calls.

### `core/security.py`
Validates uploaded PDFs by checking extensions, file size limits, and `%PDF-` magic header bytes to block spoofed or malicious uploads. Also provides user input sanitization.

### `core/logger.py` & `core/metrics.py`
Provides enterprise-grade observability:
- `core/logger.py`: Standardized UTC timestamps, module-level logging, and a `@timed` decorator.
- `core/metrics.py`: Tracks operation latencies, request counts, failure rates, and exposes system health checks.

### `core/pdf_processor.py`
Extracts text page-by-page and segments documents into 1000-character chunks with 150-character overlap. Detects academic sections (`Abstract`, `Introduction`, `Methodology`, `Results`, `Limitations`, `Conclusion`, `References`).

### `core/embedder.py`
Loads and manages the local `all-MiniLM-L6-v2` embedding model (384-dimensional dense vectors) using a cached singleton pattern.

### `core/vector_store.py`
Manages ChromaDB persistent vector database operations including chunk upserts, cosine similarity queries, and document deletions.

### `core/retriever.py`
Executes hybrid retrieval:
1. Dense semantic search via ChromaDB embeddings.
2. Sparse keyword matching via an in-memory cached BM25Okapi inverted index.
3. Reciprocal Rank Fusion (RRF, $k=60$) combining both ranking lists.

### `core/llm_handler.py`
Interfaces with Anthropic Claude 3.5 Sonnet to produce cited research answers. If the API key is missing or quota is exceeded, gracefully activates local grounded synthesis.

### `core/query_router.py` & `core/plugins.py`
Routes user requests based on slash commands:
- `/debate`: Adversarial Debate Agent (for/against arguments, robustness score).
- `/compare`: Comparative Differentiation Agent (multi-paper matrix, trade-offs).
- `/predict`: Future Prediction Agent (theoretical limits, 3–5 year forecast).
- `/workflow`: Architecture Workflow Agent (Mermaid flowcharts, ablation tables).
- Default: Direct Grounded Research Q&A.

### `core/analytics.py`
Computes author claim-to-caution ratios, empirical rigor scores, cross-paper cosine similarity matrices, stance distributions, and live inquiry discourse tension.
