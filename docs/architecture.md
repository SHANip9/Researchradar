# ResearchRadar — System Architecture

## Overview

ResearchRadar is a **RAG (Retrieval-Augmented Generation)** system for research papers.

The key idea: instead of sending an entire paper to an LLM (expensive, slow, hits token limits),
we store all paper content as searchable chunks, retrieve only the RELEVANT ones per question,
and send just those to Claude. The LLM is constrained to answer only from retrieved content.

---

## The Two Pipelines

### Pipeline 1: Indexing (runs once per PDF upload)

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                        │
│                  (runs once per upload)                     │
│                                                             │
│  📄 PDF Upload                                              │
│       │                                                     │
│       ▼                                                     │
│  pdf_processor.py ──── PDFPlumber extracts text per page   │
│       │                                                     │
│       ▼                                                     │
│  pdf_processor.py ──── LangChain splits into chunks        │
│                         512 tokens, 50 token overlap        │
│       │                                                     │
│       ▼                                                     │
│  embedder.py ────────── Sentence Transformers               │
│                         chunk text → 384-dim vector         │
│       │                                                     │
│       ▼                                                     │
│  sentiment_analyzer.py  DistilBERT per chunk               │
│                         → Optimistic/Cautious/Critical/Neutral│
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────┐                                       │
│  │   ChromaDB       │ ← all chunks + vectors + metadata     │
│  │   Vector DB      │    stored persistently on disk        │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

---

### Pipeline 2: Query (runs on every question)

```
┌─────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                          │
│                  (runs on every question)                   │
│                                                             │
│  💬 User Question                                           │
│       │                                                     │
│       ▼                                                     │
│  query_router.py ─── keyword matching → "precise" or       │
│       │               "synthesis" mode                      │
│       ▼                                                     │
│  embedder.py ────── question → 384-dim vector               │
│       │                                                     │
│       ▼                                                     │
│  retriever.py ──── HYBRID SEARCH:                           │
│       │            • Semantic: ChromaDB cosine similarity   │
│       │            • Keyword: BM25 exact term matching      │
│       │            • Fusion: RRF combines both rankings     │
│       │                                                     │
│       │            PRECISE mode → top-8 global             │
│       │            SYNTHESIS mode → top-3 per paper        │
│       │                                                     │
│       ▼                                                     │
│  llm_handler.py ─── Builds strict RAG prompt               │
│       │              Chunks + question + conversation history│
│       ▼                                                     │
│  Claude API ─────── Reads ONLY retrieved chunks            │
│       │              Answers with citations                  │
│       ▼                                                     │
│  memory/conversation ← stores this Q&A for follow-ups      │
│       │                                                     │
│       ▼                                                     │
│  🖥️ Streamlit UI ── displays answer + citations + sentiment │
└─────────────────────────────────────────────────────────────┘
```

---

## Hybrid Search Detail

```
Question: "What F1 score did the model achieve?"

SEMANTIC SEARCH (ChromaDB)             BM25 KEYWORD SEARCH
──────────────────────────             ──────────────────────
Rank 1: chunk about model performance  Rank 1: chunk with "F1: 0.94"
Rank 2: chunk about accuracy metrics   Rank 2: chunk with "F1 score of 94.2%"
Rank 3: chunk about evaluation         Rank 3: chunk about performance evaluation
...                                    ...

                    RRF FUSION
        score = 1/(60+rank_semantic) + 1/(60+rank_bm25)
                         │
                         ▼
                Final Ranked List:
                Rank 1: chunk with "F1: 0.94"  ← appears in both
                Rank 2: chunk about accuracy metrics
                Rank 3: chunk about model performance
```

---

## Sentiment Analysis Detail

```
Paper chunks (per chunk):
  DistilBERT → POSITIVE (0.91) → score > 0.75 → "optimistic"
  DistilBERT → POSITIVE (0.62) → score ≤ 0.75 → "neutral"
  DistilBERT → NEGATIVE (0.78) → score > 0.65 → "critical"
  DistilBERT → NEGATIVE (0.55) → score ≤ 0.65 → "cautious"

Aggregate (weighted majority vote):
  label with highest sum(score) across all chunks → paper-level tag
```

---

## Technology Stack

| Component | Technology | Reason |
|---|---|---|
| UI | Streamlit | Fast to build, perfect for research tools |
| PDF Extraction | PDFPlumber | Page-aware, handles multi-column layouts |
| Text Chunking | LangChain RecursiveTextSplitter | Respects paragraph/sentence boundaries |
| Embeddings | Sentence Transformers (MiniLM-L6-v2) | Free, local, 384-dim, fast |
| Vector DB | ChromaDB | Persistent, metadata filtering, one collection |
| Keyword Search | BM25 (rank-bm25) | Exact term matching, complements semantic |
| Score Fusion | RRF (Reciprocal Rank Fusion) | Standard algorithm, no extra dependencies |
| LLM | Claude (Anthropic) | Best instruction-following for strict RAG |
| Sentiment | DistilBERT (HuggingFace) | Fast, local, binary → 4-class mapping |
| Memory | Custom sliding window | Simple, bounded, no extra dependencies |
