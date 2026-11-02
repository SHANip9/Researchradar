# ResearchRadar — End-to-End Workflow

This document traces the complete execution flow from PDF upload to answer synthesis and dynamic analytical inspection.

---

## Part 1: Document Ingestion Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit UI (app.py)
    participant Sec as Security (core/security.py)
    participant PDF as PDF Processor (core/pdf_processor.py)
    participant Emb as Embedder (core/embedder.py)
    participant VStore as Vector Store (core/vector_store.py)

    User->>UI: Upload research paper (e.g. paper.pdf)
    UI->>Sec: validate_pdf_upload(file)
    Note over Sec: Verifies file size, extension (.pdf)<br/>& %PDF- magic bytes
    Sec-->>UI: Validation OK
    UI->>PDF: extract_pdf(tmp_path)
    PDF-->>UI: Dict of page numbers & text
    UI->>PDF: chunk_document(pages, paper_name)
    Note over PDF: Sliding window (1000 chars, 150 overlap)<br/>Section detection & unique SHA256 chunk IDs
    PDF-->>UI: Structured chunk objects
    UI->>Emb: embed_texts(chunk_texts)
    Emb-->>UI: 384-dimensional normalized vectors
    UI->>VStore: upsert_chunks(ids, docs, metas, vectors)
    VStore-->>UI: Ingestion complete
    UI-->>User: Paper indexed successfully (chunk count displayed)
```

---

## Part 2: Query Processing & Hybrid Multi-Agent Retrieval

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit UI (app.py)
    participant Router as Router (core/query_router.py)
    participant Ret as Retriever (core/retriever.py)
    participant Chroma as ChromaDB Vector Store
    participant BM25 as In-Memory BM25 Index
    participant Agent as Agent Plugin (core/plugins.py)
    participant LLM as LLM Handler (core/llm_handler.py)
    participant Mem as Memory (memory/conversation.py)

    User->>UI: Enter query (e.g. "/debate Scaling limits" or direct Q&A)
    UI->>Router: parse_and_route_query(user_query, active_filter)
    Router->>Agent: Invoke DebateAgent / Compare / Predict / Workflow / QA
    Agent->>Ret: hybrid_search(query, top_k=8, paper_filter)
    
    par Dense Search
        Ret->>Chroma: Cosine similarity query (top candidates)
        Chroma-->>Ret: Ranked semantic chunks
    and Sparse Search
        Ret->>BM25: BM25Okapi keyword scoring
        BM25-->>Ret: Ranked lexical chunks
    end

    Note over Ret: Reciprocal Rank Fusion (k=60)<br/>score = 1/(60+rank_sem) + 1/(60+rank_bm25)
    Ret-->>Agent: Top-K fused context chunks

    Agent->>LLM: generate_llm_response(system_prompt, prompt, chunks)
    alt Claude API Available
        LLM->>LLM: Call Claude 3.5 Sonnet API
    else API Key Missing / Quota Exceeded
        LLM->>LLM: Local Grounded Persona Synthesis with verified page citations
    end
    LLM-->>Agent: Synthesized response + citation metadata
    Agent-->>UI: Rendered response
    UI->>Mem: add_turn(user_query, response, agent)
    UI-->>User: Display markdown response, citations & interactive Mermaid diagrams
```

---

## Part 3: Live Discourse & Semantic Analytics

When the user explores the **Dynamic Semantic & Bias Dashboard**:
1. `core/analytics.py` analyzes the author's claim-to-caution ratio (measuring scientific hedging phrases vs overclaiming superlatives).
2. Computes the **Methodological Rigor Breakdown** (empirical baseline references and limitation disclosures).
3. Evaluates **Dynamic Discourse Tension** by mapping inquiry stances (Adversarial, Technical, Comparative, Exploratory) against the paper's academic sections.
4. Computes cross-paper cosine similarity matrices displayed as interactive Plotly heatmaps.
