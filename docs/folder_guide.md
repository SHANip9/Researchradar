# ResearchRadar — Folder & File Guide

> **Who this is for:** You, when you open this project a week from now and want to know what everything does without re-reading all the code.

---

## Project Root

| File / Folder | What It Does |
|---|---|
| `app.py` | **Start here.** The Streamlit UI. Run `streamlit run app.py` to launch. |
| `requirements.txt` | All Python dependencies. Install with `pip install -r requirements.txt`. |
| `.env` | Your secret API keys. Never commit this. Copy from `.env.example`. |
| `.env.example` | Template showing what keys are needed and how to get them. |
| `.gitignore` | Tells Git what to ignore (API keys, model cache, database files). |
| `README.md` | GitHub-facing documentation. What visitors see first. |

---

## `core/` — The Brain

This folder contains all the backend logic. Each file does exactly one job.

### `core/embedder.py`
**Job:** Converts text into numbers (embedding vectors).

Every chunk of text from your PDFs gets converted into a list of 384 numbers.
Similar texts have similar number patterns — this is how semantic search works.

- Uses `all-MiniLM-L6-v2` from Sentence Transformers (runs locally, free).
- Loads the model once (singleton pattern) to avoid reloading every time.
- Two functions: `embed_texts()` for batches, `embed_query()` for single questions.

**When does it run?** On upload (to embed each chunk) + on every question (to embed the question).

---

### `core/pdf_processor.py`
**Job:** Reads PDFs and splits content into manageable chunks.

Raw PDFs can be 50+ pages. We can't send that all to an LLM. So we:
1. Extract text page-by-page (using `pdfplumber`)
2. Split into ~512-token chunks with 50-token overlap (using LangChain's splitter)
3. Attach metadata to each chunk: which paper, which page, unique ID

**When does it run?** Once per PDF upload.

---

### `core/vector_store.py`
**Job:** Stores and retrieves chunks from ChromaDB (the vector database).

Think of this as the filing system. When you upload a paper, chunks go in.
When you ask a question, relevant chunks come out.

- Uses ONE collection for all papers (metadata tells them apart).
- Persistent: survives app restarts. Your papers are still there tomorrow.
- Key functions: `add_paper_chunks()`, `semantic_search()`, `get_all_chunks()`, `delete_paper()`.

**When does it run?** On upload (store) + on every question (retrieve).

---

### `core/sentiment_analyzer.py`
**Job:** Tags each paper as Optimistic / Cautious / Critical / Neutral.

Uses a DistilBERT model (pre-trained on sentiment classification).
Analyzes each chunk → maps POSITIVE/NEGATIVE to our 4-class taxonomy → aggregates per paper.

- Optimistic: paper is positive and confident about findings
- Cautious: paper has measured concerns or limitations
- Critical: paper strongly critiques existing work or reports failures
- Neutral: paper reports facts without strong tone

**When does it run?** On upload (during indexing, before storage).

---

### `core/retriever.py`
**Job:** The smart search engine. Finds the most relevant chunks for a question.

This is the most technically complex file. It combines:
1. **Semantic search** (ChromaDB + cosine similarity) — finds conceptually similar chunks
2. **BM25 keyword search** — finds chunks with exact matching words
3. **RRF fusion** — merges both ranked lists into one optimal ranking

Two retrieval modes:
- `retrieve_precise()` → top-8 chunks globally (for specific questions)
- `retrieve_synthesis()` → top-3 chunks per paper (for comparison questions)

**When does it run?** On every question, after routing.

---

### `core/query_router.py`
**Job:** Looks at your question and decides which retrieval mode to use.

Simple keyword matching:
- Contains "compare", "contrast", "which paper", "all papers" → Synthesis mode
- Everything else → Precise mode

No AI needed here — pattern matching is fast, transparent, and correct for research questions.

**When does it run?** On every question, before retrieval.

---

### `core/llm_handler.py`
**Job:** Sends the retrieved chunks to Claude and gets a cited answer back.

The most important design choice: Claude is given a **strict system prompt** that tells it:
- Answer ONLY from the provided chunks
- If the answer isn't there, say so explicitly
- Cite every claim with (Paper: name | Page: N)

This prevents hallucination by design — Claude cannot answer from outside your papers.

**When does it run?** On every question, after retrieval.

---

## `memory/` — Conversation Context

### `memory/conversation.py`
**Job:** Remembers the last 5 Q&A turns so follow-up questions work.

Without this: "Which paper disagreed with that?" fails because "that" has no context.
With this: the last 5 turns are included in every Claude prompt.

Uses a sliding window (oldest turn dropped when limit reached) to prevent prompt bloat.

**When does it run?** After every answer (to store it) + on every question (to retrieve history).

---

## `data/` — Storage

### `data/chroma_db/`
**Auto-created** when you first upload a paper. Contains ChromaDB's binary files.
Do NOT manually edit or delete files here — use the "Clear All" button in the app.
This folder is in `.gitignore` and will NOT be committed to GitHub.

---

## `docs/` — Documentation

| File | Contents |
|---|---|
| `folder_guide.md` | This file — what every file/folder does |
| `architecture.md` | System architecture with pipeline diagrams |
| `workflow.md` | Step-by-step data flow from upload to answer |

---

## Data Flow Summary

```
You upload a PDF
    ↓ pdf_processor.py    extracts text, creates chunks
    ↓ embedder.py         converts chunks → vectors
    ↓ sentiment_analyzer  tags each chunk
    ↓ vector_store.py     saves everything to ChromaDB
    ✅ Paper is indexed

You ask a question
    ↓ query_router.py     decides: precise or synthesis mode
    ↓ retriever.py        hybrid search (semantic + BM25 + RRF)
    ↓ llm_handler.py      builds prompt → calls Claude API
    ↓ memory/conversation updates conversation history
    ✅ Answer appears with citations + sentiment tags
```
