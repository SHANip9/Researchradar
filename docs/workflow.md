# ResearchRadar — Complete Workflow

> A step-by-step trace of exactly what happens, from PDF upload to final answer.

---

## Part 1: Uploading a PDF

### What you do:
Drag a PDF into the sidebar uploader.

### What happens internally:

**Step 1 — `pdf_processor.extract_text_by_page()`**
- `pdfplumber` opens the PDF and reads each page
- Text is extracted per-page and stored in a dict: `{1: "page 1 text", 2: "page 2 text", ...}`
- Image-only pages (scanned PDFs) produce no text → skipped

**Step 2 — `pdf_processor.chunk_document()`**
- LangChain's `RecursiveCharacterTextSplitter` splits each page into chunks
- Target: ~2000 characters per chunk (~512 tokens)
- 200-character overlap between consecutive chunks prevents content being cut at boundaries
- Each chunk gets metadata: `{source, page, chunk_id, char_count}`

**Step 3 — `embedder.embed_texts()`**
- All chunk texts are sent to `all-MiniLM-L6-v2` (runs locally)
- Each chunk becomes a 384-dimensional vector
- Result: a matrix of shape `(num_chunks, 384)`

**Step 4 — `sentiment_analyzer.analyze_chunks()`**
- DistilBERT processes chunks in batches of 32
- Raw output: `POSITIVE (0.89)` or `NEGATIVE (0.72)`
- Mapped to 4 labels: optimistic / cautious / critical / neutral
- `aggregate_paper_sentiment()` does weighted majority vote → one label per paper

**Step 5 — `vector_store.add_paper_chunks()`**
- All chunks, embeddings, and sentiment metadata written to ChromaDB
- ChromaDB saves to disk immediately (persistent)
- If same paper re-uploaded → `upsert()` updates existing entries (no crash)

**Result:** Paper is fully indexed. UI shows: ✅ `paper.pdf — 142 chunks | 🟢 Optimistic`

---

## Part 2: Asking a Question

### What you do:
Type a question in the text box. Click "Ask →".

### What happens internally:

**Step 1 — `query_router.route_question()`**
- Your question is checked for synthesis trigger keywords
- "What method did paper 1 use?" → **Precise mode** (no triggers found)
- "Compare how all papers approach X" → **Synthesis mode** (trigger: "compare", "all papers")

---

**If PRECISE mode:**

**Step 2a — `embedder.embed_query()`**
- Your question → 384-dimensional vector

**Step 3a — `retriever.retrieve_precise()`**
- ChromaDB semantic search: top-20 most similar chunks across ALL papers
- BM25 scores all chunks in the corpus for exact keyword matches
- RRF fusion: `score = 1/(60 + semantic_rank) + 1/(60 + bm25_rank)` for each chunk
- Top-8 highest-scoring chunks selected

---

**If SYNTHESIS mode:**

**Step 2b — For each uploaded paper:**
- Semantic search filtered to that paper: top-20 chunks
- BM25 scores filtered to that paper's chunks
- RRF fusion → top-3 chunks from this paper
- Repeat for all papers

**Step 3b — Combine:**
- Results from all papers merged: `top-3 from paper1 + top-3 from paper2 + ...`
- Each chunk clearly labeled with its source paper

---

**Step 4 — `memory.conversation.get_history()`**
- Last 5 Q&A turns retrieved as a formatted string
- If first question: returns empty string → Claude told "this is the first question"

**Step 5 — `llm_handler.build_prompt()`**
- Retrieved chunks formatted with headers: `[Chunk 1 | Paper: X | Page: 4]`
- Conversation history appended
- User question appended
- Full prompt assembled

**Step 6 — `llm_handler.get_answer()` → Claude API call**
- System prompt tells Claude: "Answer ONLY from the chunks. Cite every claim."
- User message: formatted prompt from step 5
- Claude returns answer text with inline citations like `(Paper: X | Page: 4)`

**Step 7 — Parse citations + build source list**
- Unique `(source, page)` pairs extracted from retrieved chunks
- Formatted as source cards for UI display

**Step 8 — `memory.conversation.add_turn()`**
- Question + answer stored in sliding window (last 5 turns kept)
- Used as context for the NEXT question

**Step 9 — Streamlit UI reruns**
- Answer displayed in chat panel
- Source cards shown below (paper name + page + sentiment badge)
- Mode indicator shown (🎯 Precise or 🔀 Synthesis)

---

## Part 3: Asking a Follow-Up

### What you do:
Ask "Which paper disagreed with that?"

### What makes this work:
The conversation history from Step 4 above contains:
```
Q1: [your previous question]
A1: [Claude's previous answer, truncated to 500 chars]
```

This gets injected into the new prompt. Claude sees "that" refers to the previous answer
and can correctly identify which paper disagreed.

---

## Part 4: Session Persistence

When you close and reopen the app:
- ChromaDB loads from disk automatically (`data/chroma_db/`)
- All your papers are still indexed — no need to re-upload
- Conversation history is cleared (it's stored in RAM via `st.session_state`)
- Sentiment data is recomputed from ChromaDB on first sidebar render
