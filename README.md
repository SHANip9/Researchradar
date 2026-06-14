# 🔬 ResearchRadar — AI-Powered Research Paper Analysis & Debate Platform

> **Ingest research papers. Analyse them with deep NLP. Visualise everything in Power BI. Then debate their claims with AI.**

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5%2B-FF6F00?style=for-the-badge)
![Claude API](https://img.shields.io/badge/Claude-3.5_Sonnet-D4A574?style=for-the-badge&logo=anthropic&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)

---

## 📖 What is ResearchRadar?

ResearchRadar is a **RAG-based (Retrieval-Augmented Generation) research paper analysis platform** that goes far beyond simple Q&A. It ingests PDF research papers, runs multi-dimensional NLP analysis — sentiment classification, unsupervised topic discovery, semantic keyword extraction, and paper-to-paper similarity mapping — and exports all analytics to interactive **Power BI dashboards**. Its most unique feature is an **AI-powered debate engine** with three modes: AI peer-reviews a paper's claims, the user argues against a paper while AI defends with RAG-grounded evidence, or two papers debate each other on a topic. Every answer, defence, and critique is grounded in the actual paper content — traced to the source page — ensuring intellectual honesty and zero hallucination.

---

## ✨ Key Features

| | Feature | Description |
|---|---------|-------------|
| 🔍 | **Hybrid RAG Retrieval** | Semantic search (ChromaDB cosine) + keyword search (BM25Okapi) fused via Reciprocal Rank Fusion |
| 💬 | **Grounded Answers with Citations** | Every claim traced to source paper, page number, and chunk — no hallucination |
| 🎭 | **Multi-Level Sentiment Analysis** | Chunk → Section → Paper aggregation with sentiment trajectory visualisation |
| 🗺️ | **Unsupervised Topic Modelling** | BERTopic discovers hidden themes across your paper corpus automatically |
| 🔑 | **Semantic Keyword Extraction** | KeyBERT ranks keywords by contextual importance, not just frequency |
| 🌐 | **Paper-to-Paper Similarity Heatmaps** | Cosine similarity matrices at paper-level and section-level granularity |
| ⚔️ | **AI-Powered Debate Engine** | 3 modes — AI challenges claims, user vs AI, paper vs paper — all RAG-grounded |
| 📊 | **Power BI Dashboard Integration** | 5 CSV exports → 5 coordinated dashboard pages with interactive slicers |
| 🧠 | **Conversation Memory** | Sliding-window context (last 5 turns) for coherent follow-up questions |
| 🖥️ | **Premium Streamlit UI** | Dark glassmorphism theme, Inter font, gradient headers, real-time processing |

---

## 🏗️ System Architecture

```
                              ┌──────────────────────────────────┐
                              │        📄 PDF Upload (UI)         │
                              │   User provides: file + metadata  │
                              │   (title, org, category, date)    │
                              └───────────────┬──────────────────┘
                                              │
                                              ▼
                  ┌──────────────────────────────────────────────────┐
                  │             MODULE 1 — Ingestion Pipeline        │
                  │                                                  │
                  │  pdfplumber    → Extract text page-by-page       │
                  │  TextSplitter  → Chunk (2000 chars, 200 overlap) │
                  │  Regex engine  → Detect section headings         │
                  │  Claim parser  → Extract factual assertions      │
                  │  Metadata      → Tag every chunk with 11 fields  │
                  └───────────────────────┬──────────────────────────┘
                                          │
                                          ▼
                  ┌──────────────────────────────────────────────────┐
                  │           MODULE 1.5 — Embedding Engine          │
                  │                                                  │
                  │  all-MiniLM-L6-v2  → 384-dim dense vectors       │
                  │  Batch encode (chunks) / Single encode (queries) │
                  └───────────────────────┬──────────────────────────┘
                                          │
                                          ▼
                  ┌──────────────────────────────────────────────────┐
                  │          ChromaDB Vector Store (Persistent)      │
                  │   Collection: researchradar_papers               │
                  │   Cosine similarity │ Metadata filtering         │
                  │   Chunks + Vectors + Sentiment + Full metadata   │
                  └──────┬──────────────┬───────────────┬────────────┘
                         │              │               │
           ┌─────────────▼───┐   ┌──────▼────────┐  ┌──▼─────────────────┐
           │  MODULE 2        │   │ MODULE 4      │  │ Hybrid Retriever   │
           │  Analysis Engine │   │ Debate Engine │  │ (Semantic + BM25   │
           │                  │   │               │  │  + RRF Fusion)     │
           │ • Sentiment ×3   │   │ Mode A: AI    │  └──────┬─────────────┘
           │ • Similarity     │   │   challenges  │         │
           │ • BERTopic       │   │ Mode B: User  │         ▼
           │ • KeyBERT        │   │   vs AI       │  ┌──────────────────┐
           └────────┬─────────┘   │ Mode C: Paper │  │ LLM Handler      │
                    │             │   vs Paper    │  │ (Claude API)     │
                    ▼             └──────┬────────┘  │ RAG prompt +     │
           ┌────────────────┐           │            │ citations        │
           │  MODULE 3       │           │            └────────┬─────────┘
           │  Power BI Layer │           │                     │
           │  5 CSV exports  │           │                     ▼
           │  → 5 dashboard  │           │           ┌──────────────────┐
           │    pages        │           └──────────►│ MODULE 5         │
           └────────────────┘                       │ Streamlit UI     │
                                                    │ 4 Pages:         │
                                                    │  1. Query Chat   │
                                                    │  2. Analytics    │
                                                    │  3. Debate Arena │
                                                    │  4. Comparison   │
                                                    └──────────────────┘
```

---

## 🔬 How It Works — Technical Deep Dive

### 5.1 Ingestion Pipeline

**File:** `core/pdf_processor.py`

The ingestion pipeline transforms raw PDF files into searchable, analysable, metadata-rich text chunks stored in a vector database.

#### PDF Text Extraction — pdfplumber

ResearchRadar uses **pdfplumber** instead of PyPDF2 or PyMuPDF for a specific reason: research papers frequently use multi-column layouts, embedded tables, and complex formatting. pdfplumber's layout-aware text extraction handles these reliably, extracting text **page-by-page** while preserving reading order across columns.

```python
import pdfplumber

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()   # Handles multi-column layouts
        # Each page's text is tracked with its page number
```

#### Text Chunking — RecursiveCharacterTextSplitter

Raw page text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter` with carefully tuned parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | **2000 chars** | Large enough to preserve paragraph-level context for RAG; small enough for focused retrieval |
| `chunk_overlap` | **200 chars** | Prevents context loss at chunk boundaries — a sentence split mid-thought is captured in both adjacent chunks |
| Separator priority | `["\n\n", "\n", " ", ""]` | Splits at paragraph → line → word → character, preserving logical units |

Each chunk receives a **SHA256-based unique ID** derived from its content and source, ensuring deterministic deduplication.

#### Section Detection

A regex-based heuristic identifies section headings in the extracted text. Research papers typically mark sections with patterns like `"1. Introduction"`, `"METHODOLOGY"`, or `"3 Results"`. The detector scans for:

- Numbered headings (`1.`, `2.1`, etc.)
- Title-case or ALL-CAPS lines that stand alone
- Common academic section names (Abstract, Introduction, Methodology, Results, Discussion, Conclusion)

Each chunk is tagged with the **last detected section heading above it**, creating a section-level hierarchy without requiring any ML model.

#### Metadata Enrichment

Every chunk stored in ChromaDB carries a rich metadata payload:

```python
metadata = {
    # ─── Core fields ──────────────────────────────
    "source":                "teaching_claude_why.pdf",     # Original filename
    "page":                  3,                             # Source page number
    "char_count":            1847,                          # Chunk length
    "chunk_sentiment":       "optimistic",                  # Sentiment label
    "chunk_sentiment_score": 0.87,                          # Sentiment confidence

    # ─── Enriched fields (v2) ─────────────────────
    "paper_title":           "Teaching Claude Why",         # Human-readable title
    "author_org":            "Anthropic",                   # Source organisation
    "category":              "Alignment",                   # Research category
    "date":                  "2026-05-08",                  # Publication date
    "section":               "Introduction",                # Detected section heading
    "chunk_index":           4,                             # Sequential index in paper
}
```

These fields power Power BI slicers (filter by source, category, date), section-level analysis, and structured citation formatting.

#### Claim Extraction

A lightweight regex engine scans each chunk for sentences that match academic claim language patterns:

```python
claim_patterns = [
    r"we (show|find|demonstrate|propose|achieve|observe|introduce|present)",
    r"results (indicate|suggest|show|demonstrate|confirm|reveal)",
    r"our (model|approach|method|framework|system) (achieves|outperforms|surpasses|improves)",
    r"this (paper|work|study) (presents|introduces|proposes|contributes)",
    r"(significantly|substantially|consistently) (better|worse|higher|lower|outperform)",
]
```

Extracted claims are stored as structured objects linked to their parent chunk, paper, and section — pre-computing the data the Debate Engine (Module 4) needs to challenge or defend specific assertions.

---

### 5.2 Embedding Engine

**File:** `core/embedder.py`

#### Model: `all-MiniLM-L6-v2`

| Property | Value |
|----------|-------|
| Architecture | MiniLM (distilled from BERT) |
| Output dimensions | **384** |
| Max sequence length | 256 tokens |
| Speed | ~14,200 sentences/sec on GPU |
| Cost | **Free** — runs locally, no API calls |
| Quality | Top-tier for its size class on MTEB benchmarks |

**Why this model?** It strikes the optimal balance between embedding quality and inference speed for a local system. At 384 dimensions it produces compact vectors that ChromaDB can search efficiently, while still capturing deep semantic relationships. Unlike OpenAI's `text-embedding-ada-002` (1536 dims, paid API), this runs entirely on-device.

#### Singleton Pattern

The embedding model loads once into memory on first use and is reused across all subsequent calls — preventing redundant model loading on every PDF upload or query:

```python
class Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._instance
```

#### Encoding Modes

- **Batch encoding** — Used during ingestion. All chunks from a paper are embedded in a single `model.encode(chunks)` call, leveraging batch parallelism.
- **Single encoding** — Used at query time. The user's question is embedded individually for real-time search: `model.encode(query)`.

---

### 5.3 Vector Storage — ChromaDB

**File:** `core/vector_store.py`

#### Why Cosine Similarity?

ChromaDB is configured with **cosine similarity** as the distance metric, not Euclidean or dot product. For text embeddings, cosine similarity measures the *angle* between vectors, making it magnitude-invariant — a short sentence and a long paragraph about the same topic will have similar cosine scores even if their vector magnitudes differ. Euclidean distance would penalise the length difference, producing misleading results.

#### Single-Collection Architecture

All papers are stored in a **single ChromaDB collection** (`researchradar_papers`) rather than one collection per paper. This design choice enables:

- **Cross-paper search** — One query searches across the entire corpus
- **Metadata filtering** — `where={"source": "paper_name.pdf"}` isolates a specific paper when needed
- **Simpler CRUD** — No collection lifecycle management per paper

```python
# Cross-paper search (all papers)
results = collection.query(query_embeddings=[vector], n_results=20)

# Single-paper search (filtered)
results = collection.query(
    query_embeddings=[vector],
    n_results=20,
    where={"source": "teaching_claude_why.pdf"}
)
```

#### Persistence

ChromaDB stores its data at `data/chroma_db/`, surviving app restarts. Papers uploaded in one session are immediately available in the next.

---

### 5.4 Hybrid Retrieval — Semantic + BM25 + RRF

**File:** `core/retriever.py`

This is the core of ResearchRadar's RAG pipeline. Neither pure semantic search nor pure keyword search is sufficient alone — each has blind spots the other covers.

#### Semantic Search (Dense Retrieval)

The user's question is embedded into a 384-dim vector and searched against all chunk embeddings in ChromaDB using cosine similarity. This captures **meaning** — the query *"What safety measures does the paper propose?"* retrieves chunks about *"alignment techniques and safeguards"* even though no keywords overlap.

#### Keyword Search (BM25Okapi — Sparse Retrieval)

All chunk texts are indexed using **BM25Okapi**, the same TF-IDF variant that powers Elasticsearch. BM25 scores each chunk by how well it matches the query's exact terms, with two key refinements:

- **Term Frequency saturation** — The score contribution from a term appearing 10 times vs. 5 times is minimal (diminishing returns)
- **Document length normalisation** — Longer chunks aren't unfairly boosted just because they contain more words

BM25 catches what semantic search misses: specific names, acronyms (e.g., "RLHF"), numerical values, and exact terminology.

#### Reciprocal Rank Fusion (RRF)

The two ranked lists (semantic and BM25) are merged using **RRF** with `k=60`:

```
RRF_score(chunk) = Σ  1 / (k + rank_in_list)
                   for each list containing the chunk
```

**Why RRF?** It's rank-based, not score-based — meaning it doesn't need the two scoring systems (cosine similarity vs. BM25 TF-IDF) to be on the same scale. A chunk ranked #1 in both lists gets:

```
RRF = 1/(60+1) + 1/(60+1) = 0.0328
```

A chunk ranked #1 in semantic but #50 in BM25 gets:

```
RRF = 1/(60+1) + 1/(60+50) = 0.0255
```

The fused ranking consistently outperforms either individual method.

#### Two Retrieval Modes

| Mode | Triggered By | Behaviour | Use Case |
|------|-------------|-----------|----------|
| **Precise** | Default / single-paper queries | Top-8 globally from RRF fusion | *"What does this paper say about safety?"* |
| **Synthesis** | Keywords like *"compare"*, *"which paper"*, *"vs"* | Top-3 per paper, RRF within each paper's subset | *"Compare the approaches of Paper A and Paper B"* |

#### Query Routing

**File:** `core/query_router.py`

A lightweight keyword trigger matcher determines which retrieval mode to use:

```python
synthesis_triggers = ["compare", "which paper", "vs", "difference between",
                      "across papers", "all papers", "each paper"]
```

If the query contains any trigger, the router selects Synthesis mode. Otherwise, Precise mode. No ML needed — simple string matching is sufficient for this classification.

---

### 5.5 Sentiment Analysis

**File:** `core/sentiment_analyzer.py`

#### Model: `distilbert-base-uncased-finetuned-sst-2-english`

A lightweight DistilBERT model fine-tuned on the Stanford Sentiment Treebank. It outputs a binary POSITIVE/NEGATIVE classification with a confidence score (0.0–1.0).

#### Binary → 4-Label Mapping

Raw model output is mapped to four labels that better capture the nuance of academic writing:

| Raw Output | Confidence | ResearchRadar Label |
|------------|------------|---------------------|
| POSITIVE | > 0.75 | 🟢 **Optimistic** |
| POSITIVE | ≤ 0.75 | 🔵 **Neutral** |
| NEGATIVE | > 0.65 | 🔴 **Critical** |
| NEGATIVE | ≤ 0.65 | 🟡 **Cautious** |

This mapping reflects how academic papers express sentiment: a paper might be *cautiously negative* about current approaches (not outright critical) or *neutrally positive* about incremental results (not enthusiastically optimistic).

#### Three Aggregation Levels

| Level | Method | Output |
|-------|--------|--------|
| **Chunk** | Direct model inference on each chunk | `{chunk_id, label, score}` |
| **Section** | Group chunks by `section` metadata → weighted average of sentiment scores | `{paper, section, avg_score, dominant_label}` |
| **Paper** | Weighted majority vote across all chunks | `{paper, sentiment, confidence, breakdown}` |

#### Sentiment Trajectory

For any paper, the sentiment scores plotted across chunks in sequential order (`chunk_index`) reveal *where the tone shifts* — typically an optimistic introduction, cautious methodology, and critically-discussed results. This produces the "sentiment flow line" visualisation in Power BI.

---

### 5.6 Analysis Engine (Module 2)

Four independent sub-modules that produce structured output for both in-app visuals and Power BI export.

#### 🗺️ BERTopic — Unsupervised Topic Discovery

BERTopic discovers latent themes across the entire paper corpus without any pre-defined topic labels. Its internal pipeline:

```
All chunk texts
    │
    ▼
┌───────────────────────────────────┐
│ 1. EMBED                          │
│    SentenceTransformer             │
│    all-MiniLM-L6-v2               │
│    chunks → 384-dim vectors       │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ 2. REDUCE DIMENSIONS              │
│    UMAP (384-dim → 5-dim)         │
│    Preserves local + global       │
│    structure in lower space       │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ 3. CLUSTER                        │
│    HDBSCAN                        │
│    Density-based clustering       │
│    Automatically finds # clusters │
│    Marks outliers as Topic -1     │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ 4. EXTRACT TOPIC LABELS           │
│    c-TF-IDF (class-based TF-IDF) │
│    Finds most representative      │
│    keywords per cluster           │
│    e.g., "alignment, safety,      │
│          RLHF, constitutional"    │
└───────────────────────────────────┘
```

**Output:** A table of topics, each with an ID, top keywords, chunk count, and the paper contributing most chunks to that topic.

#### 🔑 KeyBERT — Semantic Keyword Extraction

Unlike traditional keyword extraction (TF-IDF, RAKE) that counts word frequency, KeyBERT finds keywords that are *semantically representative* of the document:

1. Embeds the full concatenated text of each paper
2. Generates candidate keywords/keyphrases (1-grams and 2-grams)
3. Embeds each candidate separately
4. Ranks candidates by cosine similarity to the document embedding

**Result:** The top-15 keywords per paper, ranked by a relevance score (0.0–1.0). A word like *"alignment"* in a safety paper scores high not because it appears often, but because it's semantically central to the paper's meaning.

#### 🌐 Paper-to-Paper Similarity

Computes an N×N similarity matrix across all indexed papers:

```
Paper A (all chunks concatenated) → embed → Vector A (384-dim)
Paper B (all chunks concatenated) → embed → Vector B (384-dim)

cosine_similarity(A, B) → 0.91  (highly similar)
```

The diagonal is always 1.0 (self-similarity). Off-diagonal cells reveal which papers are conceptually closest, enabling researchers to discover related work they didn't know to look for.

#### Section-to-Section Similarity Heatmap

For a selected pair of papers, computes a finer-grained similarity matrix:

```
         Paper B Sections →
         Background  Architecture  Evaluation  Conclusion
  Intro    0.84        0.42         0.31        0.73
  Method   0.38        0.91         0.55        0.29
  Results  0.27        0.48         0.88        0.62
  ↑ Paper A Sections
```

This reveals exactly *where* two papers align and diverge at the structural level.

---

### 5.7 Debate Engine (Module 4)

The most unique feature of ResearchRadar. Three debate modes, all grounded in RAG retrieval — no hallucinated arguments.

#### ⚔️ Mode A — AI Challenges the Paper's Claims

```
User selects a paper
    → System pulls pre-extracted claims (from ingestion)
        → For each claim, Claude acts as a rigorous peer reviewer
            → Identifies: assumptions, counter-evidence, alternative explanations
```

Claude is explicitly prompted to **not be agreeable** — it must push back technically and specifically. Output is a structured `{claim, challenge}` list.

#### 🛡️ Mode B — User Debates, AI Defends the Paper

```
User selects a paper + types an argument AGAINST it
    → System retrieves most relevant chunks from that paper
      (using user's argument as query, filtered to selected paper)
        → Claude defends the paper using ONLY retrieved evidence
            → Quotes specific passages from source chunks
```

**Key constraint:** The defence is grounded. Claude cannot invent supporting evidence — it can only cite what the paper actually says. This makes the debate intellectually honest.

#### 📜 Mode C — Paper vs Paper Debate

```
User selects Paper A, Paper B, and a debate topic
    → Retrieve top-4 chunks from Paper A relevant to topic
    → Retrieve top-4 chunks from Paper B relevant to topic
        → Turn 1: Claude argues from Paper A's perspective (Paper A's evidence)
        → Turn 2: Claude argues from Paper B's perspective (responds to Turn 1)
            → Multiple rounds possible
```

Displayed as a structured chat with paper names as speaker labels — a genuine academic debate powered by real evidence from both papers.

---

### 5.8 Power BI Dashboard Integration (Module 3)

Python cannot run live inside Power BI Desktop. The connection is via **exported CSVs on disk**:

```
Python Analysis Engine → Export 5 CSVs → data/exports/ → Power BI reads CSVs
```

The Streamlit UI has an **"Export to Power BI"** button that triggers all analysis modules and writes CSVs.

#### The 5 CSV Exports

| # | File | Contents | Powers Dashboard |
|---|------|----------|-----------------|
| 1 | `sentiment_analysis.csv` | paper, section, chunk_index, label, score, raw_label | Sentiment Deep Dive |
| 2 | `paper_similarity.csv` | paper_1, paper_2, similarity_score | Semantic Similarity |
| 3 | `section_similarity.csv` | paper1, paper1_section, paper2, paper2_section, score | Section Heatmaps |
| 4 | `topic_model.csv` | topic_id, keywords, chunk_count, top_paper | Topic & Keyword Analysis |
| 5 | `keyword_scores.csv` | paper_title, keyword, relevance_score | Keyword × Paper Heatmap |

#### The 5 Power BI Dashboard Pages

1. **Overview** — KPI cards (papers indexed, chunks stored, avg sentiment), chunks-per-paper bar chart, full paper index table with slicers by source/category/date
2. **Sentiment Deep Dive** — Donut chart (overall distribution), per-category sentiment bars, per-paper stacked bars, sentiment trajectory line chart
3. **Semantic Similarity** — Paper × Paper similarity heatmap, section-to-section drill-down heatmap
4. **Topic & Keyword Analysis** — BERTopic horizontal bar chart (top-10 topics), keyword × paper heatmap, word cloud
5. **Debate Prep View** — All extracted claims table (claim text, paper, section, sentiment score), filterable by paper and topic

#### Connecting Power BI Desktop

1. Open Power BI Desktop → **Get Data** → **Text/CSV**
2. Navigate to `data/exports/` and select each CSV
3. Power BI auto-detects column types and builds the data model
4. Dashboards auto-refresh when CSVs are updated via the Export button

---

### 5.9 LLM Handler — Claude Integration

**File:** `core/llm_handler.py`

#### System Prompt Design (Anti-Hallucination Contract)

The system prompt establishes a strict RAG contract:

```
You are a research paper analyst. You ONLY answer based on the provided context chunks.
If the context doesn't contain relevant information, say "I don't have enough
information in the indexed papers to answer this."
NEVER fabricate information. NEVER reference knowledge outside the provided chunks.
Always cite your sources using [Source: filename, Page X] format.
```

This is the single most important design decision for answer quality — it eliminates hallucination by contractually binding Claude to the retrieved chunks.

#### Prompt Structure

Every prompt sent to Claude follows a three-part structure:

```
┌────────────────────────────────┐
│ CONTEXT                        │
│ Retrieved chunks with metadata │
│ [Chunk 1: source, page, text]  │
│ [Chunk 2: source, page, text]  │
│ ...                            │
├────────────────────────────────┤
│ CONVERSATION HISTORY           │
│ Q1: ...  A1: ...               │
│ Q2: ...  A2: ...               │
│ (last 5 turns, answers         │
│  truncated to 500 chars)       │
├────────────────────────────────┤
│ CURRENT QUESTION               │
│ The user's actual query        │
└────────────────────────────────┘
```

#### Citation Format

Sources are injected per-chunk in the context block. Claude is instructed to cite them in `[Source: filename, Page X]` format. The LLM handler post-processes the response to deduplicate and format source citations at the bottom of the answer.

#### API Configuration

| Setting | Default | Configurable via |
|---------|---------|-----------------|
| Model | `claude-3-5-sonnet-20241022` | `CLAUDE_MODEL` in `.env` |
| API Key | — | `ANTHROPIC_API_KEY` in `.env` |
| Key loading | **At call-time** (not at import) | Enables hot-reload |

---

### 5.10 Conversation Memory

**File:** `memory/conversation.py`

A `deque(maxlen=5)` sliding window stores the last 5 (question, answer) pairs. Before each LLM call:

1. History is formatted as a numbered Q/A block
2. Long answers are truncated to **500 characters** to conserve tokens
3. The formatted history is injected into the prompt between CONTEXT and QUESTION

This enables coherent follow-up questions like *"What about safety?"* after *"Summarise the methodology"* without re-stating the topic.

---

## 📁 Project Structure

```
ResearchRadar/
│
├── app.py                          # Streamlit entry point — main chat interface
│
├── core/                           # All backend logic
│   ├── __init__.py                 # Package marker
│   ├── pdf_processor.py            # PDF → text → chunks (pdfplumber + LangChain)
│   ├── embedder.py                 # Text → 384-dim vectors (all-MiniLM-L6-v2)
│   ├── sentiment_analyzer.py       # Chunk/section/paper sentiment (DistilBERT SST-2)
│   ├── vector_store.py             # ChromaDB CRUD, metadata filtering, semantic search
│   ├── retriever.py                # Hybrid retrieval: semantic + BM25 + RRF fusion
│   ├── query_router.py             # Keyword-based precise/synthesis mode routing
│   ├── llm_handler.py              # Claude API prompt construction + response parsing
│   │
│   ├── ingestion/                  # Extended ingestion modules
│   │   └── __init__.py             # Claim extraction + metadata enrichment
│   │
│   ├── embeddings/                 # Embedding utilities
│   │   └── __init__.py             # Model management
│   │
│   ├── analysis/                   # Analysis Engine (Module 2)
│   │   ├── sentiment_multi.py      # Section-level aggregation + trajectory
│   │   ├── similarity.py           # Paper-to-paper + section-to-section similarity
│   │   ├── topics.py               # BERTopic integration
│   │   └── keywords.py             # KeyBERT integration
│   │
│   ├── debate/                     # Debate Engine (Module 4)
│   │   ├── challenger.py           # Mode A: AI challenges paper claims
│   │   ├── defender.py             # Mode B: AI defends, user attacks
│   │   └── versus.py              # Mode C: Paper vs Paper debate
│   │
│   ├── retrieval/                  # Extended retrieval modules
│   ├── storage/                    # Extended storage modules
│   └── llm/                        # Extended LLM modules
│
├── memory/
│   ├── __init__.py
│   └── conversation.py             # Sliding window memory (deque, last 5 turns)
│
├── pages/                          # Streamlit multi-page app
│   ├── 2_Analysis_Dashboard.py     # In-app analytics + Power BI export
│   ├── 3_Debate_Arena.py           # Three debate mode tabs
│   └── 4_Paper_Comparison.py       # Side-by-side similarity heatmaps
│
├── data/
│   ├── chroma_db/                  # Persistent vector database storage
│   ├── claims/                     # Extracted claims per paper
│   └── exports/                    # Power BI CSV exports (5 files)
│
├── docs/
│   ├── architecture.md             # System architecture documentation
│   ├── folder_guide.md             # Directory structure guide
│   └── workflow.md                 # Data flow documentation
│
├── .env.example                    # Template for environment variables
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
└── README.md                       # ← You are here
```

---

## ⚙️ Tech Stack

| Layer | Technology | Why This Choice |
|-------|-----------|----------------|
| **PDF Extraction** | pdfplumber ≥0.11.0 | Handles multi-column layouts and tables in research papers |
| **Text Chunking** | LangChain TextSplitter ≥0.2.0 | Recursive splitting preserves paragraph → sentence → word hierarchy |
| **Embeddings** | `all-MiniLM-L6-v2` via SentenceTransformers ≥2.7.0 | 384-dim vectors, runs locally (free), top-tier quality for size |
| **Vector Database** | ChromaDB ≥0.5.0 | Persistent local storage, cosine similarity, metadata filtering |
| **Sentiment** | DistilBERT SST-2 via Transformers ≥4.40.0 | Lightweight, accurate, runs locally with PyTorch |
| **Deep Learning** | PyTorch ≥2.1.0 | Runtime for DistilBERT + SentenceTransformers |
| **Keyword Search** | rank-bm25 ≥0.2.2 | BM25Okapi for sparse retrieval (hybrid search partner) |
| **Topic Modelling** | BERTopic ≥0.16.0 | Unsupervised topic discovery (UMAP + HDBSCAN + c-TF-IDF) |
| **Keyword Extraction** | KeyBERT ≥0.8.0 | Semantic keyword ranking using embedding similarity |
| **LLM / RAG** | Claude 3.5 Sonnet via Anthropic SDK ≥0.25.0 | Grounded Q&A, debate, claim analysis |
| **Dashboard Export** | pandas ≥2.0.0 → CSV | Structured data for Power BI consumption |
| **BI Visualisation** | Power BI Desktop | 5 interactive dashboard pages from CSV data |
| **UI Framework** | Streamlit ≥1.35.0 | Rapid web UI with dark theme, chat interface, multi-page |
| **In-App Charts** | Plotly ≥5.18.0 | Interactive heatmaps and charts within Streamlit |
| **Environment** | python-dotenv ≥1.0.0 | Secure .env file loading for API keys |
| **Numerics** | numpy ≥1.26.0 | Array operations for embeddings, scores, matrices |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **pip** (comes with Python)
- **Anthropic API key** ([get one here](https://console.anthropic.com/))
- **Power BI Desktop** (optional, for dashboard visualisation — Windows only)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ResearchRadar.git
cd ResearchRadar

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt
```

> **Note:** First install may take 5–10 minutes. SentenceTransformers, PyTorch, and BERTopic are large packages (~2GB total). The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first run (~90MB).

### Configuration

```bash
# 1. Copy the environment template
cp .env.example .env

# 2. Edit .env and add your Anthropic API key
```

Your `.env` file should contain:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional overrides
CLAUDE_MODEL=claude-3-5-sonnet-20241022
TOP_K_PRECISE=8
TOP_K_PER_PAPER=3
```

### Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with a dark-themed glassmorphism interface.

---

## 📘 Usage Guide

### 📄 Uploading Papers

1. Open the sidebar → **Upload PDF** section
2. Select one or more research paper PDFs
3. Provide metadata (title, source organisation, category, date)
4. Click **Process** — the pipeline runs: extract → chunk → embed → analyse sentiment → store
5. Papers appear in the sidebar with sentiment badges (🟢 Optimistic, 🔵 Neutral, 🟡 Cautious, 🔴 Critical)

### 💬 Asking Questions

Type any question in the chat input:

- **Single-paper queries** → *"What methodology does this paper use?"* → Precise mode (top-8 chunks)
- **Cross-paper queries** → *"Compare the safety approaches across all papers"* → Synthesis mode (top-3 per paper)
- **Follow-up questions** → *"What about the limitations?"* → Uses conversation memory for context

Every answer includes `[Source: filename, Page X]` citations.

### 📊 Exporting to Power BI

1. Navigate to **Page 2: Analysis Dashboard**
2. Click **"Export to Power BI"** — generates all 5 CSVs in `data/exports/`
3. Open Power BI Desktop → **Get Data** → **Text/CSV** → select files from `data/exports/`
4. Build or load dashboard pages using the pre-structured data

### ⚔️ Using the Debate Engine

Navigate to **Page 3: Debate Arena** and choose a mode:

| Tab | Mode | How to Use |
|-----|------|------------|
| **Challenge** | AI vs Paper | Select a paper → AI automatically critiques each extracted claim |
| **Defend** | User vs AI | Select a paper → Type your argument against it → AI defends with evidence |
| **Versus** | Paper vs Paper | Select two papers + a debate topic → Watch AI argue from each perspective |

---

## 🔄 Data Flow Diagram

```
Step 1  │  User uploads PDF + enters metadata
        ▼
Step 2  │  pdfplumber extracts text page-by-page
        ▼
Step 3  │  RecursiveCharacterTextSplitter → ~2000-char chunks (200 overlap)
        ▼
Step 4  │  Section detection (regex headings) + claim extraction
        ▼
Step 5  │  all-MiniLM-L6-v2 embeds all chunks → 384-dim vectors (batch)
        ▼
Step 6  │  DistilBERT SST-2 classifies sentiment per chunk → 4 labels
        ▼
Step 7  │  ChromaDB upserts: text + vector + sentiment + 11 metadata fields
        ▼
Step 8  │  User asks a question
        ▼
Step 9  │  Query router classifies → Precise or Synthesis mode
        ▼
Step 10 │  Embed question → 384-dim query vector
        ▼
Step 11 │  Semantic search: ChromaDB cosine similarity → top-20 candidates
        ▼
Step 12 │  BM25 keyword search: BM25Okapi across all chunk texts
        ▼
Step 13 │  RRF fusion (k=60) → merge both ranked lists → top-K results
        ▼
Step 14 │  Build RAG prompt: CONTEXT (chunks) + HISTORY (last 5) + QUESTION
        ▼
Step 15 │  Claude API → generates grounded answer with citations
        ▼
Step 16 │  Display in Streamlit chat UI with source links
```

---

## 🗺️ Future Roadmap

| Feature | Description | Status |
|---------|-------------|--------|
| 🌐 Web search integration | Augment RAG with live web results for recent papers | Planned |
| 🤖 Multi-model support | Switch between Claude, GPT-4, Gemini at runtime | Planned |
| 📈 Auto-generated research summaries | One-click executive summary of any paper | Planned |
| 🔗 Citation graph visualisation | Map citation relationships between indexed papers | Planned |
| 🗂️ Paper tagging & collections | Organise papers into custom research collections | Planned |
| 📱 Responsive mobile layout | Optimised Streamlit layout for tablet/mobile | Planned |
| 🔄 Incremental re-analysis | Re-run analysis on new papers without full corpus rebuild | Planned |

---

<p align="center">
  <b>Built with 🧠 NLP, 🔍 RAG, and ⚔️ Debate</b><br>
  <sub>ResearchRadar — Because research papers deserve more than Ctrl+F</sub>
</p>
