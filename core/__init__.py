"""
core — ResearchRadar's processing engine

Organized into focused sub-packages:
    ingestion/   — PDF extraction, chunking, metadata tagging, claim extraction
    embeddings/  — Text-to-vector conversion (SentenceTransformers)
    storage/     — ChromaDB vector database operations
    retrieval/   — Hybrid search (semantic + BM25 + RRF fusion)
    analysis/    — NLP analysis (sentiment, similarity, topics, keywords)
    debate/      — AI-powered debate engine
    llm/         — Claude API prompt construction and response handling
"""
