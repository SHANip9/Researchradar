"""
core/engine.py — Unified Façade for Research RAG and Multi-Agent Orchestration.

This module acts as the backward-compatible entry point, delegating to specialized
modular sub-components (pdf_processor, embedder, vector_store, retriever, llm_handler).
"""

from core.config import (
    PROJECT_ROOT,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    LLM_MAX_TOKENS,
    RRF_K
)
from core.logger import get_logger, timed
from core.metrics import metrics
from core.embedder import get_embed_model, embed_texts, embed_query
from core.pdf_processor import (
    KNOWN_SECTIONS,
    extract_pdf,
    detect_section as detect_section_for_page,
    chunk_document
)
from core.vector_store import (
    get_collection,
    get_all_chunks,
    get_indexed_papers,
    delete_by_paper as delete_paper,
    clear_all as clear_database,
    upsert_chunks
)
from core.retriever import (
    hybrid_search,
    clear_bm25_cache
)
from core.llm_handler import generate_llm_response

log = get_logger("engine")


@timed
def ingest_paper(pdf_path: str, paper_name: str, metadata: dict | None = None) -> int:
    """Full ingestion pipeline: PDF extraction → chunking → embedding → ChromaDB."""
    pages = extract_pdf(pdf_path)
    if not pages:
        raise ValueError(f"Could not extract readable text from '{paper_name}'.")

    chunks = chunk_document(pages, paper_name, metadata)
    if not chunks:
        raise ValueError(f"No content chunks generated for '{paper_name}'.")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    metadatas = [{
        "paper_name": c["paper_name"],
        "paper_title": c["paper_title"],
        "author_org": c["author_org"],
        "category": c["category"],
        "page": c["page"],
        "section": c["section"],
    } for c in chunks]

    upsert_chunks(
        ids=[c["id"] for c in chunks],
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    clear_bm25_cache()
    log.info("Successfully ingested paper '%s' (%d chunks)", paper_name, len(chunks))
    return len(chunks)
