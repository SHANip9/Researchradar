"""
core/vector_store.py — ChromaDB Vector Storage Layer.

Manages the persistent vector database: initialisation, upserts, queries,
and deletions.  All ChromaDB access goes through this module so that the
rest of the codebase never touches database internals directly.
"""

import os

import chromadb
from chromadb.config import Settings

from core.config import CHROMA_DB_PATH, COLLECTION_NAME
from core.logger import get_logger, timed
from core.metrics import metrics

log = get_logger("vector_store")

from typing import Any

# ─── Singletons ──────────────────────────────────────────────
_client: Any = None
_collection: Any = None


def get_collection():
    """Initialises (once) and returns the persistent ChromaDB collection.

    Creates the data directory and collection on first call.
    Subsequent calls return the cached reference.
    """
    global _client, _collection
    if _collection is None:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "ChromaDB collection '%s' ready (%d vectors)",
            COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


# ─── Write Operations ─────────────────────────────────────────
@timed
def upsert_chunks(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
) -> int:
    """Upserts a batch of chunks into the vector store.

    Args:
        ids: Unique chunk identifiers.
        documents: Raw text of each chunk.
        metadatas: Per-chunk metadata dicts.
        embeddings: Pre-computed embedding vectors.

    Returns:
        The number of chunks upserted.
    """
    collection = get_collection()
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    count = len(ids)
    metrics.record("vector_upsert", 0, count=count)
    log.info("Upserted %d chunks into ChromaDB", count)
    return count


def delete_by_paper(paper_name: str) -> None:
    """Deletes all chunks belonging to a specific paper.

    Args:
        paper_name: The ``paper_name`` metadata value to match.
    """
    collection = get_collection()
    collection.delete(where={"paper_name": paper_name})
    log.info("Deleted all chunks for paper: %s", paper_name)


def clear_all() -> None:
    """Removes every document from the collection."""
    global _collection
    collection = get_collection()
    try:
        res = collection.get()
        all_ids = res.get("ids") if res else None
        if all_ids and len(all_ids) > 0:
            collection.delete(ids=all_ids)
            log.info("Cleared %d chunks from ChromaDB", len(all_ids))
    except Exception:
        if _client is not None:
            try:
                _client.delete_collection(COLLECTION_NAME)
                _collection = None
                get_collection()
                log.info("Recreated ChromaDB collection after clear")
            except Exception as exc:
                log.error("Failed to recreate collection: %s", exc)


# ─── Read Operations ──────────────────────────────────────────
def get_all_chunks(paper_name: str | None = None) -> list[dict]:
    """Retrieves chunks from ChromaDB, optionally filtered by paper.

    Args:
        paper_name: If provided, only chunks from this paper are returned.

    Returns:
        A list of chunk dicts with keys: ``id``, ``text``, ``paper_name``,
        ``paper_title``, ``author_org``, ``category``, ``page``,
        ``section``, ``embedding``.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    kwargs: dict = {"include": ["documents", "metadatas", "embeddings"]}
    if paper_name:
        kwargs["where"] = {"paper_name": paper_name}

    res = collection.get(**kwargs)
    docs = res.get("documents")
    if docs is None:
        return []

    raw_metas = res.get("metadatas")
    metas = raw_metas if raw_metas is not None else []
    raw_embs = res.get("embeddings")
    embeddings = raw_embs if raw_embs is not None else []
    raw_ids = res.get("ids")
    ids = raw_ids if raw_ids is not None else []

    chunks: list[dict] = []
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        emb = embeddings[i] if i < len(embeddings) and embeddings[i] is not None else None
        chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"
        chunks.append({
            "id": chunk_id,
            "text": docs[i],
            "paper_name": meta.get("paper_name", "unknown"),
            "paper_title": meta.get("paper_title", "unknown"),
            "author_org": meta.get("author_org", ""),
            "category": meta.get("category", ""),
            "page": meta.get("page", 1),
            "section": meta.get("section", "Unknown"),
            "embedding": emb,
        })
    return chunks


def get_indexed_papers() -> list[str]:
    """Returns a sorted list of all indexed paper filenames."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    res = collection.get(include=["metadatas"])
    metas = res.get("metadatas")
    if metas is None:
        return []
    return sorted({
        m.get("paper_name", "unknown")
        for m in metas
        if m and isinstance(m, dict)
    })


def semantic_query(
    query_embedding: list[float],
    n_results: int,
    paper_filter: str | None = None,
) -> dict:
    """Performs a cosine-similarity search against the vector store.

    Args:
        query_embedding: The query vector.
        n_results: Maximum number of results.
        paper_filter: Optional paper name to restrict results.

    Returns:
        Raw ChromaDB query result dict.
    """
    collection = get_collection()
    query_args: dict = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, max(collection.count(), 1)),
        "include": ["documents", "metadatas", "distances"],
    }
    if paper_filter:
        query_args["where"] = {"paper_name": paper_filter}

    try:
        return collection.query(**query_args)
    except Exception:
        # Fallback with reduced n_results on collection size mismatch
        query_args["n_results"] = min(n_results, collection.count())
        try:
            return collection.query(**query_args)
        except Exception:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

