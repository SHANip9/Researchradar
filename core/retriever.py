"""
core/retriever.py — Hybrid Retrieval Engine (Semantic + BM25 + RRF).

Implements the query pipeline's core retrieval step:
  1. Dense semantic search via ChromaDB cosine similarity.
  2. Sparse keyword search via BM25 Okapi.
  3. Reciprocal Rank Fusion (RRF) to merge both ranked lists.
"""

import time

import numpy as np
from rank_bm25 import BM25Okapi

from core.config import RRF_K
from core.embedder import embed_query
from core.logger import get_logger, timed
from core.metrics import metrics
from core.vector_store import get_all_chunks, semantic_query

log = get_logger("retriever")

# ─── BM25 Cache ──────────────────────────────────────────────
_bm25_cache: dict[str, dict] = {}


def clear_bm25_cache() -> None:
    """Clears the in-memory BM25 index cache (call after corpus changes)."""
    _bm25_cache.clear()
    log.debug("BM25 cache cleared")


# ─── Hybrid Search ────────────────────────────────────────────
@timed
def hybrid_search(
    query: str,
    top_k: int = 6,
    paper_filter: str | None = None,
) -> list[dict]:
    """Executes hybrid retrieval combining dense and sparse signals.

    Algorithm:
        1. Embed the query and run cosine-similarity search in ChromaDB.
        2. Score all candidate chunks with BM25 for exact-term matching.
        3. Fuse both ranked lists using RRF (k=60).
        4. Return the top-k chunks by fused score.

    Args:
        query: The user's search query.
        top_k: Number of chunks to return.
        paper_filter: Restrict results to a specific paper.

    Returns:
        A list of chunk dicts sorted by relevance (highest first).
    """
    start = time.perf_counter()

    # 1. Semantic Search
    query_emb = embed_query(query)
    n_candidates = max(top_k * 3, 1)

    semantic_res = semantic_query(
        query_embedding=query_emb.tolist(),
        n_results=n_candidates,
        paper_filter=paper_filter,
    )

    semantic_chunks = _parse_semantic_results(semantic_res)

    # 2. BM25 Keyword Search (cached)
    bm25_ranked = _bm25_search(query, top_k * 3, paper_filter)

    # 3. Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion(semantic_chunks, bm25_ranked, top_k)

    elapsed = (time.perf_counter() - start) * 1000
    metrics.record("hybrid_search", elapsed, chunks_returned=len(fused))
    return fused


# ─── Internal Helpers ─────────────────────────────────────────
def _parse_semantic_results(result: dict) -> list[dict]:
    """Converts raw ChromaDB query results into a flat chunk list."""
    chunks: list[dict] = []
    docs_group = result.get("documents") or [[]]
    ids_group = result.get("ids") or [[]]
    metas_group = result.get("metadatas") or [[]]
    dists_group = result.get("distances") or [[]]

    if not docs_group or not docs_group[0]:
        return chunks

    docs = docs_group[0]
    ids = ids_group[0] if ids_group else []
    metas = metas_group[0] if metas_group else []
    dists = dists_group[0] if dists_group else []

    for i in range(len(docs)):
        m = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        dist = dists[i] if i < len(dists) and dists[i] is not None else 0.5
        cid = ids[i] if i < len(ids) and ids[i] else f"sem_{i}"
        chunks.append({
            "id": cid,
            "text": docs[i],
            "paper_name": m.get("paper_name", "unknown"),
            "paper_title": m.get("paper_title", "unknown"),
            "page": m.get("page", 1),
            "section": m.get("section", "Unknown"),
            "distance": dist,
        })
    return chunks


def _bm25_search(
    query: str,
    max_results: int,
    paper_filter: str | None,
) -> list[tuple[int, dict]]:
    """Runs BM25 keyword search over the corpus (with caching).

    Returns:
        A list of ``(rank, chunk_dict)`` tuples ordered by BM25 score.
    """
    cache_key = paper_filter or "__all__"
    all_chunks = get_all_chunks(paper_name=paper_filter)
    valid = [c for c in all_chunks if c.get("text", "").strip()]
    if not valid:
        return []

    # Build or reuse cached BM25 index
    entry = _bm25_cache.get(cache_key)
    if entry and entry.get("count") == len(valid):
        bm25 = entry["model"]
    else:
        corpus = [c["text"].lower().split() for c in valid]
        try:
            bm25 = BM25Okapi(corpus)
            _bm25_cache[cache_key] = {"model": bm25, "count": len(valid)}
        except Exception as exc:
            log.warning("BM25 index build failed: %s", exc)
            return []

    tokens = [w for w in query.lower().split() if w]
    if not tokens:
        return []

    try:
        scores = bm25.get_scores(tokens)
        sorted_indices = np.argsort(scores)[::-1]
        # Only keep chunks with positive lexical overlap
        ranked = [
            (rank, valid[idx])
            for rank, idx in enumerate(sorted_indices)
            if scores[idx] > 0.0
        ]
        return ranked[:max_results]
    except Exception as exc:
        log.warning("BM25 scoring failed: %s", exc)
        return []


def _reciprocal_rank_fusion(
    semantic_chunks: list[dict],
    bm25_ranked: list[tuple[int, dict]],
    top_k: int,
) -> list[dict]:
    """Merges semantic and BM25 ranked lists using RRF.

    RRF score for each document:
        ``score = 1/(k + semantic_rank) + 1/(k + bm25_rank)``

    Args:
        semantic_chunks: Chunks from dense retrieval (ordered by similarity).
        bm25_ranked: ``(rank, chunk)`` tuples from BM25.
        top_k: Number of results to return.

    Returns:
        Top-k chunks sorted by fused RRF score.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(semantic_chunks):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in bm25_ranked:
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = {
                "id": cid,
                "text": chunk["text"],
                "paper_name": chunk["paper_name"],
                "paper_title": chunk["paper_title"],
                "page": chunk["page"],
                "section": chunk["section"],
                "distance": 0.5,
            }

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids[:top_k]]

