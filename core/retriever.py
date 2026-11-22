"""
core/retriever.py — Hybrid Retrieval Engine with Neural Cross-Encoder Reranking.

Inspired by: https://github.com/abh2050/hybrid_rag_with_bm25_reranker
Architecture:
  1. Dense Semantic Vector Search: ChromaDB with all-MiniLM-L6-v2 embeddings.
  2. Sparse Keyword Search: In-memory BM25Okapi with token caching.
  3. Candidate Pooling & Source Tracking: Merges candidates from both streams.
  4. Cross-Encoder Reranking: Neural cross-attention via ms-marco-TinyBERT-L-2-v2
     (with fallback to weighted combination: 0.4*BM25 + 0.6*Vector).
"""

import time
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

from core.config import (
    RRF_K,
    ENABLE_HYBRID_SEARCH,
    ENABLE_RERANKING,
    TOP_K_RETRIEVAL,
    RERANK_TOP_N
)
from core.embedder import embed_query
from core.logger import get_logger, timed
from core.metrics import metrics
from core.vector_store import get_all_chunks, semantic_query
from core.reranker import rerank_candidates

log = get_logger("retriever")

# ─── BM25 Cache ──────────────────────────────────────────────
_bm25_cache: dict[str, dict] = {}


def clear_bm25_cache() -> None:
    """Clears the in-memory BM25 index cache (call after corpus changes)."""
    _bm25_cache.clear()
    log.debug("BM25 cache cleared")


# ─── Hybrid Search with Neural Cross-Encoder Reranking ────────
@timed
def hybrid_search(
    query: str,
    top_k: int = RERANK_TOP_N,
    paper_filter: str | None = None,
) -> list[dict]:
    """
    Executes advanced Hybrid Retrieval with Cross-Encoder Reranking.
    (Following the architecture of abh2050/hybrid_rag_with_bm25_reranker)

    Steps:
      1. Dense vector search in ChromaDB (cosine similarity).
      2. Sparse keyword search with BM25Okapi (exact lexical matching).
      3. Candidate deduplication & provenance tracking (Vector, BM25, or Both).
      4. Neural Cross-Encoder reranking (ms-marco-TinyBERT-L-2-v2).
      5. Top-k selection sorted by cross-encoder relevance.
    """
    start = time.perf_counter()
    n_retrieve = max(top_k * 3, TOP_K_RETRIEVAL)

    candidates_by_id: Dict[str, Dict[str, Any]] = {}

    # ─── 1. Vector Search (ChromaDB) ───────────────────────────
    query_emb = embed_query(query)
    semantic_res = semantic_query(
        query_embedding=query_emb.tolist(),
        n_results=n_retrieve,
        paper_filter=paper_filter,
    )
    vector_chunks = _parse_semantic_results(semantic_res)

    for rank, chunk in enumerate(vector_chunks):
        cid = chunk["id"]
        # Similarity score: 1.0 / (1.0 + distance)
        dist = chunk.get("distance", 0.5)
        sim = float(1.0 / (1.0 + max(0.0, dist)))

        chunk_data = dict(chunk)
        chunk_data["vector_score"] = sim
        chunk_data["vector_rank"] = rank + 1
        chunk_data["bm25_score"] = 0.0
        chunk_data["sources"] = ["Vector"]
        candidates_by_id[cid] = chunk_data

    # ─── 2. Keyword Search (BM25) ─────────────────────────────
    if ENABLE_HYBRID_SEARCH:
        bm25_results = _bm25_search_scored(query, n_retrieve, paper_filter)
        for rank, score, chunk in bm25_results:
            cid = chunk["id"]
            if cid in candidates_by_id:
                candidates_by_id[cid]["bm25_score"] = score
                candidates_by_id[cid]["bm25_rank"] = rank + 1
                if "BM25" not in candidates_by_id[cid]["sources"]:
                    candidates_by_id[cid]["sources"].append("BM25")
            else:
                chunk_data = dict(chunk)
                chunk_data["vector_score"] = 0.0
                chunk_data["bm25_score"] = score
                chunk_data["bm25_rank"] = rank + 1
                chunk_data["sources"] = ["BM25"]
                candidates_by_id[cid] = chunk_data

    combined_candidates = list(candidates_by_id.values())

    if not combined_candidates:
        log.warning("Hybrid search returned 0 candidates for query: %s", query[:60])
        return []

    # ─── 3. Neural Cross-Encoder Reranking ───────────────────
    reranked = rerank_candidates(query, combined_candidates, top_n=top_k)

    elapsed = (time.perf_counter() - start) * 1000
    metrics.record("hybrid_search", elapsed, chunks_returned=len(reranked))
    log.info("Hybrid search completed in %.1fms: %d candidates -> %d reranked chunks",
             elapsed, len(combined_candidates), len(reranked))
    return reranked


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


def _bm25_search_scored(
    query: str,
    max_results: int,
    paper_filter: str | None,
) -> list[tuple[int, float, dict]]:
    """
    Runs BM25 keyword search over the corpus.
    Returns:
        List of (rank, bm25_score, chunk_dict) tuples for chunks with positive scores.
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
        corpus = [re.findall(r"\b\w+\b", c["text"].lower()) for c in valid]
        try:
            bm25 = BM25Okapi(corpus)
            _bm25_cache[cache_key] = {"model": bm25, "count": len(valid)}
        except Exception as exc:
            log.warning("BM25 index build failed: %s", exc)
            return []

    tokens = re.findall(r"\b\w+\b", query.lower())
    if not tokens:
        return []

    try:
        scores = bm25.get_scores(tokens)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Max score for normalization
        max_score = float(np.max(scores)) if len(scores) > 0 and np.max(scores) > 0 else 1.0

        ranked = []
        for rank, idx in enumerate(sorted_indices):
            raw_s = float(scores[idx])
            if raw_s > 0.0:
                norm_s = round(raw_s / max_score, 4)
                ranked.append((rank + 1, norm_s, valid[idx]))
                if len(ranked) >= max_results:
                    break
        return ranked
    except Exception as exc:
        log.warning("BM25 scoring failed: %s", exc)
        return []


def _bm25_search(
    query: str,
    max_results: int,
    paper_filter: str | None,
) -> list[tuple[int, dict]]:
    """Backward compatibility wrapper returning (rank, chunk_dict)."""
    scored = _bm25_search_scored(query, max_results, paper_filter)
    return [(rank, chunk) for rank, _, chunk in scored]
