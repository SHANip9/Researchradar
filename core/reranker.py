"""
core/reranker.py — Neural Cross-Encoder Reranker for Hybrid RAG.

Inspired by: https://github.com/abh2050/hybrid_rag_with_bm25_reranker
Combines dense semantic vector retrieval (ChromaDB) and sparse keyword retrieval (BM25)
with a state-of-the-art cross-encoder (ms-marco-TinyBERT-L-2-v2) for precision reranking.

If neural reranking is unavailable, gracefully falls back to a weighted combination:
    combined_score = (0.4 * bm25_score) + (0.6 * vector_score)
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple

from core.config import (
    CROSS_ENCODER_MODEL_NAME,
    ENABLE_RERANKING,
    RERANK_TOP_N
)
from core.logger import get_logger, timed
from core.metrics import metrics

log = get_logger("reranker")

_cross_encoder_instance = None
_cross_encoder_load_attempted = False


def get_cross_encoder():
    """
    Lazy singleton loader for the sentence-transformers CrossEncoder.
    Uses 'cross-encoder/ms-marco-TinyBERT-L-2-v2' (fast, lightweight, highly accurate on CPU).
    """
    global _cross_encoder_instance, _cross_encoder_load_attempted
    if _cross_encoder_instance is not None:
        return _cross_encoder_instance

    if _cross_encoder_load_attempted:
        return None

    _cross_encoder_load_attempted = True
    try:
        from sentence_transformers import CrossEncoder
        log.info("Loading CrossEncoder model: %s", CROSS_ENCODER_MODEL_NAME)
        start = time.perf_counter()
        _cross_encoder_instance = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        elapsed = (time.perf_counter() - start) * 1000
        log.info("CrossEncoder loaded successfully in %.1fms", elapsed)
        return _cross_encoder_instance
    except Exception as exc:
        log.warning("Failed to load CrossEncoder (%s): %s. Falling back to weighted score combination.",
                    CROSS_ENCODER_MODEL_NAME, exc)
        return None


@timed
def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = RERANK_TOP_N
) -> List[Dict[str, Any]]:
    """
    Reranks candidate chunks using CrossEncoder neural relevance scoring.
    
    Args:
        query: User search or agent prompt.
        candidates: List of chunk dicts from BM25 and ChromaDB vector search.
        top_n: Number of top chunks to return.
        
    Returns:
        List of reranked chunk dicts sorted by cross_score (highest relevance first).
    """
    if not candidates:
        return []

    start = time.perf_counter()
    ce = get_cross_encoder() if ENABLE_RERANKING else None

    if ce is not None:
        try:
            # Build query-document pairs for cross-attention
            pairs = [[query, c.get("text", "")[:1200]] for c in candidates]
            scores = ce.predict(pairs)

            reranked = []
            for idx, c in enumerate(candidates):
                c_copy = dict(c)
                score_val = float(scores[idx])
                c_copy["cross_score"] = score_val
                # Sigmoid normalization for human-readable confidence
                c_copy["confidence"] = round(1.0 / (1.0 + np.exp(-score_val)), 4)
                reranked.append(c_copy)

            # Sort descending by cross_score
            reranked.sort(key=lambda x: x["cross_score"], reverse=True)
            elapsed = (time.perf_counter() - start) * 1000
            metrics.record("cross_encoder_rerank", elapsed, candidates_scored=len(candidates))
            log.info("Cross-Encoder reranked %d candidates in %.1fms (top score=%.3f)",
                     len(candidates), elapsed, reranked[0]["cross_score"] if reranked else 0.0)
            return reranked[:top_n]
        except Exception as exc:
            log.warning("Cross-encoder prediction failed: %s. Falling back to weighted score combination.", exc)

    # Fallback to weighted score combination: (0.4 * bm25_score) + (0.6 * vector_score)
    # (Matching the exact fallback architecture of abh2050/hybrid_rag_with_bm25_reranker)
    reranked = []
    for c in candidates:
        c_copy = dict(c)
        bm25_s = float(c.get("bm25_score", 0.0))
        vec_s = float(c.get("vector_score", 0.5))
        combined_s = (0.4 * bm25_s) + (0.6 * vec_s)
        c_copy["cross_score"] = combined_s
        c_copy["confidence"] = round(vec_s, 4)
        reranked.append(c_copy)

    reranked.sort(key=lambda x: x["cross_score"], reverse=True)
    return reranked[:top_n]


def score_sentences(query: str, sentences: List[str]) -> List[Tuple[float, str]]:
    """
    Scores individual sentences against the query using CrossEncoder.
    Used by generative RAG synthesis to pinpoint the exact empirical answers
    and avoid repetitive static snippets.
    
    Returns:
        List of (score, sentence) tuples sorted descending by relevance score.
    """
    valid = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not valid:
        return []

    ce = get_cross_encoder()
    if ce is not None:
        try:
            pairs = [[query, s[:400]] for s in valid]
            scores = ce.predict(pairs)
            scored = [(float(scores[i]), valid[i]) for i in range(len(valid))]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
        except Exception as exc:
            log.debug("Sentence scoring via CrossEncoder failed: %s", exc)

    # Lexical overlap fallback
    q_words = set(query.lower().split())
    scored = []
    for s in valid:
        s_words = set(s.lower().split())
        overlap = len(q_words.intersection(s_words))
        scored.append((float(overlap), s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
