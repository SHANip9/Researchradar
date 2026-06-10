"""
retriever.py — Smart Hybrid Search: Semantic + BM25 + RRF Score Fusion

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY HYBRID SEARCH?
    Two search strategies — each catches what the other misses:

    1. SEMANTIC SEARCH (ChromaDB + vectors):
       Understands MEANING. Good for conceptual questions.
       "What method improves efficiency?" → finds chunks about "speed",
       "FLOPs", "latency" even if none contain the word "efficiency".

    2. KEYWORD SEARCH (BM25 — Best Match 25):
       Finds EXACT WORDS. Good for specific terms, numbers, names.
       "What is the reported F1 score?" → finds "F1: 0.94" reliably.
       Semantic search might rank this poorly because "F1 score" has
       no deep semantic neighborhood in embedding space.

    HYBRID (Both combined):
       A chunk scoring high in BOTH signals is almost certainly relevant.
       Complementary weaknesses cancel out.

RRF — RECIPROCAL RANK FUSION:
    The standard algorithm to merge two ranked lists into one:

    score(chunk) = 1/(k + rank_in_semantic_list) + 1/(k + rank_in_bm25_list)

    k=60 is a smoothing constant (prevents rank-1 from dominating completely).
    A chunk ranked #1 in both lists scores highest.
    A chunk ranked #20 in both lists scores very low.
    A chunk only in one list gets partial credit.

TWO RETRIEVAL MODES:
    - PRECISE:    Top-8 chunks globally (for single-question answers)
    - SYNTHESIS:  Top-3 chunks PER PAPER (for cross-paper comparison)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from rank_bm25 import BM25Okapi
from core.storage.vector_store import semantic_search, get_all_chunks, get_papers_list
from core.embeddings.embedder import embed_query
import numpy as np

# ─── Retrieval Configuration ──────────────────────────────────
TOP_K_PRECISE    = 8    # Number of chunks for single-question answers
TOP_K_PER_PAPER  = 3    # Chunks per paper for cross-paper synthesis
SEMANTIC_FETCH   = 20   # Over-fetch before RRF (more candidates = better fusion)
RRF_K            = 60   # RRF smoothing constant (standard value)


def build_bm25_index(all_chunks: list):
    """
    Builds a BM25 keyword search index over all stored chunks.

    HOW BM25 WORKS (simplified):
        BM25 scores a document by counting how often query words appear in it,
        but it penalizes:
        1. Very long documents (they have more words by chance, not meaning)
        2. Words that appear in almost every document (like "the", "is")
           → These are less informative than rare words like "transformer"

        This is the same algorithm at the core of Elasticsearch.

    Args:
        all_chunks (list): All chunk dicts from get_all_chunks()

    Returns:
        Tuple: (bm25_index, all_chunks)
        We return all_chunks because BM25 returns position indices,
        and we need the original list to map index → chunk content.
    """
    if not all_chunks:
        return None, []

    # Tokenize: lowercase + split by whitespace
    # BM25 is word-level, not character-level (unlike embeddings)
    tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_corpus)

    return bm25_index, all_chunks


def _rrf_fusion(semantic_results: list, bm25_all_chunks: list,
                bm25_scores: np.ndarray, k: int = RRF_K) -> list:
    """
    Merges semantic and BM25 results using Reciprocal Rank Fusion.

    ALGORITHM:
        For each chunk, compute: 1/(k + rank_semantic) + 1/(k + rank_bm25)
        Chunks not found by a method get rank = len(corpus) + 1 (worst rank).
        Sort all chunks by combined RRF score (descending).

    Args:
        semantic_results (list):  Ordered list from semantic_search() (best first)
        bm25_all_chunks (list):   All chunks in BM25 corpus order
        bm25_scores (ndarray):    BM25 scores for each chunk in corpus order
        k (int):                  RRF smoothing constant

    Returns:
        list of chunk dicts ordered by RRF score (best match first)
    """
    rrf_scores = {}  # chunk_text → cumulative RRF score

    # ─── Step 1: Score semantic results by their rank ────────────────────
    for rank, chunk in enumerate(semantic_results):
        key = chunk["text"]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)

    # ─── Step 2: Score BM25 results by their rank ───────────────────────
    # Sort chunks by BM25 score descending to get BM25 ranking
    bm25_ranked_indices = np.argsort(bm25_scores)[::-1]  # highest score first

    for rank, idx in enumerate(bm25_ranked_indices):
        key = bm25_all_chunks[idx]["text"]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)

    # ─── Step 3: Combine into a lookup dict for final assembly ───────────
    # Build a map from chunk text → full chunk data for output
    chunk_lookup = {}
    for chunk in semantic_results:
        chunk_lookup[chunk["text"]] = chunk
    for chunk in bm25_all_chunks:
        chunk_lookup[chunk["text"]] = chunk_lookup.get(chunk["text"], chunk)

    # Sort by RRF score (descending) and return full chunk dicts
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    return [
        {**chunk_lookup[key], "rrf_score": round(rrf_scores[key], 6)}
        for key in sorted_keys
        if key in chunk_lookup
    ]


def retrieve_precise(question: str) -> list:
    """
    Precise mode: Retrieves the top-8 most relevant chunks globally (across all papers).

    Used for: specific questions, single-paper questions, factual lookups.
    Example: "What training method did paper 2 use?"

    Pipeline:
        question → embed → semantic search (top-20) → BM25 score all chunks
        → RRF fusion → top-8 results

    Args:
        question (str): The user's question.

    Returns:
        List of up to TOP_K_PRECISE chunk dicts, best match first.
    """
    # Step 1: Embed the question into a vector
    query_vector = embed_query(question)

    # Step 2: Semantic search — fetch more than needed (we'll trim after fusion)
    semantic_results = semantic_search(query_vector, n_results=SEMANTIC_FETCH)

    if not semantic_results:
        return []

    # Step 3: Build BM25 index from all stored chunks
    all_chunks = get_all_chunks()
    bm25_index, all_chunks = build_bm25_index(all_chunks)

    if bm25_index is None:
        # Fallback: return semantic results alone if BM25 fails
        return semantic_results[:TOP_K_PRECISE]

    # Step 4: BM25 keyword scoring
    # Tokenize the question the same way we tokenized chunks
    tokenized_query = question.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)

    # Step 5: RRF fusion → trim to top-k
    fused = _rrf_fusion(semantic_results, all_chunks, bm25_scores)

    return fused[:TOP_K_PRECISE]


def retrieve_synthesis(question: str) -> list:
    """
    Synthesis mode: Retrieves top-3 chunks PER PAPER, then combines.

    Used for: cross-paper comparison, "which paper agrees/disagrees" questions.
    Example: "Compare how all papers approach the attention mechanism."

    Why per-paper retrieval?
        If we did global top-8, one dominant paper might fill all 8 slots.
        Per-paper ensures every uploaded paper contributes to the answer,
        giving the LLM equal representation from each source.

    Args:
        question (str): The user's cross-paper question.

    Returns:
        List of chunks: top-3 from paper1 + top-3 from paper2 + ...
        Each chunk includes its source paper for the LLM prompt structure.
    """
    papers = get_papers_list()

    if not papers:
        return []

    query_vector = embed_query(question)
    tokenized_query = question.lower().split()

    all_chunks_global = get_all_chunks()
    bm25_index, all_chunks_global = build_bm25_index(all_chunks_global)
    bm25_scores_global = (
        bm25_index.get_scores(tokenized_query) if bm25_index else None
    )

    synthesis_results = []

    for paper_name in papers:
        # Semantic search filtered to this paper only
        paper_semantic = semantic_search(
            query_vector, n_results=SEMANTIC_FETCH, paper_filter=paper_name
        )

        if not paper_semantic:
            continue

        # Filter BM25 corpus and scores to only this paper's chunks
        paper_chunks = [c for c in all_chunks_global if c["source"] == paper_name]

        if bm25_index and paper_chunks:
            # Get the global indices for this paper's chunks
            paper_indices = [
                i for i, c in enumerate(all_chunks_global) if c["source"] == paper_name
            ]
            # Extract only this paper's BM25 scores (aligned with paper_chunks)
            paper_bm25_scores = np.array([bm25_scores_global[i] for i in paper_indices])

            # Run RRF on just this paper's data (no wasteful full-corpus fusion)
            fused = _rrf_fusion(paper_semantic, paper_chunks, paper_bm25_scores)
        else:
            fused = paper_semantic

        synthesis_results.extend(fused[:TOP_K_PER_PAPER])

    return synthesis_results
