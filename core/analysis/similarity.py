"""similarity.py — Computes cosine similarity matrices at paper and section level."""

import numpy as np
from core.storage.vector_store import get_chunks_with_embeddings, get_papers_list


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_paper_similarity_matrix() -> list[dict]:
    """
    Computes a pairwise similarity score between all indexed papers.
    Uses the mean chunk embedding vector to represent each paper.

    Returns:
        list of dicts: [{"paper_1": str, "paper_2": str, "similarity_score": float}]
    """
    papers = get_papers_list()
    if len(papers) < 2:
        # If there's 0 or 1 paper, similarity is trivial or empty
        results = []
        for p in papers:
            results.append({
                "paper_1":          p,
                "paper_2":          p,
                "similarity_score": 1.0
            })
        return results

    # Get mean embedding for each paper
    paper_embeddings = {}
    for paper in papers:
        chunks = get_chunks_with_embeddings(paper)
        if not chunks:
            continue
        embeddings = [np.array(c["embedding"]) for c in chunks]
        paper_embeddings[paper] = np.mean(embeddings, axis=0)

    # Compute pairwise similarities
    similarity_data = []
    for i, p1 in enumerate(papers):
        for j, p2 in enumerate(papers):
            if p1 not in paper_embeddings or p2 not in paper_embeddings:
                score = 0.0
            elif p1 == p2:
                score = 1.0
            else:
                score = round(cosine_similarity(paper_embeddings[p1], paper_embeddings[p2]), 4)
            
            similarity_data.append({
                "paper_1":          p1,
                "paper_2":          p2,
                "similarity_score": score
            })

    return similarity_data


def compute_section_similarity_matrix(paper1: str, paper2: str) -> list[dict]:
    """
    Computes pairwise similarity scores between the sections of two papers.
    Uses the mean chunk embedding to represent each section.

    Returns:
        list of dicts: [
            {"paper1": str, "paper1_section": str, "paper2": str, "paper2_section": str, "score": float}
        ]
    """
    chunks1 = get_chunks_with_embeddings(paper1)
    chunks2 = get_chunks_with_embeddings(paper2)

    if not chunks1 or not chunks2:
        return []

    # Group chunk embeddings by section for paper 1
    sections1 = {}
    for c in chunks1:
        sect = c.get("section", "Unknown")
        sections1.setdefault(sect, []).append(np.array(c["embedding"]))

    # Group chunk embeddings by section for paper 2
    sections2 = {}
    for c in chunks2:
        sect = c.get("section", "Unknown")
        sections2.setdefault(sect, []).append(np.array(c["embedding"]))

    # Compute mean embedding for each section
    sect_embeddings1 = {s: np.mean(v, axis=0) for s, v in sections1.items()}
    sect_embeddings2 = {s: np.mean(v, axis=0) for s, v in sections2.items()}

    # Compute pairwise similarities
    section_similarities = []
    for s1, emb1 in sect_embeddings1.items():
        for s2, emb2 in sect_embeddings2.items():
            score = round(cosine_similarity(emb1, emb2), 4)
            section_similarities.append({
                "paper1":         paper1,
                "paper1_section": s1,
                "paper2":         paper2,
                "paper2_section": s2,
                "score":          score
            })

    return section_similarities
