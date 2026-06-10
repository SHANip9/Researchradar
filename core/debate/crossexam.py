"""crossexam.py — Multi-paper cross-examination and comparative RAG debate engine."""

import os
import re
import numpy as np
import anthropic
from dotenv import load_dotenv

from core.storage.vector_store import semantic_search, get_all_chunks, get_paper_metadata
from core.retrieval.retriever import build_bm25_index, _rrf_fusion
from core.embeddings.embedder import embed_query

# ─── Retrieval Config ──────────────────────────────────────────
TOP_K_PER_PAPER  = 4
SEMANTIC_FETCH   = 15
RRF_K            = 60


def retrieve_crossexam_chunks(question: str, paper_titles: list[str]) -> list:
    """
    Retrieves the top chunks for a specified subset of papers.
    Ensures that each of the selected papers gets represented in the context.
    """
    if not paper_titles:
        return []

    query_vector = embed_query(question)
    tokenized_query = question.lower().split()

    all_chunks_global = get_all_chunks()
    bm25_index, all_chunks_global = build_bm25_index(all_chunks_global)
    bm25_scores_global = (
        bm25_index.get_scores(tokenized_query) if bm25_index else None
    )

    synthesis_results = []

    for paper_title in paper_titles:
        # Get metadata to find the source filename
        meta = get_paper_metadata(paper_title)
        source_file = meta.get("source", paper_title)

        # Semantic search filtered to this paper source file
        paper_semantic = semantic_search(
            query_vector, n_results=SEMANTIC_FETCH, paper_filter=source_file
        )

        if not paper_semantic:
            # Try semantic search with paper_title if source file did not return results
            paper_semantic = semantic_search(
                query_vector, n_results=SEMANTIC_FETCH, paper_filter=paper_title
            )
            if not paper_semantic:
                continue

        # Filter BM25 corpus and scores to only this paper's chunks
        paper_chunks = [
            c for c in all_chunks_global 
            if c.get("paper_title") == paper_title or c.get("source") == source_file
        ]

        if bm25_index and paper_chunks:
            # Extract indices for this paper's chunks in global corpus
            paper_indices = [
                i for i, c in enumerate(all_chunks_global) 
                if c.get("paper_title") == paper_title or c.get("source") == source_file
            ]
            paper_bm25_scores = np.array([bm25_scores_global[i] for i in paper_indices])

            # RRF fusion
            fused = _rrf_fusion(paper_semantic, paper_chunks, paper_bm25_scores)
        else:
            fused = paper_semantic

        # Add top chunks
        synthesis_results.extend(fused[:TOP_K_PER_PAPER])

    return synthesis_results


def run_cross_examination(paper_titles: list[str], question: str, conversation_history: list = []) -> dict:
    """
    Executes a simulated cross-examination panel where representatives of 
    multiple papers answer the same question based on their data.
    """
    load_dotenv(override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is missing from environment config.")

    # Step 1: Retrieve context chunks
    chunks = retrieve_crossexam_chunks(question, paper_titles)
    
    if not chunks:
        return {
            "response": "No relevant citations could be located across the selected papers.",
            "retrieved_chunks": []
        }

    # Step 2: Format context blocks
    context_parts = []
    for i, c in enumerate(chunks):
        title = c.get("paper_title", c.get("source", "unknown"))
        context_parts.append(
            f"[Source: {title} | Page: {c['page']} | Section: {c.get('section', 'Unknown')}]\n"
            f"{c['text']}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    # Format history
    history_block = ""
    if conversation_history:
        history_parts = []
        for role, text in conversation_history:
            history_parts.append(f"{'Question' if role == 'user' else 'Response'}: {text}")
        history_block = "\n".join(history_parts)

    # Step 3: Prompt definition
    panel_papers = "\n".join([f"- {title}" for title in paper_titles])
    system_prompt = f"""You are an academic mediator orchestrating a panel cross-examination between researchers.
Your panel consists of representatives from the following papers:
{panel_papers}

Your task:
1. Address the user's question by cross-examining the papers.
2. Group the answer by major conceptual themes, research methodologies, or direct points of comparison.
3. For each point, let the paper representatives "speak" or present their arguments using direct assertions from their text. Format it clearly like:
   * **[Paper Title]**: "Stance / argument details" (Paper: filename.pdf | Page: N)
4. Highlight areas where the papers agree, disagree, or present contradictions.
5. Answer ONLY using information from the provided CONTEXT chunks. Do NOT make up points or use general training knowledge. If a paper has no information on the topic, state that it does not address the question.
6. End with a 2-3 sentence synthesized mediator summary comparing their contributions.

Your goal: Present a clear, structured, side-by-side debate and cross-examination."""

    prompt = f"""CONTEXT FROM INDEXED PAPERS:
{context_block}

---

PREVIOUS DISCUSSION HISTORY:
{history_block if history_block else "This is the first turn."}

---

CROSS-EXAMINATION QUESTION:
{question}

Formulate the panel responses:"""

    # Step 4: Execute API call
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=claude_model,
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer_text = response.content[0].text

    # Unique sources deduplication
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "paper": chunk.get("paper_title", chunk["source"]),
                "page": chunk["page"],
                "sentiment": chunk.get("chunk_sentiment", "neutral")
            })

    return {
        "response": answer_text,
        "sources": sources,
        "retrieved_chunks": chunks
    }
