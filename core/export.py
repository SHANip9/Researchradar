"""export.py — Orchestrates the generation of the 5 datasets and exports them to CSV for Power BI."""

import os
import pandas as pd
from core.storage.vector_store import get_all_chunks, get_papers_list, get_chunks_with_embeddings
from core.analysis.similarity import compute_paper_similarity_matrix, compute_section_similarity_matrix
from core.analysis.topics import discover_topics
from core.analysis.keywords import extract_keywords_for_all_papers

# ─── Path Calculation ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_DIR = os.path.join(PROJECT_ROOT, "data", "exports")


def map_sentiment_to_raw(label: str) -> str:
    """Maps the research-nuanced sentiment label back to DistilBERT raw binary output."""
    if label in ["optimistic", "neutral"]:
        return "POSITIVE"
    else:
        return "NEGATIVE"


def export_all_to_csv() -> dict:
    """
    Runs all Modules (sentiment, similarity, topic modeling, keyword extraction)
    and writes the results to 5 coordinated CSV files inside data/exports/.

    Returns:
        dict: A summary containing export file paths and record counts.
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)
    summary = {}

    print(f"[Export Engine] Initializing database extraction from {PROJECT_ROOT}...")

    # ─── Step 1: Retrieve All Chunks & Chunks with Embeddings ───────────
    all_chunks = get_all_chunks()
    papers = get_papers_list()

    if not all_chunks:
        print("[Export Engine] Database is empty. Export aborted.")
        return {"error": "No data in vector database to export."}

    # Fetch chunks with embeddings (needed for BERTopic topic modeling)
    chunks_with_embeddings = []
    for paper in papers:
        chunks_with_embeddings.extend(get_chunks_with_embeddings(paper))

    # ─── Dataset 1: Sentiment Analysis Trajectory ────────────────────
    print("[Export Engine] Generating Dataset 1: Sentiment Analysis...")
    sentiment_rows = []
    for chunk in all_chunks:
        label = chunk.get("chunk_sentiment", "neutral")
        sentiment_rows.append({
            "paper_title": chunk.get("paper_title", chunk.get("source", "unknown")),
            "section":     chunk.get("section", "Unknown"),
            "chunk_index": chunk.get("chunk_index", 0),
            "label":       label,
            "score":       chunk.get("chunk_sentiment_score", 0.5),
            "raw_label":   map_sentiment_to_raw(label),
            "author_org":  chunk.get("author_org", ""),
            "category":    chunk.get("category", ""),
            "date":        chunk.get("date", "")
        })
    
    df_sentiment = pd.DataFrame(sentiment_rows)
    sentiment_path = os.path.join(EXPORT_DIR, "sentiment_analysis.csv")
    df_sentiment.to_csv(sentiment_path, index=False, encoding="utf-8")
    summary["sentiment_analysis"] = {"path": sentiment_path, "records": len(df_sentiment)}

    # ─── Dataset 2: Paper-to-Paper Similarity Matrix ──────────────────
    print("[Export Engine] Generating Dataset 2: Paper-to-Paper Similarity...")
    paper_similarity_data = compute_paper_similarity_matrix()
    df_paper_sim = pd.DataFrame(paper_similarity_data)
    paper_sim_path = os.path.join(EXPORT_DIR, "paper_similarity.csv")
    df_paper_sim.to_csv(paper_sim_path, index=False, encoding="utf-8")
    summary["paper_similarity"] = {"path": paper_sim_path, "records": len(df_paper_sim)}

    # ─── Dataset 3: Section-to-Section Similarity Heatmaps ──────────────
    print("[Export Engine] Generating Dataset 3: Section-to-Section Similarity...")
    section_similarity_rows = []
    # Compute section similarity for all pairwise combinations of papers
    for p1 in papers:
        for p2 in papers:
            # We compute all pairs (including p1==p2 for self-paper sections)
            sect_sims = compute_section_similarity_matrix(p1, p2)
            section_similarity_rows.extend(sect_sims)
            
    df_sect_sim = pd.DataFrame(section_similarity_rows)
    sect_sim_path = os.path.join(EXPORT_DIR, "section_similarity.csv")
    df_sect_sim.to_csv(sect_sim_path, index=False, encoding="utf-8")
    summary["section_similarity"] = {"path": sect_sim_path, "records": len(df_sect_sim)}

    # ─── Dataset 4: Topic Modeling (BERTopic) ────────────────────────
    print("[Export Engine] Generating Dataset 4: Topic Discovery...")
    topic_data = discover_topics(chunks_with_embeddings)
    df_topics = pd.DataFrame(topic_data)
    topics_path = os.path.join(EXPORT_DIR, "topic_model.csv")
    df_topics.to_csv(topics_path, index=False, encoding="utf-8")
    summary["topic_model"] = {"path": topics_path, "records": len(df_topics)}

    # ─── Dataset 5: Keyword Relevance Scores (KeyBERT) ───────────────
    print("[Export Engine] Generating Dataset 5: Keyphrase Extraction...")
    keyword_data = extract_keywords_for_all_papers()
    df_keywords = pd.DataFrame(keyword_data)
    keywords_path = os.path.join(EXPORT_DIR, "keyword_scores.csv")
    df_keywords.to_csv(keywords_path, index=False, encoding="utf-8")
    summary["keyword_scores"] = {"path": keywords_path, "records": len(df_keywords)}

    print(f"[Export Engine] Export completed successfully. Files saved to: {EXPORT_DIR}")
    return summary
