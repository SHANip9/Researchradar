"""topics.py — Unsupervised topic modeling using BERTopic with pre-computed embeddings."""

from collections import Counter
import re
from bertopic import BERTopic


def _clean_text_simple(text: str) -> list[str]:
    """Basic tokenizer and stop-word filter for fallback topic modeling."""
    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
        "up", "down", "in", "on", "at", "by", "for", "with", "about", "against",
        "of", "to", "is", "was", "were", "are", "be", "been", "being", "have",
        "has", "had", "having", "do", "does", "did", "doing", "this", "that",
        "these", "those", "i", "we", "you", "they", "he", "she", "it", "our",
        "us", "their", "his", "her", "its", "them", "from", "as", "into",
        "through", "during", "before", "after", "above", "below", "to", "from",
        "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now"
    }
    words = re.findall(r"\b[a-z]{3,15}\b", text.lower())
    return [w for w in words if w not in stop_words]


def _get_fallback_topics(chunks: list) -> list[dict]:
    """
    Generates a fallback topic representation if there are too few chunks
    to run BERTopic (HDBSCAN requires a minimum cluster size/sample count).
    """
    if not chunks:
        return []

    # Get most common terms across all chunks
    all_text = " ".join([c["text"] for c in chunks])
    words = _clean_text_simple(all_text)
    most_common = Counter(words).most_common(10)
    keywords = ", ".join([w[0] for w in most_common]) if most_common else "research, paper, study"

    # Find the top paper (the one with the most chunks)
    papers = [c.get("paper_title", c.get("source", "unknown")) for c in chunks]
    top_paper = Counter(papers).most_common(1)[0][0] if papers else "unknown"

    return [{
        "topic_id":    0,
        "keywords":    keywords,
        "chunk_count": len(chunks),
        "top_paper":   top_paper
    }]


def discover_topics(chunks: list) -> list[dict]:
    """
    Runs BERTopic on all indexed chunks using their pre-computed embeddings.
    
    Args:
        chunks: List of chunk dicts from vector_store containing "text", "embedding",
                and metadata fields (source, paper_title).

    Returns:
        list of dicts: [{"topic_id": int, "keywords": str, "chunk_count": int, "top_paper": str}]
    """
    # ─── Robust Fallback for Small Corpora ──────────────────────────────
    # UMAP and HDBSCAN need a minimal number of chunks to run without errors.
    # We set a threshold of 10 chunks.
    if len(chunks) < 10:
        print(f"[Topic Model] Corpus too small ({len(chunks)} chunks). Using TF-IDF fallback.")
        return _get_fallback_topics(chunks)

    try:
        texts = [c["text"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]

        # Initialize BERTopic.
        # We pass empty embedding_model because we provide pre-computed embeddings during fit.
        # min_topic_size=2 helps form topics in small-to-medium datasets.
        print(f"[Topic Model] Fitting BERTopic on {len(texts)} chunks...")
        topic_model = BERTopic(embedding_model=None, min_topic_size=2)
        
        # Fit topic model using pre-computed embeddings to avoid redundant GPU/CPU runs
        topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)

        # Get topic info dataframe
        topic_info = topic_model.get_topic_info()

        # Extract results
        results = []
        for _, row in topic_info.iterrows():
            topic_id = int(row["Topic"])
            
            # Map topic representation to comma-separated string
            rep = row.get("Representation", [])
            keywords = ", ".join(rep[:5]) if rep else "unknown"
            
            # Count chunks assigned to this topic
            chunk_count = int(row["Count"])

            # Find the top paper (the paper that has the most chunks in this topic)
            assigned_papers = [
                chunks[i].get("paper_title", chunks[i].get("source", "unknown"))
                for i, t in enumerate(topics)
                if t == topic_id
            ]
            
            if assigned_papers:
                top_paper = Counter(assigned_papers).most_common(1)[0][0]
            else:
                top_paper = "unknown"

            results.append({
                "topic_id":    topic_id,
                "keywords":    keywords,
                "chunk_count": chunk_count,
                "top_paper":   top_paper
            })

        print(f"[Topic Model] Discovered {len(results)} topics.")
        return results

    except Exception as e:
        print(f"[Topic Model] BERTopic run failed: {str(e)}. Falling back to simple keyword profiling.")
        return _get_fallback_topics(chunks)
