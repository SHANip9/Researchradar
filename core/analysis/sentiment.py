"""
sentiment.py — Tags each paper as Optimistic / Cautious / Critical / Neutral

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY SENTIMENT TAGGING FOR RESEARCH PAPERS?
    Research papers take positions. A paper might:
    - Celebrate results:   "Our method achieves state-of-the-art accuracy"  → Optimistic
    - Warn about risks:    "These results must be interpreted carefully"     → Cautious
    - Critique others:     "Current approaches fundamentally fail to..."     → Critical
    - Just report facts:   "We observe that X correlates with Y"            → Neutral

    Knowing the tone helps you understand a paper's stance at a glance
    before diving into 50 pages of dense academic writing.

HOW IT WORKS (3 steps):
    1. Run each chunk through DistilBERT (fine-tuned on SST-2 movie reviews)
       → Returns POSITIVE/NEGATIVE + confidence score (0.0 to 1.0)
    2. Map binary output to our 4-label taxonomy using confidence thresholds
    3. Aggregate all chunk sentiments → one label per paper (weighted majority)

SENTIMENT MAPPING LOGIC:
    POSITIVE + score > 0.75  →  Optimistic  (confident positive stance)
    POSITIVE + score ≤ 0.75  →  Neutral     (hedged, uncertain positive)
    NEGATIVE + score > 0.65  →  Critical    (strong negative/critical stance)
    NEGATIVE + score ≤ 0.65  →  Cautious    (measured concern, limitations)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from collections import Counter
from transformers import pipeline

# ─── Configuration ────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE = 32          # Process 32 chunks at a time (prevents memory overflow)
OPTIMISTIC_THRESHOLD = 0.75   # POSITIVE above this → Optimistic
CAUTIOUS_THRESHOLD   = 0.65   # NEGATIVE above this → Critical (below → Cautious)

# Emoji labels for display in the Streamlit UI
SENTIMENT_DISPLAY = {
    "optimistic": "🟢 Optimistic",
    "cautious":   "🟡 Cautious",
    "critical":   "🔴 Critical",
    "neutral":    "⚪ Neutral",
}

# Singleton: load the HuggingFace model only once
_sentiment_pipeline = None


def get_sentiment_pipeline():
    """Loads DistilBERT sentiment model on first call, returns cached model after."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print(f"[Sentiment] Loading model: {MODEL_NAME}...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            device=-1,       # -1 = CPU (change to 0 if you have an NVIDIA GPU)
            truncation=True, # Silently truncate chunks longer than 512 tokens
            max_length=512
        )
        print("[Sentiment] Model ready.")
    return _sentiment_pipeline


def _map_to_research_label(raw_label: str, score: float) -> str:
    """
    Maps DistilBERT's binary output (POSITIVE/NEGATIVE + score) to our 4-class taxonomy.

    This mapping is a design decision:
    - DistilBERT was trained on movie reviews, not research papers.
    - We re-interpret POSITIVE/NEGATIVE in the context of academic tone.
    - The thresholds (0.75, 0.65) were chosen based on manual testing.
    """
    if raw_label == "POSITIVE":
        return "optimistic" if score > OPTIMISTIC_THRESHOLD else "neutral"
    else:  # NEGATIVE
        return "critical" if score > CAUTIOUS_THRESHOLD else "cautious"


def analyze_chunks(chunks: list) -> list:
    """
    Runs sentiment analysis on all chunks from a paper.

    Args:
        chunks (list): List of chunk dicts from pdf_processor.py

    Returns:
        list of dicts: [{"label": "optimistic", "score": 0.85, "raw": "POSITIVE"}, ...]
        One dict per chunk, in the same order as the input list.
    """
    sentiment_pipe = get_sentiment_pipeline()
    texts = [chunk["text"] for chunk in chunks]

    print(f"[Sentiment] Analyzing {len(texts)} chunks in batches of {BATCH_SIZE}...")

    all_raw_results = []

    # ─── Batch Processing ────────────────────────────────────────────────
    # Processing all chunks at once can overflow RAM for very large papers.
    # We split into batches of 32 and process sequentially.
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_results = sentiment_pipe(batch)
        all_raw_results.extend(batch_results)

    # Map each raw POSITIVE/NEGATIVE result to our 4-class label
    return [
        {
            "label": _map_to_research_label(r["label"], r["score"]),
            "score": round(r["score"], 4),
            "raw":   r["label"]
        }
        for r in all_raw_results
    ]


def aggregate_paper_sentiment(chunk_sentiments: list) -> dict:
    """
    Combines all chunk sentiments for a paper into one paper-level tag.

    AGGREGATION STRATEGY — WEIGHTED MAJORITY VOTE:
        We don't just count votes (that would be a simple majority).
        Instead, high-confidence chunks (score ≈ 0.95) carry more weight
        than uncertain ones (score ≈ 0.51).

        This prevents a long neutral introduction from drowning out a
        strongly critical conclusion section.

    Args:
        chunk_sentiments (list): Output of analyze_chunks() or get_paper_sentiment_data()

    Returns:
        dict: {
            "paper_sentiment": "optimistic" | "cautious" | "critical" | "neutral",
            "confidence": float,
            "display_label": str,   ← emoji label for UI display
            "breakdown": {"optimistic": N, "cautious": N, ...}
        }
    """
    if not chunk_sentiments:
        return {
            "paper_sentiment": "neutral",
            "confidence": 0.5,
            "display_label": SENTIMENT_DISPLAY["neutral"],
            "breakdown": {}
        }

    # Accumulate weighted scores for each label
    breakdown    = {"optimistic": 0, "cautious": 0, "critical": 0, "neutral": 0}
    weight_sums  = {"optimistic": 0.0, "cautious": 0.0, "critical": 0.0, "neutral": 0.0}

    for s in chunk_sentiments:
        label = s["label"]
        score = s["score"]
        breakdown[label]   += 1
        weight_sums[label] += score

    # The winner is the label with the highest total weighted score
    paper_sentiment = max(weight_sums, key=weight_sums.get)

    # Average confidence = total weight / count for the winning label
    count      = max(breakdown[paper_sentiment], 1)
    confidence = round(weight_sums[paper_sentiment] / count, 3)

    return {
        "paper_sentiment": paper_sentiment,
        "confidence":      confidence,
        "display_label":   SENTIMENT_DISPLAY[paper_sentiment],
        "breakdown":       breakdown,
    }


# ─── Section-Level Aggregation ────────────────────────────────

def aggregate_section_sentiment(chunks: list, sentiments: list) -> list:
    """
    Groups chunks by their 'section' metadata and computes per-section sentiment.

    Useful for understanding how tone shifts across a paper's structure
    (e.g. Introduction is neutral, Results is optimistic, Limitations is cautious).

    Args:
        chunks (list):     Chunk dicts with optional 'section' metadata key.
        sentiments (list): Parallel list of sentiment dicts from analyze_chunks().

    Returns:
        list of dicts: [
            {"section": "Introduction", "avg_score": 0.72, "dominant_label": "neutral", "chunk_count": 5},
            {"section": "Results",      "avg_score": 0.89, "dominant_label": "optimistic", "chunk_count": 8},
            ...
        ]
    """
    # ─── Group sentiment data by section name ────────────────────────────
    section_data: dict[str, list[dict]] = {}

    for chunk, sentiment in zip(chunks, sentiments):
        section = chunk.get("section", "Unknown")
        section_data.setdefault(section, []).append(sentiment)

    # ─── Compute per-section aggregates ──────────────────────────────────
    results = []

    for section, section_sentiments in section_data.items():
        scores = [s["score"] for s in section_sentiments]
        labels = [s["label"] for s in section_sentiments]

        # Dominant label = most frequent label in this section
        dominant_label = Counter(labels).most_common(1)[0][0]

        results.append({
            "section":        section,
            "avg_score":      round(sum(scores) / len(scores), 4),
            "dominant_label": dominant_label,
            "chunk_count":    len(section_sentiments),
        })

    return results
