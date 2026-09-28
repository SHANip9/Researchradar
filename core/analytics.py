"""
core/analytics.py — Advanced Research Analytics & Bias Detection Engine.
Computes:
  1. Semantic Bias & Objectivity Inspector (Hedging vs Overclaiming, Confirmation Bias, Benchmark Overfitting).
  2. Methodological Rigor Scorecard (Empirical validation, baseline diversity, limitation transparency).
  3. Pairwise Cosine Similarity Heatmap (Paper-to-Paper & Section-to-Section).
  4. Stance & Sentiment Distribution (Optimistic, Cautious, Critical, Neutral).
  5. Key Conceptual Themes & Terminology.
"""

import re
import numpy as np
import pandas as pd
from core.engine import get_all_chunks, get_indexed_papers


# ─── Bias & Objectivity Patterns ──────────────────────────────
OVERCLAIM_TERMS = [
    r"\bproves?\s+definitively\b", r"\bflawless\b", r"\bunbeatable\b",
    r"\bin\s+all\s+scenarios\b", r"\bsuperior\s+in\s+every\b", r"\brevolutionizes?\b",
    r"\bunprecedented\b", r"\bwithout\s+any\s+limitations?\b", r"\boptimal\s+in\s+all\b"
]

HEDGING_TERMS = [
    r"\bsuggests?\s+that\b", r"\bmay\s+indicate\b", r"\bpreliminary\s+results?\b",
    r"\bunder\s+specific\s+conditions\b", r"\bwe\s+hypothesize\b", r"\bfurther\s+investigation\b",
    r"\bpotential\s+limitation\b", r"\bcaveat\b", r"\bsubject\s+to\b"
]

LIMITATION_TERMS = [
    r"\blimitation\b", r"\bfailure\s+case\b", r"\bbottleneck\b",
    r"\bcompute\s+cost\b", r"\blacks?\s+generaliz\b", r"\bunderperformed\b",
    r"\bthreats?\s+to\s+validity\b", r"\bnegative\s+result\b"
]

BASELINE_TERMS = [
    r"\bbaseline\b", r"\bstate-of-the-art\b", r"\bsota\b",
    r"\bcompared\s+to\b", r"\bcomparison\s+with\b", r"\bbenchmarks?\b",
    r"\bprior\s+work\b", r"\bablations?\b"
]


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two 1D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ─── 1. Semantic Bias & Objectivity Analysis ──────────────────
def analyze_paper_bias(paper_name: str) -> dict:
    """
    Evaluates potential author bias, overclaiming ratio, and limitation transparency
    for a given research paper.
    """
    chunks = get_all_chunks(paper_name=paper_name)
    if not chunks:
        return {}

    full_text = " ".join([c["text"] for c in chunks]).lower()
    total_words = max(len(full_text.split()), 1)

    # Count pattern occurrences
    overclaim_count = sum(len(re.findall(pat, full_text)) for pat in OVERCLAIM_TERMS)
    hedging_count = sum(len(re.findall(pat, full_text)) for pat in HEDGING_TERMS)
    limitation_count = sum(len(re.findall(pat, full_text)) for pat in LIMITATION_TERMS)
    baseline_count = sum(len(re.findall(pat, full_text)) for pat in BASELINE_TERMS)

    # Normalize per 1,000 words
    scale = 1000.0 / total_words
    norm_overclaim = overclaim_count * scale
    norm_hedging = hedging_count * scale
    norm_limitation = limitation_count * scale
    norm_baseline = baseline_count * scale

    # Compute Bias & Objectivity Index
    # High overclaiming + low limitations = High bias risk
    # High hedging + high limitations + baselines = High objectivity
    bias_score = min(max(int((norm_overclaim * 3.5) - (norm_limitation * 1.5) - (norm_hedging * 0.8) + 30), 5), 95)
    objectivity_score = 100 - bias_score
    
    # Rigor components (0-100)
    empirical_rigor = min(int(norm_baseline * 12 + 40), 98)
    transparency_score = min(int(norm_limitation * 18 + norm_hedging * 8 + 35), 98)

    # Determine risk category
    if bias_score > 60:
        bias_risk = "⚠️ Elevated Bias Risk (High claim-to-caution ratio)"
        bias_color = "#ef4444"
    elif bias_score > 35:
        bias_risk = "⚖️ Balanced Academic Discourse"
        bias_color = "#f59e0b"
    else:
        bias_risk = "🛡️ Highly Objective & Well-Hedged"
        bias_color = "#10b981"

    return {
        "paper_name": paper_name,
        "bias_score": bias_score,
        "objectivity_score": objectivity_score,
        "bias_risk": bias_risk,
        "bias_color": bias_color,
        "empirical_rigor": empirical_rigor,
        "transparency_score": transparency_score,
        "overclaim_count": overclaim_count,
        "hedging_count": hedging_count,
        "limitation_count": limitation_count,
        "baseline_count": baseline_count,
        "total_words": total_words
    }


# ─── 2. Cross-Paper Similarity Matrix ─────────────────────────
def compute_similarity_matrix() -> pd.DataFrame:
    """Computes pairwise cosine similarity between all indexed papers."""
    papers = get_indexed_papers()
    if len(papers) < 1:
        return pd.DataFrame()

    paper_vectors = {}
    for p in papers:
        chunks = get_all_chunks(paper_name=p)
        embs = [np.array(c["embedding"]) for c in chunks if c.get("embedding") is not None and len(c.get("embedding")) > 0]
        if embs:
            paper_vectors[p] = np.mean(embs, axis=0)

    sim_data = []
    for p1 in papers:
        row = {"Paper": p1}
        for p2 in papers:
            if p1 in paper_vectors and p2 in paper_vectors:
                score = cosine_sim(paper_vectors[p1], paper_vectors[p2])
                row[p2] = round(score, 3)
            else:
                row[p2] = 0.0
        sim_data.append(row)

    df = pd.DataFrame(sim_data).set_index("Paper").fillna(0.0)
    return df


# ─── 3. Stance & Tone Breakdown ──────────────────────────────
def compute_paper_stance(paper_name: str) -> dict:
    """Categorizes the tone of chunks into Optimistic, Cautious, Critical, or Neutral."""
    chunks = get_all_chunks(paper_name=paper_name)
    if not chunks:
        return {"Optimistic": 25, "Cautious": 25, "Critical": 25, "Neutral": 25}

    counts = {"Optimistic": 0, "Cautious": 0, "Critical": 0, "Neutral": 0}
    for c in chunks:
        txt = c["text"].lower()
        if any(w in txt for w in ["achieve", "outperform", "state-of-the-art", "superior", "significant improvement", "best"]):
            counts["Optimistic"] += 1
        elif any(w in txt for w in ["careful", "limitation", "preliminary", "constrained", "further research", "caution"]):
            counts["Cautious"] += 1
        elif any(w in txt for w in ["fails", "flaw", "inadequate", "bottleneck", "vulnerable", "poorly", "underperforms"]):
            counts["Critical"] += 1
        else:
            counts["Neutral"] += 1

    total = max(sum(counts.values()), 1)
    return {k: round((v / total) * 100, 1) for k, v in counts.items()}


# ─── 4. Thematic Concepts & Keyphrases ───────────────────────
STOPWORDS = {
    "the", "of", "and", "in", "to", "a", "is", "for", "that", "this", "we", "with",
    "on", "as", "by", "are", "an", "be", "our", "from", "at", "which", "can", "paper",
    "using", "used", "model", "results", "based", "proposed", "method", "learning"
}

def extract_top_concepts(paper_name: str, top_n: int = 8) -> list[tuple[str, int]]:
    """Extracts high-frequency scientific terms and bigrams."""
    chunks = get_all_chunks(paper_name=paper_name)
    if not chunks:
        return []

    words = []
    for c in chunks:
        cleaned = re.findall(r"\b[a-zA-Z]{4,20}\b", c["text"].lower())
        words.extend([w for w in cleaned if w not in STOPWORDS])

    if not words:
        return []

    freq = pd.Series(words, dtype="string").value_counts()
    return list(freq.head(top_n).items())


# ─── 5. Dynamic Inquiry & Discourse Semantic Alignment ────────
def analyze_inquiry_semantic_alignment(paper_name: str, chat_history: list[dict]) -> dict:
    """
    Dynamically analyzes the semantic interaction between user/agent inquiries
    and the indexed research paper:
      - Section scrutiny distribution (which sections are being challenged/explored).
      - Inquiry tone & stance (Adversarial, Technical, Comparative, Exploratory).
      - Discourse tension & scrutiny index.
      - Active queried concepts.
    """
    user_queries = [m.get("content", "") for m in chat_history if m.get("role") == "user"]
    total_inquiries = len(user_queries)

    chunks = get_all_chunks(paper_name=paper_name)
    if not chunks:
        return {
            "total_inquiries": total_inquiries,
            "has_inquiries": total_inquiries > 0,
            "section_scrutiny": {},
            "inquiry_tone": {"Exploratory": 25, "Technical": 25, "Adversarial": 25, "Comparative": 25},
            "discourse_tension": 30,
            "tension_label": "Calm Exploratory Discourse",
            "top_queried_terms": [],
            "agent_invocations": {"Debate": 0, "Comparison": 0, "Prediction": 0, "DirectQA": 0}
        }

    # Group chunk texts by academic section
    sections_map = {}
    for c in chunks:
        sec = c.get("section", "General") or "General"
        sections_map.setdefault(sec, []).append(c.get("text", "").lower())

    # Count agent invocations and tone signals across queries
    agent_counts = {"Debate": 0, "Comparison": 0, "Prediction": 0, "DirectQA": 0}
    tone_counts = {"Adversarial": 0, "Technical": 0, "Comparative": 0, "Exploratory": 0}
    query_words = []
    cmd_words = {"debate", "adversarial", "compare", "diff", "contrast", "predict", "future"}

    for q in user_queries:
        low_q = q.lower()
        # Clean words for concept extraction
        q_clean = re.findall(r"\b[a-zA-Z]{3,20}\b", low_q)
        query_words.extend([w for w in q_clean if w not in STOPWORDS and w not in cmd_words and not w.startswith("/")])

        # Agent command classification
        if low_q.startswith("/debate") or low_q.startswith("/adversarial"):
            agent_counts["Debate"] += 1
            tone_counts["Adversarial"] += 2
        elif low_q.startswith("/compare") or low_q.startswith("/diff") or low_q.startswith("/contrast"):
            agent_counts["Comparison"] += 1
            tone_counts["Comparative"] += 2
        elif low_q.startswith("/predict") or low_q.startswith("/future"):
            agent_counts["Prediction"] += 1
            tone_counts["Technical"] += 1
        else:
            agent_counts["DirectQA"] += 1

        # Text-based tone heuristic
        if any(w in low_q for w in ["flaw", "fail", "limitation", "weakness", "overclaim", "proof", "assume", "bottleneck", "disagree", "why"]):
            tone_counts["Adversarial"] += 1
        elif any(w in low_q for w in ["how", "architecture", "math", "equation", "parameter", "loss", "metric", "ablation", "layer", "module"]):
            tone_counts["Technical"] += 1
        elif any(w in low_q for w in ["versus", "vs", "compare", "baseline", "sota", "tradeoff", "differ", "better"]):
            tone_counts["Comparative"] += 1
        else:
            tone_counts["Exploratory"] += 1

    if total_inquiries > 0:
        total_tone = max(sum(tone_counts.values()), 1)
        inquiry_tone = {k: round((v / total_tone) * 100, 1) for k, v in tone_counts.items()}
    else:
        inquiry_tone = {"Exploratory": 40.0, "Technical": 30.0, "Comparative": 15.0, "Adversarial": 15.0}

    # Compute Section Scrutiny based on query terms matching section chunks
    section_scrutiny = {sec: 0 for sec in sections_map}
    for q_word in query_words:
        for sec, text_list in sections_map.items():
            combined = " ".join(text_list)
            if q_word in combined:
                section_scrutiny[sec] += 1

    total_sec_hits = sum(section_scrutiny.values())
    if total_sec_hits > 0:
        section_scrutiny = {k: round((v / total_sec_hits) * 100, 1) for k, v in section_scrutiny.items()}
    else:
        # Default uniform distribution if no direct term match yet
        n_sec = max(len(sections_map), 1)
        section_scrutiny = {k: round(100.0 / n_sec, 1) for k in sections_map}

    # Discourse Tension Index (0 - 100)
    adversarial_weight = inquiry_tone.get("Adversarial", 0) * 0.7
    technical_weight = inquiry_tone.get("Technical", 0) * 0.3
    discourse_tension = min(max(int(adversarial_weight + technical_weight), 10), 95) if total_inquiries > 0 else 25

    if discourse_tension > 60:
        tension_label = "⚔️ High Critical Scrutiny (Adversarial stress-testing active)"
        tension_color = "#f43f5e"
    elif discourse_tension > 35:
        tension_label = "🔬 In-Depth Analytical Discourse (Active theoretical probing)"
        tension_color = "#0284c7"
    else:
        tension_label = "💡 Exploratory & Factual Review (Gentle discovery mode)"
        tension_color = "#10b981"

    top_queried_terms = list(pd.Series(query_words, dtype="string").value_counts().head(6).items()) if query_words else []

    return {
        "total_inquiries": total_inquiries,
        "has_inquiries": total_inquiries > 0,
        "section_scrutiny": section_scrutiny,
        "inquiry_tone": inquiry_tone,
        "discourse_tension": discourse_tension,
        "tension_label": tension_label,
        "tension_color": tension_color,
        "top_queried_terms": top_queried_terms,
        "agent_invocations": agent_counts
    }
