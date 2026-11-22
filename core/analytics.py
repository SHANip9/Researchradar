"""
core/analytics.py — Advanced Research Analytics & Semantic Evaluation Engine.

Provides deep, per-document and cross-corpus analytical inspection:
  1. Author Tone & Rhetorical Spectrum (Informative, Technical, Assertive/SOTA, Cautious/Critical).
  2. Tone & Argument Flow Trajectory (Reading-order spline tracking rhetoric across sections).
  3. Paper Quality & Empirical Rigor Scorecard (Objectivity, baselines, transparency, claim validity).
  4. Academic Section Anatomy (Distribution of chunks across paper sections).
  5. Filtered Domain Thematic Concepts & Keyphrases.
  6. Cross-Paper Semantic Similarity Heatmap & Comparative Corpus Matrix.
  7. Dynamic Inquiry Scrutiny & Discourse Tension (Reflection of chat prompts against paper text).
"""

import re
import numpy as np
import pandas as pd
from core.engine import get_all_chunks, get_indexed_papers


# ─── Academic & Lexical Stopwords for Concept Extraction ─────
ACADEMIC_STOPWORDS = {
    "the", "of", "and", "in", "to", "a", "is", "for", "that", "this", "we", "with",
    "on", "as", "by", "are", "an", "be", "our", "from", "at", "which", "can", "paper",
    "using", "used", "uses", "show", "shown", "shows", "see", "one", "two", "three",
    "many", "well", "high", "new", "based", "work", "data", "model", "models",
    "example", "chapter", "user", "users", "need", "make", "good", "like", "time",
    "different", "also", "into", "each", "other", "such", "than", "them", "then",
    "some", "only", "first", "their", "will", "would", "could", "should", "about",
    "above", "after", "again", "against", "all", "am", "any", "because", "before",
    "being", "below", "between", "both", "but", "cannot", "does", "down", "during",
    "few", "further", "had", "has", "have", "having", "he", "her", "here", "him",
    "his", "how", "if", "its", "itself", "just", "me", "might", "most", "must", "my",
    "no", "nor", "not", "off", "once", "or", "ours", "out", "over", "own", "same",
    "she", "so", "there", "they", "those", "through", "too", "under", "until", "up",
    "very", "was", "were", "what", "when", "where", "while", "who", "whom", "why",
    "you", "your", "page", "table", "figure", "fig", "section", "sections", "approach",
    "method", "methods", "result", "results", "these", "been", "however", "given",
    "within", "since", "across", "without", "overall", "several", "specific", "non"
}


# ─── Bias & Objectivity Lexical Patterns ──────────────────────
OVERCLAIM_TERMS = [
    r"\bproves?\s+definitively\b", r"\bflawless\b", r"\bunbeatable\b",
    r"\bin\s+all\s+scenarios\b", r"\bsuperior\s+in\s+every\b", r"\brevolutionizes?\b",
    r"\bunprecedented\b", r"\bwithout\s+any\s+limitations?\b", r"\boptimal\s+in\s+all\b",
    r"\bgroundbreaking\b", r"\bperfect\s+accuracy\b", r"\bvastly\s+superior\b"
]

HEDGING_TERMS = [
    r"\bsuggests?\s+that\b", r"\bmay\s+indicate\b", r"\bpreliminary\s+results?\b",
    r"\bunder\s+specific\s+conditions\b", r"\bwe\s+hypothesize\b", r"\bfurther\s+investigation\b",
    r"\bpotential\s+limitation\b", r"\bcaveat\b", r"\bsubject\s+to\b", r"\btentative\b",
    r"\bplausible\b", r"\bapproximate\b", r"\bto\s+some\s+extent\b", r"\bpartially\b"
]

LIMITATION_TERMS = [
    r"\blimitation\b", r"\bfailure\s+case\b", r"\bbottleneck\b",
    r"\bcompute\s+cost\b", r"\blacks?\s+generaliz\b", r"\bunderperformed\b",
    r"\bthreats?\s+to\s+validity\b", r"\bnegative\s+result\b", r"\bconstrained\b",
    r"\bout-of-distribution\b", r"\btrade-off\b", r"\bvulnerability\b", r"\bdegradation\b"
]

BASELINE_TERMS = [
    r"\bbaseline\b", r"\bstate-of-the-art\b", r"\bsota\b",
    r"\bcompared\s+to\b", r"\bcomparison\s+with\b", r"\bbenchmarks?\b",
    r"\bprior\s+work\b", r"\bablations?\b", r"\bevaluation\s+metric\b",
    r"\bempirical\s+validation\b", r"\bground\s+truth\b"
]


# ─── Tone Spectrum Patterns ───────────────────────────────────
TONE_INFORMATIVE = [
    r"\bdescrib", r"\bpresent", r"\bdemonstrat", r"\bdataset", r"\boverview",
    r"\bcollect", r"\bprovid", r"\bobserv", r"\bconsist", r"\bdefin", r"\bnoted\b",
    r"\banalyz", r"\breport", r"\bsummar", r"\bcontext", r"\bconduct"
]

TONE_TECHNICAL = [
    r"\balgorithm", r"\barchitectur", r"\bparam", r"\blayer", r"\bloss\b",
    r"\bgradien", r"\boptim", r"\bcomput", r"\bequat", r"\bmatrix",
    r"\bdimens", r"\btoken", r"\btransform", r"\bweight", r"\bprobabilit",
    r"\bdistribut", r"\bconvolut", r"\blatent", r"\binference", r"\bthroughput"
]

TONE_ASSERTIVE = [
    r"\bachiev", r"\boutperform", r"\bstate-of-the-art\b", r"\bsota\b",
    r"\bsuperior\b", r"\bsignificant", r"\bbreakthrough\b", r"\brobust\b",
    r"\bunprecedent", r"\boptimal\b", r"\bsubstantial\s+gain", r"\bdominat",
    r"\bproves?\b", r"\bflawless\b", r"\brevolutionar"
]

TONE_CAUTIOUS = [
    r"\blimitation", r"\bfail", r"\bbottleneck", r"\bvulnerab",
    r"\bunderperform", r"\bdegrad", r"\bflaw", r"\bcareful", r"\bcaveat",
    r"\bthreat", r"\bcost\b", r"\bconstrain", r"\brisk\b", r"\bpreliminary",
    r"\bpotential\s+harm", r"\bsensitive", r"\bplateau"
]


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two 1D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ─── 1. Paper Quality & Empirical Scorecard ("How Good It Is") ─
def analyze_paper_bias(paper_name: str) -> dict:
    """
    Evaluates author objectivity, empirical rigor, limitation transparency,
    and calculates a composite Paper Quality Index.
    """
    chunks = get_all_chunks(paper_name=paper_name)
    if not chunks:
        return {}

    full_text = " ".join([c["text"] for c in chunks]).lower()
    total_words = max(len(full_text.split()), 1)

    overclaim_count = sum(len(re.findall(pat, full_text)) for pat in OVERCLAIM_TERMS)
    hedging_count = sum(len(re.findall(pat, full_text)) for pat in HEDGING_TERMS)
    limitation_count = sum(len(re.findall(pat, full_text)) for pat in LIMITATION_TERMS)
    baseline_count = sum(len(re.findall(pat, full_text)) for pat in BASELINE_TERMS)

    scale = 1000.0 / total_words
    norm_overclaim = overclaim_count * scale
    norm_hedging = hedging_count * scale
    norm_limitation = limitation_count * scale
    norm_baseline = baseline_count * scale

    bias_score = min(max(int((norm_overclaim * 3.5) - (norm_limitation * 1.5) - (norm_hedging * 0.8) + 30), 5), 95)
    objectivity_score = 100 - bias_score

    empirical_rigor = min(int(norm_baseline * 12 + 40), 98)
    transparency_score = min(int(norm_limitation * 18 + norm_hedging * 8 + 35), 98)

    # Composite Quality Score (0 - 100)
    quality_score = min(max(int(empirical_rigor * 0.45 + objectivity_score * 0.35 + transparency_score * 0.20), 20), 98)

    if quality_score >= 80:
        quality_tier = "🏆 Exceptional Empirical Rigor (Peer-Review Standard)"
        quality_color = "#10b981"
    elif quality_score >= 65:
        quality_tier = "⚖️ Solid Academic Standard (Well-Validated)"
        quality_color = "#38bdf8"
    else:
        quality_tier = "⚠️ Preliminary / Needs Further Empirical Validation"
        quality_color = "#f59e0b"

    if bias_score > 60:
        bias_risk = "⚠️ Elevated Bias Risk (High claim-to-caution ratio)"
        bias_color = "#ef4444"
    elif bias_score > 35:
        bias_risk = "⚖️ Balanced Academic Discourse"
        bias_color = "#f59e0b"
    else:
        bias_risk = "🛡️ Highly Objective & Well-Hedged"
        bias_color = "#10b981"

    pages = set(c.get("page", 1) for c in chunks)
    sections = {}
    for c in chunks:
        sec = c.get("section", "General") or "General"
        sections[sec] = sections.get(sec, 0) + 1

    paper_title = chunks[0].get("paper_title", paper_name) if chunks else paper_name

    return {
        "paper_name": paper_name,
        "paper_title": paper_title,
        "total_chunks": len(chunks),
        "total_pages": len(pages),
        "total_words": total_words,
        "sections": dict(sorted(sections.items(), key=lambda x: x[1], reverse=True)),
        "quality_score": quality_score,
        "quality_tier": quality_tier,
        "quality_color": quality_color,
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
    }


# ─── 2. Author's Semantic Tone Spectrum (Pie/Donut Classifier) 
def compute_paper_stance(paper_name: str) -> dict[str, float]:
    """
    Categorizes the author's tone across:
      - 📘 Informative & Objective
      - ⚙️ Technical & Methodological
      - 🚀 Assertive / SOTA Claims
      - 🛡️ Cautious & Critical
    """
    chunks = get_all_chunks(paper_name=paper_name)
    if not chunks:
        return {
            "Informative & Objective": 45.0,
            "Technical & Methodological": 35.0,
            "Assertive / SOTA Claims": 10.0,
            "Cautious & Critical": 10.0
        }

    counts = {
        "Informative & Objective": 0,
        "Technical & Methodological": 0,
        "Assertive / SOTA Claims": 0,
        "Cautious & Critical": 0
    }

    for c in chunks:
        txt = c["text"].lower()
        s_inf = sum(len(re.findall(p, txt)) for p in TONE_INFORMATIVE)
        s_tech = sum(len(re.findall(p, txt)) for p in TONE_TECHNICAL)
        s_ass = sum(len(re.findall(p, txt)) for p in TONE_ASSERTIVE)
        s_cau = sum(len(re.findall(p, txt)) for p in TONE_CAUTIOUS)

        scores = [
            ("Informative & Objective", s_inf),
            ("Technical & Methodological", s_tech),
            ("Assertive / SOTA Claims", s_ass),
            ("Cautious & Critical", s_cau)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        if scores[0][1] == 0:
            counts["Informative & Objective"] += 1
        else:
            counts[scores[0][0]] += 1

    total = max(sum(counts.values()), 1)
    return {k: round((v / total) * 100, 1) for k, v in counts.items()}


# ─── 3. Reading-Order Tone Trajectory Spline Flow ─────────────
def compute_paper_trajectory(paper_name: str, max_points: int = 35) -> pd.DataFrame:
    """
    Constructs a sequential sentiment & tone trajectory across reading order (chunk by chunk),
    reflecting how author rhetoric shifts through sections.
    """
    chunks = get_all_chunks(paper_name=paper_name)
    if not chunks:
        return pd.DataFrame()

    total_c = len(chunks)
    step = max(1, total_c // max_points)
    sampled_indices = list(range(0, total_c, step))[:max_points]

    tone_weight_map = {
        "Cautious & Critical": -0.65,
        "Informative & Objective": 0.15,
        "Technical & Methodological": 0.45,
        "Assertive / SOTA Claims": 0.90
    }

    rows = []
    for idx in sampled_indices:
        c = chunks[idx]
        txt = c["text"].lower()
        sec = c.get("section", "General") or "General"
        page = c.get("page", 1)

        s_inf = sum(len(re.findall(p, txt)) for p in TONE_INFORMATIVE)
        s_tech = sum(len(re.findall(p, txt)) for p in TONE_TECHNICAL)
        s_ass = sum(len(re.findall(p, txt)) for p in TONE_ASSERTIVE)
        s_cau = sum(len(re.findall(p, txt)) for p in TONE_CAUTIOUS)

        scores = [
            ("Informative & Objective", s_inf),
            ("Technical & Methodological", s_tech),
            ("Assertive / SOTA Claims", s_ass),
            ("Cautious & Critical", s_cau)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        chosen_tone = scores[0][0] if scores[0][1] > 0 else "Informative & Objective"
        raw_score = tone_weight_map.get(chosen_tone, 0.0)

        preview = c["text"][:140].replace("\n", " ").strip() + "..."

        rows.append({
            "chunk_index": idx + 1,
            "section": sec,
            "page": page,
            "tone_label": chosen_tone,
            "tone_score": raw_score,
            "text_preview": preview
        })

    df = pd.DataFrame(rows)
    # Apply 3-point rolling moving average for spline smoothness
    if len(df) >= 3:
        df["smoothed_tone"] = df["tone_score"].rolling(window=3, min_periods=1, center=True).mean().round(2)
    else:
        df["smoothed_tone"] = df["tone_score"]

    return df


# ─── 4. Per-Document Section Breakdown ────────────────────────
def extract_paper_section_breakdown(paper_name: str) -> dict[str, int]:
    """
    Extracts chunk distribution across detected academic sections.
    """
    chunks = get_all_chunks(paper_name=paper_name)
    if not chunks:
        return {}

    sec_counts: dict[str, int] = {}
    for c in chunks:
        sec = c.get("section", "General") or "General"
        sec_counts[sec] = sec_counts.get(sec, 0) + 1

    return dict(sorted(sec_counts.items(), key=lambda x: x[1], reverse=True))


# ─── 5. Per-Document Thematic Concepts & Keyphrases ──────────
def extract_top_concepts(paper_name: str, top_n: int = 12) -> list[tuple[str, int]]:
    """
    Extracts high-frequency scientific terms, filtering common stopwords.
    """
    chunks = get_all_chunks(paper_name=paper_name)
    if not chunks:
        return []

    words = []
    for c in chunks:
        cleaned = re.findall(r"\b[a-zA-Z]{4,20}\b", c["text"].lower())
        words.extend([w for w in cleaned if w not in ACADEMIC_STOPWORDS])

    if not words:
        return []

    freq = pd.Series(words, dtype="string").value_counts()
    return list(freq.head(top_n).items())


# ─── 6. Multi-Document Comparative Matrix ─────────────────────
def compute_corpus_comparison() -> pd.DataFrame:
    """
    Generates a side-by-side comparative table summarizing every indexed document.
    """
    papers = get_indexed_papers()
    if not papers:
        return pd.DataFrame()

    records = []
    for p in papers:
        profile = analyze_paper_bias(p)
        stance = compute_paper_stance(p)
        dom_stance = max(stance.items(), key=lambda x: x[1])[0]

        records.append({
            "Document": p,
            "Quality Index": f"{profile.get('quality_score', 0)}/100",
            "Chunks": profile.get("total_chunks", 0),
            "Pages": profile.get("total_pages", 0),
            "Est. Words": f"{profile.get('total_words', 0):,}",
            "Objectivity": f"{profile.get('objectivity_score', 0)}/100",
            "Rigor Score": f"{profile.get('empirical_rigor', 0)}/100",
            "Baselines Cited": profile.get("baseline_count", 0),
            "Limitations": profile.get("limitation_count", 0),
            "Dominant Tone": dom_stance,
        })

    return pd.DataFrame(records)


# ─── 7. Pairwise Cosine Similarity Matrix ─────────────────────
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


# ─── 8. Dynamic Inquiry & Discourse Semantic Alignment ────────
def analyze_inquiry_semantic_alignment(paper_name: str, chat_history: list[dict]) -> dict:
    """
    Dynamically tracks user inquiries against a specific paper's content,
    evaluating section scrutiny, stance tone, and discourse tension.
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
            "discourse_tension": 25,
            "tension_label": "Calm Exploratory Discourse",
            "tension_color": "#10b981",
            "top_queried_terms": [],
            "agent_invocations": {"Debate": 0, "Comparison": 0, "Prediction": 0, "DirectQA": 0}
        }

    sections_map = {}
    for c in chunks:
        sec = c.get("section", "General") or "General"
        sections_map.setdefault(sec, []).append(c.get("text", "").lower())

    agent_counts = {"Debate": 0, "Comparison": 0, "Prediction": 0, "DirectQA": 0}
    tone_counts = {"Adversarial": 0, "Technical": 0, "Comparative": 0, "Exploratory": 0}
    query_words = []
    cmd_words = {"debate", "adversarial", "compare", "diff", "contrast", "predict", "future", "workflow", "summarize"}

    for q in user_queries:
        low_q = q.lower()
        q_clean = re.findall(r"\b[a-zA-Z]{3,20}\b", low_q)
        query_words.extend([w for w in q_clean if w not in ACADEMIC_STOPWORDS and w not in cmd_words and not w.startswith("/")])

        if low_q.startswith("/debate") or low_q.startswith("/adversarial"):
            agent_counts["Debate"] += 1
            tone_counts["Adversarial"] += 2
        elif low_q.startswith("/compare") or low_q.startswith("/diff") or low_q.startswith("/contrast"):
            agent_counts["Comparison"] += 1
            tone_counts["Comparative"] += 2
        elif low_q.startswith("/predict") or low_q.startswith("/future"):
            agent_counts["Prediction"] += 1
            tone_counts["Technical"] += 1
        elif low_q.startswith("/workflow"):
            agent_counts["DirectQA"] += 1
            tone_counts["Technical"] += 2
        else:
            agent_counts["DirectQA"] += 1

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
        n_sec = max(len(sections_map), 1)
        section_scrutiny = {k: round(100.0 / n_sec, 1) for k in sections_map}

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
