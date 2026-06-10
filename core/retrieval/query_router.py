"""
query_router.py — Detects whether a question needs Precise or Synthesis retrieval

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY DO WE NEED ROUTING?
    Not all questions are the same:

    "What accuracy did paper 1 report?"
    → PRECISE mode: fetch top-8 chunks globally. One paper, one fact.

    "Compare how all papers approach regularization."
    → SYNTHESIS mode: fetch top-3 chunks PER PAPER. Need views from all papers.

    Routing the question to the right retrieval strategy makes answers
    significantly better without the user having to think about it.

HOW ROUTING WORKS:
    Simple keyword matching. If the question contains synthesis trigger words
    (compare, contrast, which paper, across, etc.) → Synthesis mode.
    Otherwise → Precise mode.

    This is intentionally simple. LLM-based routing would be more accurate but
    adds latency and cost. Keyword matching is fast, transparent, and correct
    for ~90% of research paper questions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─── Synthesis Trigger Keywords ───────────────────────────────
# If ANY of these words appear in the question (case-insensitive),
# the router sends the question to synthesis retrieval mode.
#
# To extend: just add more strings to this list.
SYNTHESIS_TRIGGERS = [
    "compare", "contrast", "comparison",
    "across papers", "across all",
    "which paper", "which papers",
    "all papers", "both papers",
    "disagree", "agree", "agreement", "disagreement",
    "differ", "difference", "differently",
    "consensus", "conflict",
    "versus", " vs ", " vs.", "vs ", "vs.",
    "synthesize", "synthesis",
    "each paper", "every paper",
    "multiple papers",
]


def detect_mode(question: str) -> str:
    """
    Determines the retrieval mode for a given question.

    Args:
        question (str): The user's question in natural language.

    Returns:
        str: "synthesis" if the question compares across papers,
             "precise"   if it asks about specific facts or a single paper.

    Examples:
        detect_mode("What method did paper 1 use?")
        → "precise"

        detect_mode("Compare how all papers handle overfitting")
        → "synthesis"

        detect_mode("Which paper disagrees with the transformer approach?")
        → "synthesis"
    """
    question_lower = question.lower()

    # Check if any synthesis trigger phrase appears in the question
    for trigger in SYNTHESIS_TRIGGERS:
        if trigger in question_lower:
            return "synthesis"

    return "precise"


def route_question(question: str) -> dict:
    """
    Wrapper that returns both the mode and a human-readable explanation.
    Used by app.py to show the user which mode was activated.

    Returns:
        dict: {
            "mode": "precise" | "synthesis",
            "reason": str   ← short explanation for UI display
        }
    """
    mode = detect_mode(question)

    if mode == "synthesis":
        reason = "🔀 Cross-paper synthesis mode — retrieving from all papers"
    else:
        reason = "🎯 Precise mode — retrieving top matches globally"

    return {"mode": mode, "reason": reason}
