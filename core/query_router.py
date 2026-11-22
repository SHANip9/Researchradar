"""
core/query_router.py — Slash Command Parsing & Intelligent RAG Routing.

Parses user queries for slash commands:
  - /debate <user_take>: Adversarially takes the opposite stance against user
  - /workflow <aspect>: Reconstructs system architecture workflows & tables
  - /compare <topic A vs topic B>: Contrasts differences between topics/papers
  - /summarize: Generates complete structured summary of specified paper
  - /predict: 3-5 year technical forecasts
  - Direct query: Grounded precision research Q&A with 0 hallucination

Includes automatic detection and extraction of mentioned PDF paper names.
"""

import re
from core.plugins import (
    DebateAgent, ComparisonAgent, PredictionAgent,
    WorkflowTableAgent, SummaryAgent
)
from core.retriever import hybrid_search
from core.llm_handler import generate_llm_response
from core.logger import get_logger

log = get_logger("query_router")


def detect_mentioned_paper(query_text: str, papers_in_db: list[str]) -> tuple[str | None, str]:
    """
    Scans the user query for mentions of any indexed paper names or titles.
    Returns:
        (matched_paper_filename_or_None, cleaned_query_without_paper_noise)
    """
    if not papers_in_db:
        return None, query_text

    cleaned_query = query_text
    best_match = None
    longest_match_len = 0

    for paper in papers_in_db:
        # Check raw filename match
        base_name = paper.replace(".pdf", "")
        # Remove common symbols for flexible matching
        simplified_base = re.sub(r"[_\-()]", " ", base_name).strip()
        
        # Check against full filename, base name, or words
        pattern = re.compile(re.escape(paper), re.IGNORECASE)
        if pattern.search(query_text):
            if len(paper) > longest_match_len:
                best_match = paper
                longest_match_len = len(paper)
            continue

        pattern_base = re.compile(re.escape(base_name), re.IGNORECASE)
        if pattern_base.search(query_text):
            if len(base_name) > longest_match_len:
                best_match = paper
                longest_match_len = len(base_name)
            continue

        # Check for substantial title match
        words = [w for w in simplified_base.split() if len(w) > 3]
        if len(words) >= 2:
            title_sample = " ".join(words[:4])
            if re.search(re.escape(title_sample), query_text, re.IGNORECASE):
                if len(title_sample) > longest_match_len:
                    best_match = paper
                    longest_match_len = len(title_sample)

    return best_match, cleaned_query


def parse_and_route_query(user_query: str, active_filter: str | None, papers_in_db: list[str]) -> dict:
    """
    Routes query based on slash commands or defaults to grounded RAG.
    Detects explicitly mentioned paper names in the query string.
    Supported commands:
      /debate <user_take>
      /compare <topic 1 vs topic 2>
      /workflow <aspect>
      /summarize <paper_name or aspect>
      /predict <inquiry>
    """
    trimmed = user_query.strip()
    low = trimmed.lower()

    # Step 1: Detect if a specific paper was mentioned in the prompt
    detected_paper, _ = detect_mentioned_paper(trimmed, papers_in_db)
    effective_paper_filter = active_filter or detected_paper

    # Step 2: Route /debate (Takes opposite side of user)
    if low.startswith("/debate") or low.startswith("/adversarial"):
        topic = re.sub(r"^/(?:debate|adversarial)\s*", "", trimmed, flags=re.IGNORECASE).strip()
        if not topic:
            topic = "The paper's claims are completely valid, scalable, and superior to all existing baselines."
        res = DebateAgent.run(topic, target_paper=effective_paper_filter)
        return {
            "agent": "debate",
            "agent_title": "Adversarial Debate Agent (Opposite Stance)",
            "agent_icon": "🥊",
            "badge_class": "agent-badge-debate",
            "content": res["content"],
            "citations": res.get("citations", [])
        }

    # Step 3: Route /compare (Compares differences between 2 topics or papers)
    if low.startswith("/compare") or low.startswith("/diff") or low.startswith("/contrast"):
        focus = re.sub(r"^/(?:compare|diff|contrast)\s*", "", trimmed, flags=re.IGNORECASE).strip()
        if not focus:
            focus = "Compare core architectural paradigms, empirical performance differences, and computational trade-offs."
        res = ComparisonAgent.run(focus, target_paper=effective_paper_filter)
        return {
            "agent": "compare",
            "agent_title": "Comparative Differentiation Agent",
            "agent_icon": "⚖️",
            "badge_class": "agent-badge-compare",
            "content": res["content"],
            "citations": res.get("citations", [])
        }

    # Step 4: Route /workflow (Generates paper workflows & tables)
    if low.startswith("/workflow") or low.startswith("/architecture") or low.startswith("/pipeline") or low.startswith("/table"):
        focus = re.sub(r"^/(?:workflow|architecture|pipeline|table)\s*", "", trimmed, flags=re.IGNORECASE).strip()
        if not focus:
            focus = "Complete system architecture workflow and experimental ablation metrics table"
        res = WorkflowTableAgent.run(focus, target_paper=effective_paper_filter)
        return {
            "agent": "workflow",
            "agent_title": "Architecture & Workflow Agent",
            "agent_icon": "🔄",
            "badge_class": "agent-badge-predict",
            "content": res["content"],
            "citations": res.get("citations", [])
        }

    # Step 5: Route /summarize (Generates complete structured paper summary)
    if low.startswith("/summarize") or low.startswith("/summeris") or low.startswith("/summary"):
        focus = re.sub(r"^/(?:summarize|summeris|summary)\s*", "", trimmed, flags=re.IGNORECASE).strip()
        res = SummaryAgent.run(focus, target_paper=effective_paper_filter)
        return {
            "agent": "summary",
            "agent_title": "Executive Research Summary Agent",
            "agent_icon": "📑",
            "badge_class": "agent-badge-qa",
            "content": res["content"],
            "citations": res.get("citations", [])
        }

    # Step 6: Route /predict (3-5 year trajectories)
    if low.startswith("/predict") or low.startswith("/future"):
        topic = re.sub(r"^/(?:predict|future)\s*", "", trimmed, flags=re.IGNORECASE).strip()
        if not topic:
            topic = "Project theoretical bottlenecks and 3-5 year evolution."
        res = PredictionAgent.run(topic, horizon="3 to 5-Year Evolutionary Trajectory", paper_filter=effective_paper_filter)
        return {
            "agent": "predict",
            "agent_title": "Future Prediction Agent",
            "agent_icon": "🔮",
            "badge_class": "agent-badge-predict",
            "content": res["content"],
            "citations": res.get("citations", [])
        }

    # Step 7: Default Grounded Research Q&A (Near-0 hallucination, strictly relevant to user query)
    chunks = hybrid_search(trimmed, top_k=7, paper_filter=effective_paper_filter)
    system_prompt = """You are ResearchRadar, an exacting, precision research assistant.
CRITICAL ZERO-HALLUCINATION DIRECTIVES:
1. Answer the user's specific question DIRECTLY and THOROUGHLY based ONLY on the provided context passages.
2. Every fact, statistic, and conclusion MUST be directly anchored in the text.
3. For every single claim, cite the source using: (Paper: <Title> | Page: <Page Number>).
4. If the retrieved context does not contain the answer, you must state explicitly: "The uploaded paper(s) do not contain information to address this question." NEVER guess or invent details."""

    res = generate_llm_response(system_prompt, trimmed, chunks)
    return {
        "agent": "qa",
        "agent_title": "Grounded Research Q&A",
        "agent_icon": "🔬",
        "badge_class": "agent-badge-qa",
        "content": res["content"],
        "citations": res.get("citations", [])
    }
