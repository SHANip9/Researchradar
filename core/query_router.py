"""
core/query_router.py — Slash Command Parsing & Intelligent RAG Routing.

Parses user queries for slash commands (/debate, /compare, /predict, /workflow)
and routes them to the appropriate specialized multi-agent plugin or direct RAG pipeline.
"""

import re
from core.plugins import DebateAgent, ComparisonAgent, PredictionAgent, WorkflowTableAgent
from core.retriever import hybrid_search
from core.llm_handler import generate_llm_response
from core.logger import get_logger

log = get_logger("query_router")


def parse_and_route_query(user_query: str, active_filter: str | None, papers_in_db: list[str]) -> dict:
    """
    Routes query based on slash commands or defaults to grounded RAG.
    Supported commands:
      /debate <claim>
      /compare <topic>
      /predict <inquiry>
      /workflow <aspect>
    """
    trimmed = user_query.strip()
    low = trimmed.lower()

    if low.startswith("/debate") or low.startswith("/adversarial"):
        topic = re.sub(r"^/(?:debate|adversarial)\s*", "", trimmed, flags=re.IGNORECASE).strip()
        if not topic:
            topic = "Evaluate the core claims and empirical validity of the paper."
        res = DebateAgent.run(topic, posture="Skeptical Academic Peer Reviewer", paper_filter=active_filter)
        return {
            "agent": "debate",
            "agent_title": "Adversarial Debate Agent",
            "agent_icon": "🥊",
            "badge_class": "agent-badge-debate",
            "content": res["content"],
            "citations": res.get("citations", [])
        }

    if low.startswith("/compare") or low.startswith("/diff") or low.startswith("/contrast"):
        focus = re.sub(r"^/(?:compare|diff|contrast)\s*", "", trimmed, flags=re.IGNORECASE).strip()
        if not focus:
            focus = "Compare their core methodologies, performance tradeoffs, and architectural differences."
        target_papers = papers_in_db[:min(len(papers_in_db), 4)]
        if len(target_papers) < 2:
            res = DebateAgent.run(focus, posture="Objective Scientific Juror", paper_filter=active_filter)
            return {
                "agent": "compare",
                "agent_title": "Comparative Differentiation Agent",
                "agent_icon": "⚖️",
                "badge_class": "agent-badge-compare",
                "content": f"> *Note: Only 1 paper indexed ({target_papers[0] if target_papers else 'Current Paper'}). Running deep-dive trade-off analysis on this paper's internal baselines.*\n\n" + res["content"],
                "citations": res.get("citations", [])
            }
        res = ComparisonAgent.run(focus, papers=target_papers, dimension="Methodology & Model Architecture")
        return {
            "agent": "compare",
            "agent_title": "Comparative Differentiation Agent",
            "agent_icon": "⚖️",
            "badge_class": "agent-badge-compare",
            "content": res["content"],
            "citations": res.get("citations", [])
        }

    if low.startswith("/predict") or low.startswith("/future"):
        topic = re.sub(r"^/(?:predict|future)\s*", "", trimmed, flags=re.IGNORECASE).strip()
        if not topic:
            topic = "Project future scaling, hardware bottlenecks, and 3-5 year evolution."
        res = PredictionAgent.run(topic, horizon="3 to 5-Year Evolutionary Trajectory", paper_filter=active_filter)
        return {
            "agent": "predict",
            "agent_title": "Future Prediction Agent",
            "agent_icon": "🔮",
            "badge_class": "agent-badge-predict",
            "content": res["content"],
            "citations": res.get("citations", [])
        }

    if low.startswith("/workflow") or low.startswith("/architecture") or low.startswith("/pipeline") or low.startswith("/table"):
        focus = re.sub(r"^/(?:workflow|architecture|pipeline|table)\s*", "", trimmed, flags=re.IGNORECASE).strip()
        if not focus:
            focus = "System Architecture workflow and experimental benchmark tables"
        res = WorkflowTableAgent.run(focus, mode="Visual Flowchart & Ablation Dissection", paper_filter=active_filter)
        return {
            "agent": "workflow",
            "agent_title": "Architecture & Workflow Agent",
            "agent_icon": "🔄",
            "badge_class": "agent-badge-predict",
            "content": res["content"],
            "citations": res.get("citations", [])
        }

    # Default: Grounded Q&A
    chunks = hybrid_search(trimmed, top_k=6, paper_filter=active_filter)
    system_prompt = "You are ResearchRadar, a strict research assistant. Answer the user's question directly from the provided chunks with citations."
    res = generate_llm_response(system_prompt, trimmed, chunks)
    return {
        "agent": "qa",
        "agent_title": "Grounded Research Q&A",
        "agent_icon": "🔬",
        "badge_class": "agent-badge-qa",
        "content": res["content"],
        "citations": res.get("citations", [])
    }
