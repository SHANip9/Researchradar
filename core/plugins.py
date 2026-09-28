"""
core/plugins.py — Specialized Research Agent Plugins for ResearchRadar.
1. DebateAgent: Adversarial analysis, for & against arguments, claim scrutiny.
2. ComparisonAgent: Multi-paper differentiation, methodology & tradeoff contrast.
3. PredictionAgent: In-depth technical dissection, theoretical limits & future predictions.
4. WorkflowTableAgent: Visual architecture pipeline extraction (Mermaid) & tabular ablation dissection.
"""

from core.engine import hybrid_search, generate_llm_response, get_all_chunks


class DebateAgent:
    """
    Agent Plugin 1: Debate & Adversarial Sparring
    Examines paper claims from both sides (Strongest Arguments For vs. Critical Objections Against),
    identifies hidden assumptions, and stress-tests findings.
    """
    SYSTEM_PROMPT = """You are an elite academic peer reviewer and debate referee.
Your mission is to rigorously evaluate research papers from both sides:
1. ARGUMENTS FOR (Strengths, empirical validation, sound methodology).
2. ARGUMENTS AGAINST (Hidden assumptions, lack of baselines, edge cases, vulnerability to bias).
3. THE VERDICT (Where does the paper stand under adversarial pressure?).

STRICT RULES:
- Ground every single point in the provided research paper passages.
- Cite direct page numbers and exact quotes where possible.
- Never accept claims blindly; probe for potential methodological flaws."""

    @staticmethod
    def run(topic_or_claim: str, posture: str = "Adversarial Reviewer", paper_filter: str | None = None) -> dict:
        """Runs the debate agent on a specific topic or paper."""
        query = f"evidence strengths weaknesses limitations assumptions {topic_or_claim}"
        chunks = hybrid_search(query, top_k=8, paper_filter=paper_filter)
        
        user_prompt = f"""Conduct a formal academic debate from the perspective of an {posture} on:
"{topic_or_claim}"

Structure your response into:
1. 🥊 **The Case FOR (Core Defenses & Proven Claims)**
2. 🛡️ **The Case AGAINST (Key Vulnerabilities, Hidden Assumptions & Limitations)**
3. ⚖️ **Adversarial Sparring Q&A (3 Skeptical Questions a Reviewer would ask & How the paper holds up)**
4. 🏁 **Empirical Verdict & Robustness Rating (1-10)**"""

        return generate_llm_response(DebateAgent.SYSTEM_PROMPT, user_prompt, chunks)


class ComparisonAgent:
    """
    Agent Plugin 2: Multi-Paper Differentiation
    Compares 2 or more research papers side-by-side:
    highlights divergence in methodology, dataset differences, trade-offs, and points of agreement/conflict.
    """
    SYSTEM_PROMPT = """You are a senior meta-analyst and comparative research synthesizer.
Your goal is to contrast and differentiate multiple research papers with high precision:
- Compare architectural and methodological paradigms.
- Contrast datasets, benchmarks, and evaluation metrics.
- Expose direct contradictions or disagreements between authors.
- Identify the Pareto frontier (which paper wins on accuracy, speed, compute, or generalizability).

STRICT RULES:
- Attribute every claim to its exact paper source and page number.
- Use structured comparison tables and bullet points."""

    @staticmethod
    def run(focus_question: str, papers: list[str], dimension: str = "Methodology & Benchmarks") -> dict:
        """Compares multiple papers across the focus question."""
        all_selected_chunks = []
        for p in papers:
            chunks = hybrid_search(f"{dimension} {focus_question}", top_k=4, paper_filter=p)
            all_selected_chunks.extend(chunks)

        user_prompt = f"""Differentiate and contrast the selected research papers: {', '.join(papers)}
Primary Comparison Dimension: {dimension}
Focus Inquiry: "{focus_question}"

Provide a structured comparative meta-analysis:
1. 📋 **Methodology & Architectural Matrix (Side-by-Side comparison table)**
2. ⚔️ **Points of Disagreement & Divergence (Where do their findings conflict?)**
3. 🤝 **Consensus & Overlapping Truths (What do all papers confirm?)**
4. 🏆 **Trade-off Analysis (Efficiency vs Accuracy vs Scalability)**
5. 📌 **Actionable Recommendation (When to choose Paper A vs Paper B)**"""

        return generate_llm_response(ComparisonAgent.SYSTEM_PROMPT, user_prompt, all_selected_chunks)


class PredictionAgent:
    """
    Agent Plugin 3: Deep Dive & Future Scope Prediction
    Conducts in-depth theoretical dissection and projects future research trajectories,
    unaddressed bottlenecks, and 3-5 year industry/academic impact predictions.
    """
    SYSTEM_PROMPT = """You are a pioneering research futurist and chief scientist.
Your objective is to conduct a profound deep dive into the presented research papers, dissect their fundamental mechanics,
and predict their long-term impact:
- Unpack the theoretical mathematics/mechanisms beneath the empirical results.
- Uncover systemic bottlenecks the authors could not solve.
- Forecast the 3 to 5-year evolution and industry applications.
- Propose 3 high-impact follow-up research proposals that must be explored next.

STRICT RULES:
- Base extrapolations strictly on the trajectories and limitations established in the papers.
- Cite specific sections (e.g. Limitations, Future Work, Discussion)."""

    @staticmethod
    def run(topic_or_inquiry: str, horizon: str = "3 to 5-Year Horizon", paper_filter: str | None = None) -> dict:
        """Executes the future trajectory & prediction agent."""
        query = f"limitations future work theoretical mechanism scalability future scope {topic_or_inquiry}"
        chunks = hybrid_search(query, top_k=8, paper_filter=paper_filter)

        user_prompt = f"""Conduct an in-depth technical deep dive and future prediction analysis on:
"{topic_or_inquiry}"
Forecasting Horizon: {horizon}

Structure your response into:
1. 🔬 **Theoretical Deep Dive (The core mechanic & mathematical/systemic intuition)**
2. 🧱 **Unresolved Bottlenecks (What prevents this from scaling infinitely?)**
3. 🔮 **{horizon} Predictions (How will this transform the field & industry?)**
4. 🚀 **Next-Generation Research Agenda (3 concrete follow-up papers someone should write)**"""

        return generate_llm_response(PredictionAgent.SYSTEM_PROMPT, user_prompt, chunks)


class WorkflowTableAgent:
    """
    Agent Plugin 4: Architecture Workflow & Table Dissector
    Extracts multi-step architecture pipelines, generates visual Mermaid flowcharts,
    and conducts deep analytical dissections of experimental tables and ablation studies.
    """
    SYSTEM_PROMPT = """You are an elite research systems architect and quantitative data analyst.
Your objective is to extract the paper's physical/computational workflow and experimental tables:
1. Map the exact multi-step architecture pipeline into a clean, valid Mermaid flowchart syntax block (```mermaid graph TD ... ```).
2. Extract the primary experimental tables, benchmark metrics, and ablation results into clean Markdown tables.
3. Provide a rigorous quantitative dissection: Which parameter changes gave the largest delta? Where did the methodology plateau?

STRICT RULES:
- Ensure the Mermaid code is completely valid, syntactically clean, and enclosed in ```mermaid ... ```.
- Format all numeric results in clear Markdown tables with columns: [Method/Variant | Benchmark/Dataset | Metric Value | Delta/Notes].
- Ground all numbers and pipeline stages strictly in the retrieved text."""

    @staticmethod
    def run(focus_aspect: str, mode: str = "Full Workflow & Tables", paper_filter: str | None = None) -> dict:
        """Extracts architecture workflow and table comparisons."""
        query = f"table results benchmark ablation architecture pipeline framework steps {focus_aspect}"
        chunks = hybrid_search(query, top_k=8, paper_filter=paper_filter)

        user_prompt = f"""Extract the architecture workflow and table experimental results for:
"{focus_aspect}"
Extraction Focus: {mode}

Structure your response into:
1. 🔄 **System Architecture Workflow (Visual Mermaid Diagram)**:
   Include a complete, valid ```mermaid ... ``` flowchart depicting the data flow from input to output.
2. 📊 **Experimental Results & Ablation Table**:
   Construct a clean Markdown table comparing baseline models vs the proposed approach with exact numbers and metrics.
3. 🔍 **Quantitative Findings & Trade-off Dissection**:
   Analyze what the numbers reveal: compute trade-offs, accuracy gains, and points of diminishing returns."""

        return generate_llm_response(WorkflowTableAgent.SYSTEM_PROMPT, user_prompt, chunks)
