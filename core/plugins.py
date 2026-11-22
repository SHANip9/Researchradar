"""
core/plugins.py — Specialized Research Agent Plugins for ResearchRadar.
1. DebateAgent: Takes the opposite side of the user's take and vigorously debates it using retrieved evidence.
2. ComparisonAgent: Compares the differences between 2 topics or 2 papers provided in the prompt.
3. WorkflowTableAgent: Reconstructs exact system architecture workflows & tables strictly from the uploaded PDF(s).
4. SummaryAgent: Generates comprehensive multi-section executive summaries of the specified paper(s).
5. PredictionAgent: In-depth technical dissection, theoretical limits & future trajectories.
"""

from core.engine import hybrid_search, generate_llm_response, get_all_chunks


class DebateAgent:
    """
    Agent Plugin 1: Adversarial Debate Agent
    Always takes the opposite side of the user's stance/take and rigorously debates
    against the user using grounded counter-arguments, empirical vulnerabilities,
    and citations from the uploaded paper.
    """
    SYSTEM_PROMPT = """You are an elite adversarial academic debater and scientific peer reviewer.
CRITICAL BEHAVIOR:
You must ALWAYS adopt the OPPOSITE side of the user's take, stance, or hypothesis.
- If the user argues that a method/model is effective, scalable, or superior, you MUST argue against it: expose hidden assumptions, failure cases, compute bottlenecks, and empirical limitations from the paper.
- If the user critiques or doubts the paper's findings, you MUST defend the paper: cite its empirical baselines, ablation gains, and rigorous defenses.
- You must engage in a direct, rigorous point-by-point debate challenging the user's perspective.
- Every counter-argument and critique must be grounded strictly in the provided paper context with explicit page citations (Paper: Title | Page: X).
- Never fabricate external facts. If the paper lacks evidence to challenge a specific point, state that limitation clearly."""

    @staticmethod
    def run(user_take: str, target_paper: str | None = None) -> dict:
        """Runs the adversarial debate agent against the user's take."""
        query = f"evidence limitations counterarguments assumptions benchmarks {user_take}"
        chunks = hybrid_search(query, top_k=8, paper_filter=target_paper)
        
        user_prompt = f"""USER'S STANCE / ARGUMENT:
"{user_take}"

TARGET PAPER:
{target_paper if target_paper else 'Uploaded Literature'}

INSTRUCTIONS:
Take the OPPOSITE side of the user's argument above. Formulate a rigorous, intellectually robust rebuttal.
Structure your debate response into:
1. 🥊 **The Counter-Thesis (Adopting the Direct Opposite Position)**
2. 🛡️ **Empirical Rebuttals & Contradictions (Dissecting the User's Claims with Paper Passages)**
3. 🔍 **Hidden Assumptions & Vulnerabilities Exposed**
4. ⚖️ **Adversarial Cross-Examination (3 Probing Questions Challenging the User's View)**
5. 🏁 **Debate Conclusion & Empirical Strength Rating (1–10)**"""

        return generate_llm_response(DebateAgent.SYSTEM_PROMPT, user_prompt, chunks)


class ComparisonAgent:
    """
    Agent Plugin 2: Comparative Differentiation Agent
    Compares the differences between 2 topics, paradigms, or methods specified by the user
    strictly based on the uploaded paper(s).
    """
    SYSTEM_PROMPT = """You are a senior meta-analyst and comparative research synthesizer.
Your goal is to contrast and differentiate the two topics, methods, or papers provided by the user with high precision:
- Compare architectural and methodological paradigms.
- Contrast datasets, benchmarks, evaluation metrics, and performance deltas.
- Expose direct contradictions, differences in computational cost, latency, and accuracy.
- Identify the Pareto frontier (which method or approach wins on which dimension).

STRICT RULES:
- Base every single comparison point strictly on the provided research paper passages.
- Cite the source paper title and exact page number for every claim.
- If one or both topics are not addressed in the uploaded papers, clearly state: "The uploaded paper(s) do not contain information regarding [Topic]." Do not hallucinate."""

    @staticmethod
    def run(comparison_prompt: str, target_paper: str | None = None) -> dict:
        """Compares differences between topics or papers."""
        query = f"compare contrast difference trade-offs evaluation baseline {comparison_prompt}"
        chunks = hybrid_search(query, top_k=8, paper_filter=target_paper)

        user_prompt = f"""USER COMPARISON INQUIRY:
"{comparison_prompt}"

INSTRUCTIONS:
Compare the core differences between the topics, approaches, or architectures specified by the user.
Structure your comparative analysis into:
1. 📋 **Comparative Matrix (Side-by-Side Dimension Breakdown Table)**
2. ⚔️ **Key Architectural & Methodological Differences**
3. 📊 **Empirical Performance & Benchmark Divergence**
4. ⚖️ **Trade-off Analysis (Efficiency vs Accuracy vs Resource Cost)**
5. 🏆 **Synthesized Conclusion & Recommendation**"""

        return generate_llm_response(ComparisonAgent.SYSTEM_PROMPT, user_prompt, chunks)


class WorkflowTableAgent:
    """
    Agent Plugin 3: Architecture & Workflow Agent
    Extracts multi-step architecture pipelines, generates visual Mermaid flowcharts,
    and conducts deep dissections of experimental tables and ablation studies strictly from the uploaded paper(s).
    """
    SYSTEM_PROMPT = """You are an elite research systems architect and quantitative data analyst.
Your objective is to reconstruct the system architecture workflow and experimental tables strictly from the uploaded paper(s):
1. Map the exact multi-step computational pipeline into a clean, valid Mermaid flowchart syntax block (```mermaid graph TD ... ```).
2. Extract the primary experimental tables, benchmark metrics, and ablation results into clean Markdown tables.
3. Provide a quantitative dissection: Which pipeline stages or parameter changes yielded the largest delta?

STRICT RULES:
- Every pipeline step, node label, and metric MUST come directly from the uploaded paper(s).
- Ensure the Mermaid code is valid, clean, and enclosed in ```mermaid ... ```.
- Format all numeric results in clear Markdown tables with columns: [Method/Variant | Benchmark/Dataset | Metric Value | Delta/Notes].
- If the requested workflow or table is not in the uploaded papers, say so explicitly."""

    @staticmethod
    def run(focus_aspect: str, target_paper: str | None = None) -> dict:
        """Extracts architecture workflow and table comparisons."""
        query = f"architecture workflow pipeline framework steps methodology algorithm ablation table {focus_aspect}"
        chunks = hybrid_search(query, top_k=8, paper_filter=target_paper)

        user_prompt = f"""USER WORKFLOW REQUEST:
"{focus_aspect}"

INSTRUCTIONS:
Reconstruct the complete system architecture workflow and experimental tables strictly from the uploaded paper(s).
Structure your response into:
1. 🔄 **System Architecture Workflow (Visual Mermaid Diagram)**:
   Include a complete, valid ```mermaid ... ``` flowchart depicting data flow from input to output.
2. 📊 **Experimental & Ablation Metrics Table**:
   Construct a clean Markdown table comparing baseline models vs the proposed approach with exact numbers.
3. 🔍 **Quantitative Findings & Bottleneck Analysis**:
   Explain what the data reveals about system throughput, latency, and algorithmic efficiency."""

        return generate_llm_response(WorkflowTableAgent.SYSTEM_PROMPT, user_prompt, chunks)


class SummaryAgent:
    """
    Agent Plugin 4: Executive Research Summary Agent (/summarize)
    Generates a comprehensive, highly structured executive summary of the entire specified research paper(s).
    """
    SYSTEM_PROMPT = """You are a distinguished research synthesizer and principal scientific reviewer.
Your objective is to generate an authoritative, exhaustive executive summary of the specified research paper(s).

STRICT RULES:
- Ground every single point in the provided research paper passages.
- Provide explicit page citations for every major finding and section.
- Cover all core dimensions: Problem Statement, Novel Contributions, Methodology, Empirical Results, Limitations, and Future Trajectory.
- Never hallucinate external results."""

    @staticmethod
    def run(summary_prompt: str, target_paper: str | None = None) -> dict:
        """Generates comprehensive summary of the paper."""
        query = "abstract introduction contribution methodology results evaluation limitations conclusion"
        if summary_prompt:
            query = f"{summary_prompt} {query}"
        chunks = hybrid_search(query, top_k=10, paper_filter=target_paper)

        paper_name = target_paper or "the uploaded research paper"
        user_prompt = f"""USER SUMMARY INQUIRY:
"{summary_prompt if summary_prompt else f'Provide a comprehensive summary of {paper_name}'}"

INSTRUCTIONS:
Generate an exhaustive, highly structured executive summary of {paper_name} strictly from the retrieved text.
Structure your summary into:
1. 🎯 **Executive Abstract & Core Problem Statement**
2. 💡 **Primary Novel Contributions & Innovations**
3. ⚙️ **Key System Methodology & Architecture**
4. 📈 **Empirical Findings, Benchmarks & Key Metrics**
5. ⚠️ **Documented Limitations & Boundary Conditions**
6. 🏁 **Strategic Takeaway & Research Implications**"""

        return generate_llm_response(SummaryAgent.SYSTEM_PROMPT, user_prompt, chunks)


class PredictionAgent:
    """
    Agent Plugin 5: Future Horizon & Prediction Agent (/predict)
    Conducts in-depth theoretical dissection and projects 3-5 year trajectories based strictly on paper limitations.
    """
    SYSTEM_PROMPT = """You are a pioneering research futurist and chief scientist.
Your objective is to conduct an in-depth theoretical dissection of the research paper and project future research trajectories.
- Unpack theoretical mechanisms beneath the empirical results.
- Uncover systemic bottlenecks documented in the limitation disclosures.
- Forecast 3 to 5-year evolution strictly from established empirical trajectories."""

    @staticmethod
    def run(topic_or_inquiry: str, horizon: str = "3 to 5-Year Horizon", paper_filter: str | None = None) -> dict:
        """Executes future trajectory & prediction analysis."""
        query = f"limitations future work theoretical mechanism scalability future scope {topic_or_inquiry}"
        chunks = hybrid_search(query, top_k=8, paper_filter=paper_filter)

        user_prompt = f"""Conduct an in-depth technical deep dive and future prediction analysis on:
"{topic_or_inquiry}"
Forecasting Horizon: {horizon}

Structure your response into:
1. 🔬 **Theoretical Deep Dive (Core Mechanisms & Systemic Intuition)**
2. 🧱 **Unresolved Bottlenecks (Scaling Ceilings & Hardware Constraints)**
3. 🔮 **{horizon} Predictions (Industry & Academic Impact)**
4. 🚀 **Next-Generation Research Agenda (3 Concrete Follow-up Questions)**"""

        return generate_llm_response(PredictionAgent.SYSTEM_PROMPT, user_prompt, chunks)
