"""
app.py — ResearchRadar: Streamlined Research RAG & Multi-Agent Intelligence System.
Features:
  1. PDF Ingestion & Hybrid RAG Retrieval (ChromaDB + BM25).
  2. Three Specialized Agent Plugins:
     - ⚔️ Debate Agent (For & Against, adversarial scrutiny)
     - ⚖️ Comparison Agent (Multi-paper differentiation & trade-offs)
     - 🔮 Future Prediction Agent (Deep dive, theoretical limits & 3-5 year scope)
  3. Integrated Analytical Dashboard:
     - 🛡️ Semantic Bias & Objectivity Inspector
     - 📊 Stance & Sentiment Distribution
     - 🌐 Pairwise Cosine Similarity Heatmap
     - 🏆 Methodological Rigor Scorecard
"""

import os
import tempfile
import re
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px

# ─── Core Modules ─────────────────────────────────────────────
from core.ui import inject_clean_css
from core.engine import (
    ingest_paper, get_indexed_papers, get_all_chunks,
    hybrid_search, generate_llm_response, delete_paper, clear_database
)
from core.plugins import DebateAgent, ComparisonAgent, PredictionAgent, WorkflowTableAgent
from core.analytics import (
    analyze_paper_bias, compute_similarity_matrix,
    compute_paper_stance, extract_top_concepts,
    analyze_inquiry_semantic_alignment
)


def parse_and_route_query(user_query: str, active_filter: str | None, papers_in_db: list[str]) -> dict:
    """
    Routes query based on Codex-style slash commands or defaults to direct RAG.
    Supported slash commands:
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


def render_content_with_visual_mermaid(text: str):
    """
    Renders markdown text and extracts any ```mermaid ... ``` blocks
    to render them visually as interactive dark-themed flowcharts.
    """
    mermaid_blocks = re.findall(r"```mermaid\s+(.*?)```", text, re.DOTALL)
    st.markdown(text)

    if mermaid_blocks:
        st.markdown("##### 🖼️ Interactive Architecture Diagram")
        for idx, code in enumerate(mermaid_blocks):
            clean_code = code.strip().replace("`", "")
            html_code = f"""
            <div id="mermaid-box-{idx}" style="background: #0f172a; padding: 20px; border-radius: 12px; border: 1px solid #1e293b; text-align: center; overflow: auto; margin: 12px 0;">
                <pre class="mermaid" style="background: transparent;">
{clean_code}
                </pre>
            </div>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true, theme: 'dark', securityLevel: 'loose' }});
            </script>
            """
            components.html(html_code, height=380, scrolling=True)

# ─── Page Setup ───────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchRadar — Research RAG & Multi-Agent Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
inject_clean_css()

# ─── Session State Initialization ─────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "plugin_history" not in st.session_state:
    st.session_state.plugin_history = []
if "deleted_papers" not in st.session_state:
    st.session_state.deleted_papers = set()


# ─── Sidebar: Document Ingestion & Management ─────────────────
with st.sidebar:
    st.markdown('<div class="brand-title">🔬 ResearchRadar</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Autonomous Research RAG & Multi-Agent Engine</div>', unsafe_allow_html=True)
    st.divider()

    # File Upload
    st.markdown("### 📥 Ingest Research Papers")
    uploaded_files = st.file_uploader(
        "Upload one or more research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF research papers to build your local vector knowledge base."
    )

    if uploaded_files:
        current_papers = set(get_indexed_papers())
        new_indexed = 0
        for uploaded_file in uploaded_files:
            paper_name = uploaded_file.name
            if paper_name not in current_papers and paper_name not in st.session_state.deleted_papers:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                with st.spinner(f"Processing & indexing **{paper_name}**..."):
                    try:
                        chunk_count = ingest_paper(
                            pdf_path=tmp_path,
                            paper_name=paper_name,
                            metadata={"title": paper_name.replace(".pdf", "").replace("_", " ").title()}
                        )
                        st.success(f"Indexed **{paper_name}** ({chunk_count} chunks)")
                        new_indexed += 1
                    except Exception as e:
                        st.error(f"Error processing {paper_name}: {str(e)}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        if new_indexed > 0:
            st.rerun()

    st.divider()

    # Active Library
    papers_in_db = get_indexed_papers()
    st.markdown(f"### 📚 Indexed Papers ({len(papers_in_db)})")
    
    if not papers_in_db:
        st.info("No documents indexed yet. Upload a PDF above to begin.")
    else:
        for p in papers_in_db:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"📄 **{p}**")
            with c2:
                if st.button("🗑️", key=f"del_{p}", help=f"Delete {p}"):
                    st.session_state.deleted_papers.add(p)
                    delete_paper(p)
                    st.rerun()

        st.divider()
        if st.button("💥 Reset & Clear All", use_container_width=True):
            clear_database()
            st.session_state.deleted_papers.clear()
            st.session_state.chat_history = []
            st.session_state.plugin_history = []
            st.rerun()

    # Engine Status Info
    st.divider()
    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if has_api_key:
        st.caption("🟢 **Hybrid RAG Active (Local Embeddings + Claude Synthesis)**")
    else:
        st.caption("⚡ **Local Grounded RAG Active (On-Device Inference)**")


# ─── Main Interface ───────────────────────────────────────────
if not papers_in_db:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 40px 20px;">
        <h2 style="color: #38bdf8; margin-bottom: 8px;">Welcome to ResearchRadar</h2>
        <p style="color: #94a3b8; max-width: 650px; margin: 0 auto 24px auto; line-height: 1.6;">
            A streamlined, agentic RAG application built specifically for academic and technical research.
            Upload research PDFs in the sidebar to activate multi-agent intelligence and the dynamic semantic dashboard.
        </p>
        <div style="display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;">
            <div class="citation-tag">🥊 <strong>Debate Agent</strong> (<code>/debate</code>)</div>
            <div class="citation-tag">⚖️ <strong>Differentiation Agent</strong> (<code>/compare</code>)</div>
            <div class="citation-tag">🔮 <strong>Future Scope Agent</strong> (<code>/predict</code>)</div>
            <div class="citation-tag">🛡️ <strong>Bias & Rigor Inspector</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Tab Navigation ───────────────────────────────────────────
tab_chat, tab_dashboard = st.tabs([
    "💬 Research Chat & Agent Console",
    "📊 Dynamic Semantic & Bias Dashboard"
])


# ═════════════════════════════════════════════════════════════
# TAB 1: Unified Research Chat & Agent Console (Slash Commands)
# ═════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### 💬 Research Chat & Agent Console")
    st.caption("Ask questions directly or invoke specialized AI agents with slash commands (`/debate`, `/compare`, `/predict`, `/workflow`).")

    # Quick Command Launcher Chips
    st.markdown("###### ⚡ Quick Slash Commands:")
    chip_cols = st.columns(4)
    with chip_cols[0]:
        if st.button("🥊 /debate", use_container_width=True, help="Adversarial academic sparring & counter-examinations"):
            st.session_state.prefill_query = "/debate The proposed architecture achieves superior performance with minimal overhead"
            st.rerun()
    with chip_cols[1]:
        if st.button("⚖️ /compare", use_container_width=True, help="Contrasts methodologies, trade-offs, and empirical benchmarks"):
            st.session_state.prefill_query = "/compare Compare core methodologies, benchmark accuracy, and compute efficiency"
            st.rerun()
    with chip_cols[2]:
        if st.button("🔮 /predict", use_container_width=True, help="Theoretical deep dive & 3-5 year future projections"):
            st.session_state.prefill_query = "/predict Project theoretical bottlenecks and 3 to 5-year industry trajectories"
            st.rerun()
    with chip_cols[3]:
        if st.button("🔄 /workflow", use_container_width=True, help="Visual architecture pipelines (Mermaid) & ablation dissection"):
            st.session_state.prefill_query = "/workflow Reconstruct the complete system architecture workflow and experimental ablation metrics"
            st.rerun()

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # Scope Filter & Clear Chat
    paper_options = ["All Papers"] + papers_in_db
    c_scope, c_clear = st.columns([4, 1])
    with c_scope:
        selected_filter = st.selectbox("Search Scope", paper_options, index=0, key="qa_scope")
        active_filter = None if selected_filter == "All Papers" else selected_filter
    with c_clear:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Chat Messages Stream
    chat_box = st.container()
    with chat_box:
        if not st.session_state.chat_history:
            st.info("💡 **Tip:** Ask a direct research question or try a slash command like `/debate [claim]`, `/compare [topic]`, or `/workflow` above.")
        
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg-box">🧑‍💻 <strong>You:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                badge_class = msg.get("badge_class", "agent-badge-qa")
                agent_icon = msg.get("agent_icon", "🔬")
                agent_title = msg.get("agent_title", "ResearchRadar")
                
                with st.container(border=True):
                    st.markdown(f'<span class="agent-badge {badge_class}">{agent_icon} {agent_title}</span>', unsafe_allow_html=True)
                    render_content_with_visual_mermaid(msg["content"])
                    if msg.get("citations"):
                        cit_tags = "".join([f'<span class="citation-tag">📄 {cit["paper"]} (p. {cit["page"]} · {cit.get("section", "Section")})</span>' for cit in msg["citations"]])
                        st.markdown(f'<div style="margin-top: 10px;"><strong>Verified Citations:</strong><br>{cit_tags}</div>', unsafe_allow_html=True)

    # Default value if a quick-chip was clicked
    default_input_val = st.session_state.pop("prefill_query", "")

    # Input Form
    with st.form(key="chat_input_form", clear_on_submit=True):
        user_query = st.text_input(
            "Inquiry or slash command:",
            value=default_input_val,
            placeholder="Type your inquiry or enter /debate, /compare, /predict..."
        )
        col_sub1, col_sub2 = st.columns([5, 1])
        with col_sub2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_query.strip():
        q = user_query.strip()
        st.session_state.chat_history.append({"role": "user", "content": q})

        with st.spinner("Processing inquiry and coordinating agent..."):
            routed = parse_and_route_query(q, active_filter, papers_in_db)
            st.session_state.chat_history.append({
                "role": "assistant",
                "agent": routed["agent"],
                "agent_title": routed["agent_title"],
                "agent_icon": routed["agent_icon"],
                "badge_class": routed["badge_class"],
                "content": routed["content"],
                "citations": routed.get("citations", [])
            })
        st.rerun()


# ═════════════════════════════════════════════════════════════
# TAB 2: Dynamic Semantic & Bias Dashboard
# ═════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("### 📊 Dynamic Research & Discourse Analytics")
    st.caption("Algorithmic inspection of author objectivity, empirical rigor, and live semantic scrutiny reflecting your questions & agent sparring.")

    # High-level Metrics
    all_chunks = get_all_chunks()
    total_docs = len(papers_in_db)
    total_c = len(all_chunks)

    selected_dash_paper = st.selectbox("Select Focus Paper to Inspect:", papers_in_db, key="dash_paper_select")

    # Static bias and dynamic inquiry analysis
    bias_profiles = {p: analyze_paper_bias(p) for p in papers_in_db}
    profile = bias_profiles.get(selected_dash_paper, {})
    avg_objectivity = int(sum(b.get("objectivity_score", 50) for b in bias_profiles.values()) / max(total_docs, 1))
    
    # Dynamic Inquiry & Tone Semantic Alignment
    inquiry_data = analyze_inquiry_semantic_alignment(selected_dash_paper, st.session_state.chat_history)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{total_docs}</div><div class="metric-title">Papers Analyzed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{total_c}</div><div class="metric-title">Vector Chunks</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{avg_objectivity}/100</div><div class="metric-title">Corpus Objectivity</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{inquiry_data["discourse_tension"]}/100</div><div class="metric-title">Discourse Tension</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # ── Section 1: Live Discourse & Semantic Inquiry Scrutiny ──
    st.markdown("#### ⚡ Live Inquiry Semantic Scrutiny & Discourse Tension")
    st.caption("Tracks which sections are being probed, your question stance vs paper claims, and agent battle activity.")

    with st.container(border=True):
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            st.markdown(f"""
            <div style="font-size: 1.05rem; font-weight: 700; color: {inquiry_data['tension_color']}; margin-bottom: 6px;">
                {inquiry_data['tension_label']}
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;">
                Discourse Scrutiny Level: <strong>{inquiry_data['discourse_tension']}%</strong>
            </div>
            """, unsafe_allow_html=True)
            st.progress(inquiry_data["discourse_tension"] / 100.0)

        with col_t2:
            inv = inquiry_data["agent_invocations"]
            st.markdown(f"""
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.7;">
                <strong>Session Inquiries:</strong> {inquiry_data['total_inquiries']}<br>
                🥊 <strong>Debates:</strong> {inv['Debate']} | ⚖️ <strong>Compares:</strong> {inv['Comparison']}<br>
                🔮 <strong>Predictions:</strong> {inv['Prediction']}
            </div>
            """, unsafe_allow_html=True)

    col_dyn1, col_dyn2 = st.columns([1, 1])

    with col_dyn1:
        st.markdown("##### 🎯 Dynamic Section Scrutiny Map")
        st.caption("Which academic sections have received the most inquiry attention in this session:")
        sec_scrutiny = inquiry_data.get("section_scrutiny", {})
        if sec_scrutiny:
            df_sec = pd.DataFrame(list(sec_scrutiny.items()), columns=["Section", "Scrutiny Percentage"])
            fig_sec = px.bar(
                df_sec,
                x="Scrutiny Percentage",
                y="Section",
                orientation="h",
                color="Scrutiny Percentage",
                color_continuous_scale=[[0, "#1e3a8a"], [0.5, "#0284c7"], [1.0, "#38bdf8"]],
            )
            fig_sec.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans, sans-serif"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_sec, use_container_width=True)

    with col_dyn2:
        st.markdown("##### 🎭 User & Agent Inquiry Tone Distribution")
        st.caption("Stance classification of your research inquiries and agent prompts:")
        inq_tone = inquiry_data.get("inquiry_tone", {})
        df_inq = pd.DataFrame(list(inq_tone.items()), columns=["Inquiry Stance", "Percentage"])
        fig_inq = px.pie(
            df_inq,
            values="Percentage",
            names="Inquiry Stance",
            hole=0.55,
            color="Inquiry Stance",
            color_discrete_map={
                "Adversarial": "#f43f5e",
                "Technical": "#6366f1",
                "Comparative": "#0284c7",
                "Exploratory": "#10b981"
            }
        )
        fig_inq.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Plus Jakarta Sans, sans-serif"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            showlegend=True
        )
        st.plotly_chart(fig_inq, use_container_width=True)

    # Active queried concepts
    if inquiry_data.get("top_queried_terms"):
        st.markdown("##### 🏷️ Active Inquired Scientific Themes")
        q_tags = "".join([f'<span class="citation-tag"><strong>{t}</strong> ({c})</span>' for t, c in inquiry_data["top_queried_terms"]])
        st.markdown(f'<div class="glass-card" style="margin-bottom: 20px;">{q_tags}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Section 2: Semantic Bias & Objectivity Inspector ──
    st.markdown("#### 🛡️ Paper Objectivity & Methodological Rigor")
    st.caption("Author claim-to-caution ratio, confirmation bias risk, and baseline disclosure.")

    if profile:
        col_b1, col_b2 = st.columns([1, 1])

        with col_b1:
            st.markdown(f"""<div class="glass-card">
<div style="font-weight: 700; font-size: 1.1rem; color: #f8fafc; margin-bottom: 6px;">
Objectivity Rating: <span style="color: {profile['bias_color']};">{profile['objectivity_score']}/100</span>
</div>
<div style="font-size: 0.88rem; color: {profile['bias_color']}; margin-bottom: 14px;">
{profile['bias_risk']}
</div>
<div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 4px;">Hedging & Caution Density:</div>
<div class="bias-meter"><div class="bias-bar-low" style="width: {min(profile['hedging_count'] * 15, 100)}%;"></div></div>
<div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 4px; margin-top: 10px;">Overclaiming / Superlatives Density:</div>
<div class="bias-meter"><div class="bias-bar-high" style="width: {min(profile['overclaim_count'] * 20, 100)}%;"></div></div>
<div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 4px; margin-top: 10px;">Limitation & Failure Case Disclosure:</div>
<div class="bias-meter"><div class="bias-bar-med" style="width: {min(profile['limitation_count'] * 20, 100)}%;"></div></div>
</div>""", unsafe_allow_html=True)

        with col_b2:
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-weight: 700; font-size: 1.1rem; color: #f8fafc; margin-bottom: 12px;">
                    Methodological Rigor Breakdown
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #94a3b8;">Empirical Baselines Referenced:</span>
                    <strong style="color: #38bdf8;">{profile['baseline_count']} occurrences</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #94a3b8;">Explicit Limitation Statements:</span>
                    <strong style="color: #f59e0b;">{profile['limitation_count']} statements</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #94a3b8;">Scientific Hedging Phrases:</span>
                    <strong style="color: #10b981;">{profile['hedging_count']} instances</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #94a3b8;">Estimated Word Count:</span>
                    <strong style="color: #cbd5e1;">~{profile['total_words']:,} words</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # ── Section 3: Stance Distribution & Similarity Matrix ──
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.markdown("#### 🎭 Author Stance & Writing Tone")
        st.caption("Distribution of claims made by the paper authors:")
        stance = compute_paper_stance(selected_dash_paper)
        df_stance = pd.DataFrame(list(stance.items()), columns=["Stance", "Percentage"])
        
        fig_stance = px.pie(
            df_stance,
            values="Percentage",
            names="Stance",
            hole=0.55,
            color="Stance",
            color_discrete_map={
                "Optimistic": "#10b981",
                "Cautious": "#f59e0b",
                "Critical": "#f43f5e",
                "Neutral": "#64748b"
            }
        )
        fig_stance.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Plus Jakarta Sans, sans-serif"),
            margin=dict(l=10, r=10, t=20, b=10),
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig_stance, use_container_width=True)

    with col_chart2:
        st.markdown("#### 🌐 Cross-Paper Semantic Similarity Heatmap")
        sim_df = compute_similarity_matrix()
        if not sim_df.empty and len(papers_in_db) > 1:
            fig_sim = px.imshow(
                sim_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(color="Cosine Similarity")
            )
            fig_sim.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                margin=dict(l=10, r=10, t=20, b=10),
                height=300
            )
            st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.info("Upload 2 or more research papers to compute cross-paper cosine similarity matrix.")

    # ── Section 3: Thematic Concepts ──
    st.markdown("#### 🏷️ Core Thematic Terminology")
    concepts = extract_top_concepts(selected_dash_paper, top_n=10)
    if concepts:
        tags_html = "".join([f'<span class="citation-tag"><strong>{term}</strong> ({cnt})</span>' for term, cnt in concepts])
        st.markdown(f'<div class="glass-card">{tags_html}</div>', unsafe_allow_html=True)
