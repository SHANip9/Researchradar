"""
app.py — ResearchRadar: Autonomous Research RAG & Multi-Agent Intelligence System.
Features:
  1. PDF Ingestion & Hybrid RAG Retrieval (ChromaDB + BM25 via RRF).
  2. Four Specialized Agent Plugins:
     - ⚔️ Debate Agent (/debate: For & Against adversarial scrutiny)
     - ⚖️ Comparison Agent (/compare: Multi-paper differentiation & trade-offs)
     - 🔮 Future Prediction Agent (/predict: Theoretical limits & 3-5 year scope)
     - 🔄 Architecture Workflow Agent (/workflow: Mermaid diagrams & ablation dissection)
  3. Integrated Analytical Dashboard:
     - 🛡️ Semantic Bias & Objectivity Inspector
     - 📊 Stance & Tone Distribution
     - 🌐 Pairwise Cosine Similarity Heatmap
     - 🏆 Methodological Rigor Breakdown
  4. Enterprise Infrastructure:
     - 🔒 Strict PDF & Input Security Validation
     - 📈 Real-Time Operation Latency & Health Metrics
     - 🪵 Structured Logging & Memory Tracking
"""

import os
import tempfile
import re
import html
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px

# ─── Core Modules ─────────────────────────────────────────────
from core.ui import inject_clean_css
from core.engine import (
    ingest_paper, get_indexed_papers, get_all_chunks,
    delete_paper, clear_database
)
from core.query_router import parse_and_route_query
from core.analytics import (
    analyze_paper_bias, compute_similarity_matrix,
    compute_paper_stance, extract_top_concepts,
    analyze_inquiry_semantic_alignment
)
from core.security import validate_pdf_upload, sanitize_user_input, ValidationError
from core.metrics import metrics
from core.logger import get_logger
from memory.conversation import ConversationMemory

log = get_logger("app")


def render_content_with_visual_mermaid(text: str):
    """
    Renders markdown text and extracts any ```mermaid ... ``` blocks
    to render them visually as interactive dark-themed flowcharts.
    Uses strict DOM isolation & sanitization.
    """
    mermaid_blocks = re.findall(r"```mermaid\s+(.*?)```", text, re.DOTALL)
    clean_text = re.sub(r"```mermaid\s+.*?```", "", text, flags=re.DOTALL).strip()
    if clean_text:
        st.markdown(clean_text, unsafe_allow_html=True)
    elif not mermaid_blocks:
        st.markdown(text, unsafe_allow_html=True)

    if mermaid_blocks:
        st.markdown("##### 🖼️ Interactive Architecture Diagram")
        for idx, code in enumerate(mermaid_blocks):
            clean_code = code.strip().replace("`", "")
            # Safe HTML template with sandboxed mermaid rendering
            html_code = f"""
            <div id="mermaid-box-{idx}" style="background: #0f172a; padding: 20px; border-radius: 12px; border: 1px solid #1e293b; text-align: center; overflow: auto; margin: 12px 0;">
                <pre class="mermaid" style="background: transparent;">
{clean_code}
                </pre>
            </div>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true, theme: 'dark', securityLevel: 'strict' }});
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
if "deleted_papers" not in st.session_state:
    st.session_state.deleted_papers = set()
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory(max_turns=5)


# ─── Sidebar: Document Ingestion & Management ─────────────────
with st.sidebar:
    st.markdown('<div class="brand-title">🔬 ResearchRadar</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Autonomous Research RAG & Multi-Agent Engine</div>', unsafe_allow_html=True)
    st.divider()

    # File Upload with Security Validation
    st.markdown("### 📥 Ingest Research Papers")
    uploaded_files = st.file_uploader(
        "Upload one or more research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF research papers to build your local vector knowledge base. Enforces size & format security."
    )

    if uploaded_files:
        current_papers = set(get_indexed_papers())
        new_indexed = 0
        for uploaded_file in uploaded_files:
            paper_name = uploaded_file.name
            if paper_name not in current_papers and paper_name not in st.session_state.deleted_papers:
                try:
                    # Security checks: validate file type, size limits & PDF magic bytes
                    validate_pdf_upload(uploaded_file)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    with st.spinner(f"Ingesting & indexing **{paper_name}**..."):
                        try:
                            chunk_count = ingest_paper(
                                pdf_path=tmp_path,
                                paper_name=paper_name,
                                metadata={"title": paper_name.replace(".pdf", "").replace("_", " ").title()}
                            )
                            st.success(f"Indexed **{paper_name}** ({chunk_count} chunks)")
                            new_indexed += 1
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                except ValidationError as ve:
                    st.error(f"Security Alert on {paper_name}: {str(ve)}")
                except Exception as e:
                    log.error("Failed to ingest %s: %s", paper_name, str(e))
                    st.error(f"Error processing {paper_name}: {str(e)}")

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
            st.session_state.conversation_memory.clear()
            st.rerun()

    # Engine Status Info & Metrics
    st.divider()
    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if has_api_key:
        st.caption("🟢 **Hybrid RAG Active (Local Embeddings + Claude Synthesis)**")
    else:
        st.caption("⚡ **Local Grounded RAG Active (On-Device Inference)**")
    
    health = metrics.get_health()
    st.caption(f"⏱️ Uptime: {health['uptime_seconds']}s | Ops: {health['total_requests']}")


# ─── Main Interface ───────────────────────────────────────────
if not papers_in_db:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 40px 20px;">
        <h2 style="color: #38bdf8; margin-bottom: 8px;">Welcome to ResearchRadar</h2>
        <p style="color: #94a3b8; max-width: 650px; margin: 0 auto 24px auto; line-height: 1.6;">
            An autonomous, agentic RAG application built specifically for academic and technical research.
            Upload research PDFs in the sidebar to activate multi-agent intelligence and the dynamic semantic dashboard.
        </p>
        <div style="display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;">
            <div class="citation-tag">🥊 <strong>Debate Agent</strong> (<code>/debate</code>)</div>
            <div class="citation-tag">⚖️ <strong>Differentiation Agent</strong> (<code>/compare</code>)</div>
            <div class="citation-tag">🔮 <strong>Future Scope Agent</strong> (<code>/predict</code>)</div>
            <div class="citation-tag">🔄 <strong>Architecture Agent</strong> (<code>/workflow</code>)</div>
            <div class="citation-tag">🛡️ <strong>Bias & Rigor Inspector</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Tab Navigation ───────────────────────────────────────────
tab_chat, tab_dashboard, tab_telemetry = st.tabs([
    "💬 Research Chat & Agent Console",
    "📊 Dynamic Semantic & Bias Dashboard",
    "📈 System Health & Metrics"
])


# ═════════════════════════════════════════════════════════════
# TAB 1: Unified Research Chat & Agent Console (Slash Commands)
# ═════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### 💬 Research Chat & Agent Console")
    st.caption("Ask questions directly or invoke specialized AI agents with slash commands (`/debate`, `/compare`, `/predict`, `/workflow`).")

    # Ready Prompts bar right above the conversation
    st.markdown("###### ⚡ Ready Prompts (Click to Run):")
    prompt_to_run = None
    rp_cols = st.columns(4)
    with rp_cols[0]:
        if st.button("💡 Summarize Core Findings", use_container_width=True, key="rp_btn_summary"):
            prompt_to_run = "What are the primary findings, core methodology, and experimental results?"
    with rp_cols[1]:
        if st.button("🥊 /debate Scalability Claims", use_container_width=True, key="rp_btn_debate"):
            prompt_to_run = "/debate The proposed architecture achieves superior performance with minimal overhead"
    with rp_cols[2]:
        if st.button("⚖️ /compare Model Trade-offs", use_container_width=True, key="rp_btn_compare"):
            prompt_to_run = "/compare Compare core methodologies, benchmark accuracy, and compute efficiency"
    with rp_cols[3]:
        if st.button("🔄 /workflow System Pipeline", use_container_width=True, key="rp_btn_workflow"):
            prompt_to_run = "/workflow Reconstruct the complete system architecture workflow and experimental ablation metrics"

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
            st.session_state.conversation_memory.clear()
            st.rerun()

    # Chat Display Area — Contains all historical and newly generated messages
    chat_display_area = st.container()

    # Chat Input with Sanitization
    chat_prompt = st.chat_input("Ask a research question or enter /debate, /compare, /predict, /workflow...")

    query_to_process = prompt_to_run or chat_prompt

    with chat_display_area:
        if not st.session_state.chat_history and not query_to_process:
            st.info("💡 **Tip:** Ask a direct research question in the prompt box below or click any of the **Ready Prompts** above.")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                safe_user_content = html.escape(msg["content"])
                st.markdown(f'<div class="user-msg-box">🧑‍💻 <strong>You:</strong><br>{safe_user_content}</div>', unsafe_allow_html=True)
            else:
                badge_class = msg.get("badge_class", "agent-badge-qa")
                agent_icon = msg.get("agent_icon", "🔬")
                agent_title = msg.get("agent_title", "ResearchRadar")
                with st.container(border=True):
                    st.markdown(f'<span class="agent-badge {badge_class}">{agent_icon} {agent_title}</span>', unsafe_allow_html=True)
                    render_content_with_visual_mermaid(msg["content"])
                    if msg.get("citations"):
                        cit_tags = "".join([f'<span class="citation-tag">📄 {cit["paper"]} (p. {cit["page"]} · {cit.get("section", "Section")})</span>' for cit in msg["citations"]])
                        st.markdown(f'<div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid rgba(148, 163, 184, 0.16);"><span style="color: #94a3b8; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Verified Citations:</span><br>{cit_tags}</div>', unsafe_allow_html=True)

        if query_to_process:
            q = sanitize_user_input(query_to_process.strip())
            # 1. Immediately append to state and render user query in place
            st.session_state.chat_history.append({"role": "user", "content": q})
            safe_q = html.escape(q)
            st.markdown(f'<div class="user-msg-box">🧑‍💻 <strong>You:</strong><br>{safe_q}</div>', unsafe_allow_html=True)

            # 2. Immediately execute RAG and render assistant response directly below
            with st.container(border=True):
                with st.spinner("Analyzing indexed documents & coordinating research agent..."):
                    routed = parse_and_route_query(q, active_filter, papers_in_db)
                    st.markdown(f'<span class="agent-badge {routed["badge_class"]}">{routed["agent_icon"]} {routed["agent_title"]}</span>', unsafe_allow_html=True)
                    render_content_with_visual_mermaid(routed["content"])
                    if routed.get("citations"):
                        cit_tags = "".join([f'<span class="citation-tag">📄 {cit["paper"]} (p. {cit["page"]} · {cit.get("section", "Section")})</span>' for cit in routed["citations"]])
                        st.markdown(f'<div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid rgba(148, 163, 184, 0.16);"><span style="color: #94a3b8; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Verified Citations:</span><br>{cit_tags}</div>', unsafe_allow_html=True)

            # 3. Save assistant response into session state and conversation memory
            st.session_state.chat_history.append({
                "role": "assistant",
                "agent": routed["agent"],
                "agent_title": routed["agent_title"],
                "agent_icon": routed["agent_icon"],
                "badge_class": routed["badge_class"],
                "content": routed["content"],
                "citations": routed.get("citations", [])
            })
            st.session_state.conversation_memory.add_turn(
                user_query=q,
                response_content=routed["content"],
                agent=routed["agent"]
            )


# ═════════════════════════════════════════════════════════════
# TAB 2: Dynamic Semantic & Bias Dashboard
# ═════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("### 📊 Dynamic Research & Discourse Analytics")
    st.caption("Algorithmic inspection of author objectivity, empirical rigor, and live semantic scrutiny reflecting your questions & agent sparring.")

    all_chunks = get_all_chunks()
    total_docs = len(papers_in_db)
    total_c = len(all_chunks)

    selected_dash_paper = st.selectbox("Select Focus Paper to Inspect:", papers_in_db, key="dash_paper_select")

    # Bias and dynamic inquiry analysis
    bias_profiles = {p: analyze_paper_bias(p) for p in papers_in_db}
    profile = bias_profiles.get(selected_dash_paper, {})
    avg_objectivity = int(sum(b.get("objectivity_score", 50) for b in bias_profiles.values()) / max(total_docs, 1))
    
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

    # Section 1: Live Discourse & Semantic Inquiry Scrutiny
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

    # Section 2: Semantic Bias & Objectivity Inspector
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

    # Section 3: Stance Distribution & Similarity Matrix
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

    # Section 4: Thematic Concepts
    st.markdown("#### 🏷️ Core Thematic Terminology")
    concepts = extract_top_concepts(selected_dash_paper, top_n=10)
    if concepts:
        tags_html = "".join([f'<span class="citation-tag"><strong>{term}</strong> ({cnt})</span>' for term, cnt in concepts])
        st.markdown(f'<div class="glass-card">{tags_html}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TAB 3: System Health & Metrics
# ═════════════════════════════════════════════════════════════
with tab_telemetry:
    st.markdown("### 📈 System Health & Observability Metrics")
    st.caption("Live latency, operational telemetry, and component health monitoring.")

    health_stat = metrics.get_health()
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("System Health", health_stat["status"].upper(), delta="Healthy" if health_stat["status"] == "healthy" else "Degraded")
    with m_col2:
        st.metric("Total Requests", health_stat["total_requests"])
    with m_col3:
        st.metric("App Uptime", f"{health_stat['uptime_seconds']}s")
    with m_col4:
        st.metric("Error Rate", f"{health_stat['error_rate_pct']}%")

    st.markdown("#### ⏱️ Operation Latency Breakdown")
    op_summaries = metrics.get_op_summary()
    if op_summaries:
        summary_rows = [
            {"Operation": k, "Calls": v["count"], "Avg Latency (ms)": v["avg_ms"], "Errors": v["errors"]}
            for k, v in op_summaries.items()
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    else:
        st.info("No recorded operations in this session yet. Run queries or ingest PDFs to observe telemetry.")
