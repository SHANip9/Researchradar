"""
app.py — ResearchRadar v2 Main Application

Run with: streamlit run app.py

This is the entry point. It renders:
  • A polished dashboard overview (papers, stats, sentiment at a glance)
  • PDF upload with metadata input (title, source, category, date)
  • A chat-style Q&A interface with cited, grounded answers
  • Sidebar with paper management and session controls
"""

import streamlit as st
import tempfile
import os
import time
import json
from datetime import date

# ─── Internal Modules (v2 reorganized paths) ─────────────────
from core.ingestion.pdf_processor import process_pdf
from core.ingestion.claim_extractor import extract_claims, save_claims
from core.embeddings.embedder import embed_texts
from core.storage.vector_store import (
    add_paper_chunks, get_papers_list,
    delete_paper, clear_all, get_paper_sentiment_data
)
from core.analysis.sentiment import analyze_chunks, aggregate_paper_sentiment
from core.retrieval.retriever import retrieve_precise, retrieve_synthesis
from core.retrieval.query_router import route_question
from core.llm.handler import get_answer
from memory.conversation import ConversationMemory


# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION — must be the first Streamlit command
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchRadar",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── UI STYLING ───────────────────────────────────────────────
from core.ui_theme import inject_premium_css
inject_premium_css()



# ─────────────────────────────────────────────────────────────
# SESSION STATE — persists across Streamlit reruns
# ─────────────────────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_papers" not in st.session_state:
    st.session_state.processed_papers = set()
if "paper_sentiments" not in st.session_state:
    st.session_state.paper_sentiments = {}


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def run_indexing_pipeline(pdf_path: str, paper_name: str, metadata: dict):
    """
    Full ingestion pipeline: PDF → chunks → embed → sentiment → claims → store.
    The metadata dict carries user-provided info (title, source org, category, date).
    """
    # Step 1: Extract and chunk the PDF with enriched metadata
    with st.spinner(f"📄 Extracting text from **{paper_name}**..."):
        chunks = process_pdf(pdf_path, paper_metadata=metadata)

    # Step 2: Generate 384-dimensional vectors for each chunk
    with st.spinner(f"🔢 Generating embeddings ({len(chunks)} chunks)..."):
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

    # Step 3: Run DistilBERT sentiment on every chunk, aggregate to paper-level
    with st.spinner("🧠 Analyzing sentiment..."):
        sentiments = analyze_chunks(chunks)
        paper_sentiment = aggregate_paper_sentiment(sentiments)

    # Step 4: Extract academic claims (for debate engine in Phase 4)
    with st.spinner("🔍 Extracting claims..."):
        claims = extract_claims(chunks)
        if claims:
            claims_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "claims"
            )
            save_claims(claims, metadata.get("paper_title", paper_name), claims_dir)

    # Step 5: Store everything in ChromaDB
    with st.spinner("💾 Storing in ChromaDB..."):
        add_paper_chunks(chunks, embeddings, sentiments)

    # Cache results in session state for sidebar display
    st.session_state.paper_sentiments[paper_name] = paper_sentiment
    st.session_state.processed_papers.add(paper_name)

    st.success(
        f"✅ **{metadata.get('paper_title', paper_name)}** indexed — "
        f"{len(chunks)} chunks | {len(claims)} claims | "
        f"{paper_sentiment['display_label']}"
    )


def run_query_pipeline(question: str) -> dict:
    """Full query pipeline: route → retrieve → LLM → answer with sources."""
    routing = route_question(question)
    mode = routing["mode"]

    if mode == "synthesis":
        chunks = retrieve_synthesis(question)
    else:
        chunks = retrieve_precise(question)

    if not chunks:
        return {
            "answer": "❌ No relevant content found in the uploaded papers for this question.",
            "sources": [],
            "mode": mode,
            "mode_reason": routing["reason"],
        }

    history = st.session_state.memory.get_history()
    result = get_answer(question, chunks, history)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "mode": mode,
        "mode_reason": routing["reason"],
    }


def get_sentiment_badge(sentiment_label: str) -> str:
    """Returns an HTML badge for a sentiment label."""
    badges = {
        "optimistic": '<span class="badge-optimistic">🟢 Optimistic</span>',
        "cautious":   '<span class="badge-cautious">🟡 Cautious</span>',
        "critical":   '<span class="badge-critical">🔴 Critical</span>',
        "neutral":    '<span class="badge-neutral">⚪ Neutral</span>',
    }
    return badges.get(sentiment_label, badges["neutral"])


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 ResearchRadar")
    st.markdown("*Ingest · Analyse · Debate · Visualise*")
    st.divider()

    # ── PDF Upload with Metadata ─────────────────────────────
    st.markdown('<div class="section-header">📁 Upload Papers</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload research papers — any domain works.",
        label_visibility="collapsed",
    )

    # Metadata input — shown only when files are uploaded
    if uploaded_files:
        st.markdown(
            '<div class="upload-info">'
            '💡 Add metadata below for richer analytics and Power BI dashboards.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Expandable metadata section — keeps UI clean by default
        with st.expander("📝 Paper Metadata (optional)", expanded=False):
            meta_title = st.text_input(
                "Paper Title",
                placeholder="e.g. Teaching Claude Why",
                help="Human-readable title. If empty, filename is used.",
            )
            meta_org = st.text_input(
                "Source Organisation",
                placeholder="e.g. Anthropic, OpenAI, DeepMind",
            )
            meta_category = st.selectbox(
                "Category",
                options=[
                    "Uncategorized", "Alignment", "Interpretability",
                    "Policy", "Science", "Economic Research",
                    "Announcements", "Safety", "Other",
                ],
                help="Research category — powers Power BI slicers.",
            )
            meta_date = st.date_input(
                "Publication Date",
                value=date.today(),
                help="When the paper was published.",
            )

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            paper_name = uploaded_file.name

            if paper_name not in st.session_state.processed_papers:
                # Build metadata dict from user input
                metadata = {
                    "paper_title": meta_title.strip() if meta_title.strip() else paper_name.replace(".pdf", ""),
                    "author_org": meta_org.strip(),
                    "category": meta_category if meta_category != "Uncategorized" else "",
                    "date": str(meta_date),
                }

                # Save to temp file (pdfplumber needs a file path, not BytesIO)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    run_indexing_pipeline(tmp_path, paper_name, metadata)
                except Exception as e:
                    st.error(f"❌ Failed to process **{paper_name}**: {str(e)}")
                finally:
                    os.unlink(tmp_path)
            else:
                st.info(f"ℹ️ **{paper_name}** already indexed.")

    st.divider()

    # ── Indexed Papers List ──────────────────────────────────
    st.markdown('<div class="section-header">📚 Indexed Papers</div>', unsafe_allow_html=True)
    papers_in_db = get_papers_list()

    if not papers_in_db:
        st.markdown("*No papers indexed yet.*")
    else:
        for paper in papers_in_db:
            sentiment_data = st.session_state.paper_sentiments.get(paper)

            if not sentiment_data:
                # Load sentiment from DB for papers indexed in previous sessions
                chunk_sentiments = get_paper_sentiment_data(paper)
                sentiment_data = aggregate_paper_sentiment(chunk_sentiments)
                st.session_state.paper_sentiments[paper] = sentiment_data

            badge_html = get_sentiment_badge(sentiment_data.get("paper_sentiment", "neutral"))
            confidence = sentiment_data.get("confidence", 0)

            st.markdown(
                f'<div class="paper-card">'
                f'📄 <strong>{paper}</strong><br>'
                f'{badge_html} &nbsp; <small style="color:#94a3b8;">({confidence:.0%})</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️", key=f"del_{paper}", help=f"Remove {paper}"):
                    delete_paper(paper)
                    st.session_state.paper_sentiments.pop(paper, None)
                    st.session_state.processed_papers.discard(paper)
                    st.rerun()

    st.divider()

    # ── Session Controls ─────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Controls</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()
    with col2:
        if st.button("💥 Clear All", use_container_width=True, help="Remove all papers"):
            clear_all()
            st.session_state.processed_papers = set()
            st.session_state.paper_sentiments = {}
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()

    st.markdown(
        '<div class="app-footer">'
        'ResearchRadar v2.0<br>'
        'Claude · ChromaDB · HuggingFace · BM25'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 24px 0 8px 0;">
    <h1 style="font-size:2.6rem; font-weight:700;
               background: linear-gradient(90deg, #818cf8, #38bdf8, #34d399);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               margin:0;">
        🔬 ResearchRadar
    </h1>
    <p style="color:#94a3b8; font-size:1rem; margin-top:6px;">
        Upload papers · Ask anything · Get grounded, cited answers · Debate ideas
    </p>
</div>
""", unsafe_allow_html=True)


# ── Dashboard Overview (shown when papers are loaded) ────────
if papers_in_db:
    # Compute quick stats for the dashboard cards
    total_papers = len(papers_in_db)
    sentiment_counts = {"optimistic": 0, "cautious": 0, "critical": 0, "neutral": 0}
    total_confidence = 0

    for paper in papers_in_db:
        s = st.session_state.paper_sentiments.get(paper, {})
        label = s.get("paper_sentiment", "neutral")
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
        total_confidence += s.get("confidence", 0)

    avg_confidence = total_confidence / max(total_papers, 1)

    # Count total claims across all papers
    claims_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "claims")
    total_claims = 0
    if os.path.exists(claims_dir):
        for fname in os.listdir(claims_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(claims_dir, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        total_claims += len(data.get("claims", []))
                except Exception:
                    pass


    # Find dominant sentiment
    dominant = max(sentiment_counts, key=sentiment_counts.get)
    dominant_display = {
        "optimistic": "🟢 Optimistic", "cautious": "🟡 Cautious",
        "critical": "🔴 Critical", "neutral": "⚪ Neutral",
    }

    # Render 4 stat cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{total_papers}</div>'
            f'<div class="stat-label">Papers Indexed</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{total_claims}</div>'
            f'<div class="stat-label">Claims Extracted</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{avg_confidence:.0%}</div>'
            f'<div class="stat-label">Avg Confidence</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{dominant_display[dominant]}</div>'
            f'<div class="stat-label">Dominant Sentiment</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")  # Spacer

else:
    # ── Welcome Card (no papers loaded yet) ──────────────────
    st.markdown("""
    <div class="welcome-card">
        <h3 style="color:#a5b4fc; margin-bottom:14px; font-size:1.4rem;">
            Get Started in 3 Steps
        </h3>
        <p style="color:#94a3b8; font-size:0.95rem; line-height:1.7;">
            <strong>1.</strong> Upload one or more research PDFs in the sidebar<br>
            <strong>2.</strong> Wait for indexing to complete (~30-60 seconds per paper)<br>
            <strong>3.</strong> Ask any question below — get cited, grounded answers
        </p>
        <p style="color:#64748b; font-size:0.82rem; margin-top:16px;">
            Works with any domain: AI, medicine, finance, policy, science
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Chat History ─────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-msg">🧑‍💻 <strong>You</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-msg">🔬 <strong>ResearchRadar</strong>'
                f'<br><span class="mode-pill">{message.get("mode_reason", "")}</span>'
                f'<br><br>{message["content"]}</div>',
                unsafe_allow_html=True,
            )
            sources = message.get("sources", [])
            if sources:
                st.markdown("**📎 Sources Used:**")
                for src in sources:
                    badge = get_sentiment_badge(src.get("sentiment", "neutral"))
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <strong>{src["paper"]}</strong> · Page {src["page"]} &nbsp; {badge}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("")


# ── Question Input ───────────────────────────────────────────
st.divider()
papers_loaded = bool(papers_in_db)

with st.form(key="question_form", clear_on_submit=True):
    question_input = st.text_area(
        label="Ask a question",
        placeholder=(
            "e.g. 'What training method does this paper use?' or "
            "'Compare how all papers approach attention mechanisms.'"
            if papers_loaded
            else "Upload papers first, then ask a question..."
        ),
        height=90,
        disabled=not papers_loaded,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([5, 1])
    with col2:
        submit = st.form_submit_button(
            "Ask →",
            use_container_width=True,
            disabled=not papers_loaded,
        )

# ── Handle Question Submission ───────────────────────────────
if submit and question_input.strip():
    question = question_input.strip()

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    with st.spinner("🔍 Searching papers and generating answer..."):
        try:
            result = run_query_pipeline(question)
            st.session_state.memory.add_turn(question, result["answer"])

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "mode_reason": result["mode_reason"],
            })

        except Exception as e:
            error_msg = f"❌ **Error:** {str(e)}"
            if "ANTHROPIC_API_KEY" in str(e):
                error_msg += (
                    "\n\n💡 **Fix:** Create a `.env` file from `.env.example` "
                    "and add your Anthropic API key."
                )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "mode_reason": "⚠️ Error occurred",
            })

    st.rerun()
