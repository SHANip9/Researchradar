"""
3_Debate_Arena.py — Integrated AI Debate Arena

This page provides the user interface for all three debate modes:
- Tab 1: AI vs Paper (Claim Challenger)
- Tab 2: User vs AI (Interactive Sparring Partner)
- Tab 3: Paper A vs Paper B (Versus Mode)
"""

import streamlit as st
import time

# ─── Internal Modules ──────────────────────────────────────────
from core.storage.vector_store import get_papers_list, get_paper_metadata
from core.debate import (
    challenge_paper_claims,
    defend_paper_against_objection,
    defend_papers_against_objection,
    run_paper_versus_debate,
    run_cross_examination
)
from core.ui_theme import inject_premium_css


# ─── UI Styling ───────────────────────────────────────────────
inject_premium_css()

# ─── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="ResearchRadar - Debate Arena",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─── Navigation Header ────────────────────────────────────────

st.title("⚔️ AI Debate Arena")
st.write("Stress-test claims, challenge authors, and pit papers against each other in structured debates.")
st.markdown("---")

# ─── Check Database Status ────────────────────────────────────
papers = get_papers_list()

if not papers:
    st.warning("⚠️ **No research papers indexed yet.** Please go to the **Query Chat (Home)** to upload and process research PDFs first.")
    st.stop()

# ─── TAB NAVIGATION ──────────────────────────────────────────
tab_challenge, tab_defend, tab_versus, tab_crossexam = st.tabs([
    "🔍 AI Challenger (AI vs Paper)",
    "🛡️ Interactive Defender (User vs AI)",
    "🔥 Versus Mode (Paper vs Paper)",
    "🗣️ Cross-Examination (Multi-Paper Panel)"
])

# ─────────────────────────────────────────────────────────────
# TAB 1: AI CHALLENGER
# ─────────────────────────────────────────────────────────────
with tab_challenge:
    st.markdown("<div class='section-title'>🔍 AI Peer Review Critique</div>", unsafe_allow_html=True)
    st.write("AI scans claims extracted during ingestion and acts as a skeptical reviewer to critique them.")
    
    selected_papers_challenge = st.multiselect("Select Documents to Challenge", options=papers, default=papers[:1], key="challenge_select")
    
    if st.button("⚔️ Launch Peer Review Critique", type="primary"):
        if not selected_papers_challenge:
            st.error("Please select at least one document.")
        else:
            for paper in selected_papers_challenge:
                with st.expander(f"📖 Critique for: {paper}", expanded=True):
                    with st.spinner(f"Analyzing claims and generating critiques for {paper}..."):
                        results = challenge_paper_claims(paper)
                        
                    if results:
                        st.success(f"Discovered and challenged {len(results)} claims!")
                        for idx, r in enumerate(results):
                            st.markdown(f"""
                            <div class="critique-card">
                                <div class="claim-header">Claim {idx+1} — Section: {r['section']}</div>
                                <div class="claim-text">"{r['claim_text']}"</div>
                                <div class="claim-header" style="color: #ef4444;">Reviewer Objection</div>
                                <div class="critique-text">{r['critique']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info(f"No claims found in storage for '{paper}'. Try uploading the PDF again to run claim extraction.")

# ─────────────────────────────────────────────────────────────
# TAB 2: INTERACTIVE DEFENDER
# ─────────────────────────────────────────────────────────────
with tab_defend:
    st.markdown("<div class='section-title'>🛡️ Interactive Defense Sparring</div>", unsafe_allow_html=True)
    st.write("Type your objections against one or more papers. AI will defend them using **only** direct quotes and citations.")

    selected_papers_def = st.multiselect("Select Documents to Challenge & Defend", options=papers, default=papers[:1], key="defend_select")
    
    if not selected_papers_def:
        st.warning("Please select at least one paper above to defend.")
    else:
        # Sort to keep session key consistent
        selected_papers_def = sorted(selected_papers_def)
        history_key = f"debate_history_{'_'.join(selected_papers_def)}"
        if history_key not in st.session_state:
            st.session_state[history_key] = []

        col_clr, _ = st.columns([1, 4])
        with col_clr:
            if st.button("🗑️ Clear History", key="clear_defend_history"):
                st.session_state[history_key] = []
                st.rerun()

        # Rerender chat history
        for role, text in st.session_state[history_key]:
            if role == "user":
                st.chat_message("user").write(text)
            else:
                st.chat_message("assistant").write(text)

        # Chat Input
        if user_attack := st.chat_input("Enter objection (e.g. 'Their sample size is too small to generalize')", key="defend_chat_input"):
            # Display user attack
            st.chat_message("user").write(user_attack)
            
            # Convert history for Claude API (maps role -> content)
            api_history = []
            for role, text in st.session_state[history_key]:
                api_history.append((role, text))

            with st.spinner("Analyzing objection and scanning text for defense..."):
                if len(selected_papers_def) == 1:
                    single_paper = selected_papers_def[0]
                    meta = get_paper_metadata(single_paper)
                    source_file = meta.get("source", single_paper)
                    response = defend_paper_against_objection(
                        paper_title=single_paper,
                        source_file=source_file,
                        objection=user_attack,
                        conversation_history=api_history
                    )
                else:
                    sources = []
                    for p in selected_papers_def:
                        meta = get_paper_metadata(p)
                        sources.append(meta.get("source", p))
                    response = defend_papers_against_objection(
                        paper_titles=selected_papers_def,
                        source_files=sources,
                        objection=user_attack,
                        conversation_history=api_history
                    )

            defense_text = response["defense"]
            retrieved_chunks = response["retrieved_chunks"]

            # Append to state
            st.session_state[history_key].append(("user", user_attack))
            st.session_state[history_key].append(("assistant", defense_text))

            # Display AI defense
            st.chat_message("assistant").write(defense_text)

            # Show retrieved citations in an expander
            if retrieved_chunks:
                with st.expander("📚 Retrieved Evidence Chunks", expanded=False):
                    for idx, chunk in enumerate(retrieved_chunks):
                        title = chunk.get("paper_title", chunk.get("source", "unknown"))
                        st.markdown(f"**Chunk {idx+1} | Paper: {title} | Page {chunk['page']} | Section: {chunk.get('section', 'Unknown')}**")
                        st.write(chunk["text"])
                        st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TAB 3: VERSUS MODE
# ─────────────────────────────────────────────────────────────
with tab_versus:
    st.markdown("<div class='section-title'>🔥 Versus Mode (Paper vs Paper)</div>", unsafe_allow_html=True)
    st.write("Choose two papers and a debate topic. The AI will simulate a text-grounded academic debate.")

    col_vs1, col_vs2 = st.columns(2)
    with col_vs1:
        paper_a = st.selectbox("Document A", options=papers, key="vs_a")
    with col_vs2:
        default_b_idx = min(1, len(papers) - 1)
        paper_b = st.selectbox("Document B", options=papers, index=default_b_idx, key="vs_b")

    debate_topic = st.text_input(
        "Debate Topic / Question",
        value="Compare their stances on context length scaling and compute efficiency.",
        placeholder="e.g. Compare how they approach safety alignment and datasets",
        key="vs_topic"
    )

    if st.button("🔥 Start Versus Debate", type="primary"):
        if paper_a == paper_b:
            st.error("❌ Please select two different documents to run a versus debate.")
        else:
            # Resolve source filenames
            meta_a = get_paper_metadata(paper_a)
            meta_b = get_paper_metadata(paper_b)
            
            source_a = meta_a.get("source", paper_a)
            source_b = meta_b.get("source", paper_b)

            with st.spinner("Conducting versus debate (4 rounds)..."):
                transcript = run_paper_versus_debate(
                    title_a=paper_a, source_a=source_a,
                    title_b=paper_b, source_b=source_b,
                    topic=debate_topic
                )

            st.success("Debate complete!")
            st.markdown("---")

            for turn_idx, turn in enumerate(transcript):
                speaker = turn["speaker"]
                speech = turn["speech"]
                
                # Alternate bubble colors
                if speaker == paper_a:
                    bg_color = "rgba(99, 102, 241, 0.04)"
                    border_color = "rgba(99, 102, 241, 0.15)"
                    speaker_color = "#818cf8"
                else:
                    bg_color = "rgba(56, 189, 248, 0.04)"
                    border_color = "rgba(56, 189, 248, 0.15)"
                    speaker_color = "#38bdf8"

                st.markdown(f"""
                <div class="speech-bubble" style="background: {bg_color}; border: 1px solid {border_color}; border-left: 4px solid {speaker_color};">
                    <div class="speaker-name" style="color: {speaker_color};">🗣️ {speaker} (Turn {turn_idx+1})</div>
                    <div>{speech}</div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 4: PANEL CROSS-EXAMINATION
# ─────────────────────────────────────────────────────────────
with tab_crossexam:
    st.markdown("<div class='section-title'>🗣️ Panel Cross-Examination</div>", unsafe_allow_html=True)
    st.write("Cross-examine multiple papers simultaneously. The AI acts as a panel mediator directing side-by-side arguments.")

    selected_papers_exam = st.multiselect("Select Panel Documents", options=papers, default=papers[:min(2, len(papers))], key="exam_select")
    
    if not selected_papers_exam:
        st.warning("Please select at least one paper for the panel.")
    else:
        # Session state for cross-examination chat history
        exam_history_key = f"exam_history_{'_'.join(sorted(selected_papers_exam))}"
        if exam_history_key not in st.session_state:
            st.session_state[exam_history_key] = []
            
        col_ex_clr, _ = st.columns([1, 4])
        with col_ex_clr:
            if st.button("🗑️ Clear Panel Chat", key="clear_exam"):
                st.session_state[exam_history_key] = []
                st.rerun()
                
        # Rerender conversation history
        for role, text in st.session_state[exam_history_key]:
            if role == "user":
                st.markdown(f'<div class="user-msg">🧑‍💻 <strong>Objection/Question</strong><br>{text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="speech-bubble" style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-left: 4px solid #818cf8;">'
                            f'<div class="speaker-name" style="color: #818cf8;">🗣️ Panel Mediator & Discussion Summary</div>'
                            f'<div>{text}</div>'
                            f'</div>', unsafe_allow_html=True)
                            
        # Chat Input for panel Q&A
        if exam_question := st.chat_input("Ask a comparative question (e.g. 'How do your methodologies compare?')", key="exam_input_chat"):
            st.markdown(f'<div class="user-msg">🧑‍💻 <strong>Objection/Question</strong><br>{exam_question}</div>', unsafe_allow_html=True)
            
            # Form conversation history for API
            api_history = []
            for role, text in st.session_state[exam_history_key]:
                api_history.append((role, text))
                
            with st.spinner("Convening panel and gathering answers..."):
                response_data = run_cross_examination(
                    paper_titles=selected_papers_exam,
                    question=exam_question,
                    conversation_history=api_history
                )
                
            response_text = response_data["response"]
            sources = response_data["sources"]
            retrieved_chunks = response_data["retrieved_chunks"]
            
            st.session_state[exam_history_key].append(("user", exam_question))
            st.session_state[exam_history_key].append(("assistant", response_text))
            
            st.markdown(f'<div class="speech-bubble" style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-left: 4px solid #818cf8;">'
                        f'<div class="speaker-name" style="color: #818cf8;">🗣️ Panel Mediator & Discussion Summary</div>'
                        f'<div>{response_text}</div>'
                        f'</div>', unsafe_allow_html=True)
                        
            # Show citations and sources
            if sources:
                st.markdown("**📎 Sources Cited:**")
                cols = st.columns(min(4, len(sources)))
                for idx, src in enumerate(sources):
                    col_idx = idx % len(cols)
                    with cols[col_idx]:
                        badge = {"optimistic": "🟢", "cautious": "🟡", "critical": "🔴", "neutral": "⚪"}.get(src["sentiment"], "⚪")
                        st.markdown(f"""
                        <div class="source-card">
                            📄 <strong>{src["paper"]}</strong><br>
                            Page {src["page"]} &nbsp; <span style="font-size:0.75rem;">{badge} {src["sentiment"].upper()}</span>
                        </div>
                        """, unsafe_allow_html=True)
            
            if retrieved_chunks:
                with st.expander("📚 Retrieved Evidence Chunks", expanded=False):
                    for idx, chunk in enumerate(retrieved_chunks):
                        title = chunk.get("paper_title", chunk.get("source", "unknown"))
                        st.markdown(f"**Chunk {idx+1} | Paper: {title} | Page {chunk['page']} | Section: {chunk.get('section', 'Unknown')}**")
                        st.write(chunk["text"])
                        st.markdown("---")

