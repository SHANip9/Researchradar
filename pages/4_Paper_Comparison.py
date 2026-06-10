"""
4_Paper_Comparison.py — Fine-Grained Document Comparison Dashboard

This page displays side-by-side paper comparisons, including overall semantic similarity
and the section-to-section similarity heatmap matrix.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─── Internal Modules ──────────────────────────────────────────
from core.storage.vector_store import get_papers_list
from core.analysis.similarity import compute_paper_similarity_matrix, compute_section_similarity_matrix
from core.ui_theme import inject_premium_css

# ─── UI Styling ───────────────────────────────────────────────
inject_premium_css()

# ─── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="ResearchRadar - Paper Comparison",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Navigation Header ────────────────────────────────────────
st.title("🌐 Semantic Document Comparison")
st.write("Compare overall semantic overlaps and drill down into section-level alignment heatmaps.")
st.markdown("---")

# ─── Check Database Status ────────────────────────────────────
papers = get_papers_list()

if len(papers) < 1:
    st.warning("⚠️ **No research papers indexed yet.** Please go to the **Query Chat (Home)** to upload and process research PDFs first.")
    st.stop()

# ─── SELECT PAPERS FOR COMPARISON ──────────────────────────────
st.markdown("<div class='section-title'>🔍 Select Papers to Compare</div>", unsafe_allow_html=True)

col_select1, col_select2 = st.columns(2)

with col_select1:
    paper_a = st.selectbox("Document A", options=papers, index=0)

with col_select2:
    # Set default index to 1 if there are multiple papers
    default_idx = min(1, len(papers) - 1)
    paper_b = st.selectbox("Document B", options=papers, index=default_idx)

# ─── ROW 1: Overall Similarity Gauge / KPI ─────────────────────
col_kpi, col_info = st.columns([1, 1])

# Fetch paper similarity
sim_matrix = compute_paper_similarity_matrix()
overall_score = 0.0

if sim_matrix:
    for row in sim_matrix:
        if (row["paper_1"] == paper_a and row["paper_2"] == paper_b) or \
           (row["paper_1"] == paper_b and row["paper_2"] == paper_a):
            overall_score = row["similarity_score"]
            break

with col_kpi:
    percent_similarity = overall_score * 100
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{percent_similarity:.1f}%</div>
        <div class="stat-label">Semantic Similarity Score</div>
    </div>
    """, unsafe_allow_html=True)


with col_info:
    st.markdown("<div class='section-title'>📌 Semantic Interpretation</div>", unsafe_allow_html=True)
    
    if paper_a == paper_b:
        st.markdown(f"""
        <div class="insight-card" style="border-color: #10b981; background: rgba(16,185,129,0.05)">
            <strong>🔄 Identical Document selected:</strong><br>
            You are comparing <em>{paper_a}</em> with itself. The similarity is naturally 100%.
        </div>
        """, unsafe_allow_html=True)
    elif overall_score > 0.85:
        st.markdown(f"""
        <div class="insight-card" style="border-color: #10b981; background: rgba(16,185,129,0.05)">
            <strong>🟢 Strong Semantic Overlap:</strong><br>
            These papers share a significant conceptual relationship. They likely explore the same algorithms, architectures, or domains with minor differences in methodology.
        </div>
        """, unsafe_allow_html=True)
    elif overall_score > 0.65:
        st.markdown(f"""
        <div class="insight-card" style="border-color: #f59e0b; background: rgba(245,158,11,0.05)">
            <strong>🟡 Moderate Conceptual Overlap:</strong><br>
            These papers share some high-level contexts or methodologies but explore different execution paths or problem statements.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-card" style="border-color: #ef4444; background: rgba(239,68,68,0.05)">
            <strong>🔴 Weak Conceptual Relationship:</strong><br>
            These papers belong to distinct research domains or use fundamentally different techniques. Few concepts overlap.
        </div>
        """, unsafe_allow_html=True)

# ─── ROW 2: Section-to-Section Alignment Heatmap ──────────────
st.markdown("<div class='section-title'>🗺️ Section-to-Section Alignment Heatmap</div>", unsafe_allow_html=True)
st.write("This drill-down compares the sections of Document A (Y-Axis) against the sections of Document B (X-Axis).")

section_similarities = compute_section_similarity_matrix(paper_a, paper_b)

if section_similarities:
    df_sec = pd.DataFrame(section_similarities)
    
    # Pivot to create matrix grid: rows=paper1_section, columns=paper2_section, values=score
    grid_sec = df_sec.pivot_table(
        index="paper1_section",
        columns="paper2_section",
        values="score",
        fill_value=0.0
    )
    
    # Render Plotly Heatmap
    fig_sec = px.imshow(
        grid_sec,
        labels=dict(x=f"Sections of {paper_b}", y=f"Sections of {paper_a}", color="Similarity"),
        color_continuous_scale=[[0, '#eff6ff'], [1, '#1d4ed8']],
        text_auto=".2f", # Show values on the cells!
        template="plotly"
    )
    fig_sec.update_layout(
        font_family="Inter",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=10, b=20),
        height=450,
        xaxis=dict(tickfont=dict(color='#0f172a')),
        yaxis=dict(tickfont=dict(color='#0f172a'))
    )
    st.plotly_chart(fig_sec, use_container_width=True)


    # ─── Alignment Insights ──────────────────────────────────────────
    st.markdown("<div class='section-title'>💡 Primary Structural Alignments</div>", unsafe_allow_html=True)
    
    # Sort section similarities to list top matches (filtering out identity if paper_a=paper_b)
    sorted_sims = df_sec.sort_values(by="score", ascending=False)
    
    alignment_count = 0
    for _, row in sorted_sims.iterrows():
        s1 = row["paper1_section"]
        s2 = row["paper2_section"]
        score = row["score"]
        
        # Skip self-section matches if comparing the same paper to avoid noise
        if paper_a == paper_b and s1 == s2:
            continue
            
        if score > 0.70 and alignment_count < 4:
            st.markdown(
                f"🔗 **{s1}** *(Doc A)* and **{s2}** *(Doc B)* show high semantic similarity of **{score*100:.1f}%**"
            )
            alignment_count += 1
            
    if alignment_count == 0:
        st.info("No notable cross-section alignments found above the 70% threshold.")
else:
    st.info("Section-level similarity is not available (insufficient metadata or sections detected).")
