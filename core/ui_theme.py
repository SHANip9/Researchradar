"""ui_theme.py — Centralized premium light glassmorphic styling engine for ResearchRadar v2.2."""

import streamlit as st

def inject_premium_css():
    """
    Injects a unified light glassmorphism CSS design system across Streamlit pages.
    Includes clean responsive layouts, frosted glass panels, modern typography,
    glowing hover animations, customized inputs/buttons, and speech bubbles.
    """
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ─── Typography & Core Global Colors ─── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0f172a;
    }

    .stApp {
        background: radial-gradient(circle at 10% 20%, #f0f4ff 0%, #f5f3ff 50%, #fafafa 100%);
        color: #334155;
    }

    /* Custom Webkit Scrollbars for Light Mode */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.02);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.2);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.4);
    }

    /* ─── Sidebar Navigation & Panels ─── */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0, 0, 0, 0.06);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ─── General Section Headers ─── */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        color: #4f46e5;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 22px 0 10px 0;
        border-bottom: 1px solid rgba(99, 102, 241, 0.12);
        padding-bottom: 4px;
    }

    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 26px;
        margin-bottom: 16px;
        border-left: 4px solid #4f46e5;
        padding-left: 14px;
        line-height: 1.2;
    }

    /* ─── Premium Frosted Light Glassmorphic Cards ─── */
    .stat-card {
        background: rgba(255, 255, 255, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.65);
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-4px);
        box-shadow: 0 12px 35px rgba(99, 102, 241, 0.1);
    }
    .stat-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.76rem;
        color: #475569;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-top: 8px;
    }

    /* Paper/File Item Cards */
    .paper-card {
        background: rgba(255, 255, 255, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.65);
        border-radius: 12px;
        padding: 14px 16px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #334155;
        transition: all 0.2s ease;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
    }
    .paper-card:hover {
        background: rgba(255, 255, 255, 0.75);
        border-color: rgba(99, 102, 241, 0.2);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.05);
    }

    /* ─── Styled Widgets (Inputs, Dropdowns, Buttons) ─── */
    /* Selectboxes, text inputs, textareas */
    div[data-baseweb="select"], div[data-baseweb="input"], textarea {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        transition: all 0.2s ease;
    }
    div[data-baseweb="select"]:focus-within, div[data-baseweb="input"]:focus-within, textarea:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Ensure text in selectboxes is dark */
    div[data-baseweb="select"] * {
        color: #0f172a !important;
    }

    /* Primary and secondary button designs */
    .stButton > button {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 10px;
        color: #334155;
        padding: 8px 24px;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.9);
        border-color: rgba(0, 0, 0, 0.15);
        color: #0f172a;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    }
    .stButton > button:active {
        transform: scale(0.98);
    }
    
    /* Primary Action Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        border: none;
        color: #ffffff !important;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
        color: #ffffff !important;
    }

    /* Custom Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(0, 0, 0, 0.01) !important;
        border-bottom: 1px solid rgba(0, 0, 0, 0.05) !important;
        gap: 12px;
        padding: 6px 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 20px !important;
        padding: 8px 18px !important;
        color: #64748b !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a !important;
        background-color: rgba(0, 0, 0, 0.02) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #4f46e5 !important;
        background-color: rgba(99, 102, 241, 0.07) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        font-weight: 600 !important;
    }

    /* Expanders styling */
    .stMarkdown div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02) !important;
        margin: 10px 0 !important;
    }
    .stMarkdown div[data-testid="stExpander"] * {
        color: #1e293b !important;
    }

    /* ─── Chat Messages (Grounded Layout) ─── */
    .user-msg {
        background: rgba(99, 102, 241, 0.06);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-left: 4px solid #4f46e5;
        border-radius: 16px;
        padding: 16px 22px;
        margin: 12px 0;
        color: #1e293b;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    }
    .assistant-msg {
        background: rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.7);
        border-left: 4px solid #06b6d4;
        border-radius: 16px;
        padding: 16px 22px;
        margin: 12px 0;
        color: #1e293b;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    }
    .mode-pill {
        display: inline-block;
        background: rgba(6, 182, 212, 0.08);
        border: 1px solid rgba(6, 182, 212, 0.18);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.74rem;
        color: #0891b2;
        margin-bottom: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ─── Citations & Sentiment Badges ─── */
    .source-card {
        background: rgba(16, 185, 129, 0.03);
        border: 1px solid rgba(16, 185, 129, 0.1);
        border-radius: 10px;
        padding: 10px 16px;
        margin: 6px 0;
        font-size: 0.83rem;
        color: #065f46;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        background: rgba(16, 185, 129, 0.06);
        border-color: rgba(16, 185, 129, 0.2);
    }
    .badge-optimistic { color: #059669; font-weight: 600; font-family: 'Outfit', sans-serif; }
    .badge-cautious   { color: #d97706; font-weight: 600; font-family: 'Outfit', sans-serif; }
    .badge-critical   { color: #dc2626; font-weight: 600; font-family: 'Outfit', sans-serif; }
    .badge-neutral    { color: #475569; font-weight: 600; font-family: 'Outfit', sans-serif; }

    /* ─── Ingestion Upload Helpers ─── */
    .upload-info {
        background: rgba(6, 182, 212, 0.04);
        border: 1px solid rgba(6, 182, 212, 0.1);
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 0.8rem;
        color: #0891b2;
        margin: 10px 0;
        line-height: 1.4;
    }
    .welcome-card {
        background: rgba(99, 102, 241, 0.02);
        border: 1px solid rgba(99, 102, 241, 0.08);
        border-radius: 20px;
        padding: 45px 35px;
        text-align: center;
        margin: 25px 0;
        backdrop-filter: blur(10px);
        box-shadow: inset 0 0 20px rgba(99, 102, 241, 0.02);
        color: #475569;
    }

    /* ─── Critique Cards & Objections ─── */
    .critique-card {
        background: rgba(255, 255, 255, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-left: 4px solid #ef4444;
        border-radius: 14px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    }
    .claim-header {
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .claim-text {
        font-size: 1.05rem;
        font-style: italic;
        color: #1e293b;
        background: rgba(255,255,255,0.3);
        border: 1px solid rgba(0, 0, 0, 0.03);
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .critique-text {
        font-size: 0.94rem;
        color: #334155;
        line-height: 1.6;
        background: rgba(239, 68, 68, 0.02);
        border: 1px dashed rgba(239, 68, 68, 0.1);
        padding: 16px;
        border-radius: 8px;
    }

    /* Speech Bubbles for Versus & Cross-Exam Modes */
    .speech-bubble {
        border-radius: 16px;
        padding: 18px 22px;
        margin: 14px 0;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
        color: #1e293b;
    }
    .speech-bubble:hover {
        transform: translateY(-2px);
    }
    .speaker-name {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .source-citation {
        font-size: 0.78rem;
        color: #059669;
        font-weight: 600;
        margin-top: 10px;
        border-top: 1px solid rgba(0, 0, 0, 0.04);
        padding-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Insight Cards for Comparison Heatmaps */
    .insight-card {
        background: rgba(6, 182, 212, 0.02);
        border: 1px solid rgba(6, 182, 212, 0.1);
        border-radius: 12px;
        padding: 18px;
        margin: 10px 0;
        font-size: 0.9rem;
        line-height: 1.5;
        color: #334155;
    }

    /* Power BI Status Banner */
    .pbi-status-banner {
        border-radius: 12px;
        padding: 16px 20px;
        margin: 15px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        backdrop-filter: blur(10px);
    }

    .app-footer {
        font-size: 0.72rem;
        color: #64748b;
        text-align: center;
        margin-top: 35px;
        padding: 15px 0;
        border-top: 1px solid rgba(0, 0, 0, 0.04);
        letter-spacing: 0.05em;
    }

    /* ─── Top-Bordered KPI Cards (Light Mode) ─── */
    .kpi-card {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.95);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        text-align: left;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.08);
        background: rgba(255, 255, 255, 0.9);
        border-color: rgba(99, 102, 241, 0.2);
    }
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #475569;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #10b981;
        font-weight: 600;
        margin-top: 8px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .kpi-subtext-down {
        font-size: 0.78rem;
        color: #ef4444;
        font-weight: 600;
        margin-top: 8px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .kpi-subtext-neutral {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 600;
        margin-top: 8px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* ─── Paper Index Full Analysis Table ─── */
    .analysis-table-container {
        width: 100%;
        overflow-x: auto;
        margin-top: 15px;
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.06);
    }
    .analysis-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        color: #334155;
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(10px);
        text-align: left;
    }
    .analysis-table th {
        background-color: rgba(248, 250, 252, 0.8);
        text-align: left;
        padding: 12px 16px;
        font-weight: 700;
        color: #475569;
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
        text-transform: uppercase;
        font-size: 0.74rem;
        letter-spacing: 0.06em;
    }
    .analysis-table td {
        padding: 12px 16px;
        border-bottom: 1px solid rgba(0, 0, 0, 0.04);
        vertical-align: middle;
    }
    .analysis-table tr:last-child td {
        border-bottom: none;
    }
    .analysis-table tr:hover {
        background-color: rgba(255, 255, 255, 0.4);
    }
    .badge-optimistic-inline {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        border: 1px solid rgba(21, 128, 61, 0.15);
    }
    .badge-neutral-inline {
        background-color: #f1f5f9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        border: 1px solid rgba(71, 85, 105, 0.15);
    }
    .badge-cautious-inline {
        background-color: #fef9c3;
        color: #a16207;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        border: 1px solid rgba(161, 98, 7, 0.15);
    }
    .badge-critical-inline {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        border: 1px solid rgba(185, 28, 28, 0.15);
    }
    .progress-bar-container {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .progress-bar-bg {
        width: 70px;
        background-color: #e2e8f0;
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 3px;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
