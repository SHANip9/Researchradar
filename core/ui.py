"""
core/ui.py — Sleek, calm, modern design system for ResearchRadar.
Inspired by Microsoft Fluent 2, Copilot, and Google Material You/Gemini interfaces:
Calm slate surfaces, gentle acrylic glassmorphism, refined typography,
and balanced, attractive accent colors.
"""

import streamlit as st

def inject_clean_css():
    """Injects a minimalist, calm, premium design system with Microsoft/Google inspired tones."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', 'Segoe UI Mono', monospace !important;
    }

    /* Overall Canvas - Calm Slate Obsidian */
    .stApp {
        background-color: #0c1017;
        background-image: 
            radial-gradient(at 15% 15%, rgba(37, 99, 235, 0.05) 0px, transparent 50%),
            radial-gradient(at 85% 85%, rgba(14, 165, 233, 0.04) 0px, transparent 50%);
        color: #e2e8f0;
    }

    /* Subtle custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0c1017;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 6px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }

    /* Clean Sidebar - Microsoft / Azure Slate */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(148, 163, 184, 0.12) !important;
    }

    /* Brand Header - Calm Microsoft Copilot / Google DeepMind Gradient */
    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .brand-sub {
        font-size: 0.90rem;
        color: #94a3b8;
        margin-bottom: 20px;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* Cards - Soft Acrylic Mica Surface */
    .glass-card {
        background: rgba(18, 24, 38, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 16px;
        backdrop-filter: blur(14px);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.32);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    /* Stat & Metric Cards - Crisp, calm corporate data display */
    .metric-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 16px 18px;
        text-align: center;
        transition: border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.28);
    }
    .metric-num {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .metric-title {
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-top: 6px;
    }

    /* Mode Pill Tags */
    .mode-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .mode-debate { background: rgba(244, 63, 94, 0.10); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.22); }
    .mode-compare { background: rgba(14, 165, 233, 0.10); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.22); }
    .mode-predict { background: rgba(99, 102, 241, 0.10); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.22); }

    /* Bias Indicators - Calm Green, Amber, Rose */
    .bias-meter {
        background: #1e293b;
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin: 8px 0;
    }
    .bias-bar-low { background: #10b981; height: 100%; border-radius: 6px; }
    .bias-bar-med { background: #f59e0b; height: 100%; border-radius: 6px; }
    .bias-bar-high { background: #f43f5e; height: 100%; border-radius: 6px; }

    /* Citation badge */
    .citation-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.78rem;
        color: #cbd5e1;
        margin: 4px 4px 4px 0;
    }

    /* Tabs Styling - Microsoft Fluent segmented pill tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 8px 18px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0 !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: rgba(37, 99, 235, 0.14) !important;
        border: 1px solid rgba(56, 189, 248, 0.28) !important;
    }

    /* Buttons - Microsoft Fluent & Google Material Calm Theme */
    button,
    .stButton > button,
    [data-testid*="stBaseButton"],
    [data-testid*="baseButton"],
    div[data-testid="stButton"] > button {
        background: rgba(18, 26, 44, 0.88) !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    button p,
    button span,
    [data-testid*="stBaseButton"] p,
    [data-testid*="stBaseButton"] span,
    div[data-testid="stButton"] button p {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    button:hover,
    .stButton > button:hover,
    [data-testid*="stBaseButton"]:hover,
    div[data-testid="stButton"] > button:hover {
        background: rgba(30, 41, 59, 0.95) !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
        color: #38bdf8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
    }
    button:hover p,
    button:hover span,
    [data-testid*="stBaseButton"]:hover p,
    [data-testid*="stBaseButton"]:hover span {
        color: #38bdf8 !important;
    }

    /* Primary & Submit Buttons (e.g. Send -> and Clear Chat) */
    div[data-testid="stFormSubmitButton"] > button,
    button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.28) !important;
    }
    div[data-testid="stFormSubmitButton"] > button p,
    div[data-testid="stFormSubmitButton"] > button span,
    button[kind="primary"] p,
    button[kind="primary"] span {
        color: #ffffff !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover,
    button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4) !important;
    }

    /* Input & Select fields */
    .stTextInput input, .stSelectbox select {
        background-color: #0f172a !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus {
        border-color: #0284c7 !important;
        box-shadow: 0 0 0 1px #0284c7 !important;
    }

    /* Chat message styling */
    .user-msg-box {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        margin-left: 18%;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }
    .bot-msg-box {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px 12px 12px 2px;
        padding: 16px 20px;
        margin-bottom: 16px;
        margin-right: 8%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
    }

    /* File uploader enhancement */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(148, 163, 184, 0.25) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        background: rgba(15, 23, 42, 0.45) !important;
    }

    /* Agent Badges on chat bubbles - Calm, attractive tones */
    .agent-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-bottom: 8px;
    }
    .agent-badge-debate {
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.25);
    }
    .agent-badge-compare {
        background: rgba(14, 165, 233, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.25);
    }
    .agent-badge-predict {
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .agent-badge-qa {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    </style>
    """, unsafe_allow_html=True)

