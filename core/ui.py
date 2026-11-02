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
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }
    .stTextInput input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    /* Markdown Text & Content Readability */
    .stMarkdown p, .stMarkdown li {
        color: #f1f5f9 !important;
        font-size: 0.94rem !important;
        line-height: 1.65 !important;
    }
    .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        margin-top: 14px !important;
        margin-bottom: 8px !important;
        letter-spacing: -0.01em !important;
    }
    .stMarkdown strong {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    .stMarkdown ul {
        margin-top: 4px !important;
        margin-bottom: 8px !important;
        padding-left: 22px !important;
    }
    .stMarkdown li {
        margin-bottom: 4px !important;
    }
    .stMarkdown blockquote {
        border-left: 3px solid #38bdf8 !important;
        background: rgba(18, 26, 44, 0.70) !important;
        padding: 10px 16px !important;
        margin: 8px 0 16px 0 !important;
        border-radius: 0 8px 8px 0 !important;
        color: #f1f5f9 !important;
        font-style: normal !important;
        font-size: 0.94rem !important;
        line-height: 1.68 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.22) !important;
    }

    /* Clean Top Right Corner - Remove Deploy button & Streamlit branding */
    .stDeployButton,
    [data-testid="stDeployButton"],
    [data-testid="stAppDeployButton"],
    div[data-testid="stToolbarActions"],
    header [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }
    
    header[data-testid="stHeader"] {
        background: rgba(12, 16, 23, 0.85) !important;
        backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12) !important;
        height: 3.2rem !important;
    }

    header[data-testid="stHeader"]::after {
        content: "⚡ ResearchRadar v3 · Hybrid RAG Engine";
        position: absolute;
        right: 1.5rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: #38bdf8;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.28);
        padding: 4px 14px;
        border-radius: 9999px;
        pointer-events: none;
    }

    /* Chat message styling */
    .user-msg-box {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        margin-left: 15%;
        color: #f8fafc;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }
    .bot-msg-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px 12px 12px 2px;
        padding: 16px 20px;
        margin-bottom: 16px;
        margin-right: 5%;
        color: #f1f5f9;
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
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 4px 11px;
        border-radius: 9999px;
        margin-bottom: 10px;
    }
    .agent-badge-debate {
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
    .agent-badge-compare {
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.3);
    }
    .agent-badge-predict {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .agent-badge-qa {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Ready Prompts Pills styling */
    .ready-prompt-btn {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        border-radius: 20px !important;
        font-size: 0.82rem !important;
        padding: 5px 12px !important;
    }

    /* Native ChatGPT Style Message Bubbles */
    [data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.22) !important;
    }
    [data-testid="stChatMessageContent"] {
        color: #f1f5f9 !important;
    }
    
    /* Native ChatGPT Style Fixed Bottom Input */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        background: rgba(15, 23, 42, 0.95) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] button {
        color: #38bdf8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

