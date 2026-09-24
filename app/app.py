import sys
from pathlib import Path
import re
import urllib.parse

# Add project root to sys.path so src imports work reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import load_spotify_zip
from src.feature_engineering import (
    create_monthly_features,
    compute_overview_metrics,
    get_top_artists,
    get_top_tracks,
    get_hourly_distribution,
    create_listening_clock_matrix,
)
from src.clustering import run_monthly_clustering
from src.skip_prediction import evaluate_skip_models, predict_single_event
from src.recommender import generate_recommendations



# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SONOVA — Personal Music Intelligence",
    page_icon=str(Path(__file__).resolve().parent.parent / "assets" / "sonova_logo.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Helper function to render HTML safely without markdown code block indentation
def render_html(html_str: str):
    clean = re.sub(r'^[ \t]+', '', html_str, flags=re.MULTILINE)
    st.markdown(clean, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DESIGN SYSTEM & STYLES (Instrument Serif + DM Sans/Inter, Ambient Hexagon Canvas)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-dark: #050605;
    --bg-surface: #080C08;
    --bg-card: rgba(11, 18, 11, 0.72);
    --bg-card-hover: rgba(16, 26, 16, 0.88);
    --border-color: rgba(140, 255, 70, 0.12);
    --border-hover: rgba(140, 255, 70, 0.38);
    --accent-neon: #7CFF35;
    --accent-light: #B4FF5C;
    --accent-dim: rgba(124, 255, 53, 0.14);
    --text-primary: #F4F5F0;
    --text-secondary: #9BA39B;
    --text-muted: #5A635A;
}

/* Force dark background on all Streamlit containers */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp, 
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stBottom"], .main, section.main {
    background-color: var(--bg-dark) !important;
    background: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide Streamlit default sidebar, footer, status widget, and reduce header padding */
[data-testid="stSidebar"], #MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stToolbar"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1320px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Animated subtle honeycomb hexagonal ambient background */
.sonova-bg-canvas {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0;
    pointer-events: none;
    background-color: var(--bg-dark);
    overflow: hidden;
}

.sonova-hex-pattern {
    position: absolute;
    top: -60px;
    left: -60px;
    width: calc(100vw + 120px);
    height: calc(100vh + 120px);
    background-image: 
        radial-gradient(circle at 50% 25%, rgba(5, 7, 5, 0.55) 0%, rgba(5, 6, 5, 0.94) 85%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='103.923' viewBox='0 0 60 103.923'%3E%3Cpath d='M30 0 L60 17.32 L60 51.96 L30 69.28 L0 51.96 L0 17.32 Z' fill='none' stroke='rgba(140,255,70,0.05)' stroke-width='0.75'/%3E%3Cpath d='M30 51.96 L60 69.28 L60 103.92 L30 121.24 L0 103.92 L0 69.28 Z' fill='none' stroke='rgba(140,255,70,0.035)' stroke-width='0.75'/%3E%3C/svg%3E");
    background-size: 100% 100%, 60px 103.923px;
    opacity: 0.85;
    animation: hexBreathe 14s ease-in-out infinite alternate;
}

.sonova-ambient-glow {
    position: absolute;
    top: 6%;
    left: 28%;
    width: 44vw;
    height: 38vh;
    background: radial-gradient(circle, rgba(124, 255, 53, 0.045) 0%, rgba(5, 6, 5, 0) 70%);
    filter: blur(100px);
    animation: glowPulse 12s ease-in-out infinite alternate;
}

@keyframes hexBreathe {
    0% { transform: scale(1); opacity: 0.7; }
    50% { transform: scale(1.018); opacity: 0.95; }
    100% { transform: scale(1); opacity: 0.7; }
}

@keyframes glowPulse {
    0% { opacity: 0.35; transform: scale(0.95); }
    100% { opacity: 0.75; transform: scale(1.1); }
}

@media (max-width: 700px) {
  .editorial-headline { font-size: clamp(2.5rem, 10vw, 3.8rem) !important; }
}

/* Responsive layout: keep the desktop composition while making every section
   usable without horizontal page scrolling on tablets and phones. */
html, body, [data-testid="stAppViewContainer"] {
    width: 100%;
    overflow-x: hidden !important;
}
*, *::before, *::after { box-sizing: border-box; }
[data-testid="stMainBlockContainer"] {
    width: 100%;
    min-width: 0;
}

/* Editorial Brand & Headlines */
.sonova-header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1.6rem;
    background: rgba(11, 18, 11, 0.78);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-color);
    border-radius: 9999px;
    margin-bottom: 1.5rem;
}

.brand-title {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 800;
    font-size: 1.15rem;
    letter-spacing: 0.28em;
    color: var(--accent-neon);
    text-transform: uppercase;
}

.brand-subtitle {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.22em;
    color: var(--text-secondary);
    text-transform: uppercase;
}

.editorial-headline {
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-size: 3.8rem !important;
    line-height: 1.05 !important;
    font-weight: 400 !important;
    letter-spacing: -0.02em;
    color: #FFFFFF !important;
    margin: 0.3rem 0 0.8rem 0;
}

.editorial-title {
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-size: 2.5rem !important;
    line-height: 1.15 !important;
    font-weight: 400 !important;
    letter-spacing: -0.015em;
    color: #FFFFFF !important;
    margin: 0.2rem 0 0.4rem 0;
}

.storytelling-label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.24em;
    color: var(--accent-neon);
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.storytelling-lead {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem;
    color: var(--text-secondary);
    margin-bottom: 1.4rem;
    line-height: 1.5;
}

/* Cards & Containers */
.sonova-card {
    background: var(--bg-card);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.25s ease, background 0.25s ease, box-shadow 0.25s ease;
}

.sonova-card:hover {
    border-color: var(--border-hover);
    background: var(--bg-card-hover);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45);
}

.sonova-kpi-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    text-align: left;
    margin-bottom: 0.8rem;
    transition: all 0.25s ease;
}

.sonova-kpi-card:hover {
    border-color: var(--border-hover);
    background: var(--bg-card-hover);
    transform: translateY(-2px);
}

.kpi-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: var(--text-secondary);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.kpi-value {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.kpi-subtext {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* Streamlit Horizontal Radio Navigation Pills */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 0.4rem !important;
    background: rgba(11, 18, 11, 0.85) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    padding: 0.35rem 0.6rem !important;
    border-radius: 9999px !important;
    border: 1px solid var(--border-color) !important;
    margin: 0 auto 1.8rem auto !important;
    max-width: fit-content !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background: transparent !important;
    padding: 0.45rem 1.15rem !important;
    border-radius: 9999px !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin: 0 !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    background: rgba(124, 255, 53, 0.08) !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
    background: var(--accent-dim) !important;
    border: 1px solid rgba(124, 255, 53, 0.5) !important;
    box-shadow: 0 0 16px rgba(124, 255, 53, 0.2) !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {
    font-family: 'DM Sans', 'Inter', sans-serif !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover [data-testid="stMarkdownContainer"] p {
    color: #F4F5F0 !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] [data-testid="stMarkdownContainer"] p {
    color: var(--accent-neon) !important;
    font-weight: 700 !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* Onboarding & Guide components */
.privacy-notice {
    font-size: 0.78rem;
    color: var(--text-secondary);
    border-left: 2px solid var(--accent-neon);
    padding-left: 0.85rem;
    margin-top: 0.8rem;
    line-height: 1.55;
}

/* Custom Buttons */
.stButton > button {
    background: rgba(14, 24, 16, 0.85) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.1rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    border-color: var(--accent-neon) !important;
    color: var(--accent-neon) !important;
    box-shadow: 0 0 14px rgba(124, 255, 53, 0.2) !important;
    background: rgba(20, 36, 24, 0.95) !important;
}

/* File Uploader styling */
[data-testid="stFileUploader"] {
    background: rgba(11, 18, 11, 0.6) !important;
    border: 1px dashed rgba(140, 255, 70, 0.3) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(140, 255, 70, 0.6) !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(124, 255, 53, 0.12) !important;
    border: 1px solid rgba(124, 255, 53, 0.35) !important;
    color: #7CFF35 !important;
}

/* Dataframe styling */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    background: rgba(8, 12, 8, 0.8) !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    background: rgba(11, 18, 11, 0.5) !important;
    margin-bottom: 0.6rem !important;
}

/* Leaderboard custom items */
.leaderboard-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0.8rem;
    border-bottom: 1px solid rgba(140, 255, 70, 0.07);
    transition: background 0.15s ease;
}
.leaderboard-row:hover {
    background: rgba(124, 255, 53, 0.04);
}
.leaderboard-row:last-child {
    border-bottom: none;
}
.rank-num {
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text-muted);
    width: 24px;
}
.leaderboard-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding-right: 1rem;
}
.bar-track {
    width: 90px;
    height: 5px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 3px;
    overflow: hidden;
    margin-right: 1rem;
}
.bar-fill {
    height: 100%;
    background: var(--accent-neon);
    border-radius: 3px;
}
.metric-count {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-secondary);
    min-width: 48px;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* Recommendation card styling */
.rec-card {
    display: flex;
    align-items: center;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    border: 1px solid var(--border-color);
    background: rgba(11, 18, 11, 0.6);
    margin-bottom: 0.65rem;
    transition: all 0.2s ease;
}
.rec-card:hover {
    border-color: var(--border-hover);
    background: var(--bg-card-hover);
    transform: translateX(3px);
}
.rec-cover {
    width: 48px;
    height: 48px;
    border-radius: 8px;
    margin-right: 1rem;
    flex-shrink: 0;
}
.badge-pill {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(124, 255, 53, 0.1);
    color: var(--accent-neon);
    border: 1px solid rgba(124, 255, 53, 0.3);
}

@media (max-width: 900px) {
    .block-container {
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }
    .editorial-headline { font-size: clamp(2.8rem, 8vw, 3.8rem) !important; }
    .editorial-title { font-size: clamp(2rem, 5vw, 2.5rem) !important; }
    .sonova-header-bar { padding: 0.7rem 1.1rem; }
    .kpi-value { font-size: clamp(1.55rem, 4vw, 2.1rem); }
}

@media (max-width: 640px) {
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }
    .sonova-header-bar {
        border-radius: 16px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 1rem;
        gap: 0.5rem;
    }
    .brand-title { font-size: 0.95rem; letter-spacing: 0.2em; }
    .brand-subtitle { font-size: 0.62rem; letter-spacing: 0.12em; text-align: right; }
    .editorial-headline { font-size: clamp(2.55rem, 12vw, 3.4rem) !important; }
    .editorial-title { font-size: clamp(1.85rem, 8vw, 2.3rem) !important; }
    .storytelling-label { font-size: 0.64rem; letter-spacing: 0.16em; }
    .storytelling-lead { font-size: 0.9rem; margin-bottom: 1rem; }
    .sonova-card { padding: 1rem; border-radius: 10px; }
    .sonova-kpi-card { padding: 0.85rem; }
    .kpi-title { font-size: 0.64rem; letter-spacing: 0.08em; }
    .kpi-value { font-size: clamp(1.35rem, 7vw, 1.8rem); overflow-wrap: anywhere; }
    .kpi-subtext { font-size: 0.68rem; }

    /* Navigation remains one compact, touch-friendly horizontal strip. */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        width: 100% !important;
        max-width: 100% !important;
        justify-content: flex-start !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        border-radius: 14px !important;
        padding: 0.3rem !important;
        gap: 0.2rem !important;
        scrollbar-width: thin;
        margin-bottom: 1.1rem !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        flex: 0 0 auto !important;
        padding: 0.55rem 0.75rem !important;
        min-height: 40px;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {
        font-size: 0.78rem !important;
        white-space: nowrap;
    }

    /* Allow Streamlit columns and embedded custom grids to collapse cleanly. */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        align-items: stretch !important;
        gap: 0.65rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 0 !important;
        width: 100% !important;
    }
    .sonova-card [style*="grid-template-columns"],
    [data-testid="stMarkdownContainer"] [style*="grid-template-columns"] {
        grid-template-columns: minmax(0, 1fr) !important;
    }
    .leaderboard-row { padding: 0.6rem 0.45rem; }
    .leaderboard-name { padding-right: 0.5rem; font-size: 0.8rem; }
    .bar-track { width: 48px; margin-right: 0.5rem; }
    .metric-count { min-width: 40px; font-size: 0.78rem; }
    .rec-card { padding: 0.65rem; min-width: 0; }
    .rec-cover { width: 42px; height: 42px; margin-right: 0.7rem; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { max-width: 100%; overflow-x: auto; }
    [data-testid="stFileUploader"] { padding: 0.65rem !important; }
    .sonova-bg-canvas { opacity: 0.7; }
}
</style>

<!-- Fixed Ambient Background Canvas -->
<div class="sonova-bg-canvas">
    <div class="sonova-hex-pattern"></div>
    <div class="sonova-ambient-glow"></div>
</div>
"""
render_html(CUSTOM_CSS)


# -----------------------------------------------------------------------------
# PLOTLY THEME HELPER (Consistent Calm Dark Theme with #7CFF35 Accent)
# -----------------------------------------------------------------------------
def apply_sonova_theme(fig, height=340):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Inter, sans-serif", size=12, color="#9BA39B"),
        height=height,
        margin=dict(l=45, r=30, t=35, b=40),
        legend=dict(
            bgcolor="rgba(11,18,11,0.75)",
            bordercolor="rgba(140,255,70,0.14)",
            borderwidth=1,
            font=dict(color="#F4F5F0", size=11),
        ),
        xaxis=dict(
            gridcolor="rgba(140,255,70,0.06)",
            linecolor="rgba(140,255,70,0.14)",
            tickfont=dict(color="#9BA39B", size=11),
            zerolinecolor="rgba(140,255,70,0.08)",
        ),
        yaxis=dict(
            gridcolor="rgba(140,255,70,0.06)",
            linecolor="rgba(140,255,70,0.14)",
            tickfont=dict(color="#9BA39B", size=11),
            zerolinecolor="rgba(140,255,70,0.08)",
        ),
    )
    return fig


# -----------------------------------------------------------------------------
# GENERATIVE ABSTRACT VINYL ARTWORK HELPER (For Recommendations)
# -----------------------------------------------------------------------------
def get_abstract_cover_data_uri(index: int) -> str:
    """
    Generates tasteful abstract music artwork SVG with dark green gradients,
    grain-like acoustic vectors, waveforms, and vinyl grooves.
    Strictly avoids fabricating fake Spotify branding.
    """
    themes = [
        # Theme 0: Concentric acoustic rings
        """<svg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 56 56'>
            <rect width='56' height='56' rx='8' fill='#0B190D'/>
            <circle cx='28' cy='28' r='22' stroke='rgba(124,255,53,0.15)' stroke-width='1' fill='none'/>
            <circle cx='28' cy='28' r='16' stroke='rgba(124,255,53,0.3)' stroke-width='1.2' fill='none'/>
            <circle cx='28' cy='28' r='10' stroke='rgba(124,255,53,0.5)' stroke-width='1.5' fill='none'/>
            <circle cx='28' cy='28' r='4' fill='#7CFF35'/>
        </svg>""",
        # Theme 1: Soundwave audio frequency bars
        """<svg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 56 56'>
            <rect width='56' height='56' rx='8' fill='#0D1D10'/>
            <rect x='10' y='22' width='4' height='12' rx='2' fill='rgba(124,255,53,0.4)'/>
            <rect x='17' y='14' width='4' height='28' rx='2' fill='#7CFF35'/>
            <rect x='24' y='18' width='4' height='20' rx='2' fill='rgba(124,255,53,0.85)'/>
            <rect x='31' y='10' width='4' height='36' rx='2' fill='#B4FF5C'/>
            <rect x='38' y='16' width='4' height='24' rx='2' fill='rgba(124,255,53,0.6)'/>
            <rect x='45' y='24' width='4' height='8' rx='2' fill='rgba(124,255,53,0.3)'/>
        </svg>""",
        # Theme 2: Minimalist vinyl groove
        """<svg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 56 56'>
            <rect width='56' height='56' rx='8' fill='#081009'/>
            <circle cx='28' cy='28' r='23' fill='#0E1C10' stroke='rgba(140,255,70,0.15)' stroke-width='1'/>
            <circle cx='28' cy='28' r='17' stroke='rgba(140,255,70,0.25)' stroke-dasharray='3 2' stroke-width='1' fill='none'/>
            <circle cx='28' cy='28' r='11' stroke='rgba(140,255,70,0.4)' stroke-width='1' fill='none'/>
            <circle cx='28' cy='28' r='7' fill='#152A18'/>
            <circle cx='28' cy='28' r='2.5' fill='#7CFF35'/>
        </svg>""",
        # Theme 3: Resonant waveform curves
        """<svg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 56 56'>
            <rect width='56' height='56' rx='8' fill='#0B170E'/>
            <path d='M8 28 Q 18 10, 28 28 T 48 28' fill='none' stroke='#7CFF35' stroke-width='2'/>
            <path d='M8 28 Q 18 18, 28 28 T 48 28' fill='none' stroke='rgba(124,255,53,0.4)' stroke-width='1.5'/>
            <path d='M8 28 Q 18 38, 28 28 T 48 28' fill='none' stroke='rgba(124,255,53,0.25)' stroke-width='1'/>
        </svg>""",
        # Theme 4: Acoustic lattice matrix
        """<svg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 56 56'>
            <rect width='56' height='56' rx='8' fill='#0C1B0F'/>
            <polygon points='28,8 48,20 48,36 28,48 8,36 8,20' fill='none' stroke='rgba(124,255,53,0.3)' stroke-width='1.5'/>
            <polygon points='28,16 40,23 40,33 28,40 16,33 16,23' fill='none' stroke='rgba(124,255,53,0.6)' stroke-width='1.5'/>
            <circle cx='28' cy='28' r='3.5' fill='#7CFF35'/>
        </svg>""",
    ]
    selected_svg = themes[index % len(themes)]
    return "data:image/svg+xml;utf8," + urllib.parse.quote(selected_svg)


# -----------------------------------------------------------------------------
# CACHED COMPUTATION HELPERS (ML & Data Processing - Logic Untouched)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_data_artifacts(df_records):
    """Generates and caches behavioral features and metrics."""
    overview = compute_overview_metrics(df_records)
    # Enrich overview with unique albums count if column is present
    if "master_metadata_album_album_name" in df_records.columns:
        overview["unique_albums"] = df_records["master_metadata_album_album_name"].nunique()
    else:
        overview["unique_albums"] = df_records["master_metadata_album_artist_name"].nunique()

    monthly = create_monthly_features(df_records)
    top_artists = get_top_artists(df_records, n=10)
    top_tracks = get_top_tracks(df_records, n=10)
    hourly = get_hourly_distribution(df_records)
    listening_clock = create_listening_clock_matrix(df_records)
    return {
        "overview": overview,
        "monthly": monthly,
        "top_artists": top_artists,
        "top_tracks": top_tracks,
        "hourly": hourly,
        "listening_clock": listening_clock,
    }


@st.cache_data(show_spinner=False)
def compute_clustering(monthly_df):
    """Runs monthly K-Means clustering and calculates silhouette scores."""
    return run_monthly_clustering(monthly_df)


@st.cache_data(show_spinner=False)
def compute_skip_evaluation(df_records):
    """Trains and compares 5 models with 5-fold CV and train/test diagnostics."""
    return evaluate_skip_models(df_records, sample_size=3000)


@st.cache_data(show_spinner=False)
def compute_recommendations_cached(df_records, context_name):
    """Calculates behavior-based recommendations for selected context."""
    return generate_recommendations(df_records, context=context_name, n=10)


# -----------------------------------------------------------------------------
# APPLICATION TOP HEADER
# -----------------------------------------------------------------------------
render_html(
    """
    <div class="sonova-header-bar">
        <div>
            <span class="brand-title">SONOVA</span>
            <span style="color: rgba(140,255,70,0.3); margin: 0 0.8rem; font-weight: 300;">|</span>
            <span class="brand-subtitle">PERSONAL MUSIC INTELLIGENCE</span>
        </div>
        <div style="font-size: 0.78rem; color: #9BA39B; font-weight: 500; letter-spacing: 0.05em;">
            IN-MEMORY &bull; NO SPOTIFY LOGIN
        </div>
    </div>
    """
)


# -----------------------------------------------------------------------------
# SESSION STATE & AUTOMATIC DATA INITIALIZATION
# -----------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state["df"] = None
    st.session_state["data_source"] = None
    st.session_state["dataset_removed"] = False

# -----------------------------------------------------------------------------
# NAVIGATION MENU (CLEAN TOP PILLS)
# -----------------------------------------------------------------------------
nav_tabs = [
    "Overview",
    "Music DNA",
    "Analytics",
    "Listening Phases",
    "Skip Prediction",
    "Recommendations",
    "Blog",
    "About",
]

# Product announcements and release notes. Add new entries at the top.
PRODUCT_UPDATES = [
    {
        "date": "2026-09-24",
        "status": "Coming soon",
        "title": "More coming to SONOVA",
        "body": "New features are in the works. Stay tuned for updates.",
    },
]

selected_tab = st.radio(
    label="Navigation",
    options=nav_tabs,
    horizontal=True,
    label_visibility="collapsed",
)


# Helper reference to active dataset
df = st.session_state["df"]
artifacts = compute_data_artifacts(df) if df is not None else None


def render_dataset_prompt():
    """Show the empty-state prompt in SONOVA colors at the bottom of the viewport."""
    render_html(
        """
        <div style="position: fixed; left: 50%; bottom: 1.5rem; transform: translateX(-50%); z-index: 9999;
                    width: min(92vw, 680px); box-sizing: border-box; padding: 0.9rem 1.25rem;
                    color: #DCE8D5; background: rgba(12, 27, 15, 0.97);
                    border: 1px solid rgba(124, 255, 53, 0.45); border-left: 4px solid #7CFF35;
                    border-radius: 8px; box-shadow: 0 8px 28px rgba(0, 0, 0, 0.55);
                    font-size: 0.95rem; text-align: center;">
            <span style="color: #7CFF35; font-weight: 700;">INPUT DATASET</span>
            <span style="color: #9BA39B; margin-left: 0.45rem;">to start</span>
        </div>
        """
    )


# =============================================================================
# 1. OVERVIEW / UPLOAD PAGE
# =============================================================================
if selected_tab == "Overview":

    # Overview hero
    render_html(
        """
        <div style="margin-top: 0.8rem; margin-bottom: 2rem;">
            <div class="storytelling-label">PERSONAL &bull; DATA-DRIVEN &bull; INSIGHTS</div>
            <div class="editorial-headline">Your listening.<br>Decoded.</div>
            <div style="font-size: 1.05rem; color: #9BA39B; max-width: 680px; line-height: 1.6; margin-bottom: 0.5rem;">
                Upload your Spotify Extended Streaming History to uncover the patterns, phases and behaviors hidden inside your listening history.
            </div>
        </div>
        """
    )

    # Upload and capabilities
    col_upload, col_features = st.columns([1.1, 0.9], gap="large")

    with col_upload:
        # Upload card matching Panel 1 mockup
        render_html(
            """
            <div class="sonova-card">
                <div class="storytelling-label">INPUT DATASET</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFF; margin-bottom: 0.4rem;">
                    Upload Spotify Extended Streaming History ZIP
                </div>
                <div style="font-size: 0.84rem; color: #9BA39B; margin-bottom: 1.2rem; line-height: 1.5;">
                    Upload the original <code>.zip</code> package provided by Spotify. ZIP file only. No Spotify login required. SONOVA parses your listening history records in memory.
                </div>
            </div>
            """
        )

        if st.session_state.pop("reset_dataset_uploader", False):
            st.session_state.pop("spotify_history_zip", None)

        uploaded_zip = st.file_uploader(
            "Upload Spotify Extended Streaming History ZIP",
            type=["zip"],
            help="Upload the ZIP file downloaded from your Spotify Account Privacy page.",
            label_visibility="collapsed",
            key="spotify_history_zip",
        )

        if uploaded_zip is not None:
            try:
                with st.spinner("Decoding Spotify listening history in memory..."):
                    df_loaded = load_spotify_zip(uploaded_zip)
                    st.session_state["df"] = df_loaded
                    st.session_state["data_source"] = f"Uploaded Archive ({uploaded_zip.name})"
                    st.session_state["dataset_removed"] = False
                st.success(
                    f"✓ Listening history loaded: {len(df_loaded):,} events "
                    f"({df_loaded['master_metadata_track_name'].nunique():,} unique tracks). "
                    "Data prepared. Ready to explore."
                )
                st.rerun()
            except Exception as e:
                st.error(
                    f"This doesn't look like a Spotify Extended Streaming History export. "
                    f"Details: {str(e)}"
                )

        # Dataset Status Indicator
        if st.session_state["df"] is not None:
            df_curr = st.session_state["df"]
            status_col, remove_col = st.columns([5, 1])
            with status_col:
                render_html(
                    f"""
                    <div style="background: rgba(124, 255, 53, 0.08); border: 1px solid rgba(124, 255, 53, 0.3); border-radius: 8px; padding: 0.9rem 1.2rem; margin: 1rem 0;">
                        <span style="color: var(--accent-neon); font-weight: 700; font-size: 0.8rem; letter-spacing: 0.1em;">ACTIVE DATASET:</span>
                        <span style="color: #FFF; margin-left: 0.5rem; font-weight: 600;">{st.session_state['data_source']}</span>
                        <span style="color: #9BA39B; margin-left: 0.8rem; font-size: 0.82rem;">({len(df_curr):,} listening events)</span>
                    </div>
                    """
                )
            with remove_col:
                if st.button("Remove dataset", key="remove_dataset", use_container_width=True):
                    st.session_state["df"] = None
                    st.session_state["data_source"] = None
                    st.session_state["dataset_removed"] = True
                    st.session_state["reset_dataset_uploader"] = True
                    compute_data_artifacts.clear()
                    compute_clustering.clear()
                    compute_skip_evaluation.clear()
                    compute_recommendations_cached.clear()
                    st.rerun()

        # Onboarding Accordion: How to get your Spotify data
        with st.expander("HOW TO GET YOUR SPOTIFY DATA", expanded=False):
            render_html(
                """
                <ol style="font-size: 0.84rem; color: #9BA39B; line-height: 1.8; margin-top: 0.4rem; padding-left: 1.2rem;">
                    <li><strong>Open Spotify Account Privacy settings:</strong> Sign in to your account and locate the personal data section.</li>
                    <li><strong>Request Extended Streaming History:</strong> Request your personal data download (Extended Streaming History).</li>
                    <li><strong>Wait for Spotify to prepare the archive:</strong> Spotify prepares your data package.</li>
                    <li><strong>Locate Extended Streaming History:</strong> Download the full streaming archive.</li>
                    <li><strong>Upload the downloaded ZIP:</strong> Drag and drop the downloaded ZIP file directly into SONOVA.</li>
                </ol>
                <div style="margin-top: 0.8rem;">
                    <a href="https://www.spotify.com/account/privacy/" target="_blank" style="text-decoration:none;">
                        <button style="background: rgba(124,255,53,0.12); border: 1px solid rgba(124,255,53,0.35); color: #7CFF35; padding: 0.45rem 0.9rem; border-radius: 6px; font-weight:600; font-size:0.8rem; cursor:pointer;">
                            Open Spotify Privacy Page ↗
                        </button>
                    </a>
                </div>
                """
            )

        # Privacy notice
        render_html(
            """
            <div class="privacy-notice">
                <strong>Data Privacy:</strong> Your listening data is processed purely in memory within this session. 
                No Spotify login or account credentials are required, and sensitive telemetry (IP addresses, device identifiers, geographic coordinates) is stripped upon ingestion and never stored externally.
            </div>
            """
        )

    with col_features:
        # What you'll get section matching Panel 1
        render_html(
            """
            <div class="sonova-card">
                <div class="storytelling-label">CAPABILITIES</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFF; margin-bottom: 1.2rem;">
                    What you'll get
                </div>
                
                <div style="display: flex; gap: 1rem; margin-bottom: 1.2rem; align-items: flex-start;">
                    <div style="background: rgba(124, 255, 53, 0.12); border: 1px solid rgba(124, 255, 53, 0.3); border-radius: 8px; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; color: #7CFF35; font-size: 1.1rem; flex-shrink: 0;">
                        ♫
                    </div>
                    <div>
                        <div style="font-weight: 700; color: #FFF; font-size: 0.92rem;">Your Music DNA</div>
                        <div style="font-size: 0.8rem; color: #9BA39B; margin-top: 0.15rem;">Key listening habits, unique catalog metrics, and repeat archetypes</div>
                    </div>
                </div>

                <div style="display: flex; gap: 1rem; margin-bottom: 1.2rem; align-items: flex-start;">
                    <div style="background: rgba(124, 255, 53, 0.12); border: 1px solid rgba(124, 255, 53, 0.3); border-radius: 8px; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; color: #7CFF35; font-size: 1.1rem; flex-shrink: 0;">
                        ◷
                    </div>
                    <div>
                        <div style="font-weight: 700; color: #FFF; font-size: 0.92rem;">Listening Analytics</div>
                        <div style="font-size: 0.8rem; color: #9BA39B; margin-top: 0.15rem;">Longitudinal monthly volume trends, hourly rhythms, and listening clock</div>
                    </div>
                </div>

                <div style="display: flex; gap: 1rem; margin-bottom: 1.2rem; align-items: flex-start;">
                    <div style="background: rgba(124, 255, 53, 0.12); border: 1px solid rgba(124, 255, 53, 0.3); border-radius: 8px; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; color: #7CFF35; font-size: 1.1rem; flex-shrink: 0;">
                        ⚯
                    </div>
                    <div>
                        <div style="font-weight: 700; color: #FFF; font-size: 0.92rem;">Listening Phases (K-Means)</div>
                        <div style="font-size: 0.8rem; color: #9BA39B; margin-top: 0.15rem;">Unsupervised behavioral segmentation across months optimized via silhouette score</div>
                    </div>
                </div>

                <div style="display: flex; gap: 1rem; margin-bottom: 1.2rem; align-items: flex-start;">
                    <div style="background: rgba(124, 255, 53, 0.12); border: 1px solid rgba(124, 255, 53, 0.3); border-radius: 8px; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; color: #7CFF35; font-size: 1.1rem; flex-shrink: 0;">
                        ⚡
                    </div>
                    <div>
                        <div style="font-weight: 700; color: #FFF; font-size: 0.92rem;">Supervised Skip Prediction</div>
                        <div style="font-size: 0.8rem; color: #9BA39B; margin-top: 0.15rem;">5 machine learning models, 5-fold stratified CV, and overfitting diagnostics</div>
                    </div>
                </div>

                <div style="display: flex; gap: 1rem; align-items: flex-start;">
                    <div style="background: rgba(124, 255, 53, 0.12); border: 1px solid rgba(124, 255, 53, 0.3); border-radius: 8px; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; color: #7CFF35; font-size: 1.1rem; flex-shrink: 0;">
                        ✦
                    </div>
                    <div>
                        <div style="font-weight: 700; color: #FFF; font-size: 0.92rem;">Personalized Recommendations</div>
                        <div style="font-size: 0.8rem; color: #9BA39B; margin-top: 0.15rem;">Contextual candidates ranked by playback frequency, loyalty, and recency</div>
                    </div>
                </div>
            </div>
            """
        )


# =============================================================================
# 2. MUSIC DNA PAGE
# =============================================================================
elif selected_tab == "Music DNA":
    if df is None:
        render_dataset_prompt()
        st.stop()

    overview = artifacts["overview"]

    # Header
    render_html(
        """
        <div class="storytelling-label">CATALOG &bull; HABITS &bull; ARCHETYPES</div>
        <div class="editorial-headline">Your Music DNA</div>
        <div class="storytelling-lead">A snapshot of your listening identity and core habits.</div>
        """
    )

    # 5 KPI Cards in a row matching Panel 2
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_html(
            f"""
            <div class="sonova-kpi-card">
                <div class="kpi-title">Total Plays</div>
                <div class="kpi-value">{overview['total_plays']:,}</div>
                <div class="kpi-subtext">Verified audio events</div>
            </div>
            """
        )
    with c2:
        render_html(
            f"""
            <div class="sonova-kpi-card">
                <div class="kpi-title">Unique Artists</div>
                <div class="kpi-value">{overview['unique_artists']:,}</div>
                <div class="kpi-subtext">Distinct catalog creators</div>
            </div>
            """
        )
    with c3:
        render_html(
            f"""
            <div class="sonova-kpi-card">
                <div class="kpi-title">Unique Tracks</div>
                <div class="kpi-value">{overview['unique_tracks']:,}</div>
                <div class="kpi-subtext">Distinct songs heard</div>
            </div>
            """
        )
    with c4:
        render_html(
            f"""
            <div class="sonova-kpi-card">
                <div class="kpi-title">Unique Albums</div>
                <div class="kpi-value">{overview.get('unique_albums', 0):,}</div>
                <div class="kpi-subtext">Distinct releases</div>
            </div>
            """
        )
    with c5:
        render_html(
            f"""
            <div class="sonova-kpi-card">
                <div class="kpi-title">Listening Time</div>
                <div class="kpi-value">{overview['total_hours']:,.0f} <span style="font-size:1.05rem; color:#9BA39B; font-weight:400;">hrs</span></div>
                <div class="kpi-subtext">{overview['total_hours']/24:.1f} continuous days</div>
            </div>
            """
        )

    render_html("<div style='height: 1.2rem;'></div>")

    # Top Artists & Top Tracks side-by-side matching Panel 2
    col_art, col_trk = st.columns(2, gap="large")

    top_art = artifacts["top_artists"]
    top_trk = artifacts["top_tracks"]
    max_artist_plays = top_art["total_plays"].max() if len(top_art) > 0 else 1

    # Behavioral Archetype & Additional Metrics
    repeat_ratio = overview["repeat_rate"]
    if repeat_ratio > 0.40:
        archetype_title = "Focused Loyalist"
        archetype_desc = "You cultivate deep listening loyalty with favorite tracks and artists, returning to core songs frequently rather than browsing aimlessly."
    elif repeat_ratio > 0.25:
        archetype_title = "Balanced Explorer"
        archetype_desc = "You maintain a healthy equilibrium between replaying beloved catalog favorites and integrating fresh single-play discoveries."
    else:
        archetype_title = "Avid Discoverer"
        archetype_desc = "Your listening prioritizes wide breadth and exploration, rarely lingering on multiple repeat plays of the same song."

    with col_art:
        art_items_html = ""
        for idx, row in top_art.iterrows():
            rank = idx + 1
            artist = row["artist"]
            plays = row["total_plays"]
            pct = (plays / max_artist_plays) * 100.0
            art_items_html += f"""
            <div class="leaderboard-row">
                <span class="rank-num">{rank}</span>
                <span class="leaderboard-name">{artist}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width: {pct:.1f}%;"></div>
                </div>
                <span class="metric-count">{plays:,}</span>
            </div>
            """

        render_html(
            f"""
            <div class="sonova-card" style="padding: 1.2rem 1.4rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFF;">Top Artists</div>
                    <div style="font-size: 0.76rem; color: #7CFF35; font-weight: 600; text-transform: uppercase;">Ranked by Volume</div>
                </div>
                {art_items_html}
            </div>
            """
        )

        # Keep this panel directly beneath the artist list, independent of the
        # taller track list beside it.
        render_html(
            f"""
            <div class="sonova-card" style="margin-top: 1.2rem;">
                <div class="kpi-title">Behavioral Archetype</div>
                <div style="font-family: 'Instrument Serif', Georgia, serif; font-size: 1.8rem; color: #FFF; margin: 0.3rem 0;">
                    {archetype_title}
                </div>
                <div style="font-size: 0.86rem; color: #9BA39B; line-height: 1.6;">
                    {archetype_desc}
                </div>
                <div style="font-size: 0.74rem; color: #5A635A; margin-top: 0.6rem;">
                    Empirically computed from repeat play ratio ({overview['repeat_rate']*100:.1f}%) versus single-play discovery proxy ({overview['discovery_rate']*100:.1f}%).
                </div>
            </div>
            """
        )

    with col_trk:
        trk_items_html = ""
        for idx, row in top_trk.iterrows():
            rank = idx + 1
            track = row["track"]
            artist = row["artist"]
            plays = row["total_plays"]
            trk_items_html += f"""
            <div class="leaderboard-row">
                <span class="rank-num">{rank}</span>
                <div style="flex: 1; min-width: 0; padding-right: 0.8rem;">
                    <div class="leaderboard-name" style="padding-right: 0;">{track}</div>
                    <div style="font-size: 0.74rem; color: #9BA39B;">{artist}</div>
                </div>
                <span class="badge-pill" style="min-width: 50px; text-align: center;">{plays:,} plays</span>
            </div>
            """

        render_html(
            f"""
            <div class="sonova-card" style="padding: 1.2rem 1.4rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFF;">Top Tracks</div>
                    <div style="font-size: 0.76rem; color: #7CFF35; font-weight: 600; text-transform: uppercase;">Most Repeated</div>
                </div>
                {trk_items_html}
            </div>
            """
        )

        # The rate cards now sit directly below Top Tracks. Fixed-height cards
        # keep the three metrics aligned even when a title wraps.
        c_r1, c_r2, c_r3 = st.columns(3, gap="small")
        with c_r1:
            render_html(
                f"""
                <div class="sonova-kpi-card" style="height: 7rem; box-sizing: border-box; padding: 0.85rem 1rem;">
                    <div class="kpi-title" style="min-height: 1.8em;">Skip Rate</div>
                    <div class="kpi-value" style="font-size: 1.6rem; color: {'#7CFF35' if overview['skip_rate'] < 0.3 else '#F59E0B'};">{overview['skip_rate']*100:.1f}%</div>
                    <div class="kpi-subtext">Ended early</div>
                </div>
                """
            )
        with c_r2:
            render_html(
                f"""
                <div class="sonova-kpi-card" style="height: 7rem; box-sizing: border-box; padding: 0.85rem 1rem;">
                    <div class="kpi-title" style="min-height: 1.8em;">Repeat Rate</div>
                    <div class="kpi-value" style="font-size: 1.6rem;">{overview['repeat_rate']*100:.1f}%</div>
                    <div class="kpi-subtext">Played &gt; 1 time</div>
                </div>
                """
            )
        with c_r3:
            render_html(
                f"""
                <div class="sonova-kpi-card" style="height: 7rem; box-sizing: border-box; padding: 0.85rem 1rem; overflow: hidden;">
                    <div class="kpi-title" style="min-height: 1.8em;">Single-Play Discovery</div>
                    <div class="kpi-value" style="font-size: 1.6rem;">{overview['discovery_rate']*100:.1f}%</div>
                    <div class="kpi-subtext">Tracks played exactly once</div>
                </div>
                """
            )


# =============================================================================
# 3. LISTENING ANALYTICS PAGE
# =============================================================================
elif selected_tab == "Analytics":
    if df is None:
        render_dataset_prompt()
        st.stop()

    render_html(
        """
        <div class="storytelling-label">TEMPORAL RHYTHMS &bull; PATTERNS &bull; CYCLES</div>
        <div class="editorial-headline">Listening Analytics</div>
        <div class="storytelling-lead">When, how much, and how you listen.</div>
        """
    )

    # Row 1: Monthly Listening Trend & Hourly Distribution matching Panel 3
    col_trend, col_hour = st.columns([1.1, 0.9], gap="large")

    with col_trend:
        render_html('<div class="kpi-title">Monthly Listening Volume Trend</div>')
        monthly_df = artifacts["monthly"]
        fig_trend = px.line(
            monthly_df,
            x="month",
            y="total_plays",
            labels={"month": "Month", "total_plays": "Plays"},
        )
        fig_trend.update_traces(
            line_color="#7CFF35",
            line_width=2.4,
            mode="lines+markers",
            marker=dict(size=5, color="#F4F5F0"),
            fill="tozeroy",
            fillcolor="rgba(124, 255, 53, 0.08)",
            hovertemplate="<b>%{x}</b><br>Plays: %{y:,}<extra></extra>",
        )
        st.plotly_chart(apply_sonova_theme(fig_trend, height=320), use_container_width=True)

    with col_hour:
        render_html('<div class="kpi-title">Hourly Listening Distribution (0 – 23h)</div>')
        hourly_df = artifacts["hourly"]
        fig_hour = px.bar(
            hourly_df,
            x="hour",
            y="plays",
            labels={"hour": "Hour of Day", "plays": "Plays"},
            color_discrete_sequence=["#7CFF35"],
        )
        fig_hour.update_traces(
            marker_line_width=0,
            opacity=0.88,
            hovertemplate="<b>Hour %{x}:00</b><br>Plays: %{y:,}<extra></extra>",
        )
        st.plotly_chart(apply_sonova_theme(fig_hour, height=320), use_container_width=True)

    render_html("<div style='height: 1rem;'></div>")

    # Row 2: Listening Clock Heatmap & Listening Patterns Donut matching Panel 3
    col_clock, col_patterns = st.columns([1.1, 0.9], gap="large")

    with col_clock:
        render_html('<div class="kpi-title">The Listening Clock (Day of Week &times; Hour of Day)</div>')
        clock_matrix = artifacts["listening_clock"]
        fig_clock = px.imshow(
            clock_matrix,
            labels=dict(x="Hour of Day", y="Day of Week", color="Plays"),
            x=[f"{h:02d}" for h in range(24)],
            y=clock_matrix.index.tolist(),
            color_continuous_scale=[
                [0.0, "#050605"],
                [0.2, "#0B1D0E"],
                [0.5, "#1E652B"],
                [0.8, "#56DF2F"],
                [1.0, "#7CFF35"],
            ],
            aspect="auto",
        )
        fig_clock.update_traces(
            hovertemplate="<b>%{y} at %{x}:00</b><br>Plays: %{z:,}<extra></extra>"
        )
        st.plotly_chart(apply_sonova_theme(fig_clock, height=330), use_container_width=True)

    with col_patterns:
        render_html('<div class="kpi-title">Listening Patterns Breakdown</div>')
        overview = artifacts["overview"]
        
        # Calculate empirical behavioral distributions
        weekday_plays = df[df["is_weekend"] == False]["master_metadata_track_name"].count()
        weekend_plays = df[df["is_weekend"] == True]["master_metadata_track_name"].count()
        total_plays = len(df)
        
        weekday_pct = (weekday_plays / total_plays) * 100.0 if total_plays > 0 else 0
        weekend_pct = (weekend_plays / total_plays) * 100.0 if total_plays > 0 else 0
        shuffle_pct = df["shuffle"].mean() * 100.0 if "shuffle" in df.columns else 0

        pattern_labels = ["Weekday", "Weekend"]
        pattern_values = [weekday_plays, weekend_plays]
        
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=pattern_labels,
                    values=pattern_values,
                    hole=0.68,
                    marker=dict(colors=["#7CFF35", "#166534"]),
                    textinfo="none",
                    hovertemplate="<b>%{label}</b><br>Plays: %{value:,} (%{percent})<extra></extra>",
                )
            ]
        )
        fig_donut.add_annotation(
            text=f"<b style='font-size:20px; color:#FFF;'>{total_plays:,}</b><br><span style='font-size:11px; color:#9BA39B;'>Total Plays</span>",
            showarrow=False,
            font=dict(family="DM Sans, sans-serif"),
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Context metrics breakdown cards
        render_html(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.6rem; margin-top: 0.5rem;">
                <div class="sonova-kpi-card" style="padding: 0.7rem 0.85rem; margin-bottom: 0;">
                    <div style="font-size: 0.68rem; color: #9BA39B; text-transform: uppercase;">Weekday</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #FFF;">{weekday_pct:.0f}%</div>
                </div>
                <div class="sonova-kpi-card" style="padding: 0.7rem 0.85rem; margin-bottom: 0;">
                    <div style="font-size: 0.68rem; color: #9BA39B; text-transform: uppercase;">Weekend</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #7CFF35;">{weekend_pct:.0f}%</div>
                </div>
                <div class="sonova-kpi-card" style="padding: 0.7rem 0.85rem; margin-bottom: 0;">
                    <div style="font-size: 0.68rem; color: #9BA39B; text-transform: uppercase;">Shuffle</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #FFF;">{shuffle_pct:.0f}%</div>
                </div>
            </div>
            """
        )

    # Row 3: Key Insights cards matching Panel 3
    render_html('<div class="kpi-title" style="margin-top: 1.2rem;">Key Empirical Insights</div>')
    
    # 1. Most active day
    day_sums = clock_matrix.sum(axis=1)
    most_active_day = day_sums.idxmax() if len(day_sums) > 0 else "N/A"
    
    # 2. Peak listening time
    peak_hour = int(hourly_df.sort_values("plays", ascending=False).iloc[0]["hour"]) if len(hourly_df) > 0 else 20
    peak_start = int((peak_hour - 2) % 24)
    peak_end = int((peak_hour + 2) % 24)
    fmt_peak = f"{peak_start%12 or 12} {'PM' if peak_start>=12 else 'AM'} – {peak_end%12 or 12} {'PM' if peak_end>=12 else 'AM'}"
    
    # 3. Most consistent period (top year)
    top_year = str(df["year"].value_counts().idxmax()) if "year" in df.columns and len(df) > 0 else "2024"
    
    # 4. Activity intensity (Weekends vs Weekdays per day)
    wknd_avg = weekend_plays / 2.0
    wkdy_avg = weekday_plays / 5.0
    higher_act = "Weekends" if wknd_avg > wkdy_avg else "Weekdays"

    ki1, ki2, ki3, ki4 = st.columns(4)
    with ki1:
        render_html(
            f"""
            <div class="sonova-kpi-card">
                <div class="kpi-title">Most Active Day</div>
                <div style="font-size: 1.45rem; font-weight: 700; color: #7CFF35;">{most_active_day}</div>
                <div class="kpi-subtext">Highest aggregate play count</div>
            </div>
            """
        )
    with ki2:
        render_html(
            f"""
            <div class="sonova-kpi-card">
                <div class="kpi-title">Peak Listening Time</div>
                <div style="font-size: 1.45rem; font-weight: 700; color: #FFF;">{fmt_peak}</div>
                <div class="kpi-subtext">Peak 4-hour habit density</div>
            </div>
            """
        )
    with ki3:
        render_html(
            f"""
            <div class="sonova-kpi-card">
                <div class="kpi-title">Peak Volume Period</div>
                <div style="font-size: 1.45rem; font-weight: 700; color: #FFF;">{top_year}</div>
                <div class="kpi-subtext">Year with highest volume</div>
            </div>
            """
        )
    with ki4:
        render_html(
            f"""
            <div class="sonova-kpi-card">
                <div class="kpi-title">Higher Daily Activity</div>
                <div style="font-size: 1.45rem; font-weight: 700; color: #7CFF35;">{higher_act}</div>
                <div class="kpi-subtext">Average sessions per day</div>
            </div>
            """
        )


# =============================================================================
# 4. LISTENING PHASES — K-MEANS
# =============================================================================
elif selected_tab == "Listening Phases":
    if df is None:
        render_dataset_prompt()
        st.stop()

    render_html(
        """
        <div class="storytelling-label">UNSUPERVISED LEARNING &bull; BEHAVIORAL PATTERNS</div>
        <div class="editorial-headline">Listening Phases</div>
        <div class="storytelling-lead">Your listening isn't static. Different months show distinct behavioral phases.</div>
        """
    )

    monthly_df = artifacts["monthly"]
    cluster_results = compute_clustering(monthly_df)

    if "error" in cluster_results and cluster_results.get("best_k") is None:
        st.info(cluster_results["error"])
    else:
        best_k = cluster_results["best_k"]
        best_score = cluster_results["best_score"]
        scores_dict = cluster_results["silhouette_scores"]
        clustered_df = cluster_results["monthly_clustered"]
        profiles = cluster_results["cluster_profiles"]
        descriptions = cluster_results["cluster_descriptions"]

        # KPI Row matching Panel 4
        col_k, col_sc = st.columns(2)
        with col_k:
            render_html(
                f"""
                <div class="sonova-kpi-card">
                    <div class="kpi-title">Optimal Cluster Count (K)</div>
                    <div class="kpi-value">{best_k}</div>
                    <div class="kpi-subtext">Maximizes silhouette separation across candidate K ∈ [2, 8]</div>
                </div>
                """
            )
        with col_sc:
            render_html(
                f"""
                <div class="sonova-kpi-card">
                    <div class="kpi-title">Silhouette Score</div>
                    <div class="kpi-value">{best_score:.3f}</div>
                    <div class="kpi-subtext">Cluster separation metric evaluated in standardized feature space</div>
                </div>
                """
            )

        render_html("<div style='height: 0.8rem;'></div>")

        # Silhouette by K & Monthly Cluster Timeline matching Panel 4
        col_sil, col_time = st.columns([1, 1], gap="large")

        with col_sil:
            render_html('<div class="kpi-title">Silhouette Score by K</div>')
            sil_df = pd.DataFrame(
                {"K": list(scores_dict.keys()), "Silhouette Score": list(scores_dict.values())}
            )
            fig_sil = px.line(
                sil_df,
                x="K",
                y="Silhouette Score",
                markers=True,
                labels={"K": "Number of Clusters (K)", "Silhouette Score": "Score"},
            )
            fig_sil.update_traces(
                line_color="#7CFF35",
                line_width=2.4,
                marker=dict(size=7, color="#FFFFFF"),
                hovertemplate="<b>K = %{x}</b><br>Score: %{y:.3f}<extra></extra>",
            )
            st.plotly_chart(apply_sonova_theme(fig_sil, height=290), use_container_width=True)

        with col_time:
            render_html('<div class="kpi-title">Monthly Cluster Timeline</div>')
            # Use monochromatic green shades only — no random colors!
            green_palette = ["#7CFF35", "#166534", "#4ADE80", "#15803D", "#86EFAC"]
            fig_time = px.scatter(
                clustered_df,
                x="month",
                y="cluster",
                color=clustered_df["cluster"].astype(str),
                labels={"month": "Month", "cluster": "Assigned Phase"},
                color_discrete_sequence=green_palette,
            )
            fig_time.update_traces(
                marker=dict(size=9, opacity=0.9),
                hovertemplate="<b>%{x}</b><br>Phase: %{y}<extra></extra>",
            )
            fig_time.update_yaxes(tickmode="linear", tick0=0, dtick=1)
            st.plotly_chart(apply_sonova_theme(fig_time, height=290), use_container_width=True)

        # Cluster Profiles Cards matching Panel 4
        render_html('<div class="kpi-title" style="margin-top: 1.2rem;">Evidence-Based Cluster Profiles</div>')

        prof_cols = st.columns(len(descriptions))
        total_months = len(clustered_df)

        for idx, (c_id, desc) in enumerate(descriptions.items()):
            col_target = prof_cols[idx] if idx < len(prof_cols) else prof_cols[0]
            count_m = (clustered_df["cluster"] == c_id).sum()
            pct_m = (count_m / total_months) * 100.0 if total_months > 0 else 0
            
            # Split description traits
            traits = [t.strip() for t in desc.split("—")]
            lead_trait = traits[0] if len(traits) > 0 else f"Phase {c_id}"
            sub_traits = traits[1:] if len(traits) > 1 else []

            with col_target:
                traits_html = "".join([f"<li style='margin-bottom:0.25rem;'>{t.capitalize()}</li>" for t in sub_traits])
                render_html(
                    f"""
                    <div class="sonova-card" style="padding: 1.2rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-weight: 700; color: #7CFF35; font-size: 1.05rem;">Phase {c_id}</span>
                            <span class="badge-pill">{count_m} months ({pct_m:.0f}%)</span>
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #FFF; margin-bottom: 0.7rem;">
                            {lead_trait}
                        </div>
                        <ul style="font-size: 0.82rem; color: #9BA39B; padding-left: 1.1rem; margin-bottom: 0.9rem; line-height: 1.5;">
                            {traits_html}
                        </ul>
                        <div class="bar-track" style="width: 100%; height: 5px; margin-top: 0.4rem;">
                            <div class="bar-fill" style="width: {pct_m}%;"></div>
                        </div>
                    </div>
                    """
                )

        # Feature Averages Table
        with st.expander("View Numerical Cluster Feature Means", expanded=False):
            st.dataframe(
                profiles.rename(
                    columns={
                        "cluster": "Cluster ID",
                        "total_plays": "Avg Plays",
                        "unique_tracks": "Avg Tracks",
                        "unique_artists": "Avg Artists",
                        "total_minutes": "Avg Minutes",
                        "average_minutes": "Track Duration (min)",
                        "skip_rate": "Skip Rate",
                        "night_listening_ratio": "Night %",
                        "weekend_listening_ratio": "Weekend %",
                        "repeat_rate": "Repeat Rate",
                        "discovery_rate": "Discovery Rate",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# 5. SKIP PREDICTION (SUPERVISED ML)
# =============================================================================
elif selected_tab == "Skip Prediction":
    if df is None:
        render_dataset_prompt()
        st.stop()

    render_html(
        """
        <div class="storytelling-label">SUPERVISED LEARNING &bull; EVENT-LEVEL CLASSIFICATION</div>
        <div class="editorial-headline">Supervised Skip Prediction</div>
        <div class="storytelling-lead">Binary event-level classification predicting whether a track will be skipped based on non-leaking contextual signals.</div>
        """
    )

    with st.spinner("Evaluating supervised ML models and cross-validation..."):
        ml_results = compute_skip_evaluation(df)

    if "error" in ml_results:
        st.error(ml_results["error"])
    else:
        perf_df = ml_results["performance_df"]
        cv_df = ml_results["cv_df"]
        tt_df = ml_results["train_test_df"]
        final_model_name = ml_results["final_model_name"]
        cm_data = ml_results["confusion_matrix"]
        final_pipe = ml_results["final_pipeline"]
        best_row = perf_df[perf_df["Model"] == final_model_name].iloc[0]

        # Selected Final Model Banner matching Panel 5
        render_html(
            f"""
            <div class="sonova-card" style="border: 1px solid rgba(124, 255, 53, 0.45); background: rgba(14, 24, 16, 0.9);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap;">
                    <div class="storytelling-label" style="margin-bottom: 0;">FINAL MODEL — {final_model_name}</div>
                    <span class="badge-pill" style="font-size: 0.74rem;">Test Set Evaluation</span>
                </div>
                <div style="font-size: 2rem; font-family: 'Instrument Serif', Georgia, serif; color: #FFF; margin: 0.2rem 0 0.8rem 0;">
                    {final_model_name}
                </div>
                
                <div style="display: flex; gap: 2.2rem; flex-wrap: wrap; margin-bottom: 1rem; border-top: 1px solid rgba(140,255,70,0.1); padding-top: 0.8rem;">
                    <div>
                        <div class="kpi-title">Accuracy</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #FFF;">{best_row['Accuracy']:.3f}</div>
                        <div style="font-size: 0.7rem; color: #9BA39B;">Overall correct</div>
                    </div>
                    <div>
                        <div class="kpi-title">Precision</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #FFF;">{best_row['Precision']:.3f}</div>
                        <div style="font-size: 0.7rem; color: #9BA39B;">True skip accuracy</div>
                    </div>
                    <div>
                        <div class="kpi-title">Recall</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #FFF;">{best_row['Recall']:.3f}</div>
                        <div style="font-size: 0.7rem; color: #9BA39B;">Skips detected</div>
                    </div>
                    <div>
                        <div class="kpi-title" style="color: var(--accent-neon);">Test F1 (Primary)</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: var(--accent-neon);">{best_row['F1']:.3f}</div>
                        <div style="font-size: 0.7rem; color: #7CFF35;">Precision & Recall balance</div>
                    </div>
                    <div>
                        <div class="kpi-title">ROC-AUC</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #FFF;">{best_row['ROC-AUC']:.3f}</div>
                        <div style="font-size: 0.7rem; color: #9BA39B;">Class discrimination</div>
                    </div>
                </div>
                
                <div style="font-size: 0.82rem; color: #9BA39B; line-height: 1.55;">
                    Selected using <strong>test-set F1</strong> as the project's primary evaluation criterion. 
                    Because the skip target is imbalanced (~26% skipped), F1 provides a more rigorous indicator of minority-class detection than accuracy alone.
                </div>
            </div>
            """
        )

        render_html("<div style='height: 0.8rem;'></div>")

        # Model Comparison Table & Metric Selector Chart matching Panel 5
        col_tbl, col_chart = st.columns([1.1, 0.9], gap="large")

        with col_tbl:
            render_html('<div class="kpi-title">Model Comparison Matrix (Test Set)</div>')
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

        with col_chart:
            render_html('<div class="kpi-title">Compare by Metric</div>')
            chosen_metric = st.selectbox(
                "Select Metric to Compare",
                ["F1", "Accuracy", "Precision", "Recall", "ROC-AUC"],
                label_visibility="collapsed",
            )
            # Use shades of green only!
            fig_metric = px.bar(
                perf_df,
                x="Model",
                y=chosen_metric,
                color="Model",
                color_discrete_sequence=["#7CFF35", "#8DFF45", "#22C55E", "#16A34A", "#15803D"],
            )
            fig_metric.update_traces(
                marker_line_width=0,
                opacity=0.9,
                hovertemplate="<b>%{x}</b><br>" + chosen_metric + ": %{y:.4f}<extra></extra>",
            )
            fig_metric.update_layout(showlegend=False)
            st.plotly_chart(apply_sonova_theme(fig_metric, height=270), use_container_width=True)

        render_html("<div style='height: 1rem;'></div>")

        # 5-Fold Stratified Cross-Validation & Confusion Matrix matching Panel 5
        col_cv, col_cm = st.columns([1.1, 0.9], gap="large")

        with col_cv:
            render_html('<div class="kpi-title">5-Fold Stratified Cross-Validation (F1 Metric)</div>')
            st.dataframe(cv_df, use_container_width=True, hide_index=True)
            st.caption("Evaluated across 5 stratified folds on the training set to verify stability across data slices.")

        with col_cm:
            render_html(f'<div class="kpi-title">Confusion Matrix: {final_model_name} (Test Set)</div>')
            tn = cm_data["tn"]
            fp = cm_data["fp"]
            fn = cm_data["fn"]
            tp = cm_data["tp"]

            # Visual 2x2 grid matching Panel 5
            render_html(
                f"""
                <div class="sonova-card" style="padding: 1.1rem;">
                    <div style="display: grid; grid-template-columns: 80px 1fr 1fr; gap: 0.5rem; text-align: center; align-items: center;">
                        <div></div>
                        <div style="font-size: 0.76rem; font-weight: 700; color: #9BA39B; text-transform: uppercase;">Pred: No Skip</div>
                        <div style="font-size: 0.76rem; font-weight: 700; color: #9BA39B; text-transform: uppercase;">Pred: Skip</div>
                        
                        <div style="font-size: 0.76rem; font-weight: 700; color: #9BA39B; text-align: right; padding-right: 0.5rem;">Act: No Skip</div>
                        <div style="background: rgba(124, 255, 53, 0.14); border: 1px solid rgba(124, 255, 53, 0.35); border-radius: 8px; padding: 0.85rem 0.5rem;">
                            <div style="font-size: 1.35rem; font-weight: 700; color: #FFF;">{tn:,}</div>
                            <div style="font-size: 0.68rem; color: #7CFF35;">True Negative</div>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.85rem 0.5rem;">
                            <div style="font-size: 1.35rem; font-weight: 700; color: #9BA39B;">{fp:,}</div>
                            <div style="font-size: 0.68rem; color: #6B7280;">False Positive</div>
                        </div>
                        
                        <div style="font-size: 0.76rem; font-weight: 700; color: #9BA39B; text-align: right; padding-right: 0.5rem;">Act: Skip</div>
                        <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.85rem 0.5rem;">
                            <div style="font-size: 1.35rem; font-weight: 700; color: #9BA39B;">{fn:,}</div>
                            <div style="font-size: 0.68rem; color: #6B7280;">False Negative</div>
                        </div>
                        <div style="background: rgba(124, 255, 53, 0.22); border: 1px solid rgba(124, 255, 53, 0.5); border-radius: 8px; padding: 0.85rem 0.5rem;">
                            <div style="font-size: 1.35rem; font-weight: 700; color: #7CFF35;">{tp:,}</div>
                            <div style="font-size: 0.68rem; color: #7CFF35;">True Positive</div>
                        </div>
                    </div>
                    <div style="font-size: 0.74rem; color: #6B7280; text-align: center; margin-top: 0.8rem;">
                        Evaluated on {ml_results['test_sample_count']:,} hold-out test listening events.
                    </div>
                </div>
                """
            )

        # Generalization Diagnostics Table
        render_html('<div class="kpi-title" style="margin-top: 1.2rem;">Training vs. Test F1 (Generalization Diagnostic)</div>')
        st.dataframe(
            tt_df[["Model", "Train F1", "Test F1", "Gap", "Diagnostic"]],
            use_container_width=True,
            hide_index=True,
        )

        render_html("<hr style='border-color: rgba(140,255,70,0.1); margin: 2rem 0;'>")

        # Practical Skip Prediction Interactive Tool
        render_html('<div class="kpi-title">Practical Skip Prediction Simulator</div>')
        st.caption("Simulate an upcoming listening event using the trained pipeline.")

        col_sim1, col_sim2, col_sim3 = st.columns(3)
        with col_sim1:
            sim_hour = st.slider("Hour of Day (0–23)", 0, 23, 21)
            sim_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
        with col_sim2:
            sim_shuffle = st.checkbox("Shuffle Active?", value=True)
            sim_reason = st.selectbox("Playback Start Reason", ["trackdone", "fwdbtn", "appload", "backbtn", "clickrow", "unknown"], index=0)
        with col_sim3:
            popular_artists = artifacts["top_artists"]["artist"].tolist() if len(artifacts["top_artists"]) > 0 else ["BTS"]
            sim_artist = st.selectbox("Artist", popular_artists, index=0)

        is_wknd = 1 if sim_day in ["Saturday", "Sunday"] else 0
        outcome, prob = predict_single_event(
            final_pipe,
            hour=sim_hour,
            day_of_week=sim_day,
            is_weekend=is_wknd,
            shuffle=1 if sim_shuffle else 0,
            reason_start=sim_reason,
            artist=sim_artist,
        )

        outcome_color = "#EF4444" if "SKIP" in outcome else "#7CFF35"

        render_html(
            f"""
            <div class="sonova-card" style="text-align: center; padding: 1.5rem; margin-top: 1rem; border-color: {outcome_color};">
                <div class="kpi-title">MODEL INFERENCE RESULT</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {outcome_color}; letter-spacing: 0.05em; margin: 0.3rem 0;">
                    {outcome}
                </div>
                <div style="font-size: 1.05rem; color: #FFF;">
                    Predicted Skip Probability: <strong>{prob:.1f}%</strong>
                </div>
                <div style="font-size: 0.78rem; color: #9BA39B; margin-top: 0.6rem;">
                    Prediction based on behavioral patterns learned from the uploaded listening history.
                </div>
            </div>
            """
        )


# =============================================================================
# 6. PERSONALIZED RECOMMENDATIONS PAGE
# =============================================================================
elif selected_tab == "Recommendations":
    if df is None:
        render_dataset_prompt()
        st.stop()

    render_html(
        """
        <div class="storytelling-label">BEHAVIOR-BASED CANDIDATE RANKING</div>
        <div class="editorial-headline">Made for your listening.</div>
        <div class="storytelling-lead">Recommendations based on patterns found in your listening history.</div>
        """
    )

    # Context Selector Options matching Panel 6
    context_options = [
        "General (All-Round Favorites)",
        "Night (22:00 – 05:00)",
        "Weekend Sessions",
        "Exploration (Hidden Gems)",
        "Repeat Favorites",
    ]

    selected_context = st.selectbox(
        "Select Listening Context",
        context_options,
        index=0,
        label_visibility="collapsed",
    )

    recs_df = compute_recommendations_cached(df, selected_context)

    render_html("<div style='height: 0.6rem;'></div>")

    # Render editorial music recommendation cards matching Panel 6
    rec_cards_html = ""
    for idx, row in recs_df.iterrows():
        rank = idx + 1
        track = row["Track Name"]
        artist = row["Artist"]
        plays = row["Total Plays"]
        skip_rate = row["Skip Rate"]
        score = row["Score"]
        ctx_desc = row["Behavioral Context"]
        artwork_uri = get_abstract_cover_data_uri(idx)

        rec_cards_html += f"""
        <div class="rec-card">
            <span class="rank-num" style="font-size: 0.95rem; margin-right: 0.8rem;">{rank}</span>
            <img src="{artwork_uri}" class="rec-cover" alt="cover"/>
            <div style="flex: 1; min-width: 0; padding-right: 1.2rem;">
                <div style="font-size: 0.96rem; font-weight: 700; color: #FFF; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {track}
                </div>
                <div style="font-size: 0.78rem; color: #9BA39B; margin-top: 0.15rem;">
                    {artist}
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 1.5rem; flex-shrink: 0;">
                <div style="text-align: right; min-width: 70px;">
                    <div style="font-size: 0.7rem; color: #5A635A; text-transform: uppercase;">Plays</div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #FFF;">{plays:,}</div>
                </div>
                <div style="text-align: right; min-width: 70px;">
                    <div style="font-size: 0.7rem; color: #5A635A; text-transform: uppercase;">Skip Rate</div>
                    <div style="font-size: 0.88rem; font-weight: 600; color: #7CFF35;">{skip_rate}</div>
                </div>
                <div style="text-align: right; min-width: 60px;">
                    <div style="font-size: 0.7rem; color: #5A635A; text-transform: uppercase;">Score</div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #FFF;">{score:.2f}</div>
                </div>
                <span class="badge-pill" style="min-width: 140px; text-align: center;">{ctx_desc}</span>
            </div>
        </div>
        """

    render_html(rec_cards_html)

    # Context Exploration Highlights matching Panel 6
    render_html('<div class="kpi-title" style="margin-top: 1.6rem;">Explore Different Contexts</div>')
    c_ctx1, c_ctx2, c_ctx3, c_ctx4, c_ctx5 = st.columns(5)
    with c_ctx1:
        render_html(
            """
            <div class="sonova-kpi-card" style="padding: 0.9rem;">
                <div style="font-weight: 700; color: #FFF; font-size: 0.85rem;">General</div>
                <div style="font-size: 0.74rem; color: #9BA39B; margin-top: 0.2rem;">All-round lifetime favorites</div>
            </div>
            """
        )
    with c_ctx2:
        render_html(
            """
            <div class="sonova-kpi-card" style="padding: 0.9rem;">
                <div style="font-weight: 700; color: #7CFF35; font-size: 0.85rem;">Night</div>
                <div style="font-size: 0.74rem; color: #9BA39B; margin-top: 0.2rem;">Late night 22:00 – 05:00</div>
            </div>
            """
        )
    with c_ctx3:
        render_html(
            """
            <div class="sonova-kpi-card" style="padding: 0.9rem;">
                <div style="font-weight: 700; color: #FFF; font-size: 0.85rem;">Weekend</div>
                <div style="font-size: 0.74rem; color: #9BA39B; margin-top: 0.2rem;">Weekend session energy</div>
            </div>
            """
        )
    with c_ctx4:
        render_html(
            """
            <div class="sonova-kpi-card" style="padding: 0.9rem;">
                <div style="font-weight: 700; color: #7CFF35; font-size: 0.85rem;">Exploration</div>
                <div style="font-size: 0.74rem; color: #9BA39B; margin-top: 0.2rem;">Low-frequency hidden gems</div>
            </div>
            """
        )
    with c_ctx5:
        render_html(
            """
            <div class="sonova-kpi-card" style="padding: 0.9rem;">
                <div style="font-weight: 700; color: #FFF; font-size: 0.85rem;">Repeat Favorites</div>
                <div style="font-size: 0.74rem; color: #9BA39B; margin-top: 0.2rem;">Deep cuts with high loyalty</div>
            </div>
            """
        )

    with st.expander("Inspect Raw Recommendation Dataframe", expanded=False):
        st.dataframe(recs_df, use_container_width=True, hide_index=True)


# =============================================================================
# 7. BLOG PAGE (Product Updates)
# =============================================================================
elif selected_tab == "Blog":
    render_html(
        """
        <div class="storytelling-label">PRODUCT NEWS &bull; WHAT'S NEXT</div>
        <div class="editorial-headline">SONOVA Updates</div>
        <div class="storytelling-lead">Upcoming features and recent releases, all in one place.</div>
        """
    )

    if not PRODUCT_UPDATES:
        render_html(
            """
            <div class="sonova-card" style="color: #9BA39B; text-align: center; padding: 2rem;">
                More updates coming soon.
            </div>
            """
        )
    else:
        for update in PRODUCT_UPDATES:
            status_color = "#B4FF5C" if update["status"] == "Coming soon" else "#7CFF35"
            render_html(
                f"""
                <div class="sonova-card" style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.75rem;">
                        <span style="color: {status_color}; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;">
                            {update['status']}
                        </span>
                        <time style="color: #5A635A; font-size: 0.8rem;" datetime="{update['date']}">{update['date']}</time>
                    </div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #F4F5F0; margin-bottom: 0.4rem;">
                        {update['title']}
                    </div>
                    <div style="font-size: 0.9rem; color: #9BA39B; line-height: 1.6;">
                        {update['body']}
                    </div>
                </div>
                """
            )


# =============================================================================
# 8. ABOUT PAGE (Authorship, Methodology & Privacy)
# =============================================================================
elif selected_tab == "About":

    render_html(
        """
        <div class="storytelling-label">PRODUCT ARCHITECTURE &bull; METHODOLOGY</div>
        <div class="editorial-headline">SONOVA</div>
        <div class="brand-subtitle" style="margin-bottom: 1.2rem;">PERSONAL MUSIC INTELLIGENCE</div>
        <div style="font-size: 1.05rem; color: #9BA39B; max-width: 760px; line-height: 1.6; margin-bottom: 2rem;">
            SONOVA transforms your Spotify listening history into a visual picture of your listening behavior, 
            patterns and preferences using data analysis and machine learning.
        </div>
        """
    )

    # Visual Architecture Pipeline Flow
    render_html('<div class="kpi-title" style="margin-bottom: 1rem;">End-to-End Intelligence Pipeline</div>')
    
    pipeline_steps = [
        ("1. Your Spotify History", "Extended Streaming History JSON archive"),
        ("2. Data Processing", "In-memory parsing, telemetry stripped"),
        ("3. Behavioral Features", "Temporal rhythms, skip ratios, repeat frequency"),
        ("4. Listening Analysis", "Exploratory analytics, listening clock heatmap"),
        ("5. K-Means Phases", "Unsupervised behavioral clusters across months"),
        ("6. Skip Prediction", "5 supervised models & generalization diagnostics"),
        ("7. Recommendations", "Multi-objective contextual candidate scoring"),
    ]

    cols_pipe = st.columns(len(pipeline_steps))
    for idx, (title, sub) in enumerate(pipeline_steps):
        with cols_pipe[idx]:
            render_html(
                f"""
                <div class="sonova-kpi-card" style="padding: 0.85rem; height: 110px;">
                    <div style="color: #7CFF35; font-weight: 700; font-size: 0.78rem;">{title}</div>
                    <div style="font-size: 0.72rem; color: #9BA39B; margin-top: 0.4rem; line-height: 1.4;">{sub}</div>
                </div>
                """
            )

    render_html("<div style='height: 1.5rem;'></div>")

    # Core System Pillars
    col_p1, col_p2 = st.columns(2, gap="large")

    with col_p1:
        render_html(
            """
            <div class="sonova-card">
                <div class="kpi-title">01 &bull; In-Memory Processing & Data Privacy</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #FFF; margin: 0.3rem 0;">
                    Zero Cloud Ingestion & In-Memory Processing
                </div>
                <div style="font-size: 0.84rem; color: #9BA39B; line-height: 1.6;">
                    SONOVA processes personal listening archives entirely within memory during your active session. Sensitive network metadata 
                    such as IP addresses, user agents, and geographic coordinates are dropped immediately during ingestion. 
                    No external web APIs or remote Spotify credential logins are required.
                </div>
            </div>

            <div class="sonova-card">
                <div class="kpi-title">02 &bull; Unsupervised Behavioral Segmentation</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #FFF; margin: 0.3rem 0;">
                    K-Means Listening Phase Discovery
                </div>
                <div style="font-size: 0.84rem; color: #9BA39B; line-height: 1.6;">
                    Rather than imposing static genre labels, SONOVA aggregates listening into monthly behavioral vectors. 
                    The unsupervised K-Means algorithm evaluates candidate cluster counts across K &isin; [2, 8], selecting 
                    the optimal partition based on maximum silhouette score and generating evidence-based descriptions.
                </div>
            </div>
            """
        )

    with col_p2:
        render_html(
            """
            <div class="sonova-card">
                <div class="kpi-title">03 &bull; Supervised Predictive Modeling</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #FFF; margin: 0.3rem 0;">
                    Rigorous Skip Classification & Diagnostics
                </div>
                <div style="font-size: 0.84rem; color: #9BA39B; line-height: 1.6;">
                    Event-level skip prediction compares five classifier families (Logistic Regression, Decision Tree, 
                    K-NN, Calibrated LinearSVC, and Random Forest). Performance is evaluated with 5-fold stratified cross-validation 
                    and train/test generalization gap diagnostics, selecting the final model on test-set F1.
                </div>
            </div>

            <div class="sonova-card">
                <div class="kpi-title">04 &bull; Contextual Candidate Engine</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #FFF; margin: 0.3rem 0;">
                    Multi-Objective Behavioral Ranking
                </div>
                <div style="font-size: 0.84rem; color: #9BA39B; line-height: 1.6;">
                    Recommendations are ranked through mathematical composite scoring balancing logarithmic play frequency, 
                    empirical completion rate (1 &minus; skip rate), and time decay recency, with dynamic context weighting for 
                    night, weekend, exploration, and loyalty scenarios.
                </div>
            </div>
            """
        )

# -----------------------------------------------------------------------------
# GLOBAL FOOTER
# -----------------------------------------------------------------------------
render_html(
    """
    <div style="text-align: center; margin-top: 3.5rem; border-top: 1px solid rgba(140,255,70,0.08); padding: 2.5rem 1rem 1.5rem; line-height: 1.6;">
        <div style="font-size: 0.78rem; font-weight: 700; letter-spacing: 0.16em; color: #7CFF35;">
            SONOVA <span style="color: #5A635A;">&bull;</span> <span style="color: #9BA39B;">PERSONAL MUSIC INTELLIGENCE</span>
        </div>
        <div style="font-size: 0.75rem; color: #5A635A; margin-top: 1.15rem;">&copy; 2026 SONOVA. All rights reserved.</div>
    </div>
    """
)
