"""Immunopeptidomics QC Webapp — navigation shell.

Handles page config, session-state initialisation, prototype gate, sidebar
layout, and page routing via st.navigation. All analysis logic lives in the
individual page files under pages/.

Run:
    streamlit run app.py
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from modules.database import init_db
from modules.ui_utils import check_prototype_gate, inject_sidebar_css, render_sidebar_auth

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Immunopeptidomics QC",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ── Session state initialisation ──────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "step": 1,
    "df_raw": None,
    "df": None,
    "delimiter": "\t",
    "mapping": None,
    "ms_df": None,
    "user_id": None,
    "username": None,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Always-visible sidebar CSS ────────────────────────────────────────────────

inject_sidebar_css()

# ── Prototype gate (stops rendering if not authenticated) ─────────────────────

check_prototype_gate()

# ── Sidebar: application title (appears above navigation links) ───────────────

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 0.75rem 0;">
            <div style="font-size: 1.4rem; font-weight: 700; line-height: 1.2;">IPQC</div>
            <div style="font-size: 0.85rem; color: #888; margin-top: 2px;">Immunopeptidomics QC</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Navigation ────────────────────────────────────────────────────────────────

pg = st.navigation([
    st.Page("pages/2_About.py",           title="About",           icon=":material/info:"),
    st.Page("pages/3_Glossary.py",        title="Glossary",        icon=":material/menu_book:"),
    st.Page("pages/4_My_Runs.py",         title="My Runs",         icon=":material/folder_open:"),
    st.Page("pages/5_Compare_Runs.py",    title="Compare Runs",    icon=":material/compare_arrows:"),
    st.Page("pages/analysis.py",          title="MS Analysis",     icon=":material/science:"),
    st.Page("pages/8_HLA_Typing.py",      title="HLA Typing",      icon=":material/genetics:"),
    st.Page("pages/6_MHC_Prediction.py",  title="MHC Prediction",  icon=":material/biotech:"),
    st.Page("pages/7_Diagnostics.py",     title="Diagnostics",     icon=":material/monitor_heart:"),
])

# ── Sidebar: auth block + footer (appears below navigation links) ─────────────

with st.sidebar:
    st.divider()
    render_sidebar_auth()
    st.divider()
    st.caption("Developed by the He Lab at the Princess Margaret Cancer Centre 2026")

# ── Run the selected page ─────────────────────────────────────────────────────

pg.run()
