"""Immunopeptidomics QC Webapp — navigation shell.

Handles page config, session-state initialisation, prototype gate, sidebar
layout, and page routing via st.navigation. All analysis logic lives in the
individual page files under pages/.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import base64
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

# ── Sidebar logo/title — rendered above navigation links via st.logo() ────────

_LOGO_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" width="240" height="52" viewBox="0 0 240 52">
  <text x="2" y="32"
    font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    font-size="26" font-weight="700" fill="#0f172a">IPQC</text>
  <text x="3" y="48"
    font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
    font-size="13" fill="#64748b">Immunopeptidomics QC</text>
</svg>"""

_LOGO_URL = "data:image/svg+xml;base64," + base64.b64encode(_LOGO_SVG.encode()).decode()
st.logo(_LOGO_URL, size="large")

# ── Navigation ────────────────────────────────────────────────────────────────

pg = st.navigation([
    st.Page("pages/2_About.py",           title="About",           icon=":material/info:"),
    st.Page("pages/3_Glossary.py",        title="Glossary",        icon=":material/menu_book:"),
    st.Page("pages/analysis.py",          title="Analysis",        icon=":material/science:"),
    st.Page("pages/4_My_Runs.py",         title="My Runs",         icon=":material/folder_open:"),
    st.Page("pages/5_Compare_Runs.py",    title="Compare Runs",    icon=":material/compare_arrows:"),
    st.Page("pages/6_IEDB_Prediction.py", title="MHC-I Prediction",icon=":material/biotech:"),
])

# ── Sidebar: auth block + footer (appears below navigation links) ─────────────

with st.sidebar:
    st.divider()
    render_sidebar_auth()
    st.divider()
    st.caption("Developed by the He Lab at the Princess Margaret Cancer Centre 2026")

# ── Run the selected page ─────────────────────────────────────────────────────

pg.run()
