"""My Runs page — view, reopen, delete, and compare saved runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from modules.database import delete_run, get_run, get_runs_for_user, init_db
from modules.storage import deserialize_run

init_db()

st.markdown(
    "<style>div[data-testid='column'] button p { white-space: nowrap; }</style>",
    unsafe_allow_html=True,
)

st.title("My Runs")

if not st.session_state.get("user_id"):
    st.info(
        "Log in from the sidebar to save and access your runs. "
        "Use the sidebar to create an account or log in."
    )
    st.stop()

user_id: int = int(st.session_state["user_id"])
runs = get_runs_for_user(user_id)

if not runs:
    st.markdown(
        "No saved runs yet. Complete an analysis on the Analysis page and click "
        "**Save to My Runs** to store it here."
    )
    st.stop()

# ── Comparison selection state ─────────────────────────────────────────────────

if "compare_a" not in st.session_state:
    st.session_state["compare_a"] = None
if "compare_b" not in st.session_state:
    st.session_state["compare_b"] = None

# ── Table header ──────────────────────────────────────────────────────────────

st.markdown(f"**{len(runs)} saved run(s)**")
st.markdown("")

hdr = st.columns([3, 2, 1, 2, 2, 3])
hdr[0].markdown("**Run Name**")
hdr[1].markdown("**Date**")
hdr[2].markdown("**Samples**")
hdr[3].markdown("**Median Peptides**")
hdr[4].markdown("**Median MBR Rate**")
hdr[5].markdown("**Actions**")
st.divider()

# ── Run rows ──────────────────────────────────────────────────────────────────

for run in runs:
    run_id: int = run["id"]
    summary: dict = json.loads(run["summary_json"] or "{}")

    col_name, col_date, col_n, col_med, col_mbr, col_actions = st.columns(
        [3, 2, 1, 2, 2, 3]
    )
    with col_name:
        st.markdown(f"**{run['name']}**")
    with col_date:
        st.markdown(run["upload_date"][:10])
    with col_n:
        st.markdown(str(run["sample_count"]))
    with col_med:
        med = summary.get("median_peptide_count")
        st.markdown(f"{med:,.0f}" if med is not None else "—")
    with col_mbr:
        mbr = summary.get("median_mbr_rate")
        st.markdown(f"{mbr:.1f}%" if mbr is not None else "—")
    with col_actions:
        ca, cb, cc = st.columns(3)

        with ca:
            if st.button("Open", key=f"open_{run_id}"):
                rec = get_run(run_id, user_id)
                if rec and rec.get("data_dir"):
                    try:
                        df_loaded, mapping_loaded, ms_df_loaded = deserialize_run(rec["data_dir"])
                        st.session_state["df"] = df_loaded
                        st.session_state["mapping"] = mapping_loaded
                        st.session_state["ms_df"] = ms_df_loaded
                        st.session_state["step"] = 3
                        st.session_state["run_label"] = rec["name"]
                        st.success("Run loaded. Navigate to Analysis to view the report.")
                    except Exception as exc:
                        st.error(f"Failed to load run: {exc}")
                else:
                    st.error("Run data not found on disk.")

        with cb:
            if st.button("Delete", key=f"del_{run_id}"):
                st.session_state[f"confirm_del_{run_id}"] = True
                st.rerun()

        with cc:
            a_id = st.session_state["compare_a"]
            b_id = st.session_state["compare_b"]
            is_a = a_id == run_id
            is_b = b_id == run_id
            btn_label = "Selected A" if is_a else ("Selected B" if is_b else "Compare")
            if st.button(btn_label, key=f"cmp_{run_id}"):
                if is_a:
                    st.session_state["compare_a"] = None
                elif is_b:
                    st.session_state["compare_b"] = None
                elif st.session_state["compare_a"] is None:
                    st.session_state["compare_a"] = run_id
                elif st.session_state["compare_b"] is None:
                    st.session_state["compare_b"] = run_id
                st.rerun()

    # Delete confirmation
    if st.session_state.get(f"confirm_del_{run_id}"):
        st.warning(f"Delete run '{run['name']}'? This action cannot be undone.")
        cc1, cc2, _ = st.columns([1, 1, 4])
        with cc1:
            if st.button("Confirm delete", key=f"yes_del_{run_id}", type="primary"):
                delete_run(run_id, user_id)
                st.session_state.pop(f"confirm_del_{run_id}", None)
                if st.session_state.get("compare_a") == run_id:
                    st.session_state["compare_a"] = None
                if st.session_state.get("compare_b") == run_id:
                    st.session_state["compare_b"] = None
                st.rerun()
        with cc2:
            if st.button("Cancel", key=f"no_del_{run_id}"):
                st.session_state.pop(f"confirm_del_{run_id}", None)
                st.rerun()

    st.divider()

# ── Comparison action ─────────────────────────────────────────────────────────

a_id = st.session_state["compare_a"]
b_id = st.session_state["compare_b"]

if a_id is not None and b_id is not None:
    st.success(
        f"Two runs selected (IDs: {a_id} and {b_id}). "
        "Navigate to Compare Runs to view the side-by-side comparison."
    )
elif a_id is not None:
    st.info("Run A selected. Click **Compare** on a second run to select run B.")
