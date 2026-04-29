"""Compare Runs page — side-by-side QC comparison of two saved runs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from modules.charts import (
    chart_contaminant_rate,
    chart_length_all,
    chart_mbr_rate,
    chart_msms_mbr,
)
from modules.database import get_run, init_db
from modules.mapping import ColumnMapping
from modules.metrics import (
    compute_contaminant_summary,
    compute_dataset_stats,
    compute_sample_summary,
)
from modules.storage import deserialize_run

init_db()

st.title("Compare Runs")

if not st.session_state.get("user_id"):
    st.info("Log in to compare saved runs.")
    st.stop()

user_id: int = int(st.session_state["user_id"])
a_id: int | None = st.session_state.get("compare_a")
b_id: int | None = st.session_state.get("compare_b")

if a_id is None or b_id is None:
    st.info(
        "Select two runs from the My Runs page to compare them here. "
        "Use the **Compare** button on each run row to mark runs A and B."
    )
    st.stop()

# ── Load both runs ────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading run data...")
def _load_run(
    run_id: int, uid: int
) -> tuple[pd.DataFrame, ColumnMapping, pd.DataFrame | None, dict] | None:
    rec = get_run(run_id, uid)
    if not rec or not rec.get("data_dir"):
        return None
    try:
        df, mapping, ms_df = deserialize_run(rec["data_dir"])
        return df, mapping, ms_df, rec
    except Exception:
        return None


run_a = _load_run(a_id, user_id)
run_b = _load_run(b_id, user_id)

if run_a is None or run_b is None:
    st.error(
        "One or both runs could not be loaded. Return to My Runs and verify your selections."
    )
    st.stop()

df_a, mapping_a, ms_a, rec_a = run_a
df_b, mapping_b, ms_b, rec_b = run_b

label_a = f"{rec_a['name']} ({rec_a['upload_date'][:10]})"
label_b = f"{rec_b['name']} ({rec_b['upload_date'][:10]})"

st.markdown(f"Comparing **{label_a}** against **{label_b}**.")
st.markdown("")

# ── Summary metrics table ─────────────────────────────────────────────────────

st.subheader("Dataset-Level Summary Metrics")

stats_a = dict(compute_dataset_stats(df_a, mapping_a))
stats_b = dict(compute_dataset_stats(df_b, mapping_b))
all_keys = list(dict.fromkeys(list(stats_a.keys()) + list(stats_b.keys())))
compare_rows = [
    {"Metric": k, label_a: stats_a.get(k, "—"), label_b: stats_b.get(k, "—")}
    for k in all_keys
]
st.dataframe(pd.DataFrame(compare_rows).set_index("Metric"), use_container_width=True)

# ── Per-sample summaries ──────────────────────────────────────────────────────

st.subheader("Per-Sample Identification Rates")

sum_a = compute_sample_summary(df_a, mapping_a)
sum_b = compute_sample_summary(df_b, mapping_b)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**{label_a}**")
    if not sum_a.empty:
        display_cols_a = [c for c in sum_a.columns if c not in ("MBR Rate %", "Contam Rate %")]
        st.dataframe(sum_a[display_cols_a], use_container_width=True, height=260)
    else:
        st.caption("No per-sample data available.")
with col_b:
    st.markdown(f"**{label_b}**")
    if not sum_b.empty:
        display_cols_b = [c for c in sum_b.columns if c not in ("MBR Rate %", "Contam Rate %")]
        st.dataframe(sum_b[display_cols_b], use_container_width=True, height=260)
    else:
        st.caption("No per-sample data available.")

# ── MS/MS vs MBR ─────────────────────────────────────────────────────────────

st.subheader("MS/MS vs Match Between Runs")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**{label_a}**")
    if not sum_a.empty:
        st.plotly_chart(chart_msms_mbr(sum_a), use_container_width=True)
    else:
        st.caption("No identification data available.")
with col_b:
    st.markdown(f"**{label_b}**")
    if not sum_b.empty:
        st.plotly_chart(chart_msms_mbr(sum_b), use_container_width=True)
    else:
        st.caption("No identification data available.")

col_a2, col_b2 = st.columns(2)
with col_a2:
    if not sum_a.empty:
        st.plotly_chart(chart_mbr_rate(sum_a), use_container_width=True)
with col_b2:
    if not sum_b.empty:
        st.plotly_chart(chart_mbr_rate(sum_b), use_container_width=True)

# ── Peptide length distribution ───────────────────────────────────────────────

st.subheader("Peptide Length Distribution")

has_len_a = "_length" in df_a.columns
has_len_b = "_length" in df_b.columns

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**{label_a}**")
    if has_len_a:
        st.plotly_chart(chart_length_all(df_a), use_container_width=True)
    else:
        st.caption("Length data not available for this run.")
with col_b:
    st.markdown(f"**{label_b}**")
    if has_len_b:
        st.plotly_chart(chart_length_all(df_b), use_container_width=True)
    else:
        st.caption("Length data not available for this run.")

# ── Contaminant rates ─────────────────────────────────────────────────────────

st.subheader("Contaminant Rates")

contam_a = compute_contaminant_summary(df_a, mapping_a)
contam_b = compute_contaminant_summary(df_b, mapping_b)
has_contam_a = "_protein" in df_a.columns and mapping_a.protein_col is not None
has_contam_b = "_protein" in df_b.columns and mapping_b.protein_col is not None

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**{label_a}**")
    if has_contam_a and not contam_a.empty:
        st.plotly_chart(chart_contaminant_rate(contam_a), use_container_width=True)
    else:
        st.caption("Contaminant data not available for this run.")
with col_b:
    st.markdown(f"**{label_b}**")
    if has_contam_b and not contam_b.empty:
        st.plotly_chart(chart_contaminant_rate(contam_b), use_container_width=True)
    else:
        st.caption("Contaminant data not available for this run.")

# ── Per-sample delta metrics ──────────────────────────────────────────────────

st.subheader("Per-Sample Delta Metrics")

shared_samples: list[str] = []
if not sum_a.empty and not sum_b.empty:
    shared_samples = sorted(
        set(sum_a["Sample"].tolist()) & set(sum_b["Sample"].tolist())
    )

if shared_samples:
    st.markdown(
        f"{len(shared_samples)} sample name(s) are present in both runs. "
        "Delta = run A minus run B for numeric columns."
    )
    merged = (
        sum_a[sum_a["Sample"].isin(shared_samples)]
        .set_index("Sample")
        .join(
            sum_b[sum_b["Sample"].isin(shared_samples)].set_index("Sample"),
            lsuffix=" (A)",
            rsuffix=" (B)",
        )
    )
    for base in ["MS/MS", "MBR", "Total Detected"]:
        ca_col, cb_col = f"{base} (A)", f"{base} (B)"
        if ca_col in merged.columns and cb_col in merged.columns:
            merged[f"{base} delta"] = merged[ca_col] - merged[cb_col]
    st.dataframe(merged, use_container_width=True)
else:
    st.info(
        "No samples share the same name across the two runs; per-sample delta metrics "
        "are not shown. Use consistent sample names across runs to enable this view."
    )

# ── PCA note ─────────────────────────────────────────────────────────────────

st.subheader("PCA")
st.info(
    "PCA is computed separately within each run on that run's intensity matrix. "
    "The principal component axes are not directly comparable across runs with different "
    "sample sets or peptide populations, and overlaying them would produce misleading "
    "visualisations. Use the per-run Report view to inspect PCA within each run individually."
)
