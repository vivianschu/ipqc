"""MHC Class I Stability Prediction — standalone page.

Uses NetMHCstabpan 1.0 to predict pMHC-I complex half-life and stability rank.
Stability is a complement to binding affinity: a peptide that binds tightly but
dissociates quickly may not trigger a T cell response.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.prediction import (
    COMMON_ALLELES,
    SUPPORTED_LENGTHS,
    postprocess,
    run_prediction,
    validate_allele_format,
    validate_peptides,
)
from modules.predictors.registry import get_predictors_by_type

# ── Constants ─────────────────────────────────────────────────────────────────

_STABILITY_HELP = """\
**Stability vs Affinity**

MHC binding affinity (IC50) measures how tightly a peptide binds to HLA, but
not how long that complex survives on the cell surface.

**Stability** (half-life, t½) measures how long the pMHC complex persists —
a stronger correlate of actual T cell priming than affinity alone.

| Class | %Rank_Stab |
|-------|-----------|
| **SB** Strong (stable) | ≤ 0.5 % |
| **WB** Weak (semi-stable) | ≤ 2.0 % |
| **NB** Non-stable | > 2.0 % |

*Rasmussen et al., J Immunology 2016 — stability correlates with T cell immunogenicity
independently of affinity.*
"""

_PEPTIDE_COL_NAMES = frozenset({
    "peptide sequence", "peptide_sequence",
    "peptide", "sequence", "seq",
    "modified sequence", "modified_sequence",
    "annotated sequence",
})

_LEVEL_COLORS = {"SB": "#e74c3c", "WB": "#f39c12", "NB": "#bdc3c7"}
_LEVEL_ORDER  = ["SB", "WB", "NB"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_peptide_file(content: str, filename: str) -> list[str]:
    import csv, io
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if not lines:
        return []
    sample = "\n".join(lines[:200])
    lower = filename.lower()
    if lower.endswith(".tsv"):
        delimiters = "\t"
    elif lower.endswith(".csv"):
        delimiters = ","
    else:
        delimiters = ",\t|;"
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=delimiters)
    except csv.Error:
        return [ln.strip() for ln in lines]
    has_header = csv.Sniffer().has_header(sample)
    reader = csv.reader(io.StringIO(content), dialect=dialect)
    rows = list(reader)
    if not rows:
        return []
    if has_header:
        headers = [h.strip() for h in rows[0]]
        col_idx = next(
            (i for i, h in enumerate(headers) if h.lower() in _PEPTIDE_COL_NAMES), 0
        )
        data_rows = rows[1:]
    else:
        col_idx = 0
        data_rows = rows
    return [row[col_idx].strip() for row in data_rows if len(row) > col_idx and row[col_idx].strip()]


def _render_stability_metrics(df: pd.DataFrame) -> None:
    if "binding_level" not in df.columns:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Stable (SB)", int((df["binding_level"] == "SB").sum()),
              help="%Rank_Stab ≤ 0.5 %")
    c2.metric("Semi-stable (WB)", int((df["binding_level"] == "WB").sum()),
              help="%Rank_Stab ≤ 2.0 %")
    c3.metric("Non-stable (NB)", int((df["binding_level"] == "NB").sum()),
              help="%Rank_Stab > 2.0 %")


def _render_results_table(df: pd.DataFrame, sort_by: str = "rank", key: str = "stab_table") -> None:
    display_cols = [
        c for c in ["allele", "peptide", "score", "rank", "thalf", "binding_level", "model_info"]
        if c in df.columns
    ]
    display_df = df[display_cols].copy()
    col_cfg: dict = {}
    if "score" in display_df.columns:
        col_cfg["score"] = st.column_config.NumberColumn("Stability Score", format="%.4f")
    if "rank" in display_df.columns:
        col_cfg["rank"] = st.column_config.NumberColumn("%Rank_Stab", format="%.2f")
    if "thalf" in display_df.columns:
        col_cfg["thalf"] = st.column_config.NumberColumn("t½ (h)", format="%.2f")
    if "binding_level" in display_df.columns:
        col_cfg["binding_level"] = st.column_config.TextColumn("Class")
    sort_col = sort_by if sort_by in display_df.columns else display_df.columns[0]
    sorted_df = display_df.sort_values(sort_col, ascending=True, na_position="last")
    st.dataframe(
        sorted_df, use_container_width=True, hide_index=True,
        height=420, column_config=col_cfg, key=key,
    )


def _plot_stability_scatter(df: pd.DataFrame, key: str) -> None:
    """%Rank_Stab (x) vs t½ hours (y), coloured by stability class."""
    plot_df = df.dropna(subset=["rank", "thalf"]).copy()
    if plot_df.empty:
        return
    fig = px.scatter(
        plot_df,
        x="rank",
        y="thalf",
        color="binding_level",
        color_discrete_map=_LEVEL_COLORS,
        category_orders={"binding_level": _LEVEL_ORDER},
        hover_data=["peptide", "allele", "score"],
        labels={
            "rank": "Stability %Rank (lower = more stable)",
            "thalf": "Half-life t½ (hours)",
            "binding_level": "Class",
        },
        opacity=0.75,
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#e74c3c", opacity=0.5,
                  annotation_text="SB", annotation_position="top right")
    fig.add_vline(x=2.0, line_dash="dash", line_color="#f39c12", opacity=0.5,
                  annotation_text="WB", annotation_position="top right")
    fig.update_layout(
        height=380,
        margin=dict(t=20, b=40, l=60, r=20),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#ececec"),
        yaxis=dict(gridcolor="#ececec"),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def _download_button(df: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        "⬇ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Page
# ══════════════════════════════════════════════════════════════════════════════

st.title("MHC Class I Stability Prediction")
st.markdown(
    "Predict pMHC-I complex stability (half-life and %Rank) using **NetMHCstabpan 1.0**.  "
    "Stability is a key correlate of T cell immunogenicity, complementing binding affinity."
)

with st.expander("Why stability matters"):
    st.markdown(_STABILITY_HELP)

# ── Tool availability ─────────────────────────────────────────────────────────

_stability_predictors = get_predictors_by_type("stability")

if not _stability_predictors:
    from modules.predictors.netmhcstabpan_predictor import NetMHCstabpanPredictor
    st.error("NetMHCstabpan is not installed or its data files are missing.")
    with st.expander("Installation instructions"):
        st.code(NetMHCstabpanPredictor.install_hint, language="bash")
    st.stop()

_stab_cls = _stability_predictors[0]
st.caption(f"Tool: **{_stab_cls.name}** · Version: `{_stab_cls.version()}`  \n{_stab_cls.description}")

st.divider()

# ── 1. Peptide Sequences ──────────────────────────────────────────────────────

st.subheader("1. Peptide Sequences")

_input_mode = st.radio(
    "Input method",
    ["Paste sequences", "Upload file"],
    horizontal=True,
    key="stab_input_mode",
)

_raw_sequences: list[str] = []

if _input_mode == "Paste sequences":
    _raw = st.text_area(
        "One peptide per line (standard amino acids only)",
        height=140,
        placeholder="GILGFVFTL\nSLYNTVATL\nKLVVVGAVG\nYLEPGPVTA",
        key="stab_seq_text",
    )
    if _raw.strip():
        _raw_sequences = [s.strip() for s in _raw.splitlines() if s.strip()]
else:
    _upload = st.file_uploader(
        "Plain-text (.txt), CSV, or TSV — one peptide per line, or a table with a peptide column",
        type=["txt", "csv", "tsv"],
        key="stab_seq_file",
    )
    if _upload is not None:
        _raw_sequences = _parse_peptide_file(
            _upload.read().decode("utf-8", errors="replace"),
            _upload.name,
        )

_sequences: list[str] = []
_seq_issues: list[str] = []
if _raw_sequences:
    _sequences, _seq_issues = validate_peptides(_raw_sequences)
    _dupes = len(_raw_sequences) - len(set(s.upper() for s in _sequences))
    _unique_seqs = list(dict.fromkeys(s.upper() for s in _sequences))

    _info_parts = [f"{len(_raw_sequences):,} line(s) read"]
    if _seq_issues:
        _info_parts.append(f"{len(_seq_issues)} skipped (invalid amino acids)")
    if _dupes:
        _info_parts.append(f"{_dupes} duplicate(s) collapsed")
    _info_parts.append(f"**{len(_unique_seqs):,} unique peptide(s) to score**")
    st.caption("  ·  ".join(_info_parts))

    if _seq_issues:
        with st.expander(f"⚠ {len(_seq_issues)} sequence issue(s)"):
            for iss in _seq_issues[:20]:
                st.caption(iss)
            if len(_seq_issues) > 20:
                st.caption(f"… and {len(_seq_issues) - 20} more")
else:
    _unique_seqs = []

# ── 2. HLA Alleles ────────────────────────────────────────────────────────────

st.subheader("2. HLA Alleles")

_col_a, _col_c = st.columns([3, 2])
with _col_a:
    _selected_alleles: list[str] = st.multiselect(
        "Common alleles",
        options=COMMON_ALLELES,
        default=["HLA-A*02:01"],
        key="stab_alleles",
    )
with _col_c:
    _custom_raw = st.text_input(
        "Custom allele(s) — comma-separated",
        placeholder="HLA-A*68:01, HLA-B*53:01",
        key="stab_custom_allele",
    )

_all_alleles: list[str] = list(_selected_alleles)
_allele_issues: list[str] = []
if _custom_raw.strip():
    _seen: set[str] = set(_all_alleles)
    for _a in _custom_raw.strip().split(","):
        _a = _a.strip()
        if not _a:
            continue
        _fmt_err = validate_allele_format(_a)
        if _fmt_err:
            _allele_issues.append(_fmt_err)
        elif _a not in _seen:
            _all_alleles.append(_a)
            _seen.add(_a)

if _allele_issues:
    for _iss in _allele_issues:
        st.warning(f"Allele format: {_iss}")

# ── 3. Peptide Lengths ────────────────────────────────────────────────────────

st.subheader("3. Peptide Lengths")

_selected_lengths: list[int] = st.multiselect(
    "Include peptides of length (mer)",
    options=SUPPORTED_LENGTHS,
    default=[9],
    key="stab_lengths",
    help="NetMHCstabpan supports 8–11-mers.  Non-9-mer predictions use approximations.",
)

if _unique_seqs and _selected_lengths:
    _length_set = set(_selected_lengths)
    _matching = [p for p in _unique_seqs if len(p) in _length_set]
    _excluded = len(_unique_seqs) - len(_matching)
    if _excluded:
        st.caption(
            f"{len(_matching):,} of {len(_unique_seqs):,} peptides match selected length(s)"
            f" — {_excluded} excluded."
        )

st.divider()

# ── Pre-run validation ────────────────────────────────────────────────────────

_can_run = True

if not _unique_seqs:
    _can_run = False
elif not _selected_lengths:
    st.error("Select at least one peptide length.")
    _can_run = False
elif not [p for p in _unique_seqs if len(p) in set(_selected_lengths)]:
    st.error("No loaded peptides match the selected length(s).  Adjust lengths above.")
    _can_run = False

if not _all_alleles:
    st.error("Select at least one HLA allele.")
    _can_run = False

_peptides_for_run = [p for p in _unique_seqs if len(p) in set(_selected_lengths)] if _selected_lengths else []

if _can_run:
    _n_pep = len(_peptides_for_run)
    _n_pairs = _n_pep * len(_all_alleles)
    _est_secs = (_n_pep / 1_000) * 30 * len(_all_alleles)
    if _n_pep > 5_000:
        _est_str = f"{_est_secs / 60:.0f} min" if _est_secs >= 90 else f"{_est_secs:.0f} s"
        st.warning(
            f"{_n_pep:,} peptides × {len(_all_alleles)} allele(s) = {_n_pairs:,} predictions.  "
            f"Estimated run time: **~{_est_str}**"
        )

# ── Run button ────────────────────────────────────────────────────────────────

_run_clicked = st.button(
    f"Run {_stab_cls.name}",
    type="primary",
    disabled=not _can_run,
    key="stab_run",
)

if _run_clicked and _can_run:
    with st.status(
        f"Running {_stab_cls.name} — "
        f"{len(_peptides_for_run):,} peptide(s) × {len(_all_alleles)} allele(s)…",
        expanded=True,
    ) as _status:
        _df, _errs = run_prediction(
            _peptides_for_run, _all_alleles, _selected_lengths, _stab_cls.name
        )
        if _errs:
            for e in _errs:
                st.write(f":x: {e}")
            _status.update(label="Prediction failed", state="error")
        elif _df.empty:
            st.write(":warning: No results returned.")
            _status.update(label="No results", state="error")
        else:
            st.write(f":white_check_mark: {len(_df):,} predictions complete.")
            _status.update(label="Done", state="complete", expanded=False)

    if not _df.empty:
        st.session_state["stab_result"] = _df
        st.session_state["stab_n_filtered"] = len(_peptides_for_run)
    for e in _errs:
        st.warning(e)

# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════

if "stab_result" in st.session_state:
    _result_df = postprocess(st.session_state["stab_result"])

    st.divider()
    st.subheader(f"Results — {len(_result_df):,} predictions")

    _render_stability_metrics(_result_df)

    # Per-allele filter
    if "allele" in _result_df.columns and _result_df["allele"].nunique() > 1:
        _allele_filter = st.multiselect(
            "Filter by allele",
            options=sorted(_result_df["allele"].unique()),
            default=sorted(_result_df["allele"].unique()),
            key="stab_allele_filter",
        )
        _view_df = _result_df[_result_df["allele"].isin(_allele_filter)]
    else:
        _view_df = _result_df

    _sort_col = st.selectbox(
        "Sort by",
        options=["rank", "thalf", "score", "binding_level", "peptide"],
        key="stab_sort",
    )

    _plot_stability_scatter(_view_df, key="stab_scatter")
    _render_results_table(_view_df, sort_by=_sort_col, key="stab_table")

    with st.expander("Stability class reference"):
        st.markdown(_STABILITY_HELP)

    _download_button(
        _result_df,
        "mhci_stability_netmhcstabpan.csv",
        key="stab_dl",
    )
