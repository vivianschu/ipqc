"""MHC Class I Binding / Presentation Prediction — standalone page.

Supports single-tool and multi-tool (compare) modes using locally-installed
open-source ML predictors.  No external API is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.prediction import (
    COMMON_ALLELES,
    SUPPORTED_LENGTHS,
    calculate_binding_score,
    make_consensus_table,
    postprocess,
    predictor_status_table,
    run_multi_prediction,
    run_prediction,
    validate_allele_format,
    validate_peptides,
)
from modules.predictors.registry import ALL_PREDICTORS, get_available_predictors, get_predictors_by_type

# ── Constants ─────────────────────────────────────────────────────────────────

_BINDING_LEVEL_HELP = (
    "**Binding level thresholds**\n\n"
    "| Class | %Rank (EL) | IC50 (nM) |\n"
    "|-------|-----------|----------|\n"
    "| **SB** Strong Binder | ≤ 0.5 % | < 50 nM |\n"
    "| **WB** Weak Binder | ≤ 2.0 % | < 500 nM |\n"
    "| **NB** Non-Binder | > 2.0 % | ≥ 500 nM |\n\n"
    "Scores from different tools are **not directly comparable** — always "
    "compare within the same tool."
)

_BINDING_SCORE_HELP = """\
**Binding Score (BF Score)**

A sample-level QC metric: the fraction of length-acceptable peptides
predicted to bind at least one of the selected HLA alleles.

**How it is calculated:**

1. Peptides are filtered to the selected length range (e.g. 8–11-mers).
2. For each peptide the *best* binding level across all alleles is determined:
   - **SB** if any allele gives EL %Rank ≤ 0.5 %
   - **WB** if any allele gives EL %Rank ≤ 2.0 %
   - **NB** otherwise
3. **Binding Score = (# peptides classified SB or WB) ÷ (# length-filtered peptides)**

A higher score indicates that more of the sample's peptides are predicted to be
MHC-presented.  Values close to 0 may indicate poor sample quality, a low-complexity
peptide pool, or an allele mismatch between the sample donor and the selected alleles.

*This metric mirrors the BF (binding fraction) score used by MhcVizPipe.*
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

_PEPTIDE_COL_NAMES = frozenset({
    "peptide sequence", "peptide_sequence",
    "peptide", "sequence", "seq",
    "modified sequence", "modified_sequence",
    "annotated sequence",
})


def _parse_peptide_file(content: str, filename: str) -> list[str]:
    """Extract peptide sequences from a plain-text, CSV, or TSV file.

    For delimited files the function searches column headers for a known
    peptide-column name (case-insensitive); if none match it falls back to the
    first column.  For plain-text files each non-empty line is returned as-is.
    """
    import csv
    import io

    lines = [ln for ln in content.splitlines() if ln.strip()]
    if not lines:
        return []

    sample = "\n".join(lines[:200])

    # Determine delimiter from filename or by sniffing
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
        # No delimiter found — treat as one peptide per line
        return [ln.strip() for ln in lines]

    has_header = csv.Sniffer().has_header(sample)

    reader = csv.reader(io.StringIO(content), dialect=dialect)
    rows = list(reader)
    if not rows:
        return []

    if has_header:
        headers = [h.strip() for h in rows[0]]
        col_idx = next(
            (i for i, h in enumerate(headers) if h.lower() in _PEPTIDE_COL_NAMES),
            0,  # fall back to first column
        )
        data_rows = rows[1:]
    else:
        col_idx = 0
        data_rows = rows

    return [row[col_idx].strip() for row in data_rows if len(row) > col_idx and row[col_idx].strip()]


def _fmt_num(x, decimals: int = 3) -> str:
    try:
        f = float(x)
        return f"{f:.{decimals}f}" if pd.notna(f) and str(f) != "nan" else "—"
    except (TypeError, ValueError):
        return "—"


def _binding_badge(level: str) -> str:
    return {"SB": "🔴 SB", "WB": "🟡 WB", "NB": "⚪ NB"}.get(level, level)


def _render_summary_metrics(df: pd.DataFrame) -> None:
    if "binding_level" not in df.columns:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Strong Binders (SB)", int((df["binding_level"] == "SB").sum()),
              help="%Rank ≤ 0.5 % or IC50 < 50 nM")
    c2.metric("Weak Binders (WB)", int((df["binding_level"] == "WB").sum()),
              help="%Rank ≤ 2.0 % or IC50 < 500 nM")
    c3.metric("Non-Binders (NB)", int((df["binding_level"] == "NB").sum()),
              help="%Rank > 2.0 % or IC50 ≥ 500 nM")


def _render_binding_score(df: pd.DataFrame, n_length_filtered: int) -> None:
    """Display the binding fraction (BF) score and its definition box."""
    score = calculate_binding_score(df, n_length_filtered)
    if score is None:
        return

    score_pct = f"{score:.0%}"
    col, _ = st.columns([1, 3])
    col.metric(
        "Binding Score (BF)",
        score_pct,
        help="Fraction of length-filtered peptides predicted to bind ≥ 1 allele (SB or WB).",
    )
    with st.expander("What is the Binding Score?"):
        st.markdown(_BINDING_SCORE_HELP)


def _render_results_table(
    df: pd.DataFrame,
    sort_by: str = "rank",
    height: int = 420,
    key: str = "results_table",
) -> None:
    """Render the results dataframe with formatted numeric columns."""
    display_cols = [
        c for c in ["tool", "allele", "peptide", "score", "rank", "thalf",
                    "ic50", "binding_level", "model_info"]
        if c in df.columns
    ]
    display_df = df[display_cols].copy()

    col_cfg: dict = {}
    if "score" in display_df.columns:
        col_cfg["score"] = st.column_config.NumberColumn("Score", format="%.4f")
    if "rank" in display_df.columns:
        col_cfg["rank"] = st.column_config.NumberColumn("%Rank", format="%.2f")
    if "thalf" in display_df.columns:
        col_cfg["thalf"] = st.column_config.NumberColumn("t½ (h)", format="%.2f")
    if "ic50" in display_df.columns:
        col_cfg["ic50"] = st.column_config.NumberColumn("IC50 (nM)", format="%.1f")
    if "binding_level" in display_df.columns:
        col_cfg["binding_level"] = st.column_config.TextColumn("Class")

    sort_col = sort_by if sort_by in display_df.columns else display_df.columns[0]
    sorted_df = display_df.sort_values(sort_col, ascending=True, na_position="last")

    st.dataframe(
        sorted_df,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config=col_cfg,
        key=key,
    )


_LEVEL_COLORS = {"SB": "#e74c3c", "WB": "#f39c12", "NB": "#bdc3c7"}
_LEVEL_ORDER  = ["SB", "WB", "NB"]


def _plot_scatter(df: pd.DataFrame, color_col: str, key: str) -> None:
    """EL %Rank vs log₁₀ IC50 scatter, coloured by binding_level or tool."""
    import numpy as np
    plot_df = df.dropna(subset=["rank", "ic50"]).copy()
    plot_df = plot_df[plot_df["ic50"] > 0]
    if plot_df.empty:
        return
    plot_df["log_ic50"] = np.log10(plot_df["ic50"])
    color_map = _LEVEL_COLORS if color_col == "binding_level" else None
    cat_orders = {color_col: _LEVEL_ORDER} if color_col == "binding_level" else {}
    fig = px.scatter(
        plot_df,
        x="rank",
        y="log_ic50",
        color=color_col,
        color_discrete_map=color_map,
        category_orders=cat_orders,
        hover_data=["peptide", "allele"],
        labels={
            "rank": "EL %Rank",
            "log_ic50": "log₁₀ IC50 (nM)",
            "binding_level": "Class",
            "tool": "Tool",
        },
        opacity=0.75,
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#e74c3c", opacity=0.5,
                  annotation_text="SB", annotation_position="top right")
    fig.add_vline(x=2.0, line_dash="dash", line_color="#f39c12", opacity=0.5,
                  annotation_text="WB", annotation_position="top right")
    fig.add_hline(y=np.log10(50),  line_dash="dash", line_color="#e74c3c", opacity=0.3)
    fig.add_hline(y=np.log10(500), line_dash="dash", line_color="#f39c12", opacity=0.3)
    fig.update_layout(
        height=380,
        margin=dict(t=20, b=40, l=50, r=20),
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

st.title("MHC Class I Binding Prediction")
st.markdown(
    "Score peptide–HLA binding and antigen presentation using "
    "locally-installed open-source ML tools.  All computation runs on this "
    "server — no data is sent to any external service."
)

# ── Tool status ───────────────────────────────────────────────────────────────

available = get_predictors_by_type("binding")
available_names = [cls.name for cls in available]
_status_rows = [r for r in predictor_status_table() if r["Tool"] != "NetMHCstabpan"]

_all_unavailable = not available
_status_expanded = _all_unavailable  # force open when nothing is installed

_binding_predictors_all = [c for c in ALL_PREDICTORS if c.predictor_type == "binding"]
with st.expander(
    f"Tool status — {len(available)}/{len(_binding_predictors_all)} available",
    expanded=_status_expanded,
):
    status_df = pd.DataFrame(_status_rows)
    st.dataframe(
        status_df[["Tool", "Status", "Description"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn("Status"),
        },
    )
    for row in _status_rows:
        if row["Install"]:
            with st.expander(f"Install {row['Tool']}"):
                st.code(row["Install"], language="bash")

if _all_unavailable:
    st.error(
        "No prediction tools are installed.  Install at least one of the tools "
        "listed above, then restart the app."
    )
    st.stop()

# ── Mode selection ────────────────────────────────────────────────────────────

_mode = st.radio(
    "Mode",
    ["Single tool", "Compare tools"],
    horizontal=True,
    help=(
        "**Single tool** — run one predictor and view its results.  \n"
        "**Compare tools** — run all selected installed predictors and show "
        "results side-by-side."
    ),
    key="pred_mode",
)

st.divider()

# ── 1. Peptide Sequences ──────────────────────────────────────────────────────

st.subheader("1. Peptide Sequences")

_input_mode = st.radio(
    "Input method",
    ["Paste sequences", "Upload file"],
    horizontal=True,
    key="pred_input_mode",
)

_raw_sequences: list[str] = []

if _input_mode == "Paste sequences":
    _raw = st.text_area(
        "One peptide per line (standard amino acids only)",
        height=140,
        placeholder="GILGFVFTL\nSLYNTVATL\nKLVVVGAVG\nYLEPGPVTA",
        key="pred_seq_text",
    )
    if _raw.strip():
        _raw_sequences = [s.strip() for s in _raw.splitlines() if s.strip()]
else:
    _upload = st.file_uploader(
        "Plain-text (.txt), CSV, or TSV — one peptide per line, or a table with a peptide column",
        type=["txt", "csv", "tsv"],
        key="pred_seq_file",
    )
    if _upload is not None:
        _raw_sequences = _parse_peptide_file(
            _upload.read().decode("utf-8", errors="replace"),
            _upload.name,
        )

# Validate sequences
_sequences: list[str] = []
_seq_issues: list[str] = []
if _raw_sequences:
    _sequences, _seq_issues = validate_peptides(_raw_sequences)
    _dupes = len(_raw_sequences) - len(set(s.upper() for s in _sequences))
    _unique_seqs = list(dict.fromkeys(s.upper() for s in _sequences))  # deduplicated, order-preserving

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
        key="pred_alleles",
    )
with _col_c:
    _custom_raw = st.text_input(
        "Custom allele(s) — comma-separated",
        placeholder="HLA-A*68:01, HLA-B*53:01",
        key="pred_custom_allele",
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
    key="pred_lengths",
    help="Peptides whose length is not in this list are excluded from scoring.",
)

# Length filter preview
if _unique_seqs and _selected_lengths:
    _length_set = set(_selected_lengths)
    _matching = [p for p in _unique_seqs if len(p) in _length_set]
    _excluded = len(_unique_seqs) - len(_matching)
    if _excluded:
        st.caption(
            f"{len(_matching):,} of {len(_unique_seqs):,} peptides match selected length(s)"
            f" — {_excluded} excluded."
        )

# ── 4. Tool selection ─────────────────────────────────────────────────────────

st.subheader("4. Prediction Tool" if _mode == "Single tool" else "4. Prediction Tools")

_tools_to_run: list[str] = []
if _mode == "Single tool":
    _sel_tool = st.selectbox(
        "Tool",
        options=available_names,
        key="pred_single_tool",
        help="Only installed tools are listed.",
    )
    _sel_cls = next(cls for cls in available if cls.name == _sel_tool)
    st.caption(
        f"{_sel_cls.description}  \n"
        f"Version: `{_sel_cls.version()}`"
    )
    _tools_to_run = [_sel_tool]
else:
    _compare_choices = {
        cls.name: st.checkbox(
            f"**{cls.name}** — {cls.description}",
            value=True,
            key=f"pred_compare_{cls.name}",
        )
        for cls in available
    }
    _tools_to_run = [name for name, checked in _compare_choices.items() if checked]
    if not _tools_to_run:
        st.warning("Select at least one tool to run.")

st.divider()

# ── Pre-run validation summary ────────────────────────────────────────────────

_can_run = True

if not _unique_seqs:
    _can_run = False
elif not _selected_lengths:
    st.error("Select at least one peptide length.")
    _can_run = False
elif not [p for p in _unique_seqs if len(p) in set(_selected_lengths)]:
    st.error(
        "No loaded peptides match the selected length(s).  "
        "Adjust the Peptide Lengths above."
    )
    _can_run = False

if not _all_alleles:
    st.error("Select at least one HLA allele.")
    _can_run = False

if not _tools_to_run:
    _can_run = False

# Warn about large jobs — estimate based on ~23 s / 1,000 peptides / allele (NetMHCpan benchmark)
if _can_run:
    _n_pep = len([p for p in _unique_seqs if len(p) in set(_selected_lengths)])
    _n_pairs = _n_pep * len(_all_alleles)
    _est_secs = (_n_pep / 1_000) * 23 * len(_all_alleles) * len(_tools_to_run)
    if _n_pep > 10_000:
        _est_str = (
            f"{_est_secs / 60:.0f} min" if _est_secs >= 90
            else f"{_est_secs:.0f} s"
        )
        st.warning(
            f"{_n_pep:,} peptides × {len(_all_alleles)} allele(s) = "
            f"{_n_pairs:,} predictions.  "
            f"Estimated run time: **~{_est_str}** — do not close this tab."
        )
    elif _n_pep > 2_000:
        st.info(
            f"{_n_pep:,} peptides × {len(_all_alleles)} allele(s) = {_n_pairs:,} predictions."
        )

# ── Run button ────────────────────────────────────────────────────────────────

_peptides_for_run = [p for p in _unique_seqs if len(p) in set(_selected_lengths)] if _selected_lengths else []

_run_label = (
    f"Run {_tools_to_run[0]}" if len(_tools_to_run) == 1
    else f"Compare {len(_tools_to_run)} tools"
)
_run_clicked = st.button(
    _run_label,
    type="primary",
    disabled=not _can_run,
    key="pred_run",
)

if _run_clicked and _can_run:
    if _mode == "Single tool":
        with st.status(
            f"Running {_tools_to_run[0]} — "
            f"{len(_peptides_for_run):,} peptide(s) × {len(_all_alleles)} allele(s)…",
            expanded=True,
        ) as _status:
            _df, _errs = run_prediction(
                _peptides_for_run, _all_alleles, _selected_lengths, _tools_to_run[0]
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
            st.session_state["pred_single_result"] = _df
            st.session_state["pred_single_tool_used"] = _tools_to_run[0]
            st.session_state["pred_n_length_filtered"] = len(_peptides_for_run)
        for e in _errs:
            st.warning(e)

    else:  # compare mode
        _compare_results: dict[str, tuple[pd.DataFrame, list[str]]] = {}
        _total_tools = len(_tools_to_run)

        with st.status(
            f"Running {_total_tools} tool(s)…",
            expanded=True,
        ) as _status:
            for _i, _tool in enumerate(_tools_to_run):
                st.write(
                    f"**{_tool}** ({_i + 1}/{_total_tools})  "
                    f"{len(_peptides_for_run):,} peptide(s) × {len(_all_alleles)} allele(s)"
                )
                _df_t, _errs_t = run_prediction(
                    _peptides_for_run, _all_alleles, _selected_lengths, _tool
                )
                _compare_results[_tool] = (_df_t, _errs_t)
                if _errs_t:
                    for e in _errs_t:
                        st.write(f"  :x: {e}")
                elif _df_t.empty:
                    st.write(f"  :warning: No results from {_tool}.")
                else:
                    st.write(f"  :white_check_mark: {len(_df_t):,} predictions.")

            _any_ok = any(not df.empty for df, _ in _compare_results.values())
            _status.update(
                label="Done" if _any_ok else "All tools failed",
                state="complete" if _any_ok else "error",
                expanded=not _any_ok,
            )

        st.session_state["pred_compare_results"] = _compare_results
        st.session_state["pred_n_length_filtered"] = len(_peptides_for_run)

# ══════════════════════════════════════════════════════════════════════════════
# Results — Single tool
# ══════════════════════════════════════════════════════════════════════════════

if _mode == "Single tool" and "pred_single_result" in st.session_state:
    _result_df = postprocess(st.session_state["pred_single_result"])
    _tool_used = st.session_state.get("pred_single_tool_used", "")

    st.divider()
    st.subheader(f"Results — {_tool_used}  ·  {len(_result_df):,} predictions")

    _render_summary_metrics(_result_df)
    _render_binding_score(
        _result_df,
        st.session_state.get("pred_n_length_filtered", 0),
    )

    # Per-allele filter
    if "allele" in _result_df.columns and _result_df["allele"].nunique() > 1:
        _allele_filter = st.multiselect(
            "Filter by allele",
            options=sorted(_result_df["allele"].unique()),
            default=sorted(_result_df["allele"].unique()),
            key="pred_single_allele_filter",
        )
        _view_df = _result_df[_result_df["allele"].isin(_allele_filter)]
    else:
        _view_df = _result_df

    _sort_col = st.selectbox(
        "Sort by",
        options=["rank", "ic50", "score", "binding_level", "peptide"],
        key="pred_single_sort",
    )

    _plot_scatter(_view_df, color_col="binding_level", key="single_scatter")

    _render_results_table(_view_df, sort_by=_sort_col, key="pred_single_table")

    with st.expander("Binding level reference"):
        st.markdown(_BINDING_LEVEL_HELP)

    _download_button(
        _result_df,
        f"mhci_{_tool_used.lower()}_predictions.csv",
        key="pred_single_dl",
    )

# ══════════════════════════════════════════════════════════════════════════════
# Results — Compare tools
# ══════════════════════════════════════════════════════════════════════════════

elif _mode == "Compare tools" and "pred_compare_results" in st.session_state:
    _cmp = st.session_state["pred_compare_results"]
    _ok_tools = {name: df for name, (df, _) in _cmp.items() if not df.empty}
    _failed_tools = {name: errs for name, (df, errs) in _cmp.items() if df.empty or errs}

    st.divider()
    st.subheader(f"Results — {len(_ok_tools)} tool(s) completed")

    # Error summary
    for _name, _errs in _failed_tools.items():
        for _e in _errs:
            st.warning(f"**{_name}:** {_e}")

    if not _ok_tools:
        st.error("All selected tools failed.  See messages above.")
    else:
        # Merge all results into one frame for the "All" tab
        _all_frames = []
        for name, df in _ok_tools.items():
            _all_frames.append(postprocess(df))
        _merged_df = pd.concat(_all_frames, ignore_index=True)

        # ── Tabs ──────────────────────────────────────────────────────────────
        _tab_names = ["Summary"] + list(_ok_tools.keys())
        _tabs = st.tabs(_tab_names)

        # Summary tab
        with _tabs[0]:
            _plot_scatter(
                _merged_df.assign(tool=_merged_df["tool"]),
                color_col="tool",
                key="cmp_summary_scatter",
            )
            st.markdown("#### Per-tool binding-level breakdown")
            _n_lf = st.session_state.get("pred_n_length_filtered", 0)
            _summary_rows = []
            for _t, _df in _ok_tools.items():
                _pp = postprocess(_df)
                _bs = calculate_binding_score(_pp, _n_lf)
                _summary_rows.append({
                    "Tool": _t,
                    "Version": next(
                        (cls.version() for cls in ALL_PREDICTORS if cls.name == _t), "—"
                    ),
                    "Predictions": len(_pp),
                    "Strong Binders (SB)": int((_pp["binding_level"] == "SB").sum()),
                    "Weak Binders (WB)": int((_pp["binding_level"] == "WB").sum()),
                    "Non-Binders (NB)": int((_pp["binding_level"] == "NB").sum()),
                    "Binding Score": f"{_bs:.0%}" if _bs is not None else "—",
                })
            st.dataframe(
                pd.DataFrame(_summary_rows),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("What is the Binding Score?"):
                st.markdown(_BINDING_SCORE_HELP)

            st.markdown("#### Consensus table (peptide × allele × tool)")
            st.caption(
                "Each cell shows the binding class assigned by that tool.  "
                "Empty cells mean the tool did not return a result for that pair."
            )
            _consensus = make_consensus_table({n: postprocess(d) for n, d in _ok_tools.items()})
            if not _consensus.empty:
                st.dataframe(
                    _consensus,
                    use_container_width=True,
                    hide_index=True,
                    height=350,
                    key="pred_cmp_consensus",
                )
            else:
                st.info("Not enough data to build a consensus table.")

            _download_button(
                _merged_df,
                "mhci_comparison_all_tools.csv",
                key="pred_cmp_dl_all",
            )

        # Per-tool tabs
        for _tab, (_t_name, _t_df) in zip(_tabs[1:], _ok_tools.items()):
            with _tab:
                _pp_df = postprocess(_t_df)
                _t_cls = next((cls for cls in ALL_PREDICTORS if cls.name == _t_name), None)
                if _t_cls:
                    st.caption(f"Version: `{_t_cls.version()}`  ·  {_t_cls.description}")

                _render_summary_metrics(_pp_df)
                _render_binding_score(_pp_df, st.session_state.get("pred_n_length_filtered", 0))
                _plot_scatter(_pp_df, color_col="binding_level", key=f"cmp_{_t_name}_scatter")
                _render_results_table(_pp_df, key=f"pred_cmp_table_{_t_name}")
                _download_button(
                    _pp_df,
                    f"mhci_{_t_name.lower()}_predictions.csv",
                    key=f"pred_cmp_dl_{_t_name}",
                )

        with st.expander("Binding level reference"):
            st.markdown(_BINDING_LEVEL_HELP)
