"""HLA Typing — infer HLA alleles from immunopeptidomics peptide ligands."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import logomaker
    import matplotlib.pyplot as plt
    _HAS_LOGOMAKER = True
except ImportError:
    _HAS_LOGOMAKER = False

from modules.hla_typing import (
    CANDIDATE_ALLELES_CLASS_I,
    clean_peptides,
    infer_hla_class,
    length_distribution,
)
from modules.motif_decon import (
    annotate_clusters,
    pwm_to_logo_df,
    run_deconvolution,
    top_anchor_residues,
)

# ── Page header ───────────────────────────────────────────────────────────────

st.title("HLA Typing from Immunopeptidomics")
st.markdown(
    "Infer likely HLA alleles from immunopeptidomics MS/MS peptide ligands. "
    "Unsupervised motif deconvolution discovers binding motifs and matches them against "
    "a curated panel of common class I alleles. "
    "All computation runs locally — no data is sent to any external service."
)
st.warning(
    "**Clinical disclaimer:** MS-inferred HLA calls are *motif-compatible allele candidates*, "
    "not a definitive genotype.  Validate with DNA or RNA-level HLA typing before any clinical use.",
    icon="⚠️",
)

st.divider()

# ── 1. Peptide Input ──────────────────────────────────────────────────────────

st.subheader("1. Peptide Sequences")

_input_mode = st.radio(
    "Input method",
    ["Paste sequences", "Upload file"],
    horizontal=True,
    key="hla_input_mode",
)

_raw_sequences: list[str] = []
_sample_seq_map: dict[str, list[str]] = {}

if _input_mode == "Paste sequences":
    _raw_text = st.text_area(
        "One peptide per line (standard amino acids only)",
        height=140,
        placeholder="GILGFVFTL\nSLYNTVATL\nKLVVVGAVG\nYLEPGPVTA",
        key="hla_seq_text",
    )
    if _raw_text.strip():
        _raw_sequences = [s.strip() for s in _raw_text.splitlines() if s.strip()]
else:
    _upload = st.file_uploader(
        "CSV, TSV, or plain-text file",
        type=["txt", "csv", "tsv"],
        key="hla_seq_file",
    )
    if _upload is not None:
        _file_bytes = _upload.getvalue()
        _sep = "\t" if _upload.name.lower().endswith(".tsv") else ","

        @st.cache_data(show_spinner=False)
        def _detect_cols(data: bytes, sep: str) -> list[str]:
            import io as _io
            try:
                return pd.read_csv(_io.BytesIO(data), sep=sep, nrows=0).columns.tolist()
            except Exception:
                return []

        @st.cache_data(show_spinner=False)
        def _read_col(data: bytes, sep: str, col: str) -> list[str]:
            import io as _io2
            df = pd.read_csv(_io2.BytesIO(data), sep=sep, usecols=[col], dtype=str)
            return df[col].dropna().str.strip().tolist()

        @st.cache_data(show_spinner=False)
        def _read_sample_map(data: bytes, sep: str, pep_col: str, samp_col: str) -> dict[str, list[str]]:
            import io as _io3
            df = pd.read_csv(_io3.BytesIO(data), sep=sep, usecols=[pep_col, samp_col], dtype=str)
            df = df.dropna(subset=[pep_col])
            result: dict[str, list[str]] = {}
            for samp, grp in df.groupby(samp_col, sort=False):
                result[str(samp)] = grp[pep_col].str.strip().dropna().tolist()
            return result

        _csv_cols = _detect_cols(_file_bytes, _sep)

        if _csv_cols:
            _col_pep_ui, _col_samp_ui = st.columns(2)
            with _col_pep_ui:
                _pep_col = st.selectbox(
                    "Peptide column",
                    options=_csv_cols,
                    key="hla_pep_col",
                )
            _other_cols = [c for c in _csv_cols if c != _pep_col]
            _SAMPLE_HINTS = frozenset({
                "sample", "sample_id", "sample_name", "run", "experiment",
                "file", "filename", "condition", "replicate",
            })
            _auto_samp = next(
                (c for c in _other_cols
                 if c.lower().replace(" ", "_").replace("-", "_") in _SAMPLE_HINTS),
                None,
            )
            _samp_options = ["(none — single pool)"] + _other_cols
            _default_samp_idx = _samp_options.index(_auto_samp) if _auto_samp in _samp_options else 0
            with _col_samp_ui:
                _samp_col_sel = st.selectbox(
                    "Sample column (optional)",
                    options=_samp_options,
                    index=_default_samp_idx,
                    key="hla_samp_col",
                    help="Select to run HLA typing per sample.",
                )
            _samp_col = None if _samp_col_sel.startswith("(none") else _samp_col_sel

            if _samp_col:
                _sample_seq_map = _read_sample_map(_file_bytes, _sep, _pep_col, _samp_col)
                _raw_sequences = [s for seqs in _sample_seq_map.values() for s in seqs]
                _samp_labels = sorted(_sample_seq_map)
                st.caption(
                    f"Detected **{len(_sample_seq_map)} sample(s)**: "
                    + ", ".join(_samp_labels[:6])
                    + ("…" if len(_samp_labels) > 6 else "")
                )
            else:
                _raw_sequences = _read_col(_file_bytes, _sep, _pep_col)
        else:
            _lines = _file_bytes.decode("utf-8", errors="replace").splitlines()
            _raw_sequences = [ln.strip() for ln in _lines if ln.strip() and not ln.startswith("#")]

# Clean, validate, deduplicate
_peptides: list[str] = []
_clean_issues: list[str] = []
_n_dupes = 0

if _raw_sequences:
    _peptides, _clean_issues, _n_dupes = clean_peptides(_raw_sequences, min_len=7, max_len=25)
    _info_parts = [f"{len(_raw_sequences):,} line(s) read"]
    if _clean_issues:
        _info_parts.append(f"{len(_clean_issues)} invalid (skipped)")
    if _n_dupes:
        _info_parts.append(f"{_n_dupes} duplicate(s) collapsed")
    _info_parts.append(f"**{len(_peptides):,} valid unique peptide(s)**")
    st.caption("  ·  ".join(_info_parts))

    if _clean_issues:
        with st.expander(f"⚠ {len(_clean_issues)} sequence issue(s)"):
            for iss in _clean_issues[:20]:
                st.caption(iss)
            if len(_clean_issues) > 20:
                st.caption(f"… and {len(_clean_issues) - 20} more")

# ── 2. Optional Metadata ──────────────────────────────────────────────────────

st.subheader("2. Optional Metadata")

with st.expander("Expand to add metadata (optional)"):
    _col_m1, _col_m2 = st.columns(2)
    with _col_m1:
        _hla_class_input = st.selectbox(
            "HLA class",
            ["Auto-detect", "Class I", "Class II"],
            key="hla_class_sel",
            help="Auto-detect infers from peptide length distribution.",
        )
        _sample_type = st.selectbox(
            "Sample type",
            ["Unknown", "Tumor", "Normal", "Cell line", "Organoid", "Mixed tissue"],
            key="hla_sample_type",
        )
    with _col_m2:
        _external_hla_raw = st.text_input(
            "External HLA typing (DNA/RNA) — comma-separated",
            placeholder="HLA-A*02:01, HLA-B*07:02, HLA-C*07:02",
            key="hla_external",
            help="Known alleles from WES/WGS/RNA-seq to guide and validate inference.",
        )
        _pop_prior = st.selectbox(
            "Population/ancestry prior",
            ["None", "European", "East Asian", "South Asian", "African", "Latino/Hispanic", "Other"],
            key="hla_pop_prior",
        )

_external_hla: list[str] = []
if _external_hla_raw.strip():
    _external_hla = [a.strip() for a in _external_hla_raw.split(",") if a.strip()]

# ── 3. Preprocessing ──────────────────────────────────────────────────────────

_is_class_i = False
_dist: dict[int, int] = {}
_class_info: dict = {}

if _peptides:
    st.divider()
    st.subheader("3. Preprocessing")

    _dist = length_distribution(_peptides)
    _class_info = infer_hla_class(_peptides)

    # Resolve effective HLA class
    if _hla_class_input == "Class I":
        _effective_class = "I"
    elif _hla_class_input == "Class II":
        _effective_class = "II"
    else:
        _effective_class = _class_info["inferred_class"].split()[0]  # "I", "II", or "unknown"

    _is_class_i = _effective_class == "I"

    # Length distribution chart + class info side-by-side
    _col_chart, _col_info = st.columns([3, 2])

    with _col_chart:
        _dist_df = pd.DataFrame(
            [{"Length (aa)": str(l), "Count": c} for l, c in _dist.items()]
        )
        fig_dist = px.bar(
            _dist_df,
            x="Length (aa)",
            y="Count",
            title="Peptide Length Distribution",
            color_discrete_sequence=["#4C78A8"],
        )
        fig_dist.update_layout(
            height=300,
            margin=dict(t=35, b=30, l=45, r=10),
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#ececec"),
            yaxis=dict(gridcolor="#ececec"),
        )
        st.plotly_chart(fig_dist, use_container_width=True, key="hla_dist_chart")

    with _col_info:
        st.markdown("**Class inference**")
        if _hla_class_input != "Auto-detect":
            st.info(f"Manually set to **Class {_effective_class}**")
        else:
            _badge = {
                "I": ":green[Class I]",
                "II": ":orange[Class II]",
                "I (uncertain)": ":orange[Class I (uncertain)]",
                "unknown": ":gray[Unknown]",
            }.get(_class_info["inferred_class"], _class_info["inferred_class"])
            st.markdown(f"Inferred: {_badge}")
        st.caption(_class_info.get("note", ""))

        st.markdown("**Length summary**")
        st.caption(
            f"Peak: **{_class_info['peak_length']} aa**  ·  "
            f"7–11 aa: {_class_info['class_i_fraction']:.0%}  ·  "
            f"13–25 aa: {_class_info['class_ii_fraction']:.0%}"
        )
        if _external_hla:
            st.markdown("**External HLA typing provided**")
            for _a in _external_hla:
                st.caption(f"• {_a}")

    if not _is_class_i:
        st.warning(
            "Class II or ambiguous length distribution detected.  "
            "Allele scoring requires class I peptides (7–11 aa peak).  "
            "Override with the HLA class selector above if needed."
        )

# ── 4. Motif Deconvolution ────────────────────────────────────────────────────

if _peptides and _is_class_i:
    st.divider()
    st.subheader("4. Motif Deconvolution")
    st.markdown(
        "Unsupervised EM clustering discovers binding motifs without a candidate allele panel. "
        "Each cluster corresponds to one HLA allele's characteristic anchor preferences. "
        "For a diploid sample expect up to **6 motifs** (2 HLA-A · 2 HLA-B · 2 HLA-C); "
        "fewer appear when alleles are weakly expressed, have overlapping motifs, or are lost."
    )

    _md_col1, _md_col2, _md_col3 = st.columns(3)
    with _md_col1:
        _md_length = st.selectbox(
            "Peptide length",
            options=[9, 8, 10, 11],
            key="md_length",
            help="9-mers are the primary class I length.",
        )
    with _md_col2:
        _md_k_mode = st.radio(
            "K (# clusters) selection",
            ["Auto", "Fixed"],
            horizontal=True,
            key="md_k_mode",
            help="Auto uses an anchor-aware information criterion.",
        )
    with _md_col3:
        _md_fixed_k = st.number_input(
            "Fixed K",
            min_value=1, max_value=8, value=6,
            key="md_fixed_k",
            disabled=(_md_k_mode == "Auto"),
            help="Expected ≤6 for diploid (2A + 2B + 2C).",
        )

    _md_n_seqs = len([p for p in _peptides if len(p) == _md_length])
    st.caption(f"{_md_n_seqs:,} {_md_length}-mer(s) available for clustering.")

    if _md_n_seqs < 50:
        st.warning(
            f"Too few {_md_length}-mers ({_md_n_seqs}) for reliable deconvolution.  "
            "Provide ≥200 peptides for best results."
        )
    else:
        _md_run = st.button("Run Motif Deconvolution", key="md_run_btn", type="primary")

        if _md_run:
            with st.spinner("Running EM motif deconvolution…"):
                _md_result = run_deconvolution(
                    _peptides,
                    length=int(_md_length),
                    fixed_k=int(_md_fixed_k) if _md_k_mode == "Fixed" else None,
                    n_restarts=5,
                    n_iter=300,
                )

            if _md_result is None:
                st.error("Deconvolution failed — too few valid peptides.")
            else:
                annotate_clusters(_md_result, CANDIDATE_ALLELES_CLASS_I)
                st.session_state["hla_motif_result"] = _md_result
                st.success(
                    f"Found **{_md_result.k_selected} cluster(s)** from "
                    f"{_md_result.n_peptides:,} {_md_length}-mers."
                )

# ── Motif Deconvolution Results ───────────────────────────────────────────────

if "hla_motif_result" in st.session_state:
    _mdr: object = st.session_state["hla_motif_result"]  # DeconvolutionResult

    st.divider()
    st.subheader("Motif Deconvolution Results")

    # BIC/AIC curve
    if len(_mdr.bic_curve) > 1:
        _bic_df = pd.DataFrame(
            [{"K": k, "Score (lower = better)": v} for k, v in sorted(_mdr.bic_curve.items())]
        )
        _sel_k = _mdr.k_selected
        fig_bic = px.line(
            _bic_df, x="K", y="Score (lower = better)",
            markers=True, title="Anchor-AIC curve — selected K marked",
        )
        fig_bic.add_vline(x=_sel_k, line_dash="dash", line_color="steelblue",
                          annotation_text=f"K={_sel_k}", annotation_position="top right")
        fig_bic.update_layout(height=260, margin=dict(t=40, b=30, l=60, r=20), plot_bgcolor="white")
        st.plotly_chart(fig_bic, use_container_width=True, key="md_bic_chart")

    # Cluster summary table
    _cl_summ = []
    for _ci in _mdr.clusters:
        _p2, _pc = top_anchor_residues(_ci.pwm, n=3)
        _top_match = _ci.allele_matches[0] if _ci.allele_matches else {}
        _cl_summ.append({
            "Cluster": _ci.cluster_id,
            "Peptides": len(_ci.peptides),
            "Fraction": f"{_ci.weight:.1%}",
            "P2 anchors": "  ".join(f"{aa}({f:.0%})" for aa, f in _p2),
            "PΩ anchors": "  ".join(f"{aa}({f:.0%})" for aa, f in _pc),
            "Top allele match": _top_match.get("allele", "—"),
            "Match score": f"{_top_match.get('combined_score', 0):.3f}",
        })
    st.dataframe(
        pd.DataFrame(_cl_summ),
        use_container_width=True, hide_index=True,
        key="md_summ_table",
    )

    st.markdown("### Per-Cluster Sequence Logos")

    if not _HAS_LOGOMAKER:
        st.info("Install `logomaker` to view sequence logos: `pip install logomaker`")
    else:
        _md_cols = st.columns(min(3, len(_mdr.clusters)))
        for _col_i, _ci in enumerate(_mdr.clusters):
            with _md_cols[_col_i % len(_md_cols)]:
                _pfm = pwm_to_logo_df(_ci.pwm)
                try:
                    _fig_logo, _ax_logo = plt.subplots(figsize=(5, 1.8))
                    logomaker.Logo(_pfm, ax=_ax_logo, color_scheme="chemistry")
                    _ax_logo.set_title(
                        f"Cluster {_ci.cluster_id}  ({len(_ci.peptides):,} peps)",
                        fontsize=9,
                    )
                    _ax_logo.set_xlabel("Position", fontsize=7)
                    _ax_logo.tick_params(labelsize=6)
                    plt.tight_layout()
                    st.pyplot(_fig_logo)
                    plt.close(_fig_logo)
                except Exception as _e:
                    st.caption(f"Logo render error: {_e}")

                if _ci.allele_matches:
                    _top3 = _ci.allele_matches[:3]
                    st.caption(
                        "**Top allele matches:** "
                        + "  ·  ".join(
                            f"{m['allele']} ({m['combined_score']:.3f})"
                            for m in _top3
                        )
                    )
                    st.caption(
                        f"P2: {'  '.join(f'{aa}({f:.0%})' for aa,f in top_anchor_residues(_ci.pwm)[0][:3])}  "
                        f"PΩ: {'  '.join(f'{aa}({f:.0%})' for aa,f in top_anchor_residues(_ci.pwm)[1][:3])}"
                    )

    st.markdown("### Allele Calls by Cluster")
    _allele_rows = []
    for _ci in _mdr.clusters:
        for _rank_i, _m in enumerate(_ci.allele_matches[:3]):
            _allele_rows.append({
                "Cluster": _ci.cluster_id,
                "Rank": _rank_i + 1,
                "Allele": _m["allele"],
                "Locus": _m["locus"],
                "P2 score": _m["p2_score"],
                "PΩ score": _m["pc_score"],
                "Combined": _m["combined_score"],
                "Cluster size": len(_ci.peptides),
            })
    if _allele_rows:
        st.dataframe(
            pd.DataFrame(_allele_rows),
            use_container_width=True, hide_index=True,
            column_config={
                "P2 score": st.column_config.NumberColumn(format="%.3f"),
                "PΩ score": st.column_config.NumberColumn(format="%.3f"),
                "Combined": st.column_config.NumberColumn(format="%.3f"),
            },
            key="md_allele_table",
        )

    # Download
    _md_dl_col1, _md_dl_col2 = st.columns(2)
    with _md_dl_col1:
        _md_json = {
            "length": _mdr.length,
            "k_selected": _mdr.k_selected,
            "n_peptides": _mdr.n_peptides,
            "clusters": [
                {
                    "cluster_id": ci.cluster_id,
                    "n_peptides": len(ci.peptides),
                    "weight": ci.weight,
                    "allele_matches": ci.allele_matches,
                    "anchor_p2": [{"aa": aa, "freq": round(f, 4)} for aa, f in top_anchor_residues(ci.pwm)[0]],
                    "anchor_pc": [{"aa": aa, "freq": round(f, 4)} for aa, f in top_anchor_residues(ci.pwm)[1]],
                }
                for ci in _mdr.clusters
            ],
        }
        st.download_button(
            "⬇ Download motif calls (JSON)",
            data=json.dumps(_md_json, indent=2).encode("utf-8"),
            file_name="hla_motif_decon.json",
            mime="application/json",
            key="md_dl_json",
        )
    with _md_dl_col2:
        if _allele_rows:
            st.download_button(
                "⬇ Download allele calls (CSV)",
                data=pd.DataFrame(_allele_rows).to_csv(index=False).encode("utf-8"),
                file_name="hla_motif_allele_calls.csv",
                mime="text/csv",
                key="md_dl_csv",
            )

