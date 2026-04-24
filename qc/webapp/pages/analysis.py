"""Analysis page — three-step QC workflow (Upload → Configure → Report)."""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from modules.auth import hash_password, password_strength_error, verify_password
from modules.charts import (
    SAMPLE_COLORS,
    chart_aa_frequency,
    chart_aa_heatmap,
    chart_charge_per_sample,
    chart_charge_pie,
    chart_contaminant_rate,
    chart_correlation_heatmap,
    chart_genes_per_sample,
    chart_jaccard_heatmap,
    chart_length_all,
    chart_length_per_sample,
    chart_mbr_rate,
    chart_msms_mbr,
    chart_msstats_intensity_violin,
    chart_msstats_missing,
    chart_sample_intensity_correlation,
    chart_pca,
    chart_pca_variance,
    chart_peptide_prevalence,
    chart_protein_source_per_sample,
    chart_protein_source_pie,
    chart_sample_intensity_histogram,
    chart_sample_overlap_bar,
    chart_sample_spectral_histogram,
    chart_sequence_logo,
    chart_shared_heatmap,
    chart_spectral_violin,
    chart_top_proteins,
    chart_venn2,
)
from modules.database import (
    create_user,
    get_user_by_id,
    get_user_by_username,
    init_db,
    save_run as db_save_run,
    update_run_data_dir,
    username_exists,
)
from modules.mapping import (
    ColumnMapping,
    SampleDef,
    apply_column_mapping,
    detect_sample_columns,
    suggest_column,
    validate_mapping,
)
from modules.metrics import (
    AAS,
    compute_aa_composition,
    compute_charge_distribution,
    compute_contaminant_proteins,
    compute_contaminant_summary,
    compute_dataset_stats,
    compute_overlap,
    compute_pca,
    compute_sample_summary,
    parse_charges,
)
from modules.prediction import (
    COMMON_ALLELES as PRED_ALLELES,
    SUPPORTED_LENGTHS as PRED_LENGTHS,
    postprocess as pred_postprocess,
    predictor_status_table,
    run_prediction,
)
from modules.predictors.registry import get_available_predictors
from modules.parsing import detect_delimiter, load_table
from modules.report import build_csv_summary, build_html_report
from modules.storage import serialize_run


# ── Shared UI helpers ─────────────────────────────────────────────────────────

def _step_indicator(current: int) -> None:
    labels = ["1 · Upload", "2 · Configure", "3 · Report"]
    cols = st.columns(3)
    for i, (col, label) in enumerate(zip(cols, labels)):
        with col:
            if i + 1 == current:
                st.markdown(
                    f"<div style='text-align:center;font-weight:700;font-size:1rem;"
                    f"color:#4C72B0;border-bottom:3px solid #4C72B0;padding-bottom:4px'>"
                    f"{label}</div>",
                    unsafe_allow_html=True,
                )
            elif i + 1 < current:
                st.markdown(
                    f"<div style='text-align:center;color:#aaa;font-size:.9rem'>"
                    f"Done: {label}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:center;color:#aaa;font-size:.9rem'>{label}</div>",
                    unsafe_allow_html=True,
                )
    st.divider()


def _none_opt(val: str) -> str | None:
    return None if val == "(none)" else val


def _col_selector(
    label: str,
    key: str,
    columns: list[str],
    default: str | None = None,
    required: bool = False,
    help_text: str = "",
) -> str | None:
    opts = ["(none)"] + columns if not required else columns
    idx = 0
    if default and default in columns:
        idx = (opts.index(default)) if not required else columns.index(default)
    sel = st.selectbox(label, opts, index=idx, key=key, help=help_text)
    return _none_opt(sel) if not required else sel


def _optional_fig(fig: Any, label: str = "") -> None:
    if fig is None or (hasattr(fig, "data") and len(fig.data) == 0):
        st.caption(f"_{label or 'Chart not available for this dataset.'}_")
    else:
        st.plotly_chart(fig, use_container_width=True)


def _sample_filter(section_key: str, all_samples: list[str], min_required: int = 1) -> list[str]:
    """Render a per-section sample multiselect with All/Clear convenience buttons."""
    state_key = f"filter_{section_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = list(all_samples)

    col1, col2 = st.columns([5, 1])
    with col1:
        selected: list[str] = st.multiselect(
            "Samples included in this section",
            options=all_samples,
            key=state_key,
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("All", key=f"{state_key}_all"):
                st.session_state[state_key] = list(all_samples)
                st.rerun()
        with bc2:
            if st.button("Clear", key=f"{state_key}_clear"):
                st.session_state[state_key] = []
                st.rerun()

    n_sel, n_all = len(selected), len(all_samples)
    if n_sel < n_all:
        st.caption(f"Showing {n_sel} of {n_all} samples")

    if n_sel < min_required:
        st.info(f"Select at least {min_required} sample(s) to display this section.")
        return []
    return selected


def _filter_pca_data(pca_data: dict[str, Any], selected_names: list[str]) -> dict[str, Any] | None:
    all_samples: list[str] = pca_data.get("samples", [])
    scatter_idx = [i for i, s in enumerate(all_samples) if s in selected_names]
    if len(scatter_idx) < 2:
        return None

    corr_samples: list[str] = pca_data.get("samples_ordered", [])
    corr_idx = [i for i, s in enumerate(corr_samples) if s in selected_names]

    result = dict(pca_data)
    result["samples"] = [all_samples[i] for i in scatter_idx]
    result["coords"] = pca_data["coords"][scatter_idx, :]
    if corr_idx:
        result["samples_ordered"] = [corr_samples[i] for i in corr_idx]
        result["corr_ordered"] = pca_data["corr_ordered"][np.ix_(corr_idx, corr_idx)]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Upload
# ══════════════════════════════════════════════════════════════════════════════

def render_upload() -> None:
    st.header("Step 1: Upload FragPipe Peptide Table")

    with st.expander("What file to upload", expanded=False):
        st.markdown(
            "Upload the **combined_peptide.tsv** (or equivalent) exported by FragPipe. "
            "The file must contain:\n"
            "- One row per unique peptide\n"
            "- Per-sample columns such as `<sample> Match Type`, "
            "`<sample> Spectral Count`, and optionally `<sample> Intensity`\n\n"
            "Optionally upload **msstats.csv** to enable the intensity distribution section."
        )

    uploaded = st.file_uploader(
        "FragPipe peptide table (.tsv / .csv / .txt)",
        type=["tsv", "csv", "txt"],
        key="uploader_main",
    )

    if uploaded is not None:
        raw = uploaded.read()
        detected_delim = detect_delimiter(raw)
        st.caption(
            f"Auto-detected delimiter: `{'TAB' if detected_delim == chr(9) else ','}`"
        )
        delim_choice = st.radio(
            "Delimiter",
            options=["Tab (\\t)", "Comma (,)"],
            index=0 if detected_delim == "\t" else 1,
            horizontal=True,
            key="delim_radio",
        )
        delimiter = "\t" if "Tab" in delim_choice else ","

        try:
            df_raw = load_table(raw, delimiter)
        except Exception as exc:
            st.error(f"Failed to parse file: {exc}")
            return

        st.success(
            f"Loaded **{len(df_raw):,} rows x {len(df_raw.columns)} columns**"
        )
        st.dataframe(df_raw.head(6), use_container_width=True, height=220)

        st.markdown(f"**Columns ({len(df_raw.columns)})**")
        st.code("  ·  ".join(df_raw.columns.tolist()), language=None)

        st.markdown("---")
        st.subheader("Optional: MSstats Intensity File")
        ms_uploaded = st.file_uploader(
            "msstats.csv (optional — enables intensity distribution section)",
            type=["csv"],
            key="uploader_ms",
        )
        ms_df: pd.DataFrame | None = None
        if ms_uploaded is not None:
            try:
                ms_df = pd.read_csv(ms_uploaded)
                required_ms_cols = {"Intensity", "Run", "Condition"}
                if not required_ms_cols.issubset(ms_df.columns):
                    st.warning(
                        f"MSstats file is missing columns: "
                        f"{required_ms_cols - set(ms_df.columns)}. "
                        "Intensity section will be skipped."
                    )
                    ms_df = None
                else:
                    st.success(
                        f"MSstats loaded: {len(ms_df):,} rows, "
                        f"{ms_df['Run'].nunique()} runs, "
                        f"{ms_df['Condition'].nunique()} conditions."
                    )
            except Exception as exc:
                st.warning(f"Could not load MSstats file: {exc}")

        if st.button("Next: Configure Column Mapping", type="primary"):
            st.session_state.df_raw = df_raw
            st.session_state.delimiter = delimiter
            st.session_state.ms_df = ms_df
            st.session_state.step = 2
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Column + sample mapping
# ══════════════════════════════════════════════════════════════════════════════

def render_mapping() -> None:
    df = st.session_state.df_raw
    if df is None:
        st.warning("No data loaded. Please go back to step 1.")
        if st.button("Back to Upload"):
            st.session_state.step = 1
            st.rerun()
        return

    st.header("Step 2: Configure Column and Sample Mapping")
    cols_list = df.columns.tolist()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Required and Optional Columns")

        pep_default = suggest_column(
            ["Peptide Sequence", "Peptide", "peptide_sequence", "sequence"], cols_list
        )
        pep_col = st.selectbox(
            "Peptide Sequence column *",
            cols_list,
            index=cols_list.index(pep_default) if pep_default else 0,
            key="map_pep",
            help="Column containing the amino acid sequence string.",
        )

        protein_default = suggest_column(
            ["Protein", "Proteins", "protein", "proteinID"], cols_list
        )
        protein_col = _col_selector(
            "Protein column",
            "map_protein",
            cols_list,
            protein_default,
            help_text="Used for contaminant detection (Cont_ prefix) and source classification.",
        )

        gene_default = suggest_column(["Gene", "gene", "Gene Names"], cols_list)
        gene_col = _col_selector("Gene column", "map_gene", cols_list, gene_default)

        length_default = suggest_column(
            ["Peptide Length", "length", "pep_length"], cols_list
        )
        length_col = _col_selector(
            "Peptide Length column",
            "map_length",
            cols_list,
            length_default,
            help_text="Leave as (none) to auto-compute from sequence length.",
        )

        charge_default = suggest_column(["Charges", "Charge", "charge"], cols_list)
        charge_col = _col_selector("Charge column", "map_charge", cols_list, charge_default)

        entry_default = suggest_column(["Entry Name", "entry_name", "UniProt Entry"], cols_list)
        entry_col = _col_selector("Entry Name column", "map_entry", cols_list, entry_default)

        desc_default = suggest_column(
            ["Protein Description", "description", "protein_description"], cols_list
        )
        desc_col = _col_selector(
            "Protein Description column", "map_desc", cols_list, desc_default
        )

    with right:
        st.subheader("Sample Mapping")

        st.markdown(
            "Define each sample and map it to its FragPipe columns. "
            "**Match Type** is required; Spectral Count and Intensity are optional."
        )

        if st.button("Auto-detect from column names"):
            detected = detect_sample_columns(df)
            if detected:
                st.session_state["samples_state"] = [
                    {
                        "name": sd.name,
                        "match_col": sd.match_col or "",
                        "spectral_col": sd.spectral_col or "",
                        "intensity_col": sd.intensity_col or "",
                    }
                    for sd in detected
                ]
                st.success(f"Auto-detected {len(detected)} sample(s).")
            else:
                st.warning(
                    "No columns ending in ' Match Type' found. "
                    "Add samples manually below."
                )

        with st.expander("Paste sample names (one per line or comma-separated)"):
            pasted = st.text_area(
                "Sample names",
                key="paste_names",
                placeholder="PBMC_1\nPBMC_2\nTumor_1",
                height=80,
            )
            if st.button("Apply pasted names"):
                raw_names = [
                    n.strip()
                    for n in re.split(r"[,\n]", pasted)
                    if n.strip()
                ]
                existing = st.session_state.get("samples_state", [])
                existing_names = {r["name"] for r in existing}
                for n in raw_names:
                    if n not in existing_names:
                        existing.append(
                            {"name": n, "match_col": "", "spectral_col": "", "intensity_col": ""}
                        )
                st.session_state["samples_state"] = existing
                st.rerun()

        if "samples_state" not in st.session_state:
            auto = detect_sample_columns(df)
            st.session_state["samples_state"] = [
                {
                    "name": sd.name,
                    "match_col": sd.match_col or "",
                    "spectral_col": sd.spectral_col or "",
                    "intensity_col": sd.intensity_col or "",
                }
                for sd in auto
            ]

        samples_state: list[dict[str, str]] = st.session_state["samples_state"]
        col_opts_with_blank = [""] + cols_list

        rows_to_remove: list[int] = []
        for i, row in enumerate(samples_state):
            st.markdown(f"**Sample {i + 1}**")
            c1, c2, c3, c4, c5 = st.columns([2, 3, 3, 3, 0.5])
            with c1:
                row["name"] = st.text_input(
                    "Name", row.get("name", ""), key=f"sname_{i}"
                )
            with c2:
                mc_idx = (
                    col_opts_with_blank.index(row.get("match_col", ""))
                    if row.get("match_col", "") in col_opts_with_blank
                    else 0
                )
                row["match_col"] = st.selectbox(
                    "Match Type col",
                    col_opts_with_blank,
                    index=mc_idx,
                    key=f"smatch_{i}",
                )
            with c3:
                sc_idx = (
                    col_opts_with_blank.index(row.get("spectral_col", ""))
                    if row.get("spectral_col", "") in col_opts_with_blank
                    else 0
                )
                row["spectral_col"] = st.selectbox(
                    "Spectral Count col",
                    col_opts_with_blank,
                    index=sc_idx,
                    key=f"ssc_{i}",
                )
            with c4:
                ic_idx = (
                    col_opts_with_blank.index(row.get("intensity_col", ""))
                    if row.get("intensity_col", "") in col_opts_with_blank
                    else 0
                )
                row["intensity_col"] = st.selectbox(
                    "Intensity col",
                    col_opts_with_blank,
                    index=ic_idx,
                    key=f"sint_{i}",
                )
            with c5:
                # Push button down to align with selectbox values (label height ≈ 1.75rem)
                st.markdown(
                    '<div style="height:1.75rem"></div>',
                    unsafe_allow_html=True,
                )
                if st.button("×", key=f"srem_{i}", help="Remove this sample"):
                    rows_to_remove.append(i)

        for idx in sorted(rows_to_remove, reverse=True):
            del samples_state[idx]
        if rows_to_remove:
            st.rerun()

        if st.button("Add sample"):
            samples_state.append(
                {"name": "", "match_col": "", "spectral_col": "", "intensity_col": ""}
            )
            st.rerun()

    st.markdown("---")
    run_label = st.text_input(
        "Run label (used in report title)",
        value=datetime.now().strftime("Run %Y-%m-%d"),
        key="run_label",
    )

    nav_l, nav_r = st.columns([1, 1])
    with nav_l:
        if st.button("Back to Upload"):
            st.session_state.step = 1
            st.rerun()
    with nav_r:
        if st.button("Run QC Analysis", type="primary"):
            sample_defs = [
                SampleDef(
                    name=r["name"],
                    match_col=r["match_col"] or None,
                    spectral_col=r["spectral_col"] or None,
                    intensity_col=r["intensity_col"] or None,
                )
                for r in samples_state
            ]
            mapping = ColumnMapping(
                peptide_col=pep_col,
                protein_col=protein_col,
                gene_col=gene_col,
                length_col=length_col,
                charge_col=charge_col,
                entry_name_col=entry_col,
                protein_desc_col=desc_col,
                samples=sample_defs,
            )
            errors = validate_mapping(df, mapping)
            if errors:
                for err in errors:
                    st.error(err)
            else:
                st.session_state.mapping = mapping
                st.session_state.df = apply_column_mapping(df, mapping)
                st.session_state.step = 3
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MHC-I Prediction tab (embedded inside Screen 3)
# ══════════════════════════════════════════════════════════════════════════════

def _render_mhci_tab(df: pd.DataFrame) -> None:
    """MHC-I binding/presentation prediction using locally-installed open-source tools."""
    st.markdown(
        "Run MHC Class I binding and presentation predictions on the peptides in this "
        "dataset using locally-installed open-source ML tools — no external API required."
    )

    available = get_available_predictors()
    available_names = [cls.name for cls in available]

    # ── Predictor availability status ─────────────────────────────────────────
    with st.expander("Predictor availability"):
        status_rows = predictor_status_table()
        st.dataframe(
            pd.DataFrame(status_rows)[["Tool", "Available", "Description"]],
            use_container_width=True,
            hide_index=True,
        )

    if not available:
        st.warning(
            "No prediction tools are installed.  "
            "Install MHCflurry (`pip install mhcflurry && mhcflurry-downloads fetch`), "
            "NetMHCpan (https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/), "
            "BigMHC (`pip install bigmhc`), TransHLA (`pip install TransHLA`), "
            "or UniPMT, then restart the app."
        )
        return

    # ── Configuration widgets ─────────────────────────────────────────────────
    for _cfg_key, _cfg_default in [
        ("mhci_cfg_tool", available_names[0]),
        ("mhci_cfg_alleles", ["HLA-A*02:01"]),
        ("mhci_cfg_lengths", [9]),
        ("mhci_cfg_custom", ""),
    ]:
        if _cfg_key not in st.session_state:
            st.session_state[_cfg_key] = _cfg_default

    _it, _ia, _il = st.columns([2, 3, 2])
    with _it:
        st.selectbox(
            "Prediction tool",
            options=available_names,
            key="mhci_cfg_tool",
        )
    with _ia:
        st.multiselect(
            "HLA Alleles",
            options=PRED_ALLELES,
            key="mhci_cfg_alleles",
        )
        st.text_input(
            "Custom allele (comma-separated)",
            key="mhci_cfg_custom",
            placeholder="HLA-A*68:01",
        )
    with _il:
        st.multiselect(
            "Peptide Lengths (mer)",
            options=PRED_LENGTHS,
            key="mhci_cfg_lengths",
        )

    sel_tool: str = st.session_state.get("mhci_cfg_tool", available_names[0])
    sel_alleles: list[str] = list(st.session_state.get("mhci_cfg_alleles", []))
    custom_raw: str = st.session_state.get("mhci_cfg_custom", "").strip()
    if custom_raw:
        seen: set[str] = set(sel_alleles)
        for a in custom_raw.split(","):
            a = a.strip()
            if a and a not in seen:
                sel_alleles.append(a)
                seen.add(a)
    sel_lengths: list[int] = list(st.session_state.get("mhci_cfg_lengths", []))

    if not sel_alleles or not sel_lengths:
        st.warning("Select at least one HLA allele and one peptide length to run predictions.")
        return

    # ── Derive peptides from the loaded dataset ───────────────────────────────
    unique_peptides: list[str] = (
        df["_peptide"].dropna().unique().tolist() if "_peptide" in df.columns else []
    )
    matching_peptides: list[str] = [p for p in unique_peptides if len(p) in set(sel_lengths)]

    st.caption(
        f"{len(unique_peptides):,} unique peptides in dataset · "
        f"{len(matching_peptides):,} match the selected length(s)"
    )

    if len(matching_peptides) > 5_000:
        st.warning(
            f"{len(matching_peptides):,} peptides will be scored — "
            "this may take several minutes depending on the tool and your hardware."
        )

    # ── Run button ────────────────────────────────────────────────────────────
    if st.button("Run Prediction", type="primary", key="mhci_tab_run"):
        if not matching_peptides:
            st.error(
                "No peptides in the dataset match the selected lengths. "
                "Adjust the Peptide Lengths above."
            )
        else:
            with st.status(
                f"Running {sel_tool} on {len(matching_peptides):,} peptide(s) × "
                f"{len(sel_alleles)} allele(s)…",
                expanded=True,
            ) as status:
                result_df, errors = run_prediction(
                    matching_peptides, sel_alleles, sel_lengths, sel_tool
                )
                if errors:
                    for err in errors:
                        st.write(f":x: {err}")
                    status.update(label="Prediction failed", state="error")
                elif result_df.empty:
                    st.write(":warning: No results returned.")
                    status.update(label="No results", state="error")
                else:
                    st.write(f":white_check_mark: {len(result_df):,} predictions complete.")
                    status.update(label="Done", state="complete", expanded=False)

            if not result_df.empty:
                st.session_state["mhci_tab_results"] = result_df
            for err in errors:
                st.warning(err)

    # ── Results ───────────────────────────────────────────────────────────────
    if "mhci_tab_results" in st.session_state:
        result_df = pred_postprocess(st.session_state["mhci_tab_results"])

        display_cols = [
            c for c in ["tool", "allele", "peptide", "score", "rank", "ic50", "binding_level"]
            if c in result_df.columns
        ]

        if not display_cols:
            st.warning("Unexpected result format.")
            st.code(str(result_df.columns.tolist()))
        else:
            st.subheader(f"Results — {len(result_df):,} predictions")

            if "binding_level" in result_df.columns:
                m1, m2, m3 = st.columns(3)
                m1.metric("Strong Binders (SB)", int((result_df["binding_level"] == "SB").sum()))
                m2.metric("Weak Binders (WB)", int((result_df["binding_level"] == "WB").sum()))
                m3.metric("Non-Binders (NB)", int((result_df["binding_level"] == "NB").sum()))

            display_df = result_df[display_cols].copy()
            for num_col in ("score", "rank", "ic50"):
                if num_col in display_df.columns:
                    display_df[num_col] = display_df[num_col].apply(
                        lambda x: f"{float(x):.4f}" if pd.notna(x) and str(x) != "nan" else "—"
                    )

            sort_col = next(
                (c for c in ["rank", "score"] if c in display_df.columns), display_df.columns[0]
            )
            st.dataframe(
                display_df.sort_values(sort_col, ascending=True, na_position="last"),
                use_container_width=True,
                height=420,
            )
            st.caption(
                "**Binding level thresholds:** SB = Strong Binder (EL %rank ≤ 0.5% or IC50 < 50 nM), "
                "WB = Weak Binder (EL %rank ≤ 2% or IC50 < 500 nM), NB = Non-Binder.  "
                "Probability-based tools (BigMHC, TransHLA, UniPMT) use score ≥ 0.9 for SB, ≥ 0.5 for WB."
            )
            st.download_button(
                "Download Results CSV",
                data=result_df[display_cols].to_csv(index=False).encode("utf-8"),
                file_name="mhci_analysis_predictions.csv",
                mime="text/csv",
                key="mhci_tab_download",
            )

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — QC Report
# ══════════════════════════════════════════════════════════════════════════════

def render_report() -> None:
    df: pd.DataFrame | None = st.session_state.df
    mapping: ColumnMapping | None = st.session_state.mapping
    ms_df: pd.DataFrame | None = st.session_state.ms_df

    if df is None or mapping is None:
        st.warning("Analysis not configured. Please start from Step 1.")
        if st.button("Back to Upload"):
            st.session_state.step = 1
            st.rerun()
        return

    run_label: str = st.session_state.get("run_label", "QC Report")
    sample_names = [sd.name for sd in mapping.samples]

    if st.session_state.get("_report_df_id") != id(df):
        for key in list(st.session_state.keys()):
            if key.startswith("filter_"):
                del st.session_state[key]
        st.session_state["_report_df_id"] = id(df)

    nav_l, _ = st.columns([1, 4])
    with nav_l:
        if st.button("Back to Mapping"):
            st.session_state.step = 2
            st.rerun()

    st.title(run_label)
    st.caption(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
        f"{len(df):,} peptides  ·  {len(sample_names)} samples"
    )

    # ── Pre-compute all metrics with a visible progress bar ──────────────────
    _load_heading = st.empty()
    _load_bar = st.empty()
    _load_heading.subheader("Generating report…")
    _prog = _load_bar.progress(0, text="Computing sample summary…")

    summary_df = compute_sample_summary(df, mapping)
    _prog.progress(15, text="Computing dataset statistics…")

    dataset_stats = compute_dataset_stats(df, mapping)
    _prog.progress(30, text="Analysing contaminants…")

    contam_df = compute_contaminant_summary(df, mapping)
    contam_proteins = compute_contaminant_proteins(df, mapping)
    _prog.progress(50, text="Computing peptide overlap…")

    jaccard, shared, detected_sets = compute_overlap(df, mapping)
    _prog.progress(65, text="Computing amino acid composition…")

    has_length = "_length" in df.columns
    has_source = "_source" in df.columns
    has_protein = "_protein" in df.columns and mapping.protein_col is not None
    has_charge = mapping.charge_col is not None and mapping.charge_col in df.columns
    has_gene = "_gene" in df.columns and mapping.gene_col is not None

    pos_freq, all_aa_freq, mers9 = (
        compute_aa_composition(df, 9)
        if has_length
        else (np.zeros((9, 20)), pd.Series(dtype=float), [])
    )
    _prog.progress(80, text="Running PCA…")

    pca_data = compute_pca(df, mapping)
    _prog.progress(90, text="Computing charge distribution…")

    charge_series: pd.Series | None = None
    if has_charge:
        charge_series = compute_charge_distribution(df, mapping.charge_col)

    intensity_pairs = [
        sd for sd in mapping.samples
        if sd.intensity_col and sd.intensity_col in df.columns
    ]

    _prog.progress(100, text="Report ready!")
    _load_heading.empty()
    _load_bar.empty()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Summary",
        "MS/MS and MBR",
        "Length",
        "Spectral Counts",
        "Intensity",
        "Contaminants",
        "Overlap",
        "Protein Source",
        "Amino Acids",
        "Charge",
        "PCA",
        "Per-Sample",
        "MHC-I Prediction",
    ])

    # ── Tab 0: Summary ────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Dataset Overview")
        c1, c2 = st.columns(2)
        with c1:
            for label, val in dataset_stats:
                st.metric(label, val)
        with c2:
            if not summary_df.empty:
                display_cols = [
                    c for c in summary_df.columns
                    if c not in ("MBR Rate %", "Contam Rate %")
                ]
                st.dataframe(summary_df[display_cols], use_container_width=True)

    # ── Tab 1: MS/MS & MBR ───────────────────────────────────────────────────
    with tabs[1]:
        st.markdown(
            "MS/MS identifications originate from directly sequenced spectra. "
            "MBR (Match Between Runs) transfers identifications across samples using accurate "
            "mass and retention time alignment. A high MBR proportion (commonly above 30%) "
            "may indicate low spectral quality or poorly overlapping peptide sets; "
            "the 30% threshold is a heuristic, not a universal rule."
        )
        if not summary_df.empty:
            sel = _sample_filter("msms_mbr", sample_names, min_required=1)
            if sel:
                fsummary = summary_df[summary_df["Sample"].isin(sel)]
                _optional_fig(chart_msms_mbr(fsummary))
                _optional_fig(chart_mbr_rate(fsummary))

    # ── Tab 2: Peptide length ─────────────────────────────────────────────────
    with tabs[2]:
        st.markdown(
            "MHC class I-associated peptides are most commonly 8-11 amino acids in length, "
            "with 9-mers predominating in many HLA-I alleles, though the optimal length range "
            "varies by allele. MHC class II-associated peptides typically range from "
            "approximately 13 to 25 amino acids, with considerable variation depending on "
            "allele and cleavage context. Length thresholds shown are heuristic guidelines "
            "and should be interpreted in the context of the specific assay and HLA alleles present."
        )
        if has_length:
            sel = _sample_filter("length", sample_names, min_required=1)
            if sel:
                fsamples = [sd for sd in mapping.samples if sd.name in sel]
                _optional_fig(chart_length_all(df))
                _optional_fig(chart_length_per_sample(df, fsamples))
        else:
            st.warning("No length column available.")

    # ── Tab 3: Spectral counts ────────────────────────────────────────────────
    with tabs[3]:
        sel = _sample_filter("spectral", sample_names, min_required=1)
        if sel:
            fsamples = [sd for sd in mapping.samples if sd.name in sel]
            _optional_fig(chart_spectral_violin(df, fsamples))

    # ── Tab 4: Intensity (MSstats) ────────────────────────────────────────────
    with tabs[4]:
        if ms_df is not None:
            _optional_fig(chart_msstats_missing(ms_df))
            _optional_fig(chart_msstats_intensity_violin(ms_df))
        else:
            st.info(
                "MSstats file not uploaded. Return to Step 1 and upload msstats.csv "
                "to enable this section."
            )
        if len(intensity_pairs) >= 2:
            st.subheader("Sample Intensity Correlations (from peptide table)")
            sel = _sample_filter("intensity", sample_names, min_required=2)
            if sel:
                fsamples = [sd for sd in intensity_pairs if sd.name in sel]
                if len(fsamples) >= 2:
                    _optional_fig(chart_sample_intensity_correlation(df, fsamples))

    # ── Tab 5: Contaminants ───────────────────────────────────────────────────
    with tabs[5]:
        st.markdown(
            "Contaminants are proteins flagged with the `Cont_` prefix in the FASTA database "
            "(commonly keratins, trypsin, and albumin). Rates above approximately 1-2% may "
            "indicate sample handling issues, though the expected rate depends on sample type "
            "and preparation protocol."
        )
        if has_protein and not contam_df.empty:
            sel = _sample_filter("contam", sample_names, min_required=1)
            if sel:
                fcontam = contam_df[contam_df["Sample"].isin(sel)]
                _optional_fig(chart_contaminant_rate(fcontam))
                if not contam_proteins.empty:
                    st.subheader("Top Contaminant Proteins")
                    st.dataframe(contam_proteins, use_container_width=True)
        else:
            st.warning("Protein column not mapped or no contaminants detected.")

    # ── Tab 6: Overlap ────────────────────────────────────────────────────────
    with tabs[6]:
        sel = _sample_filter("overlap", sample_names, min_required=2)
        if sel:
            sel_idx = [sample_names.index(n) for n in sel if n in sample_names]
            fjaccard = jaccard[np.ix_(sel_idx, sel_idx)]
            fshared = shared[np.ix_(sel_idx, sel_idx)]
            fsets = {n: detected_sets[n] for n in sel if n in detected_sets}
            _optional_fig(chart_jaccard_heatmap(fjaccard, sel))
            _optional_fig(chart_shared_heatmap(fshared, sel))
            _optional_fig(chart_peptide_prevalence(fsets, df["_peptide"]))

    # ── Tab 7: Protein source ─────────────────────────────────────────────────
    with tabs[7]:
        if has_source:
            sel = _sample_filter("source", sample_names, min_required=1)
            if sel:
                fsamples = [sd for sd in mapping.samples if sd.name in sel]
                _optional_fig(chart_protein_source_pie(df))
                _optional_fig(chart_protein_source_per_sample(df, fsamples))
                if has_gene:
                    _optional_fig(chart_genes_per_sample(df, fsamples))
        else:
            st.warning("Protein column not mapped — source analysis unavailable.")

    # ── Tab 8: Amino acids ────────────────────────────────────────────────────
    with tabs[8]:
        st.markdown(
            "The sequence logo shows amino acid frequencies at each position in 9-mer peptides. "
            "For many HLA-A and HLA-B alleles, positions P2 and P9 (highlighted) are "
            "commonly reported primary anchor positions; the specific enriched residues "
            "vary by allele."
        )
        if mers9:
            _optional_fig(chart_aa_heatmap(pos_freq, len(mers9)))
            b64 = chart_sequence_logo(pos_freq, len(mers9))
            st.image(f"data:image/png;base64,{b64}", use_column_width=True)
        else:
            st.warning("No 9-mer peptides found — check length column mapping.")

        if not all_aa_freq.empty:
            _optional_fig(chart_aa_frequency(all_aa_freq))

    # ── Tab 9: Charge ─────────────────────────────────────────────────────────
    with tabs[9]:
        if has_charge and charge_series is not None and not charge_series.empty:
            sel = _sample_filter("charge", sample_names, min_required=1)
            if sel:
                fsamples = [sd for sd in mapping.samples if sd.name in sel]
                _optional_fig(chart_charge_pie(charge_series))
                charge_vals = sorted(charge_series.index.tolist())
                _optional_fig(
                    chart_charge_per_sample(df, fsamples, mapping.charge_col, charge_vals)
                )
        else:
            st.warning("Charge column not mapped — charge analysis unavailable.")

    # ── Tab 10: PCA ───────────────────────────────────────────────────────────
    with tabs[10]:
        st.markdown(
            "PCA is computed on log-transformed intensities of peptides detected in at least "
            "min(3, n_samples) samples; missing values are imputed with the per-peptide "
            "minimum. Biological replicates are expected to cluster together. Outlier samples "
            "may indicate quality differences or batch effects."
        )
        if pca_data is not None:
            sel = _sample_filter("pca", sample_names, min_required=2)
            if sel:
                fpca = _filter_pca_data(pca_data, sel)
                if fpca is not None:
                    _optional_fig(chart_pca(fpca))
                    _optional_fig(chart_pca_variance(fpca))
                    _optional_fig(chart_correlation_heatmap(fpca))
                else:
                    st.info(
                        "Fewer than 2 samples with intensity data in the current selection. "
                        "Select more samples to enable PCA."
                    )
        else:
            st.warning(
                "Intensity columns not available for 2 or more samples — PCA unavailable. "
                "Map Intensity columns in the Sample Mapping step."
            )

    # ── Tab 11: Per-sample ────────────────────────────────────────────────────
    with tabs[11]:
        for i, sd in enumerate(mapping.samples):
            color = SAMPLE_COLORS[i % len(SAMPLE_COLORS)]
            with st.expander(f"**{sd.name}**", expanded=False):
                row_stats = (
                    summary_df[summary_df["Sample"] == sd.name]
                    if not summary_df.empty
                    else pd.DataFrame()
                )
                if not row_stats.empty:
                    r = row_stats.iloc[0]
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("MS/MS", f"{r['MS/MS']:,}")
                    mc2.metric("MBR", f"{r['MBR']:,}")
                    mc3.metric("MBR Rate", r["MBR Rate"])
                    mc4.metric("Contam Rate", r["Contam Rate"])

                pc1, pc2 = st.columns(2)
                with pc1:
                    if has_length and sd.match_col and sd.match_col in df.columns:
                        detected = df[df[sd.match_col] != "unmatched"]
                        len_counts = detected["_length"].value_counts().sort_index().dropna()
                        lengths = [int(l) for l in len_counts.index]
                        import plotly.graph_objects as _go
                        fig_sl = _go.Figure()
                        fig_sl.add_bar(
                            x=lengths,
                            y=len_counts.values.tolist(),
                            marker_color=[
                                "#2ca02c" if 8 <= l <= 11 else "#aec7e8" for l in lengths
                            ],
                        )
                        fig_sl.update_layout(
                            title=f"Length Distribution — {sd.name}",
                            template="plotly_white", height=300,
                            xaxis=dict(tickmode="linear", dtick=1),
                        )
                        st.plotly_chart(fig_sl, use_container_width=True)

                    if has_source and sd.match_col and sd.match_col in df.columns:
                        detected = df[df[sd.match_col] != "unmatched"]
                        sc_fig = chart_protein_source_pie(detected)
                        sc_fig.update_layout(title=f"Protein Source — {sd.name}", height=280)
                        st.plotly_chart(sc_fig, use_container_width=True)

                with pc2:
                    _optional_fig(chart_sample_spectral_histogram(df, sd, color))
                    _optional_fig(chart_sample_intensity_histogram(df, sd, color))

                _optional_fig(
                    chart_sample_overlap_bar(sd.name, detected_sets, color)
                )
                if has_protein and sd.match_col and sd.match_col in df.columns:
                    detected = df[df[sd.match_col] != "unmatched"]
                    _optional_fig(chart_top_proteins(detected, sd.name, color))

                if has_length and sd.match_col and sd.match_col in df.columns:
                    detected = df[df[sd.match_col] != "unmatched"]
                    s_pos_freq, _, s_mers9 = compute_aa_composition(detected, 9)
                    if s_mers9:
                        st.subheader(f"9-mer Motif — {sd.name}")
                        b64 = chart_sequence_logo(s_pos_freq, len(s_mers9))
                        st.image(f"data:image/png;base64,{b64}", use_column_width=True)
                    else:
                        st.caption("No 9-mer peptides detected in this sample.")

    # ── Tab 12: MHC-I Prediction ──────────────────────────────────────────────
    with tabs[12]:
        _render_mhci_tab(df)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Download")
    dl_l, dl_r = st.columns(2)

    with dl_l:
        @st.cache_data(show_spinner="Building HTML report...")
        def _build_html(run_label: str, _df_id: int) -> str:
            figs: dict[str, Any] = {}
            if not summary_df.empty:
                figs["msms_mbr"] = chart_msms_mbr(summary_df)
                figs["mbr_rate"] = chart_mbr_rate(summary_df)
            if has_length:
                figs["length_all"] = chart_length_all(df)
                figs["length_per_sample"] = chart_length_per_sample(df, mapping.samples)
            figs["spectral_violin"] = chart_spectral_violin(df, mapping.samples)
            if ms_df is not None:
                figs["msstats_missing"] = chart_msstats_missing(ms_df)
                figs["msstats_violin"] = chart_msstats_intensity_violin(ms_df)
            if len(intensity_pairs) >= 2:
                figs["intensity_corr"] = chart_sample_intensity_correlation(df, mapping.samples)
            if has_protein and not contam_df.empty:
                figs["contam_rate"] = chart_contaminant_rate(contam_df)
                figs["contam_proteins"] = contam_proteins
            figs["jaccard"] = chart_jaccard_heatmap(jaccard, sample_names)
            figs["shared"] = chart_shared_heatmap(shared, sample_names)
            figs["prevalence"] = chart_peptide_prevalence(detected_sets, df["_peptide"])
            if has_source:
                figs["source_pie"] = chart_protein_source_pie(df)
                figs["source_per_sample"] = chart_protein_source_per_sample(df, mapping.samples)
                if has_gene:
                    figs["genes_per_sample"] = chart_genes_per_sample(df, mapping.samples)
            if mers9:
                figs["aa_heatmap"] = chart_aa_heatmap(pos_freq, len(mers9))
                figs["seq_logo"] = chart_sequence_logo(pos_freq, len(mers9))
            if not all_aa_freq.empty:
                figs["aa_freq"] = chart_aa_frequency(all_aa_freq)
            if has_charge and charge_series is not None and not charge_series.empty:
                figs["charge_pie"] = chart_charge_pie(charge_series)
                figs["charge_per_sample"] = chart_charge_per_sample(
                    df, mapping.samples, mapping.charge_col, sorted(charge_series.index.tolist())
                )
            if pca_data is not None:
                figs["pca"] = chart_pca(pca_data)
                figs["pca_variance"] = chart_pca_variance(pca_data)
                figs["correlation"] = chart_correlation_heatmap(pca_data)

            for i, sd in enumerate(mapping.samples):
                sid = re.sub(r"[^a-zA-Z0-9_-]", "_", sd.name)
                color = SAMPLE_COLORS[i % len(SAMPLE_COLORS)]
                figs[f"sample_{sid}_sc"] = chart_sample_spectral_histogram(df, sd, color)
                figs[f"sample_{sid}_int"] = chart_sample_intensity_histogram(df, sd, color)
                figs[f"sample_{sid}_overlap"] = chart_sample_overlap_bar(
                    sd.name, detected_sets, color
                )
                if has_protein and sd.match_col and sd.match_col in df.columns:
                    det = df[df[sd.match_col] != "unmatched"]
                    figs[f"sample_{sid}_prot"] = chart_top_proteins(det, sd.name, color)
                if has_length and sd.match_col and sd.match_col in df.columns:
                    det = df[df[sd.match_col] != "unmatched"]
                    spf, _, smers = compute_aa_composition(det, 9)
                    if smers:
                        figs[f"sample_{sid}_logo"] = chart_sequence_logo(spf, len(smers))

            return build_html_report(run_label, dataset_stats, summary_df, figs, sample_names)

        html_bytes = _build_html(run_label, id(df)).encode("utf-8")
        st.download_button(
            "Download HTML Report",
            data=html_bytes,
            file_name=f"immunopeptidomics_qc_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
        )

    with dl_r:
        if not summary_df.empty:
            csv_bytes = build_csv_summary(summary_df).encode("utf-8")
            st.download_button(
                "Download Summary CSV",
                data=csv_bytes,
                file_name=f"qc_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

    # ── Save run ──────────────────────────────────────────────────────────────
    st.divider()
    if st.session_state.get("user_id"):
        st.subheader("Save Run")
        st.markdown("Save this analysis to your account to revisit it later from My Runs.")
        save_name = st.text_input("Run name", value=run_label, key="save_run_name")
        if st.button("Save to My Runs", type="secondary", key="save_run_btn"):
            summary_dict: dict[str, float] = {}
            if not summary_df.empty and "Total Detected" in summary_df.columns:
                summary_dict["median_peptide_count"] = float(
                    summary_df["Total Detected"].median()
                )
                if "MBR Rate (numeric)" in summary_df.columns:
                    summary_dict["median_mbr_rate"] = float(
                        summary_df["MBR Rate (numeric)"].median()
                    )
            run_id = db_save_run(
                user_id=int(st.session_state["user_id"]),
                name=save_name.strip() or run_label,
                sample_names=sample_names,
                summary=summary_dict,
                data_dir="",
            )
            data_dir = serialize_run(
                user_id=int(st.session_state["user_id"]),
                run_id=run_id,
                df=df,
                mapping=mapping,
                ms_df=ms_df,
            )
            update_run_data_dir(
                run_id=run_id,
                user_id=int(st.session_state["user_id"]),
                data_dir=data_dir,
            )
            st.success(f"Run saved as '{save_name}'.")
    else:
        st.caption("Log in via the sidebar to save this run to your account.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _step_indicator(st.session_state.step)

    if st.session_state.step == 1:
        render_upload()
    elif st.session_state.step == 2:
        render_mapping()
    elif st.session_state.step == 3:
        render_report()


main()
