"""MHC Class I Binding Prediction — standalone page using IEDB Tools REST API."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from modules.iedb import (
    COMMON_ALLELES,
    IEDB_EMAIL,
    LONG_JOB_THRESHOLD,
    METHODS,
    postprocess,
    run_batched,
)

# ── Page ──────────────────────────────────────────────────────────────────────

st.title("MHC Class I Binding Prediction")
st.markdown(
    "Predict MHC Class I peptide–HLA binding using the "
    "[IEDB Tools REST API](https://tools.iedb.org/main/tools-api/)."
)

st.info(
    "**IEDB usage guidelines**  \n"
    "- Jobs are submitted **one at a time** and each must complete before the next is sent.  \n"
    "- Inputs estimated to take **> 10 minutes** automatically include an email address "
    f"(`{IEDB_EMAIL}`) so IEDB can notify on completion.  \n"
    "- Large requests are split into per-allele / per-length **batches** automatically."
)

# ── Input: sequences ──────────────────────────────────────────────────────────
st.subheader("1. Peptide Sequences")

input_mode = st.radio(
    "Input method",
    ["Paste sequences", "Upload file"],
    horizontal=True,
    key="iedb_input_mode",
)

sequences: list[str] = []

if input_mode == "Paste sequences":
    raw = st.text_area(
        "One peptide per line",
        height=150,
        placeholder="GILGFVFTL\nSLYNTVATL\nKLVVVGAVG",
        key="iedb_seq_text",
    )
    if raw.strip():
        sequences = [s.strip().upper() for s in raw.splitlines() if s.strip()]
else:
    f = st.file_uploader(
        "Plain-text file, one peptide per line (.txt / .csv)",
        type=["txt", "csv"],
        key="iedb_seq_file",
    )
    if f is not None:
        content = f.read().decode("utf-8", errors="replace")
        sequences = [s.strip().upper() for s in content.splitlines() if s.strip()]

if sequences:
    st.caption(f"{len(sequences):,} sequence(s) loaded")

# ── Input: alleles + lengths ──────────────────────────────────────────────────
st.subheader("2. Alleles and Peptide Lengths")

col_a, col_l = st.columns(2)

with col_a:
    selected_alleles: list[str] = st.multiselect(
        "HLA Alleles",
        options=COMMON_ALLELES,
        default=["HLA-A*02:01"],
        key="iedb_alleles",
    )
    custom = st.text_input(
        "Custom allele (e.g. HLA-A*68:01)",
        key="iedb_custom_allele",
        placeholder="HLA-A*68:01",
    )
    if custom.strip():
        seen: set[str] = set(selected_alleles)
        for a in custom.strip().split(","):
            a = a.strip()
            if a and a not in seen:
                selected_alleles.append(a)
                seen.add(a)

with col_l:
    selected_lengths: list[int] = st.multiselect(
        "Peptide Lengths (mer)",
        options=[8, 9, 10, 11],
        default=[9],
        key="iedb_lengths",
    )

# ── Method ────────────────────────────────────────────────────────────────────
st.subheader("3. Prediction Method")

method: str = st.selectbox(
    "Method",
    options=METHODS,
    index=0,
    key="iedb_method",
    help=(
        "'recommended' selects the best available method per allele. "
        "Other options include NetMHCpan (EL/BA), ANN, SMM, and SMMPMBEC."
    ),
)

# ── Run ───────────────────────────────────────────────────────────────────────
if st.button("Run Prediction", type="primary", key="iedb_run"):
    if not sequences:
        st.error("Enter at least one peptide sequence.")
    elif not selected_alleles:
        st.error("Select at least one HLA allele.")
    elif not selected_lengths:
        st.error("Select at least one peptide length.")
    else:
        n_comb = len(sequences) * len(selected_alleles) * len(selected_lengths)
        if n_comb > LONG_JOB_THRESHOLD:
            st.info(
                f"{n_comb:,} sequence–allele–length combinations detected "
                f"(threshold: {LONG_JOB_THRESHOLD:,}). "
                f"Email `{IEDB_EMAIL}` will be included in each request."
            )

        batches = [(a, l) for a in selected_alleles for l in selected_lengths]
        total = len(batches)
        progress = st.progress(0.0, text="Starting…")

        all_frames: list[pd.DataFrame] = []
        batch_errors: list[str] = []

        for idx, (allele, length) in enumerate(batches):
            progress.progress(
                idx / total,
                text=f"Predicting {allele} {length}-mer ({idx + 1}/{total})…",
            )
            from modules.iedb import LONG_JOB_THRESHOLD as _T, call_iedb_mhci
            import time
            use_email = n_comb > _T
            try:
                all_frames.append(call_iedb_mhci(sequences, allele, length, method, use_email))
            except Exception as exc:
                batch_errors.append(f"{allele} / {length}-mer: {exc}")
            if idx < total - 1:
                time.sleep(0.5)

        progress.progress(1.0, text="Complete.")

        for err in batch_errors:
            st.warning(err)

        if all_frames:
            st.session_state["iedb_results"] = pd.concat(all_frames, ignore_index=True)
        elif not batch_errors:
            st.error("No results returned.")

# ── Results ───────────────────────────────────────────────────────────────────
if "iedb_results" in st.session_state:
    result_df = postprocess(st.session_state["iedb_results"])

    display_cols = [
        c for c in ["allele", "peptide", "rank", "ic50", "binding_level"]
        if c in result_df.columns
    ]

    if not display_cols:
        st.warning("Could not identify expected columns in the API response.")
        st.code(str(result_df.columns.tolist()))
    else:
        st.subheader(f"Results — {len(result_df):,} predictions")

        if "binding_level" in result_df.columns:
            m1, m2, m3 = st.columns(3)
            m1.metric("Strong Binders (SB, IC50 < 50 nM)", int((result_df["binding_level"] == "SB").sum()))
            m2.metric("Weak Binders (WB, IC50 50–500 nM)", int((result_df["binding_level"] == "WB").sum()))
            m3.metric("Non-Binders (NB, IC50 ≥ 500 nM)", int((result_df["binding_level"] == "NB").sum()))

        display_df = result_df[display_cols].copy()
        for num_col in ("ic50", "rank"):
            if num_col in display_df.columns:
                display_df[num_col] = display_df[num_col].apply(
                    lambda x: f"{float(x):.2f}" if pd.notna(x) else "—"
                )

        st.dataframe(display_df, use_container_width=True, height=420)
        st.download_button(
            "Download Results CSV",
            data=result_df[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="iedb_mhci_predictions.csv",
            mime="text/csv",
        )
