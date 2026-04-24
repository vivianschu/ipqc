"""MHC Class I Binding Prediction — IEDB Tools REST API."""
from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
import streamlit as st

IEDB_MHCI_URL = "https://tools.iedb.org/tools_api/mhci/"
IEDB_EMAIL = "unechoed@gmail.com"

# Submit email when estimated combinations exceed this threshold (rough 10-min proxy)
_LONG_JOB_THRESHOLD = 500

COMMON_ALLELES = [
    "HLA-A*01:01", "HLA-A*02:01", "HLA-A*03:01", "HLA-A*11:01",
    "HLA-A*24:02", "HLA-A*26:01", "HLA-B*07:02", "HLA-B*08:01",
    "HLA-B*15:01", "HLA-B*27:05", "HLA-B*35:01", "HLA-B*40:01",
    "HLA-B*44:02", "HLA-B*57:01", "HLA-C*07:01", "HLA-C*07:02",
]

METHODS = [
    "recommended",
    "netmhcpan_el",
    "netmhcpan_ba",
    "ann",
    "smmpmbec",
    "smm",
    "comblib_sidney2008",
]


def _call_iedb(
    sequences: list[str],
    allele: str,
    length: int,
    method: str,
    include_email: bool,
) -> pd.DataFrame:
    """POST one prediction job to the IEDB MHC-I API and return parsed results."""
    fasta = "\n".join(f">seq{i + 1}\n{seq}" for i, seq in enumerate(sequences))
    data: dict[str, str] = {
        "method": method,
        "sequence_text": fasta,
        "allele": allele,
        "length": str(length),
    }
    if include_email:
        data["email_address"] = IEDB_EMAIL

    resp = requests.post(IEDB_MHCI_URL, data=data, timeout=600)
    resp.raise_for_status()

    text = resp.text.strip()
    if not text or text.lower().startswith("error"):
        raise ValueError(text[:300] or "Empty response from IEDB API")

    return pd.read_csv(StringIO(text), sep="\t")


def _binding_level(ic50: float) -> str:
    if ic50 < 50:
        return "SB"
    if ic50 < 500:
        return "WB"
    return "NB"


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
        # deduplicate while preserving order
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
        n_combinations = len(sequences) * len(selected_alleles) * len(selected_lengths)
        use_email = n_combinations > _LONG_JOB_THRESHOLD

        if use_email:
            st.info(
                f"{n_combinations:,} sequence–allele–length combinations detected "
                f"(threshold: {_LONG_JOB_THRESHOLD:,}). "
                f"Email `{IEDB_EMAIL}` will be included in each request."
            )

        batches = [
            (allele, length)
            for allele in selected_alleles
            for length in selected_lengths
        ]

        all_frames: list[pd.DataFrame] = []
        batch_errors: list[str] = []
        total = len(batches)
        progress = st.progress(0.0, text="Starting…")

        for idx, (allele, length) in enumerate(batches):
            progress.progress(
                idx / total,
                text=f"Predicting {allele} {length}-mer ({idx + 1}/{total})…",
            )
            try:
                df_batch = _call_iedb(sequences, allele, length, method, use_email)
                all_frames.append(df_batch)
            except Exception as exc:
                batch_errors.append(f"{allele} / {length}-mer: {exc}")

            # Wait for each job to finish before submitting the next
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
    raw_df: pd.DataFrame = st.session_state["iedb_results"]

    # Normalise column names across IEDB API versions
    rename: dict[str, str] = {}
    for src, dst in [
        ("percentile_rank", "rank"),
        ("ann_ic50", "ic50"),      # fallback; real ic50 takes priority
        ("ic50", "ic50"),
    ]:
        if src in raw_df.columns and dst not in rename.values():
            rename[src] = dst

    result_df = raw_df.rename(columns=rename)

    if "ic50" in result_df.columns:
        result_df["binding_level"] = result_df["ic50"].apply(
            lambda x: _binding_level(float(x)) if pd.notna(x) else "NB"
        )

    display_cols = [
        c for c in ["allele", "peptide", "rank", "ic50", "binding_level"]
        if c in result_df.columns
    ]

    if not display_cols:
        st.warning("Could not identify expected columns in the API response.")
        st.code(str(raw_df.columns.tolist()))
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
