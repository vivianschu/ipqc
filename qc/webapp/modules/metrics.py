"""Core metric computation functions for immunopeptidomics QC.

All functions operate on a DataFrame that has already been normalised by
:func:`mapping.apply_column_mapping` (i.e. it contains ``_peptide``,
``_length``, ``_is_contam``, ``_source``, etc.).

None of these functions mutate their inputs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .mapping import ColumnMapping

# ── Constants ─────────────────────────────────────────────────────────────────

AAS: list[str] = list("ACDEFGHIKLMNPQRSTVWY")

# UniProt human proteome approximate background frequencies
HUMAN_BG: dict[str, float] = {
    "A": 0.0707, "C": 0.0227, "D": 0.0526, "E": 0.0628, "F": 0.0391,
    "G": 0.0695, "H": 0.0228, "I": 0.0591, "K": 0.0577, "L": 0.0988,
    "M": 0.0228, "N": 0.0405, "P": 0.0472, "Q": 0.0397, "R": 0.0553,
    "S": 0.0694, "T": 0.0550, "V": 0.0687, "W": 0.0120, "Y": 0.0293,
}


# ── Per-sample summary ────────────────────────────────────────────────────────

def compute_sample_summary(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """Return a DataFrame with one row per sample: MS/MS, MBR, contaminant counts."""
    rows: list[dict[str, Any]] = []
    for sd in mapping.samples:
        if not sd.match_col or sd.match_col not in df.columns:
            continue
        col = df[sd.match_col]
        msms = int((col == "MS/MS").sum())
        mbr = int((col == "MBR").sum())
        total = msms + mbr
        contam = (
            int(((col != "unmatched") & df["_is_contam"]).sum())
            if "_is_contam" in df.columns
            else 0
        )
        rows.append({
            "Sample": sd.name,
            "MS/MS": msms,
            "MBR": mbr,
            "Total Detected": total,
            "MBR Rate": f"{mbr / total * 100:.1f}%" if total > 0 else "—",
            "MBR Rate %": mbr / total * 100 if total > 0 else 0.0,
            "Contaminants": contam,
            "Contam Rate": f"{contam / total * 100:.1f}%" if total > 0 else "—",
            "Contam Rate %": contam / total * 100 if total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def compute_dataset_stats(
    df: pd.DataFrame, mapping: ColumnMapping
) -> list[tuple[str, str]]:
    """Return dataset-level summary statistics as (label, value) pairs."""
    total = len(df)
    canonical = int((~df["_is_contam"]).sum()) if "_is_contam" in df.columns else total

    stats: list[tuple[str, str]] = [
        ("Total unique peptides", f"{total:,}"),
        ("Canonical (non-contaminant)", f"{canonical:,}"),
    ]

    if "_length" in df.columns:
        pep_8_11 = int(df["_length"].between(8, 11).sum())
        pep_13_25 = int(df["_length"].between(13, 25).sum())
        stats += [
            (
                "MHC-I length range (8–11 aa)",
                f"{pep_8_11:,} ({pep_8_11 / total * 100:.1f}%)" if total else "0",
            ),
            (
                "MHC-II length range (13–25 aa)",
                f"{pep_13_25:,} ({pep_13_25 / total * 100:.1f}%)" if total else "0",
            ),
        ]

    stats.append(("Number of samples", str(len(mapping.samples))))
    return stats


# ── Contaminants ──────────────────────────────────────────────────────────────

def compute_contaminant_summary(
    df: pd.DataFrame, mapping: ColumnMapping
) -> pd.DataFrame:
    """Return per-sample contaminant detection rate."""
    if "_is_contam" not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sd in mapping.samples:
        if not sd.match_col or sd.match_col not in df.columns:
            continue
        col = df[sd.match_col]
        total_det = int((col != "unmatched").sum())
        n_contam = int(((col != "unmatched") & df["_is_contam"]).sum())
        pct = n_contam / total_det * 100 if total_det > 0 else 0.0
        rows.append({
            "Sample": sd.name,
            "Contaminant Peptides": n_contam,
            "Total Detected": total_det,
            "Contam %": pct,
        })
    return pd.DataFrame(rows).sort_values("Contam %", ascending=True)


def compute_contaminant_proteins(
    df: pd.DataFrame, mapping: ColumnMapping
) -> pd.DataFrame:
    """Return top 20 contaminant proteins ranked by peptide count."""
    if "_is_contam" not in df.columns or "_protein" not in df.columns:
        return pd.DataFrame()

    contam_df = df[df["_is_contam"]].copy()
    if contam_df.empty:
        return pd.DataFrame()

    group_cols = ["_protein"]
    rename: dict[str, str] = {"_protein": "Protein"}

    if mapping.entry_name_col and mapping.entry_name_col in df.columns:
        group_cols.append(mapping.entry_name_col)
        rename[mapping.entry_name_col] = "Entry Name"
    if mapping.protein_desc_col and mapping.protein_desc_col in df.columns:
        group_cols.append(mapping.protein_desc_col)
        rename[mapping.protein_desc_col] = "Description"

    return (
        contam_df.groupby(group_cols)
        .size()
        .reset_index(name="Peptide Count")
        .sort_values("Peptide Count", ascending=False)
        .head(20)
        .rename(columns=rename)
    )


# ── Peptide overlap ───────────────────────────────────────────────────────────

def compute_overlap(
    df: pd.DataFrame, mapping: ColumnMapping
) -> tuple[np.ndarray, np.ndarray, dict[str, set[str]]]:
    """Compute Jaccard similarity and shared peptide counts between all sample pairs.

    Returns
    -------
    jaccard : ndarray, shape (n_samples, n_samples)
    shared  : ndarray, shape (n_samples, n_samples), dtype int
    detected_sets : dict mapping sample name → set of detected peptide strings
    """
    sample_names = [sd.name for sd in mapping.samples]
    detected_sets: dict[str, set[str]] = {}
    for sd in mapping.samples:
        if sd.match_col and sd.match_col in df.columns:
            mask = df[sd.match_col] != "unmatched"
            detected_sets[sd.name] = set(df.loc[mask, "_peptide"].dropna())
        else:
            detected_sets[sd.name] = set()

    n = len(sample_names)
    jaccard = np.zeros((n, n))
    shared = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            a = detected_sets[sample_names[i]]
            b = detected_sets[sample_names[j]]
            union = len(a | b)
            inter = len(a & b)
            jaccard[i, j] = inter / union if union > 0 else 0
            shared[i, j] = inter

    return jaccard, shared, detected_sets


# ── Amino acid composition ────────────────────────────────────────────────────

def compute_aa_composition(
    df: pd.DataFrame, mer_length: int = 9
) -> tuple[np.ndarray, pd.Series, list[str]]:
    """Compute positional AA frequency and overall AA frequency.

    Parameters
    ----------
    df          : normalised DataFrame with ``_length`` and ``_peptide`` columns
    mer_length  : length of peptides to use for the positional matrix (default 9 for MHC-I)

    Returns
    -------
    pos_freq   : ndarray shape (mer_length, 20) — positional frequency matrix
    all_aa_freq: Series over AAS — overall frequency in canonical detected peptides
    mers       : list of peptide strings used in pos_freq
    """
    mers = df.loc[df["_length"] == mer_length, "_peptide"].dropna().tolist()

    pos_aa = np.zeros((mer_length, 20))
    for pep in mers:
        for pos, aa in enumerate(pep[:mer_length]):
            if aa in AAS:
                pos_aa[pos, AAS.index(aa)] += 1

    row_sums = pos_aa.sum(axis=1, keepdims=True)
    pos_freq = np.divide(pos_aa, row_sums, out=np.zeros_like(pos_aa), where=row_sums > 0)

    if "_is_contam" in df.columns:
        canonical = df[~df["_is_contam"]]["_peptide"].dropna()
    else:
        canonical = df["_peptide"].dropna()

    all_aa_counts = pd.Series(list("".join(canonical))).value_counts()
    all_aa_counts = all_aa_counts.reindex(AAS, fill_value=0)
    total = all_aa_counts.sum()
    all_aa_freq = all_aa_counts / total if total > 0 else all_aa_counts * 0.0

    return pos_freq, all_aa_freq, mers


# ── Charge distribution ───────────────────────────────────────────────────────

def parse_charges(val: object) -> list[int]:
    """Parse a comma-separated charge string like '2,3' into a list of ints."""
    return [int(c.strip()) for c in str(val).split(",") if c.strip().isdigit()]


def compute_charge_distribution(df: pd.DataFrame, charge_col: str) -> pd.Series:
    """Return a value-counts Series over all charge states in the dataset."""
    all_charges: list[int] = []
    for raw in df[charge_col].dropna():
        all_charges.extend(parse_charges(raw))
    return pd.Series(all_charges).value_counts().sort_index()


# ── PCA / clustering ──────────────────────────────────────────────────────────

def compute_pca(df: pd.DataFrame, mapping: ColumnMapping) -> dict[str, Any] | None:
    """Compute PCA and sample Pearson correlation from log₂ intensity data.

    Returns ``None`` if fewer than 2 samples have intensity columns or if
    there are too few peptides after filtering.

    The returned dict contains:
    ``samples``, ``coords``, ``var_exp``, ``n_comp``,
    ``corr_ordered``, ``samples_ordered``.
    """
    valid_pairs = [
        (sd.name, sd.intensity_col)
        for sd in mapping.samples
        if sd.intensity_col and sd.intensity_col in df.columns
    ]
    if len(valid_pairs) < 2:
        return None

    sample_names = [p[0] for p in valid_pairs]
    intensity_cols = [p[1] for p in valid_pairs]

    int_mat = df[intensity_cols].copy().replace(0, np.nan)
    int_mat.columns = sample_names
    log2_mat = np.log2(int_mat.where(int_mat > 0))

    min_detect = min(3, len(sample_names))
    n_detected = (int_mat > 0).sum(axis=1)
    log2_filt = log2_mat[n_detected >= min_detect]

    if log2_filt.shape[0] < 2:
        return None

    X = log2_filt.values.T  # n_samples × n_peptides
    col_mins = np.nanmin(X, axis=0)
    nan_mask = np.isnan(X)
    X_imp = X.copy()
    X_imp[nan_mask] = np.take(col_mins, np.where(nan_mask)[1])

    n_comp = min(5, X_imp.shape[0] - 1)
    if n_comp < 1:
        return None

    X_scaled = StandardScaler().fit_transform(X_imp)
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    corr_mat = np.corrcoef(X_imp)
    if len(sample_names) >= 2:
        order = leaves_list(linkage(X_imp, method="ward", metric="euclidean"))
    else:
        order = list(range(len(sample_names)))

    corr_ordered = corr_mat[np.ix_(order, order)]
    samples_ordered = [sample_names[i] for i in order]

    return {
        "samples": sample_names,
        "coords": coords,
        "var_exp": var_exp,
        "n_comp": n_comp,
        "corr_ordered": corr_ordered,
        "samples_ordered": samples_ordered,
    }
