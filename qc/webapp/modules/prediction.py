"""Orchestration helpers for local MHC-I binding/presentation prediction."""
from __future__ import annotations

import math
import re
from typing import Callable

import pandas as pd

from .predictors.base import OUTPUT_COLUMNS, assign_binding_level
from .predictors.registry import ALL_PREDICTORS, get_available_predictors

# Common HLA-A/B/C alleles used as defaults in the UI
COMMON_ALLELES: list[str] = [
    "HLA-A*01:01", "HLA-A*02:01", "HLA-A*03:01", "HLA-A*11:01",
    "HLA-A*24:02", "HLA-A*26:01", "HLA-B*07:02", "HLA-B*08:01",
    "HLA-B*15:01", "HLA-B*27:05", "HLA-B*35:01", "HLA-B*40:01",
    "HLA-B*44:02", "HLA-B*57:01", "HLA-C*07:01", "HLA-C*07:02",
]

SUPPORTED_LENGTHS: list[int] = [8, 9, 10, 11]

# Standard amino acid alphabet (no ambiguous codes)
_VALID_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")

# HLA allele format: HLA-[gene]*group:protein  e.g. HLA-A*02:01
_ALLELE_RE = re.compile(r"^HLA-[A-Z]\*\d{1,3}:\d{2,3}$")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_peptides(sequences: list[str]) -> tuple[list[str], list[str]]:
    """Return (valid_cleaned_sequences, list_of_issue_strings).

    Filters to sequences containing only the 20 standard amino acids.
    Duplicates are preserved (deduplication is the caller's choice).
    """
    valid: list[str] = []
    issues: list[str] = []
    for raw in sequences:
        seq = raw.strip().upper()
        if not seq:
            continue
        bad = sorted({c for c in seq if c not in _VALID_AAS})
        if bad:
            issues.append(
                f"'{seq[:24]}{'…' if len(seq) > 24 else ''}': "
                f"non-standard amino acid(s) {bad} — skipped"
            )
        else:
            valid.append(seq)
    return valid, issues


def validate_allele_format(allele: str) -> str | None:
    """Return an error string if *allele* doesn't match the expected format, else None.

    Accepted format: HLA-[gene]*[group]:[protein]  e.g. HLA-A*02:01
    """
    if not _ALLELE_RE.match(allele.strip()):
        return (
            f"'{allele}': expected format HLA-A*02:01 "
            "(gene*group:protein, e.g. HLA-A*02:01, HLA-B*07:02).  "
            "Mouse H-2 and non-HLA alleles are not supported by all tools."
        )
    return None


# ── Single-predictor run ──────────────────────────────────────────────────────

def run_prediction(
    peptides: list[str],
    alleles: list[str],
    lengths: list[int],
    predictor_name: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Run a single predictor and return (results_df, error_strings).

    Args:
        peptides:       Peptide sequences to predict (already validated).
        alleles:        HLA alleles in standard format (e.g. "HLA-A*02:01").
        lengths:        Only peptides of these lengths are scored.
        predictor_name: Name matching BaseMHCIPredictor.name.

    Returns:
        (DataFrame with OUTPUT_COLUMNS, list of error message strings)
    """
    from .predictors.registry import get_predictor_by_name

    cls = get_predictor_by_name(predictor_name)
    if cls is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), [f"Unknown predictor: '{predictor_name}'"]
    if not cls.is_available():
        hint = cls.install_hint.split("\n")[0]  # first line of install hint
        return pd.DataFrame(columns=OUTPUT_COLUMNS), [
            f"{predictor_name} is not installed or its model files are missing.  "
            f"Install hint: {hint}"
        ]

    predictor = cls()
    try:
        result = predictor.predict(peptides, alleles, lengths)
    except Exception as exc:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), [
            f"{predictor_name} prediction failed: {exc}"
        ]

    return result, []


# ── Multi-predictor run ───────────────────────────────────────────────────────

def run_multi_prediction(
    peptides: list[str],
    alleles: list[str],
    lengths: list[int],
    predictor_names: list[str],
    on_progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """Run several predictors sequentially.

    Args:
        peptides:        Peptide sequences to predict.
        alleles:         HLA alleles.
        lengths:         Peptide length filter.
        predictor_names: Ordered list of predictor names to run.
        on_progress:     Optional callback(tool_name, current_idx, total).

    Returns:
        Dict mapping predictor_name → (results_df, error_strings).
    """
    results: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    total = len(predictor_names)
    for i, name in enumerate(predictor_names):
        if on_progress:
            on_progress(name, i, total)
        df, errors = run_prediction(peptides, alleles, lengths, name)
        results[name] = (df, errors)
    return results


# ── Post-processing ───────────────────────────────────────────────────────────

def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure binding_level is populated and numeric columns are properly typed."""
    if df.empty:
        return df

    df = df.copy()

    if "binding_level" not in df.columns:
        ic50_col = next((c for c in ["ic50"] if c in df.columns), None)
        rank_col = next((c for c in ["rank"] if c in df.columns), None)
        df["binding_level"] = [
            assign_binding_level(
                row[rank_col] if rank_col else None,
                row[ic50_col] if ic50_col else None,
            )
            for _, row in df.iterrows()
        ]

    for col in ["score", "rank", "ic50"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── Binding score ─────────────────────────────────────────────────────────────

def calculate_binding_score(df: pd.DataFrame, n_length_filtered: int) -> float | None:
    """Compute the binding fraction (BF) score.

    For each unique peptide in *df*, find the best binding level across all
    alleles (SB > WB > NB).  The score is the fraction of *n_length_filtered*
    peptides that are SB or WB for at least one allele.

    Returns None when *df* is empty or *n_length_filtered* is zero.
    """
    if df.empty or n_length_filtered == 0 or "binding_level" not in df.columns:
        return None

    def _best_level(levels: pd.Series) -> str:
        vals = set(levels)
        if "SB" in vals:
            return "SB"
        if "WB" in vals:
            return "WB"
        return "NB"

    peptide_best = df.groupby("peptide")["binding_level"].apply(_best_level)
    n_binders = int((peptide_best != "NB").sum())
    return round(n_binders / n_length_filtered, 2)


# ── Utility ───────────────────────────────────────────────────────────────────

def predictor_status_table() -> list[dict]:
    """Return a list of dicts describing all predictors and their status."""
    available_names = {cls.name for cls in get_available_predictors()}
    rows = []
    for cls in ALL_PREDICTORS:
        is_avail = cls.name in available_names
        rows.append({
            "Tool": cls.name,
            "Status": "Ready" if is_avail else "Not installed",
            "Description": cls.description,
            "Install": "" if is_avail else cls.install_hint,
        })
    return rows


def make_consensus_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a (peptide, allele) pivot showing each tool's binding_level.

    Only includes predictors that returned non-empty results.
    """
    frames = []
    for tool, df in results.items():
        if df.empty:
            continue
        keep = df[["peptide", "allele", "binding_level", "rank", "ic50"]].copy()
        keep = keep.rename(columns={
            "binding_level": f"{tool}_class",
            "rank": f"{tool}_rank",
            "ic50": f"{tool}_ic50",
        })
        frames.append(keep.set_index(["peptide", "allele"]))

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")

    merged = merged.reset_index().sort_values(
        [c for c in merged.columns if c.endswith("_rank")][:1] or ["peptide"],
        na_position="last",
    )
    return merged
