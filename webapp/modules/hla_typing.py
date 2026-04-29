"""HLA allele inference from immunopeptidomics ligand motifs."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# Class I: 7–14 aa; Class II: 12–25 aa
CLASS_I_LENGTHS = set(range(7, 15))
CLASS_II_LENGTHS = set(range(12, 26))

# Curated panel of common class I alleles across HLA-A, -B, -C
CANDIDATE_ALLELES_CLASS_I: list[str] = [
    # HLA-A
    "HLA-A*01:01", "HLA-A*02:01", "HLA-A*02:03", "HLA-A*02:06",
    "HLA-A*03:01", "HLA-A*11:01", "HLA-A*23:01", "HLA-A*24:02",
    "HLA-A*26:01", "HLA-A*29:02", "HLA-A*30:01", "HLA-A*30:02",
    "HLA-A*31:01", "HLA-A*32:01", "HLA-A*33:01", "HLA-A*68:01",
    "HLA-A*68:02",
    # HLA-B
    "HLA-B*07:02", "HLA-B*08:01", "HLA-B*13:01", "HLA-B*14:02",
    "HLA-B*15:01", "HLA-B*27:05", "HLA-B*35:01", "HLA-B*37:01",
    "HLA-B*38:01", "HLA-B*39:01", "HLA-B*40:01", "HLA-B*40:02",
    "HLA-B*44:02", "HLA-B*44:03", "HLA-B*46:01", "HLA-B*48:01",
    "HLA-B*51:01", "HLA-B*52:01", "HLA-B*53:01", "HLA-B*57:01",
    "HLA-B*57:03", "HLA-B*58:01",
    # HLA-C
    "HLA-C*01:02", "HLA-C*03:03", "HLA-C*03:04", "HLA-C*04:01",
    "HLA-C*05:01", "HLA-C*06:02", "HLA-C*07:01", "HLA-C*07:02",
    "HLA-C*08:02", "HLA-C*12:03", "HLA-C*14:02", "HLA-C*15:02",
    "HLA-C*16:01",
]


def clean_peptides(
    raw: list[str],
    min_len: int = 8,
    max_len: int = 25,
) -> tuple[list[str], list[str], int]:
    """Validate, uppercase, deduplicate. Returns (valid_unique, issues, n_duplicates)."""
    valid: list[str] = []
    issues: list[str] = []
    seen: set[str] = set()
    n_dupes = 0
    for seq in raw:
        s = seq.strip().upper()
        if not s:
            continue
        bad = [c for c in s if c not in _VALID_AA]
        if bad:
            issues.append(f"'{seq}' — invalid character(s): {', '.join(sorted(set(bad)))}")
            continue
        if len(s) < min_len:
            issues.append(f"'{seq}' — too short ({len(s)} aa, min {min_len})")
            continue
        if len(s) > max_len:
            issues.append(f"'{seq}' — too long ({len(s)} aa, max {max_len})")
            continue
        if s in seen:
            n_dupes += 1
            continue
        valid.append(s)
        seen.add(s)
    return valid, issues, n_dupes


def length_distribution(peptides: list[str]) -> dict[int, int]:
    return dict(sorted(Counter(len(p) for p in peptides).items()))


def infer_hla_class(peptides: list[str]) -> dict:
    """Infer likely HLA class from length distribution."""
    dist = length_distribution(peptides)
    total = len(peptides)
    if total == 0:
        return {
            "inferred_class": "unknown",
            "class_i_fraction": 0.0,
            "class_ii_fraction": 0.0,
            "peak_length": 0,
            "note": "No peptides provided.",
        }

    # 7–11 aa strongly enriched in class I immunopeptidomics
    ci_count  = sum(dist.get(l, 0) for l in range(7, 12))
    cii_count = sum(dist.get(l, 0) for l in range(13, 26))
    ci_frac   = ci_count / total
    cii_frac  = cii_count / total
    peak_len  = max(dist, key=dist.get)

    if ci_frac >= 0.6 and peak_len in (7, 8, 9, 10, 11):
        inferred = "I"
        note = (
            f"Strong class I signal: {ci_frac:.0%} of peptides are 7–11 aa "
            f"with peak at {peak_len} aa."
        )
    elif cii_frac >= 0.5:
        inferred = "II"
        note = f"Class II-like distribution: {cii_frac:.0%} of peptides are 13–25 aa."
    elif ci_frac >= 0.3:
        inferred = "I (uncertain)"
        note = (
            f"Partial class I signal ({ci_frac:.0%} in 7–11 aa range). "
            "Consider adding more peptides or manual class override."
        )
    else:
        inferred = "unknown"
        note = "Ambiguous length distribution. Could not reliably infer HLA class."

    return {
        "inferred_class": inferred,
        "class_i_fraction": round(ci_frac, 3),
        "class_ii_fraction": round(cii_frac, 3),
        "peak_length": peak_len,
        "note": note,
    }


def compute_pfm(peptides: list[str], length: int) -> pd.DataFrame:
    """Position frequency matrix (positions × AA) for peptides of a given length.

    Returns a DataFrame suitable for logomaker.Logo().
    """
    seqs = [p for p in peptides if len(p) == length]
    if not seqs:
        return pd.DataFrame()
    aa_idx = {aa: i for i, aa in enumerate(_AA_ORDER)}
    n_aa = len(_AA_ORDER)
    matrix = np.zeros((length, n_aa), dtype=float)
    for seq in seqs:
        for pos, aa in enumerate(seq):
            if aa in aa_idx:
                matrix[pos, aa_idx[aa]] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = np.where(row_sums > 0, matrix / row_sums, 0.0)
    return pd.DataFrame(
        freq,
        index=[f"P{i + 1}" for i in range(length)],
        columns=list(_AA_ORDER),
    )


def rank_alleles(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-allele statistics from a postprocessed prediction DataFrame."""
    if pred_df.empty or "allele" not in pred_df.columns:
        return pd.DataFrame()

    rows = []
    for allele, grp in pred_df.groupby("allele"):
        total = len(grp)
        sb = int((grp["binding_level"] == "SB").sum()) if "binding_level" in grp.columns else 0
        wb = int((grp["binding_level"] == "WB").sum()) if "binding_level" in grp.columns else 0
        explained = sb + wb
        # Extract locus (e.g. "HLA-A" from "HLA-A*02:01")
        parts = str(allele).split("*")
        locus = parts[0] if parts else str(allele)
        rows.append({
            "allele": allele,
            "locus": locus,
            "strong_binders": sb,
            "weak_binders": wb,
            "explained": explained,
            "total_scored": total,
            "explained_fraction": round(explained / total, 4) if total > 0 else 0.0,
            "sb_fraction": round(sb / total, 4) if total > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("sb_fraction", ascending=False).reset_index(drop=True)


def top_alleles_per_locus(ranked_df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Return the top-n alleles per locus sorted by SB fraction."""
    if ranked_df.empty:
        return pd.DataFrame()
    return (
        ranked_df
        .sort_values("sb_fraction", ascending=False)
        .groupby("locus", group_keys=False)
        .head(n)
        .reset_index(drop=True)
        .sort_values("sb_fraction", ascending=False)
    )


def assign_confidence(
    supporting_peptides: int,
    sb_fraction: float,
    explained_fraction: float,
    has_external_typing: bool,
    is_class_i: bool,
) -> str:
    if not is_class_i:
        return "low"
    if has_external_typing and sb_fraction >= 0.05:
        return "high"
    if supporting_peptides >= 100 and sb_fraction >= 0.10:
        return "high" if has_external_typing else "medium"
    if supporting_peptides >= 30 and sb_fraction >= 0.05:
        return "medium"
    return "low"


def build_inference_result(
    sample_id: str,
    hla_class: str,
    input_count: int,
    filtered_count: int,
    length_dist: dict[int, int],
    ranked_alleles: pd.DataFrame,
    external_typing: list[str],
    predictor_name: str,
    is_class_i: bool,
    global_warnings: list[str],
) -> dict:
    """Assemble the structured JSON output defined in the spec."""
    inferred: list[dict] = []
    if not ranked_alleles.empty:
        for _, row in ranked_alleles.iterrows():
            allele = row["allele"]
            sb_frac = float(row["sb_fraction"])
            expl_frac = float(row["explained_fraction"])
            conf = assign_confidence(
                supporting_peptides=int(row["strong_binders"]),
                sb_fraction=sb_frac,
                explained_fraction=expl_frac,
                has_external_typing=bool(external_typing),
                is_class_i=is_class_i,
            )
            locus = row["locus"]
            same_locus_alts = ranked_alleles[
                (ranked_alleles["locus"] == locus) & (ranked_alleles["allele"] != allele)
            ]["allele"].head(3).tolist()

            warnings: list[str] = []
            if sb_frac < 0.05:
                warnings.append("Very few strong binders — allele motif match is weak.")
            if conf == "low":
                warnings.append("Low confidence: provide more peptides or external HLA typing.")

            inferred.append({
                "locus": locus,
                "call": f"{allele}-like",
                "resolution": "allele_family_or_motif_group",
                "confidence": conf,
                "supporting_peptide_count": int(row["strong_binders"]),
                "explained_fraction": expl_frac,
                "top_alternatives": same_locus_alts,
                "warnings": warnings,
            })

    return {
        "sample_id": sample_id,
        "hla_inference_source": "immunopeptidomics_ligand_motif",
        "predictor_used": predictor_name,
        "hla_class": hla_class,
        "input_peptide_count": input_count,
        "filtered_peptide_count": filtered_count,
        "length_distribution": {str(k): v for k, v in length_dist.items()},
        "external_hla_typing_used": bool(external_typing),
        "inferred_alleles": inferred,
        "global_warnings": global_warnings,
    }
