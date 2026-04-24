"""Column mapping types and utilities for FragPipe immunopeptidomics tables.

The central types here are :class:`SampleDef` and :class:`ColumnMapping`.
Once the user fills these in through the UI they are passed to the metric
and chart layers, which never reference raw column names directly.

Example mapping object::

    mapping = ColumnMapping(
        peptide_col="Peptide Sequence",
        protein_col="Protein",
        gene_col="Gene",
        length_col=None,           # None → derived from peptide string length
        charge_col="Charges",
        entry_name_col="Entry Name",
        protein_desc_col="Protein Description",
        samples=[
            SampleDef("PBMC_1",
                      match_col="PBMC_1 Match Type",
                      spectral_col="PBMC_1 Spectral Count",
                      intensity_col="PBMC_1 Intensity"),
            SampleDef("PBMC_2",
                      match_col="PBMC_2 Match Type",
                      spectral_col="PBMC_2 Spectral Count",
                      intensity_col="PBMC_2 Intensity"),
        ],
    )
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SampleDef:
    """One sample and its associated FragPipe column names."""

    name: str
    match_col: str | None = None
    spectral_col: str | None = None
    intensity_col: str | None = None


@dataclass
class ColumnMapping:
    """Full mapping from user-chosen column names to standardised internals.

    Fields set to ``None`` are treated as absent; the analysis layer
    skips or degrades gracefully for every optional field.
    """

    peptide_col: str
    protein_col: str | None = None
    gene_col: str | None = None
    # None means derive from peptide string length at apply time
    length_col: str | None = None
    charge_col: str | None = None
    entry_name_col: str | None = None
    protein_desc_col: str | None = None
    samples: list[SampleDef] = field(default_factory=list)


# ── Auto-detection ────────────────────────────────────────────────────────────

def detect_sample_columns(df: pd.DataFrame) -> list[SampleDef]:
    """Infer sample definitions from the FragPipe convention.

    Looks for columns ending in " Match Type" and attempts to pair each one
    with matching Spectral Count and Intensity columns.
    """
    match_cols = [c for c in df.columns if c.endswith(" Match Type")]
    samples: list[SampleDef] = []
    for mc in match_cols:
        name = mc[: -len(" Match Type")]
        sc = f"{name} Spectral Count" if f"{name} Spectral Count" in df.columns else None
        ic = f"{name} Intensity" if f"{name} Intensity" in df.columns else None
        samples.append(SampleDef(name=name, match_col=mc, spectral_col=sc, intensity_col=ic))
    return samples


def suggest_column(candidates: list[str], df_columns: list[str]) -> str | None:
    """Return the first candidate that appears (case-insensitively) in df_columns."""
    lower_cols = {c.lower(): c for c in df_columns}
    for cand in candidates:
        hit = lower_cols.get(cand.lower())
        if hit is not None:
            return hit
    return None


# ── Validation ────────────────────────────────────────────────────────────────

def validate_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> list[str]:
    """Return a list of human-readable error strings.

    An empty list means the mapping is fully valid.
    """
    errors: list[str] = []
    cols = set(df.columns)

    if mapping.peptide_col not in cols:
        errors.append(f"Peptide column '{mapping.peptide_col}' not found in the uploaded table.")

    for opt_col, label in [
        (mapping.protein_col, "Protein"),
        (mapping.gene_col, "Gene"),
        (mapping.charge_col, "Charge"),
        (mapping.entry_name_col, "Entry Name"),
        (mapping.protein_desc_col, "Protein Description"),
    ]:
        if opt_col is not None and opt_col not in cols:
            errors.append(f"{label} column '{opt_col}' not found in the uploaded table.")

    if not mapping.samples:
        errors.append("No samples defined. Add at least one sample in the Sample Mapping section.")

    for sd in mapping.samples:
        if not sd.name.strip():
            errors.append("One or more samples have an empty name.")
        if sd.match_col and sd.match_col not in cols:
            errors.append(f"Sample '{sd.name}': Match Type column '{sd.match_col}' not found.")
        if sd.spectral_col and sd.spectral_col not in cols:
            errors.append(f"Sample '{sd.name}': Spectral Count column '{sd.spectral_col}' not found.")
        if sd.intensity_col and sd.intensity_col not in cols:
            errors.append(f"Sample '{sd.name}': Intensity column '{sd.intensity_col}' not found.")

    return errors


# ── Column normalisation ──────────────────────────────────────────────────────

def apply_column_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """Return a copy of *df* with standardised derived columns added.

    Added columns
    -------------
    ``_peptide``    Peptide sequence as string.
    ``_length``     Peptide length (int); derived from sequence if no dedicated column.
    ``_is_contam``  bool — True when protein annotation contains ``Cont_``.
    ``_source``     Protein source category string.
    ``_protein``    Alias for protein column (may be all-None if absent).
    ``_gene``       Alias for gene column (may be all-None if absent).
    """
    out = df.copy()
    out["_peptide"] = out[mapping.peptide_col].astype(str)

    if mapping.length_col and mapping.length_col in df.columns:
        out["_length"] = pd.to_numeric(out[mapping.length_col], errors="coerce")
    else:
        out["_length"] = out["_peptide"].str.len()

    if mapping.protein_col and mapping.protein_col in df.columns:
        out["_protein"] = out[mapping.protein_col].astype(str)
        out["_is_contam"] = out["_protein"].str.contains("Cont_", regex=False, na=False)
        out["_source"] = out["_protein"].apply(_classify_source)
    else:
        out["_protein"] = pd.NA
        out["_is_contam"] = False
        out["_source"] = "Unknown"

    if mapping.gene_col and mapping.gene_col in df.columns:
        out["_gene"] = out[mapping.gene_col]
    else:
        out["_gene"] = pd.NA

    return out


def _classify_source(protein: str) -> str:
    if re.search(r"Cont_", str(protein)):
        return "Contaminant"
    if str(protein).startswith("sp|"):
        return "SwissProt (sp)"
    if str(protein).startswith("tr|"):
        return "TrEMBL (tr)"
    return "Other"
