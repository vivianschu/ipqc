"""Abstract base class and shared utilities for MHC-I prediction backends."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import pandas as pd

# Standard %rank cutoffs (NetMHCpan / EL convention, widely adopted)
_RANK_SB = 0.5
_RANK_WB = 2.0

# IC50 (nM) cutoffs — fallback when %rank is unavailable
_IC50_SB = 50.0
_IC50_WB = 500.0

# Canonical output columns every predictor must return
OUTPUT_COLUMNS = [
    "peptide",
    "allele",
    "score",        # primary tool score (higher = stronger binder; scale varies by tool)
    "rank",         # %rank EL or affinity (lower = stronger binder; NaN if unavailable)
    "ic50",         # binding affinity IC50 in nM (lower = stronger; NaN if unavailable)
    "binding_level",# SB / WB / NB
    "tool",         # predictor name string
    "model_info",   # version / model-file description
]


def binding_level_from_rank(rank: float) -> str:
    if math.isnan(rank):
        return "NB"
    if rank <= _RANK_SB:
        return "SB"
    if rank <= _RANK_WB:
        return "WB"
    return "NB"


def binding_level_from_ic50(ic50: float) -> str:
    if math.isnan(ic50):
        return "NB"
    if ic50 < _IC50_SB:
        return "SB"
    if ic50 < _IC50_WB:
        return "WB"
    return "NB"


def assign_binding_level(rank: float | None, ic50: float | None) -> str:
    """Classify as SB/WB/NB, preferring %rank; fall back to IC50."""
    r = _to_float(rank)
    i = _to_float(ic50)
    if not math.isnan(r):
        return binding_level_from_rank(r)
    if not math.isnan(i):
        return binding_level_from_ic50(i)
    return "NB"


def _to_float(val) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


class BaseMHCIPredictor(ABC):
    """Interface every MHC-I prediction backend must implement."""

    name: str = "Base"
    description: str = ""
    install_hint: str = ""
    predictor_type: str = "binding"  # "binding" or "stability"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True when all required packages / binaries / model files are present."""
        ...

    @classmethod
    def version(cls) -> str:
        """Return a human-readable version string, or 'unknown' if not determinable."""
        return "unknown"

    @classmethod
    def supported_alleles(cls) -> list[str]:
        """Return the list of HLA alleles this tool explicitly supports, or [] if open-ended."""
        return []

    @abstractmethod
    def predict(
        self,
        peptides: list[str],
        alleles: list[str],
        lengths: list[int],
    ) -> pd.DataFrame:
        """Run prediction and return a DataFrame with OUTPUT_COLUMNS.

        Args:
            peptides: Unique peptide sequences (valid amino acids only).
            alleles:  HLA alleles in standard format, e.g. "HLA-A*02:01".
            lengths:  Only peptides whose length is in this set are scored.

        Returns:
            DataFrame with columns matching OUTPUT_COLUMNS.  Rows where the
            tool could not produce a prediction should be omitted (not filled
            with NaN) so that callers can distinguish "no result" from "NB".
        """
        ...

    def _empty_result(self) -> pd.DataFrame:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
