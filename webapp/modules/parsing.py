"""File parsing utilities for FragPipe peptide tables."""
from __future__ import annotations

import io

import pandas as pd


def detect_delimiter(raw: bytes) -> str:
    """Detect TSV vs CSV from the first line of raw bytes."""
    try:
        first = raw.split(b"\n")[0].decode("utf-8", errors="replace")
    except Exception:
        return "\t"
    return "\t" if first.count("\t") >= first.count(",") else ","


def load_table(raw: bytes, delimiter: str) -> pd.DataFrame:
    """Parse raw bytes into a DataFrame using the given delimiter."""
    return pd.read_csv(io.BytesIO(raw), sep=delimiter, low_memory=False)
