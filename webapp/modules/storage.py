"""Run data serialization and deserialization for persistent storage.

Run data is stored under:
  <data_root>/runs/<user_id>/<run_id>/df.parquet
  <data_root>/runs/<user_id>/<run_id>/mapping.json
  <data_root>/runs/<user_id>/<run_id>/ms_df.parquet  (optional)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .mapping import ColumnMapping, SampleDef

DATA_ROOT = Path(__file__).parent.parent / "data"


def run_data_dir(user_id: int, run_id: int) -> Path:
    return DATA_ROOT / "runs" / str(user_id) / str(run_id)


def serialize_run(
    user_id: int,
    run_id: int,
    df: pd.DataFrame,
    mapping: ColumnMapping,
    ms_df: pd.DataFrame | None,
) -> str:
    """Persist run data to disk and return the directory path string."""
    out_dir = run_data_dir(user_id, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_dir / "df.parquet", index=False)

    mapping_dict: dict = {
        "peptide_col": mapping.peptide_col,
        "protein_col": mapping.protein_col,
        "gene_col": mapping.gene_col,
        "length_col": mapping.length_col,
        "charge_col": mapping.charge_col,
        "entry_name_col": mapping.entry_name_col,
        "protein_desc_col": mapping.protein_desc_col,
        "samples": [
            {
                "name": sd.name,
                "match_col": sd.match_col,
                "spectral_col": sd.spectral_col,
                "intensity_col": sd.intensity_col,
            }
            for sd in mapping.samples
        ],
    }
    (out_dir / "mapping.json").write_text(json.dumps(mapping_dict, indent=2))

    if ms_df is not None:
        ms_df.to_parquet(out_dir / "ms_df.parquet", index=False)

    return str(out_dir)


def deserialize_run(
    data_dir: str,
) -> tuple[pd.DataFrame, ColumnMapping, pd.DataFrame | None]:
    """Load persisted run data from *data_dir*. Returns (df, mapping, ms_df|None)."""
    run_dir = Path(data_dir)

    df = pd.read_parquet(run_dir / "df.parquet")

    raw = json.loads((run_dir / "mapping.json").read_text())
    samples = [
        SampleDef(
            name=s["name"],
            match_col=s.get("match_col"),
            spectral_col=s.get("spectral_col"),
            intensity_col=s.get("intensity_col"),
        )
        for s in raw["samples"]
    ]
    mapping = ColumnMapping(
        peptide_col=raw["peptide_col"],
        protein_col=raw.get("protein_col"),
        gene_col=raw.get("gene_col"),
        length_col=raw.get("length_col"),
        charge_col=raw.get("charge_col"),
        entry_name_col=raw.get("entry_name_col"),
        protein_desc_col=raw.get("protein_desc_col"),
        samples=samples,
    )

    ms_path = run_dir / "ms_df.parquet"
    ms_df = pd.read_parquet(ms_path) if ms_path.exists() else None

    return df, mapping, ms_df
