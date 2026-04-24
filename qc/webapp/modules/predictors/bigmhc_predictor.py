"""BigMHC MHC-I eluted ligand and immunogenicity predictor.

Install:
    pip install bigmhc

GitHub: https://github.com/KarchinLab/bigmhc
"""
from __future__ import annotations

import math
from itertools import product as iterproduct

import pandas as pd

from .base import BaseMHCIPredictor, assign_binding_level

_MODEL_INFO = "BigMHC (eluted ligand)"


class BigMHCPredictor(BaseMHCIPredictor):
    name = "BigMHC"
    description = (
        "BigMHC — transformer-based pan-allele MHC-I eluted ligand and "
        "immunogenicity predictor (Karchin Lab, 2023)."
    )
    install_hint = "pip install bigmhc"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import bigmhc  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def version(cls) -> str:
        try:
            import bigmhc
            return getattr(bigmhc, "__version__", "unknown")
        except Exception:
            return "unknown"

    def predict(
        self,
        peptides: list[str],
        alleles: list[str],
        lengths: list[int],
    ) -> pd.DataFrame:
        import bigmhc

        target_lengths = set(lengths)
        filtered = [p for p in peptides if len(p) in target_lengths]
        if not filtered or not alleles:
            return self._empty_result()

        # BigMHC expects a DataFrame with columns 'pep' and 'mhc'
        pairs = list(iterproduct(filtered, alleles))
        input_df = pd.DataFrame({"pep": [p for p, _ in pairs], "mhc": [a for _, a in pairs]})

        try:
            # bigmhc.predict returns the input_df with prediction columns added.
            # Column name is 'EL' for eluted-ligand mode.
            result_df = bigmhc.predict(
                pep_df=input_df,
                mhc_col="mhc",
                pep_col="pep",
                model="el",       # eluted ligand model
                ba=False,         # no binding affinity (EL only)
            )
        except TypeError:
            # Older API: positional args
            result_df = bigmhc.predict(input_df, "mhc", "pep", "el")
        except Exception as exc:
            raise RuntimeError(f"BigMHC prediction failed: {exc}") from exc

        rows = []
        for _, row in result_df.iterrows():
            # Score column may be 'EL', 'EL_pred', or 'pred' depending on version
            score_col = next(
                (c for c in ["EL", "EL_pred", "pred", "score"] if c in row.index), None
            )
            score = _safe_float(row[score_col]) if score_col else float("nan")

            # BigMHC does not natively output %rank or IC50
            rows.append({
                "peptide": row.get("pep", ""),
                "allele": row.get("mhc", ""),
                "score": score,
                "rank": float("nan"),
                "ic50": float("nan"),
                # BigMHC EL score > 0.5 is used as a soft SB/WB threshold
                "binding_level": _bigmhc_level(score),
                "tool": self.name,
                "model_info": _MODEL_INFO,
            })

        return pd.DataFrame(rows) if rows else self._empty_result()


def _bigmhc_level(score: float) -> str:
    """Classify BigMHC EL score (0–1) as SB / WB / NB.

    Thresholds follow the authors' suggested cutoffs (score ≥ 0.9 → SB,
    ≥ 0.5 → WB).  These are approximate and may be tuned by the user.
    """
    if math.isnan(score):
        return "NB"
    if score >= 0.9:
        return "SB"
    if score >= 0.5:
        return "WB"
    return "NB"


def _safe_float(val) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")
