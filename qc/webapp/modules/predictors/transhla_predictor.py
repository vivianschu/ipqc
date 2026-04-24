"""TransHLA MHC-I binding predictor.

TransHLA uses a transformer architecture with ESM protein language model
embeddings to predict HLA-peptide binding.

Install:
    pip install TransHLA

GitHub: https://github.com/SkylerLinn/TransHLA
Paper: TransHLA: a transformer-based model for HLA-I binding prediction
"""
from __future__ import annotations

import math
from itertools import product as iterproduct

import pandas as pd

from .base import BaseMHCIPredictor, assign_binding_level

_MODEL_INFO = "TransHLA (HLA-I binding)"


class TransHLAPredictor(BaseMHCIPredictor):
    name = "TransHLA"
    description = (
        "TransHLA — transformer + ESM-based pan-allele HLA-I binding predictor. "
        "Returns a binding probability score (0–1)."
    )
    install_hint = "pip install TransHLA"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import TransHLA  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def version(cls) -> str:
        try:
            import TransHLA
            return getattr(TransHLA, "__version__", "unknown")
        except Exception:
            return "unknown"

    def predict(
        self,
        peptides: list[str],
        alleles: list[str],
        lengths: list[int],
    ) -> pd.DataFrame:
        # TransHLA import — the exact public API varies by release.
        # This implementation targets the TransHLA 1.x PyPI package.
        try:
            from TransHLA import TransHLA_I
        except ImportError:
            try:
                from transhla import TransHLA_I  # alternate casing
            except ImportError as exc:
                raise RuntimeError(
                    "TransHLA package found but TransHLA_I class not importable. "
                    f"Check your TransHLA installation: {exc}"
                ) from exc

        target_lengths = set(lengths)
        filtered = [p for p in peptides if len(p) in target_lengths]
        if not filtered or not alleles:
            return self._empty_result()

        try:
            model = TransHLA_I()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise TransHLA_I model: {exc}") from exc

        all_rows: list[dict] = []
        for allele in alleles:
            try:
                # Most TransHLA APIs accept a list of peptides + a single allele string
                scores = model.predict(peptides=filtered, allele=allele)
            except TypeError:
                # Alternate signature: predict(sequences, allele)
                scores = model.predict(filtered, allele)
            except Exception as exc:
                raise RuntimeError(
                    f"TransHLA prediction failed for {allele}: {exc}"
                ) from exc

            # scores should be a list/array of floats aligned with filtered
            for pep, score_val in zip(filtered, scores):
                score = _safe_float(score_val)
                all_rows.append({
                    "peptide": pep,
                    "allele": allele,
                    "score": score,
                    "rank": float("nan"),
                    "ic50": float("nan"),
                    "binding_level": _transhla_level(score),
                    "tool": self.name,
                    "model_info": _MODEL_INFO,
                })

        return pd.DataFrame(all_rows) if all_rows else self._empty_result()


def _transhla_level(score: float) -> str:
    """Map TransHLA binding probability to SB/WB/NB.

    Thresholds are based on common usage: score ≥ 0.9 → SB, ≥ 0.5 → WB.
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
