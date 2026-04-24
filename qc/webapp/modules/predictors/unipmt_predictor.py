"""UniPMT MHC-I presentation predictor.

UniPMT (Unified Peptide-MHC-I Transformer) is a pan-allele MHC-I
presentation predictor integrating peptide and MHC pseudo-sequence features.

Install (from GitHub — not currently on PyPI):
    git clone https://github.com/IEDB-group/UniPMT
    cd UniPMT && pip install -e .

Alternatively, if published to PyPI:
    pip install unipmt

GitHub: https://github.com/IEDB-group/UniPMT
Paper: UniPMT: A Unified MHC-I Presentation Predictor
"""
from __future__ import annotations

import math
from itertools import product as iterproduct

import pandas as pd

from .base import BaseMHCIPredictor, assign_binding_level

_MODEL_INFO = "UniPMT (MHC-I presentation)"


class UniPMTPredictor(BaseMHCIPredictor):
    name = "UniPMT"
    description = (
        "UniPMT — unified transformer-based pan-allele MHC-I presentation predictor. "
        "Returns a presentation probability score (0–1)."
    )
    install_hint = (
        "UniPMT is not yet on PyPI.  Install from GitHub:\n"
        "  git clone https://github.com/IEDB-group/UniPMT\n"
        "  cd UniPMT && pip install -e ."
    )

    @classmethod
    def is_available(cls) -> bool:
        try:
            import unipmt  # noqa: F401
            return True
        except ImportError:
            try:
                import UniPMT  # noqa: F401
                return True
            except ImportError:
                return False

    @classmethod
    def version(cls) -> str:
        for mod_name in ("unipmt", "UniPMT"):
            try:
                import importlib
                m = importlib.import_module(mod_name)
                return getattr(m, "__version__", "unknown")
            except Exception:
                continue
        return "unknown"

    def predict(
        self,
        peptides: list[str],
        alleles: list[str],
        lengths: list[int],
    ) -> pd.DataFrame:
        # Try both common import styles
        try:
            import unipmt as _mod
        except ImportError:
            import UniPMT as _mod  # type: ignore[no-redef]

        target_lengths = set(lengths)
        filtered = [p for p in peptides if len(p) in target_lengths]
        if not filtered or not alleles:
            return self._empty_result()

        pairs = list(iterproduct(filtered, alleles))
        pep_list = [p for p, _ in pairs]
        allele_list = [a for _, a in pairs]

        try:
            # Attempt the most common API pattern seen in transformer prediction tools
            result = _mod.predict(peptides=pep_list, alleles=allele_list)
        except TypeError:
            result = _mod.predict(pep_list, allele_list)
        except AttributeError:
            # Module may expose a class instead
            try:
                predictor = _mod.UniPMT()
                result = predictor.predict(pep_list, allele_list)
            except Exception as exc:
                raise RuntimeError(f"Could not find a usable UniPMT predict API: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"UniPMT prediction failed: {exc}") from exc

        # result is expected to be a list/array of scores or a DataFrame
        rows: list[dict] = []
        if isinstance(result, pd.DataFrame):
            for _, row in result.iterrows():
                score_col = next(
                    (c for c in ["score", "pred", "presentation_score", "prob"] if c in row.index),
                    None,
                )
                score = _safe_float(row[score_col]) if score_col else float("nan")
                pep = row.get("peptide", row.get("pep", ""))
                allele = row.get("allele", row.get("mhc", ""))
                rows.append({
                    "peptide": pep,
                    "allele": allele,
                    "score": score,
                    "rank": float("nan"),
                    "ic50": float("nan"),
                    "binding_level": _score_to_level(score),
                    "tool": self.name,
                    "model_info": _MODEL_INFO,
                })
        else:
            # Assume list of scalars aligned with pairs
            scores = list(result)
            for (pep, allele), score_val in zip(pairs, scores):
                score = _safe_float(score_val)
                rows.append({
                    "peptide": pep,
                    "allele": allele,
                    "score": score,
                    "rank": float("nan"),
                    "ic50": float("nan"),
                    "binding_level": _score_to_level(score),
                    "tool": self.name,
                    "model_info": _MODEL_INFO,
                })

        return pd.DataFrame(rows) if rows else self._empty_result()


def _score_to_level(score: float) -> str:
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
