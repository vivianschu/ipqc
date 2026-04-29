"""MHCflurry 2.x MHC-I binding/presentation predictor.

Install:
    pip install mhcflurry
    mhcflurry-downloads fetch          # downloads ~500 MB of model weights
"""
from __future__ import annotations

import math

import pandas as pd

from .base import BaseMHCIPredictor, _to_float, assign_binding_level


class MHCflurryPredictor(BaseMHCIPredictor):
    name = "MHCflurry"
    description = (
        "Pan-allele MHC-I affinity and presentation predictor.  "
        "Outputs IC50 (nM), affinity %rank, processing score, and presentation score."
    )
    install_hint = (
        "pip install mhcflurry\n"
        "mhcflurry-downloads fetch   # ~500 MB model weights"
    )

    @classmethod
    def is_available(cls) -> bool:
        try:
            from mhcflurry import Class1PresentationPredictor  # noqa: F401
            p = Class1PresentationPredictor.load()
            return p is not None
        except Exception:
            return False

    @classmethod
    def version(cls) -> str:
        try:
            import mhcflurry
            return getattr(mhcflurry, "__version__", "unknown")
        except Exception:
            return "unknown"

    @classmethod
    def supported_alleles(cls) -> list[str]:
        try:
            from mhcflurry import Class1PresentationPredictor
            p = Class1PresentationPredictor.load()
            return sorted(p.supported_alleles)
        except Exception:
            return []

    def predict(
        self,
        peptides: list[str],
        alleles: list[str],
        lengths: list[int],
    ) -> pd.DataFrame:
        from mhcflurry import Class1PresentationPredictor

        target_lengths = set(lengths)
        filtered = [p for p in peptides if len(p) in target_lengths]
        if not filtered or not alleles:
            return self._empty_result()

        predictor = Class1PresentationPredictor.load()

        # Validate alleles against what MHCflurry knows
        supported = set(predictor.supported_alleles)
        bad_alleles = [a for a in alleles if a not in supported]
        if bad_alleles:
            raise ValueError(
                f"MHCflurry does not recognise allele(s): {bad_alleles}.  "
                "Check spelling (e.g. HLA-A*02:01) or run "
                "Class1PresentationPredictor.load().supported_alleles for the full list."
            )

        # Class1PresentationPredictor treats `alleles` as a genotype (≤6 alleles,
        # returns best prediction per peptide).  Call once per allele for per-allele results.
        model_info = f"MHCflurry {self.version()}"
        rows = []
        for allele in alleles:
            try:
                raw = predictor.predict(
                    peptides=filtered,
                    alleles=[allele],
                    include_affinity_percentile=True,
                )
            except Exception as exc:
                raise RuntimeError(f"MHCflurry prediction call failed: {exc}") from exc

            for _, row in raw.iterrows():
                ic50 = _to_float(row.get("affinity"))
                aff_rank = _to_float(row.get("affinity_percentile"))
                pres_rank = _to_float(row.get("presentation_percentile"))
                pres_score = _to_float(row.get("presentation_score"))

                best_rank = pres_rank if not math.isnan(pres_rank) else aff_rank
                score = pres_score if not math.isnan(pres_score) else (
                    1.0 - aff_rank / 100.0 if not math.isnan(aff_rank) else float("nan")
                )

                rows.append({
                    "peptide": row.get("peptide", ""),
                    "allele": allele,
                    "score": score,
                    "rank": best_rank,
                    "ic50": ic50,
                    "binding_level": assign_binding_level(best_rank, ic50),
                    "tool": self.name,
                    "model_info": model_info,
                })

        return pd.DataFrame(rows)
