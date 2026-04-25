"""NetMHCpan 4.1 MHC-I predictor via subprocess.

NetMHCpan is not pip-installable.  Download from:
    https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/

After installing, make sure 'netMHCpan' is on your PATH.
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from .base import BaseMHCIPredictor, assign_binding_level

_BINARY = "netMHCpan"
_MODEL_INFO = "NetMHCpan 4.2c"


def _netmhcpan_version() -> str:
    try:
        r = subprocess.run([_BINARY, "-h"], capture_output=True, text=True, timeout=10)
        for line in (r.stdout + r.stderr).splitlines():
            if "version" in line.lower() or "NetMHCpan" in line:
                return line.strip()
        return "NetMHCpan (version unknown)"
    except Exception:
        return "NetMHCpan (version unknown)"


def _allele_to_netmhcpan(allele: str) -> str:
    """Convert 'HLA-A*02:01' → 'HLA-A02:01' (NetMHCpan drops the asterisk)."""
    return allele.replace("*", "")


def _parse_netmhcpan_output(text: str, allele: str) -> list[dict]:
    """Parse NetMHCpan tabular output into a list of row dicts.

    Column positions are detected from the header line so this works with both
    NetMHCpan 4.1 (Aff(nM) before %Rank_BA) and 4.2c (order swapped, no Exp col
    on Darwin).
    """
    # Defaults match 4.2c Darwin_arm64 layout; overridden when header is found.
    col_score_el = 11
    col_rank_el = 12
    col_rank_ba = 14
    col_ic50 = 15

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        parts = line.split()
        if not parts:
            continue

        # Detect column positions from the table header line (starts with "Pos")
        if parts[0] == "Pos":
            try:
                col_score_el = parts.index("Score_EL")
                col_rank_el  = parts.index("%Rank_EL")
                if "%Rank_BA" in parts:
                    col_rank_ba = parts.index("%Rank_BA")
                if "Aff(nM)" in parts:
                    col_ic50 = parts.index("Aff(nM)")
            except ValueError:
                pass
            continue

        if not parts[0].isdigit():
            continue

        try:
            peptide  = parts[2]
            score_el = float(parts[col_score_el])
            rank_el  = float(parts[col_rank_el])
            ic50     = float("nan")
            rank_ba  = float("nan")

            if len(parts) > col_ic50:
                rank_ba = float(parts[col_rank_ba])
                ic50    = float(parts[col_ic50])

            rank = rank_el if not math.isnan(rank_el) else rank_ba

            rows.append({
                "peptide":       peptide,
                "allele":        allele,
                "score":         score_el,
                "rank":          rank,
                "ic50":          ic50,
                "binding_level": assign_binding_level(rank, ic50),
                "tool":          "NetMHCpan",
                "model_info":    _MODEL_INFO,
            })
        except (IndexError, ValueError):
            continue
    return rows


class NetMHCpanPredictor(BaseMHCIPredictor):
    name = "NetMHCpan"
    description = (
        "NetMHCpan 4.2c — state-of-the-art pan-allele MHC-I binding predictor "
        "from DTU Health Tech.  Returns EL score, EL %rank, and (with -BA) IC50/BA %rank."
    )
    install_hint = (
        "Download NetMHCpan 4.2c (not 4.2cstatic) from:\n"
        "  https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/\n"
        "Version 4.2c includes Darwin_arm64 binaries for Apple Silicon.\n"
        "Add the install directory to PATH, then restart the app."
    )

    @classmethod
    def is_available(cls) -> bool:
        if shutil.which(_BINARY) is None:
            return False
        # The wrapper script may be on PATH but lack platform binaries (e.g. Darwin arm64).
        # Do a cheap probe to confirm it actually runs.
        try:
            r = subprocess.run(
                [_BINARY, "-h"], capture_output=True, text=True, timeout=10
            )
            combined = r.stdout + r.stderr
            if "no binaries found" in combined or "no binaries" in combined.lower():
                return False
        except Exception:
            return False
        return True

    @classmethod
    def version(cls) -> str:
        return _netmhcpan_version()

    def predict(
        self,
        peptides: list[str],
        alleles: list[str],
        lengths: list[int],
    ) -> pd.DataFrame:
        target_lengths = set(lengths)
        filtered = [p for p in peptides if len(p) in target_lengths]
        if not filtered or not alleles:
            return self._empty_result()

        all_rows: list[dict] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for length in sorted(target_lengths):
                length_peptides = [p for p in filtered if len(p) == length]
                if not length_peptides:
                    continue

                fasta_path = Path(tmpdir) / f"peptides_{length}.faa"
                fasta_path.write_text(
                    "\n".join(f">pep{i}\n{seq}" for i, seq in enumerate(length_peptides)) + "\n"
                )

                for allele in alleles:
                    netmhcpan_allele = _allele_to_netmhcpan(allele)
                    cmd = [
                        _BINARY,
                        "-a", netmhcpan_allele,
                        "-l", str(length),
                        "-f", str(fasta_path),
                        "-BA",
                    ]
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=300,
                        )
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(
                            f"NetMHCpan timed out for {allele} {length}-mer"
                        )
                    except FileNotFoundError:
                        raise RuntimeError(
                            f"'{_BINARY}' not found — install NetMHCpan and add it to PATH."
                        )

                    if result.returncode != 0:
                        raise RuntimeError(
                            f"NetMHCpan failed ({allele} {length}-mer): "
                            f"{(result.stderr or result.stdout)[:500]}"
                        )

                    rows = _parse_netmhcpan_output(result.stdout, allele)
                    if not rows and result.stdout.strip():
                        raise RuntimeError(
                            f"NetMHCpan returned output that could not be parsed "
                            f"({allele} {length}-mer). First 500 chars:\n"
                            f"{result.stdout[:500]}"
                        )
                    all_rows.extend(rows)

        return pd.DataFrame(all_rows) if all_rows else self._empty_result()
