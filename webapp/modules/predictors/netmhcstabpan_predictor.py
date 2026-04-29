"""NetMHCstabpan 1.0 pMHC stability predictor via subprocess.

NetMHCstabpan predicts MHC-I peptide complex stability (half-life, %Rank_Stab).
It depends on a NetMHCpan binary internally for allele pseudo-sequences.

The bundled copy is expected at:
  <webapp_root>/tools/netMHCstabpan-1.0/netMHCstabpan
The wrapper is a tcsh script that must have NMHOME and NetMHCpan paths set
correctly.  This module patches those paths automatically on first use.
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
from .netmhcpan_predictor import _BUNDLED_BINARY as _NETMHCPAN_BINARY

_TOOL_DIR: Path = Path(__file__).parent.parent.parent / "tools" / "netMHCstabpan-1.0"
_WRAPPER: Path = _TOOL_DIR / "netMHCstabpan"

# Platform-specific binary: only Darwin_x86_64 is distributed (runs via Rosetta
# on Apple Silicon).  Linux users get Linux_x86_64.
_PLATFORM_DIRS = ["Darwin_x86_64", "Linux_x86_64"]
_BINARY: Path | None = next(
    (d / "bin" / "netMHCstabpan" for name in _PLATFORM_DIRS
     if (d := _TOOL_DIR / name).is_dir() and (d / "bin" / "netMHCstabpan").is_file()),
    None,
)

_MODEL_INFO = "NetMHCstabpan 1.0"


def _platform_data_dir() -> Path | None:
    """Return the platform-specific data directory, creating a symlink if needed.

    The data tarball (data.tar.gz from CBS) extracts to netMHCstabpan-1.0/data/.
    The binary expects data at <platform-dir>/data/.  We create a symlink
    Darwin_x86_64/data → ../data automatically if the top-level data dir exists.
    """
    for name in _PLATFORM_DIRS:
        platform_dir = _TOOL_DIR / name
        if not platform_dir.is_dir():
            continue
        data_dir = platform_dir / "data"
        if data_dir.exists():
            return data_dir
        # Top-level data dir present but symlink missing — create it
        top_data = _TOOL_DIR / "data"
        if top_data.is_dir():
            try:
                data_dir.symlink_to("../data")
                return data_dir
            except OSError:
                pass
    return None


def _configure_wrapper() -> bool:
    """Patch NMHOME and NetMHCpan paths in the wrapper script.

    The distributed script hard-codes CBS server paths.  We rewrite them to
    point to the bundled copies so the tool works out-of-the-box.  Also inserts
    a Darwin_arm64 → Darwin_x86_64 fallback so the x86_64 binary runs via
    Rosetta on Apple Silicon.

    Returns True if the wrapper file exists and was successfully configured.
    """
    if not _WRAPPER.is_file():
        return False

    content = _WRAPPER.read_text()
    original = content

    # Update NMHOME to the actual tool directory
    content = re.sub(
        r"(setenv\s+NMHOME\s+)\S+",
        rf"\g<1>{_TOOL_DIR}",
        content,
    )

    # Update NetMHCpan path to our bundled binary
    netmhcpan_path = str(_NETMHCPAN_BINARY) if _NETMHCPAN_BINARY.is_file() else "netMHCpan"
    content = re.sub(
        r"(set\s+NetMHCpan\s*=\s*)\S+",
        rf"\g<1>{netmhcpan_path}",
        content,
    )

    # Add Darwin_arm64 → Darwin_x86_64 fallback (inserted before the binary call)
    arm64_fix = (
        "if ( $PLATFORM == Darwin_arm64 ) then\n"
        "\tset PLATFORM = Darwin_x86_64\n"
        "endif\n"
    )
    sentinel = "setenv NETMHCstabpan $NMHOME/$PLATFORM"
    if arm64_fix not in content and sentinel in content:
        content = content.replace(sentinel, arm64_fix + sentinel)

    if content != original:
        _WRAPPER.write_text(content)

    # Ensure the platform data directory is reachable (symlink if needed)
    _platform_data_dir()
    return True


def _version() -> str:
    version_file = _TOOL_DIR / "Darwin_x86_64" / "data" / "version"
    if not version_file.is_file():
        version_file = _TOOL_DIR / "Linux_x86_64" / "data" / "version"
    try:
        return version_file.read_text().strip()
    except Exception:
        return "NetMHCstabpan 1.0"


def _allele_to_netmhc(allele: str) -> str:
    """'HLA-A*02:01' → 'HLA-A02:01' (drops the asterisk)."""
    return allele.replace("*", "")


def _parse_output(text: str, allele: str) -> list[dict]:
    """Parse NetMHCstabpan tabular output.

    Expected columns (auto-detected from header):
      pos  HLA  peptide  Identity  Pred  Thalf(h)  %Rank_Stab  Exp_stab  [BindLevel]
    """
    col_pred  = 4
    col_thalf = 5
    col_rank  = 6

    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        parts = line.split()
        if not parts:
            continue

        if parts[0] == "pos":
            try:
                col_pred  = parts.index("Pred")
                col_thalf = parts.index("Thalf(h)")
                col_rank  = parts.index("%Rank_Stab")
            except ValueError:
                pass
            continue

        if not parts[0].isdigit():
            continue

        try:
            peptide   = parts[2]
            pred      = float(parts[col_pred])
            thalf     = float(parts[col_thalf])
            rank_stab = float(parts[col_rank])

            rows.append({
                "peptide":       peptide,
                "allele":        allele,
                "score":         pred,
                "rank":          rank_stab,
                "ic50":          float("nan"),
                "thalf":         thalf,
                "binding_level": assign_binding_level(rank_stab, None),
                "tool":          "NetMHCstabpan",
                "model_info":    _MODEL_INFO,
            })
        except (IndexError, ValueError):
            continue

    return rows


class NetMHCstabpanPredictor(BaseMHCIPredictor):
    name = "NetMHCstabpan"
    predictor_type = "stability"
    description = (
        "NetMHCstabpan 1.0 — predicts pMHC-I complex stability (half-life and %Rank_Stab) "
        "from DTU Health Tech.  Stability correlates with T cell immunogenicity independently "
        "of binding affinity."
    )
    install_hint = (
        "1. Download NetMHCstabpan 1.0 binaries:\n"
        "     https://services.healthtech.dtu.dk/services/NetMHCstabpan-1.0/\n"
        f"   Extract to: {_TOOL_DIR}\n\n"
        "2. Download data files (required, ~42 MB):\n"
        "     http://www.cbs.dtu.dk/services/NetMHCstabpan-1.0/data.tar.gz\n"
        f"   Extract to: {_TOOL_DIR}  (creates {_TOOL_DIR}/data/)\n\n"
        "NetMHCpan must also be installed (bundled copy is used automatically).\n"
        "Requires tcsh: brew install tcsh"
    )

    @classmethod
    def is_available(cls) -> bool:
        if not _WRAPPER.is_file():
            return False
        if _BINARY is None:
            return False
        if not _configure_wrapper():
            return False
        # Require the data directory (neural network weights)
        if _platform_data_dir() is None:
            return False
        # Quick sanity probe — stability binary should print usage on bad args
        try:
            r = subprocess.run(
                [str(_WRAPPER), "-h"],
                capture_output=True, text=True, timeout=15,
            )
            combined = r.stdout + r.stderr
            if "no binaries found" in combined.lower():
                return False
        except Exception:
            return False
        return True

    @classmethod
    def version(cls) -> str:
        return _version()

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
        length_flag = ",".join(str(l) for l in sorted(target_lengths))

        with tempfile.TemporaryDirectory() as tmpdir:
            pep_path = Path(tmpdir) / "peptides.pep"
            pep_path.write_text("\n".join(filtered) + "\n")

            # ~0.1 s/peptide (conservative); minimum 120 s
            _timeout = max(120, len(filtered) * 0.1 * len(alleles))

            for allele in alleles:
                stab_allele = _allele_to_netmhc(allele)
                cmd = [
                    str(_WRAPPER),
                    "-p", str(pep_path),
                    "-a", stab_allele,
                    "-l", length_flag,
                ]
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        stdin=subprocess.DEVNULL,
                        timeout=_timeout,
                    )
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"NetMHCstabpan timed out for {allele} "
                        f"({len(filtered):,} peptides, limit {_timeout:.0f} s)"
                    )
                except FileNotFoundError:
                    raise RuntimeError(
                        f"'{_WRAPPER}' not found — install NetMHCstabpan."
                    )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"NetMHCstabpan failed ({allele}): "
                        f"{(result.stderr or result.stdout)[:500]}"
                    )

                rows = _parse_output(result.stdout, allele)
                if not rows and result.stdout.strip():
                    raise RuntimeError(
                        f"NetMHCstabpan returned output that could not be parsed "
                        f"({allele}). First 500 chars:\n{result.stdout[:500]}"
                    )
                all_rows.extend(rows)

        if not all_rows:
            return self._empty_result()

        df = pd.DataFrame(all_rows)
        # Ensure thalf column is present even if parsing missed it
        if "thalf" not in df.columns:
            df["thalf"] = float("nan")
        return df
