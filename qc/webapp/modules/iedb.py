"""Shared IEDB MHC Class I Tools API helpers."""
from __future__ import annotations

import time
from io import StringIO

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

IEDB_MHCI_URL = "https://tools.iedb.org/tools_api/mhci/"
IEDB_EMAIL = "unechoed@gmail.com"

# Combinations above this threshold include an email address (rough 10-min proxy)
LONG_JOB_THRESHOLD = 500

# TCP connect timeout (seconds) — fail fast if the host is unreachable
_CONNECT_TIMEOUT = 15
# Read timeout (seconds) — large jobs can take several minutes to compute
_READ_TIMEOUT = 600

# Retry on transient network errors: 3 attempts with exponential backoff (2 s, 4 s, 8 s)
_RETRY = Retry(
    total=3,
    backoff_factor=2,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST"],
    raise_on_status=False,
)


def _session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

COMMON_ALLELES = [
    "HLA-A*01:01", "HLA-A*02:01", "HLA-A*03:01", "HLA-A*11:01",
    "HLA-A*24:02", "HLA-A*26:01", "HLA-B*07:02", "HLA-B*08:01",
    "HLA-B*15:01", "HLA-B*27:05", "HLA-B*35:01", "HLA-B*40:01",
    "HLA-B*44:02", "HLA-B*57:01", "HLA-C*07:01", "HLA-C*07:02",
]

METHODS = [
    "recommended",
    "netmhcpan_el",
    "netmhcpan_ba",
    "ann",
    "smmpmbec",
    "smm",
    "comblib_sidney2008",
]


def call_iedb_mhci(
    sequences: list[str],
    allele: str,
    length: int,
    method: str,
    include_email: bool,
) -> pd.DataFrame:
    """POST one prediction job to the IEDB MHC-I API and return a parsed DataFrame.

    Uses a short connect timeout (15 s) so a dead host fails fast, and a long
    read timeout (600 s) for large payloads.  Three automatic retries with
    exponential backoff cover transient 5xx / network blips.
    """
    fasta = "\n".join(f">seq{i + 1}\n{seq}" for i, seq in enumerate(sequences))
    data: dict[str, str] = {
        "method": method,
        "sequence_text": fasta,
        "allele": allele,
        "length": str(length),
    }
    if include_email:
        data["email_address"] = IEDB_EMAIL

    try:
        resp = _session().post(
            IEDB_MHCI_URL,
            data=data,
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
    except requests.exceptions.ConnectTimeout:
        raise RuntimeError(
            f"Could not reach {IEDB_MHCI_URL} within {_CONNECT_TIMEOUT} s. "
            "Check your internet connection or try again later."
        )
    except requests.exceptions.ReadTimeout:
        raise RuntimeError(
            f"IEDB API did not respond within {_READ_TIMEOUT} s. "
            "The server may be overloaded — try a smaller batch or retry later."
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Network error contacting IEDB: {exc}") from exc

    resp.raise_for_status()

    text = resp.text.strip()
    if not text or text.lower().startswith("error"):
        raise ValueError(text[:300] or "Empty response from IEDB API")

    return pd.read_csv(StringIO(text), sep="\t")


def binding_level(ic50: float) -> str:
    """Classify IC50 (nM) into SB / WB / NB."""
    if ic50 < 50:
        return "SB"
    if ic50 < 500:
        return "WB"
    return "NB"


def run_batched(
    sequences: list[str],
    alleles: list[str],
    lengths: list[int],
    method: str,
    inter_batch_delay: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    """Run all allele × length batches sequentially, one job at a time.

    Returns (combined_results_df, list_of_error_strings).
    """
    n_combinations = len(sequences) * len(alleles) * len(lengths)
    use_email = n_combinations > LONG_JOB_THRESHOLD

    batches = [(a, l) for a in alleles for l in lengths]
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for idx, (allele, length) in enumerate(batches):
        try:
            df = call_iedb_mhci(sequences, allele, length, method, use_email)
            frames.append(df)
        except Exception as exc:
            errors.append(f"{allele} / {length}-mer: {exc}")
        if idx < len(batches) - 1:
            time.sleep(inter_batch_delay)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, errors


def postprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise IEDB column names and add a binding_level column."""
    rename: dict[str, str] = {}
    for src, dst in [("percentile_rank", "rank"), ("ann_ic50", "ic50"), ("ic50", "ic50")]:
        if src in raw.columns and dst not in rename.values():
            rename[src] = dst
    df = raw.rename(columns=rename)
    if "ic50" in df.columns:
        df["binding_level"] = df["ic50"].apply(
            lambda x: binding_level(float(x)) if pd.notna(x) else "NB"
        )
    return df
