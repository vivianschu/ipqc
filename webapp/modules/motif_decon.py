"""
Unsupervised motif deconvolution for class I immunopeptidomics.

Implements a GibbsCluster / EM-style algorithm in pure NumPy:
  - Soft-assignment EM with Dirichlet-smoothed PWMs
  - K selection via BIC (or fixed K)
  - Multiple restarts with anchor-aware K-means++ seeding
  - Anchor-based allele matching against a curated reference database
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA_IDX: dict[str, int] = {aa: i for i, aa in enumerate(_AA_ORDER)}
_N_AA = 20
_PSEUDOCOUNT = 0.5      # Dirichlet pseudocount per amino acid per position
_MAX_SAMPLE = 3_000     # downsample for EM if N exceeds this
_MIN_CLUSTER_FRAC = 0.02  # prune clusters below this fraction during EM


# ── Allele anchor reference database ─────────────────────────────────────────
# Curated from published immunopeptidomics data and IEDB eluted ligand motifs.
# Format: allele -> (P2_preferred_AAs, PΩ_preferred_AAs)
# Listed in rough order of preference strength (most preferred first).
_ANCHOR_DB: dict[str, tuple[str, str]] = {
    # HLA-A
    "HLA-A*01:01": ("TSA",    "YFW"),
    "HLA-A*02:01": ("LMV",    "VLAI"),
    "HLA-A*02:03": ("LMV",    "VLA"),
    "HLA-A*02:06": ("LMV",    "VLA"),
    "HLA-A*03:01": ("VLMI",   "KR"),
    "HLA-A*11:01": ("VTIL",   "KR"),
    "HLA-A*23:01": ("ILY",    "FWY"),
    "HLA-A*24:02": ("YFI",    "FLY"),
    "HLA-A*26:01": ("EVQ",    "KRL"),
    "HLA-A*29:02": ("ELD",    "YF"),
    "HLA-A*30:01": ("DE",     "YF"),
    "HLA-A*30:02": ("DE",     "KRY"),
    "HLA-A*31:01": ("VAR",    "KRL"),
    "HLA-A*32:01": ("HLI",    "KRL"),
    "HLA-A*33:01": ("ERA",    "RKF"),
    "HLA-A*68:01": ("VTE",    "KRL"),
    "HLA-A*68:02": ("LMV",    "VLK"),
    # HLA-B
    "HLA-B*07:02": ("P",      "LMI"),
    "HLA-B*08:01": ("KRN",    "LKR"),
    "HLA-B*13:01": ("AS",     "LFI"),
    "HLA-B*14:02": ("RK",     "RL"),
    "HLA-B*15:01": ("QVL",    "FYW"),
    "HLA-B*27:05": ("R",      "KRL"),
    "HLA-B*35:01": ("P",      "YFLW"),
    "HLA-B*37:01": ("DI",     "YFW"),
    "HLA-B*38:01": ("HR",     "LVI"),
    "HLA-B*39:01": ("RH",     "LVI"),
    "HLA-B*40:01": ("ED",     "LVI"),
    "HLA-B*40:02": ("ED",     "LVI"),
    "HLA-B*44:02": ("ED",     "FYL"),
    "HLA-B*44:03": ("ED",     "FYL"),
    "HLA-B*46:01": ("P",      "LVI"),
    "HLA-B*48:01": ("ATS",    "LIV"),
    "HLA-B*51:01": ("PA",     "ILF"),
    "HLA-B*52:01": ("PVL",    "ILF"),
    "HLA-B*53:01": ("P",      "KRL"),
    "HLA-B*57:01": ("IA",     "FWY"),
    "HLA-B*57:03": ("IA",     "FWY"),
    "HLA-B*58:01": ("AST",    "WFY"),
    # HLA-C
    "HLA-C*01:02": ("AS",     "LVI"),
    "HLA-C*03:03": ("TAS",    "LFI"),
    "HLA-C*03:04": ("TAS",    "LFI"),
    "HLA-C*04:01": ("YFH",    "LVF"),
    "HLA-C*05:01": ("YF",     "LVI"),
    "HLA-C*06:02": ("YFH",    "LVF"),
    "HLA-C*07:01": ("RK",     "LFW"),
    "HLA-C*07:02": ("RK",     "LFW"),
    "HLA-C*08:02": ("TAS",    "LFI"),
    "HLA-C*12:03": ("AST",    "LVF"),
    "HLA-C*14:02": ("RQ",     "LVI"),
    "HLA-C*15:02": ("VL",     "LVF"),
    "HLA-C*16:01": ("AS",     "LVI"),
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ClusterInfo:
    cluster_id: int
    pwm: np.ndarray          # (L, 20) probability matrix
    peptides: list[str]
    weight: float            # fraction of input peptides
    log_lik_per_seq: float   # mean log-likelihood (motif sharpness)
    allele_matches: list[dict] = field(default_factory=list)  # from anchor matching


@dataclass
class DeconvolutionResult:
    length: int
    k_selected: int
    clusters: list[ClusterInfo]  # sorted by size descending
    bic_curve: dict[int, float]  # k -> BIC value
    n_peptides: int


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_peptides(peptides: list[str], length: int) -> tuple[np.ndarray, list[str]]:
    """Encode peptides of given length as (N, L, 20) one-hot float32 array."""
    seqs = [p for p in peptides if len(p) == length and all(c in _AA_IDX for c in p)]
    N = len(seqs)
    if N == 0:
        return np.zeros((0, length, _N_AA), dtype=np.float32), seqs
    X = np.zeros((N, length, _N_AA), dtype=np.float32)
    for n, seq in enumerate(seqs):
        for pos, aa in enumerate(seq):
            X[n, pos, _AA_IDX[aa]] = 1.0
    return X, seqs


# ── EM core ───────────────────────────────────────────────────────────────────

def _build_log_pwms(X: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M-step: return (log_pwms (K,L,20), log_pi (K,)) from soft assignments."""
    N = float(X.shape[0])
    counts = np.einsum("nk,nla->kla", gamma, X) + _PSEUDOCOUNT
    totals = counts.sum(axis=-1, keepdims=True)
    log_pwms = np.log(counts / totals + 1e-12)
    nk = gamma.sum(axis=0)
    log_pi = np.log(nk / N + 1e-12)
    return log_pwms, log_pi


def _e_step(X: np.ndarray, log_pwms: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    """E-step: return gamma (N, K) soft assignments (rows sum to 1)."""
    log_resp = np.einsum("nla,kla->nk", X, log_pwms) + log_pi[np.newaxis, :]
    log_resp -= log_resp.max(axis=1, keepdims=True)
    gamma = np.exp(log_resp)
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def _kmeans_pp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> list[int]:
    """K-means++ seeding using anchor positions (P2 and C-terminal)."""
    N, L, _ = X.shape
    # Reduce to anchor features only: position 1 (P2) and position L-1 (PΩ)
    feat = X[:, [1, L - 1], :].reshape(N, -1).astype(np.float64)  # (N, 40)
    chosen = [int(rng.integers(N))]
    for _ in range(1, K):
        diffs = feat - feat[chosen[-1]]           # broadcast: (N, 40)
        min_d2 = np.full(N, np.inf)
        for c in chosen:
            d2 = np.sum((feat - feat[c]) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)
        total = min_d2.sum()
        if total == 0:
            chosen.append(int(rng.integers(N)))
        else:
            chosen.append(int(rng.choice(N, p=min_d2 / total)))
    return chosen


def _marginal_log_lik(X: np.ndarray, log_pwms: np.ndarray, log_pi: np.ndarray) -> float:
    """Marginal log-likelihood: sum_n log( sum_k pi_k * p(x_n | k) ).
    Used for BIC — monotone non-decreasing in K, unlike ELBO.
    """
    log_joint = np.einsum("nla,kla->nk", X, log_pwms) + log_pi[np.newaxis, :]  # (N, K)
    mx = log_joint.max(axis=1, keepdims=True)
    return float((mx.squeeze() + np.log(np.exp(log_joint - mx).sum(axis=1))).sum())


def _run_em_trial(
    X: np.ndarray,
    K: int,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Single EM trial. Returns (log_pwms, log_pi, gamma, marginal_log_lik)."""
    N, L, _ = X.shape

    # Hard nearest-seed initialization: assign every peptide to its nearest seed
    # using anchor features (P2 and C-terminal), then run soft EM from there.
    seed_idxs = _kmeans_pp_init(X, K, rng)
    feat = X[:, [1, L - 1], :].reshape(N, -1).astype(np.float64)  # (N, 40)
    seed_feats = feat[seed_idxs]                                   # (K, 40)
    dists = np.sum((feat[:, np.newaxis, :] - seed_feats[np.newaxis, :, :]) ** 2, axis=2)  # (N, K)
    hard_assign = dists.argmin(axis=1)
    gamma = np.zeros((N, K), dtype=np.float64)
    for n in range(N):
        gamma[n, hard_assign[n]] = 1.0
    gamma += 1e-3
    gamma /= gamma.sum(axis=1, keepdims=True)

    log_pwms, log_pi = _build_log_pwms(X, gamma)
    prev_mll = -np.inf

    for _ in range(n_iter):
        gamma = _e_step(X, log_pwms, log_pi)

        # Prune near-empty clusters
        alive = gamma.sum(axis=0) >= N * _MIN_CLUSTER_FRAC
        if alive.sum() < K and alive.sum() >= 1:
            gamma = gamma[:, alive]
            gamma /= gamma.sum(axis=1, keepdims=True)
            K = int(alive.sum())

        log_pwms, log_pi = _build_log_pwms(X, gamma)
        mll = _marginal_log_lik(X, log_pwms, log_pi)
        if abs(mll - prev_mll) < 1e-4 * abs(prev_mll + 1):
            break
        prev_mll = mll

    return log_pwms, log_pi, gamma, prev_mll


def _aic_anchors(marginal_ll: float, K: int, N: int) -> float:
    # Count only P2 and PΩ anchor positions as discriminating parameters;
    # standard BIC over-penalises when middle positions are mostly uniform noise.
    n_params = K * 2 * (_N_AA - 1) + (K - 1)  # 38K + K-1
    return -2.0 * marginal_ll + n_params * np.log(max(N, 2))


# ── Main entry point ──────────────────────────────────────────────────────────

def run_deconvolution(
    peptides: list[str],
    length: int = 9,
    k_min: int = 1,
    k_max: int = 8,
    n_restarts: int = 5,
    n_iter: int = 300,
    fixed_k: int | None = None,
    seed: int = 42,
) -> DeconvolutionResult | None:
    """
    Cluster peptides of *length* into K motifs via soft-assignment EM.
    K is selected by BIC unless *fixed_k* is provided.
    Returns None if there are too few peptides.
    """
    X_full, seqs_full = encode_peptides(peptides, length)
    N_full = len(seqs_full)
    if N_full < 20:
        return None

    rng = np.random.default_rng(seed)

    # Subsample for parameter estimation when dataset is large
    if N_full > _MAX_SAMPLE:
        idxs = rng.choice(N_full, _MAX_SAMPLE, replace=False)
        X = X_full[idxs]
    else:
        X = X_full
    N = X.shape[0]

    k_range = [fixed_k] if fixed_k is not None else list(range(k_min, min(k_max + 1, N // 8 + 2)))
    k_range = [k for k in k_range if 1 <= k <= N // 4]
    if not k_range:
        k_range = [1]

    bic_curve: dict[int, float] = {}
    best_bic = np.inf
    best_result: DeconvolutionResult | None = None

    for k in k_range:
        best_ll = -np.inf
        best_log_pwms = best_log_pi = best_gamma = None

        for _ in range(n_restarts):
            log_pwms, log_pi, gamma, ll = _run_em_trial(X, k, n_iter, rng)
            if ll > best_ll:
                best_ll = ll
                best_log_pwms, best_log_pi, best_gamma = log_pwms, log_pi, gamma

        eff_K = best_gamma.shape[1]
        bic = _aic_anchors(best_ll, eff_K, N)
        bic_curve[k] = bic

        if bic < best_bic:
            best_bic = bic

            # Assign ALL peptides (including held-out subsample) to final clusters
            gamma_full = _e_step(X_full, best_log_pwms, best_log_pi)
            assignments = gamma_full.argmax(axis=1)

            clusters: list[ClusterInfo] = []
            for c in range(eff_K):
                mask = assignments == c
                c_seqs = [seqs_full[i] for i in range(N_full) if mask[i]]
                pwm_c = np.exp(best_log_pwms[c])
                ll_c = (
                    float(np.einsum("nla,la->n", X_full[mask], best_log_pwms[c]).mean())
                    if mask.any() else 0.0
                )
                clusters.append(ClusterInfo(
                    cluster_id=c + 1,
                    pwm=pwm_c,
                    peptides=c_seqs,
                    weight=float(mask.sum()) / N_full,
                    log_lik_per_seq=ll_c,
                ))

            clusters.sort(key=lambda ci: -ci.weight)
            for i, ci in enumerate(clusters):
                ci.cluster_id = i + 1

            best_result = DeconvolutionResult(
                length=length,
                k_selected=eff_K,
                clusters=clusters,
                bic_curve=dict(bic_curve),
                n_peptides=N_full,
            )

    if best_result is not None:
        best_result.bic_curve = bic_curve
    return best_result


# ── Allele matching ───────────────────────────────────────────────────────────

def _anchor_score(pwm_col: np.ndarray, preferred_aas: str) -> float:
    """Sum of PWM probability for the preferred amino acids at one position."""
    return float(sum(pwm_col[_AA_IDX[aa]] for aa in preferred_aas if aa in _AA_IDX))


def match_allele_by_anchors(
    pwm: np.ndarray,
    candidate_alleles: list[str] | None = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Rank alleles by how well their anchor preferences match the discovered motif PWM.

    Scores P2 (index 1) and C-terminal (index -1) anchor positions independently,
    then combines them. Returns top_n matches as list of dicts with keys:
    allele, locus, p2_score, pc_score, combined_score.
    """
    if candidate_alleles is None:
        candidate_alleles = list(_ANCHOR_DB.keys())

    results = []
    for allele in candidate_alleles:
        if allele not in _ANCHOR_DB:
            continue
        p2_pref, pc_pref = _ANCHOR_DB[allele]
        p2_score = _anchor_score(pwm[1], p2_pref)
        pc_score = _anchor_score(pwm[-1], pc_pref)
        combined = (p2_score + pc_score) / 2.0
        locus = allele.split("*")[0]
        results.append({
            "allele": allele,
            "locus": locus,
            "p2_score": round(p2_score, 3),
            "pc_score": round(pc_score, 3),
            "combined_score": round(combined, 3),
        })

    results.sort(key=lambda r: -r["combined_score"])
    return results[:top_n]


def annotate_clusters(result: DeconvolutionResult, candidate_alleles: list[str] | None = None) -> None:
    """In-place: add allele_matches to each ClusterInfo in result."""
    for ci in result.clusters:
        ci.allele_matches = match_allele_by_anchors(ci.pwm, candidate_alleles, top_n=5)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def pwm_to_logo_df(pwm: np.ndarray) -> pd.DataFrame:
    """Convert (L, 20) PWM to logomaker-compatible frequency DataFrame."""
    L = pwm.shape[0]
    return pd.DataFrame(
        pwm,
        index=list(range(1, L + 1)),
        columns=list(_AA_ORDER),
    )


def top_anchor_residues(pwm: np.ndarray, n: int = 4) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return top-n (AA, frequency) at P2 and C-terminal anchor positions."""
    p2 = sorted(zip(_AA_ORDER, pwm[1].tolist()), key=lambda x: -x[1])[:n]
    pc = sorted(zip(_AA_ORDER, pwm[-1].tolist()), key=lambda x: -x[1])[:n]
    return p2, pc
