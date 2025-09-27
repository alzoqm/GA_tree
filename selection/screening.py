from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import erf, sqrt, pi
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng

from .splits import CrossFitSplit
from .math_utils import chi2_sf
from .utils import cliffs_delta

# --- Optional SciPy acceleration for Kendall tau (uses Knight's O(n log n) algorithm) ---
try:  # noqa: SIM105
    from scipy.stats import kendalltau as _scipy_kendalltau  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional
    _HAVE_SCIPY = False
    _scipy_kendalltau = None

# ==================== Data classes ====================

@dataclass(slots=True)
class ScreeningDetail:
    fold_id: str
    feature: str
    method: str
    effect: float
    pvalue: float
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ScreeningResult:
    pvalues: Dict[str, float]
    effects: Dict[str, float]
    cliffs_delta: Dict[str, float]
    details: List[ScreeningDetail]
    passing_features: List[str]
    methods: List[str]


# ==================== Utilities ====================

def _as_float32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32, order="C")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with mergesort; returns float32 to cut memory."""
    values = _as_float32(values)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    if sorted_vals.size == 1:
        ranks = np.array([1.0], dtype=np.float32)
    else:
        diff = np.diff(sorted_vals)
        boundaries = np.concatenate(([0], np.nonzero(diff)[0] + 1, [sorted_vals.size]))
        counts = np.diff(boundaries)
        starts = boundaries[:-1]
        ends = boundaries[1:]
        avg_ranks = (starts + ends - 1) * 0.5 + 1.0
        ranks_sorted = np.repeat(avg_ranks, counts)
        ranks = np.empty_like(ranks_sorted, dtype=np.float32)
        ranks[order] = ranks_sorted.astype(np.float32, copy=False)
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray, *, y_ctx: Optional[Tuple[np.ndarray, float]] = None) -> float:
    """Compute Spearman's rho; reuses precomputed y_ctx=(y_centered_ranks, ||y||) if provided."""
    if x.size <= 1 or y.size <= 1:
        return 0.0

    rx = _average_ranks(_as_float32(x))
    x_centered = rx - rx.mean(dtype=np.float32)

    if y_ctx is None:
        ry = _average_ranks(_as_float32(y))
        y_centered = ry - ry.mean(dtype=np.float32)
        denom_y = float(np.sqrt(np.sum(y_centered * y_centered, dtype=np.float64)))
    else:
        y_centered, denom_y = y_ctx

    denom_x = float(np.sqrt(np.sum(x_centered * x_centered, dtype=np.float64)))
    denom = denom_x * denom_y
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def _approx_spearman_pvalue(r: float, n: int) -> float:
    if n <= 3:
        return 1.0
    # Normal approximation of Spearman's rho (works for n>10 reasonably well)
    z = r * sqrt(n - 1)
    # two-sided using normal CDF via error function
    return float(max(0.0, min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2)))))))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


# ==================== Kendall tau ====================

def _kendall_tau(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Kendall's tau with SciPy fast path and a precise Python fallback."""
    x_arr = _as_float32(x)
    y_arr = _as_float32(y)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_valid = x_arr[mask]
    y_valid = y_arr[mask]
    n = int(x_valid.size)
    if n <= 1:
        return 0.0, 0

    if _HAVE_SCIPY:
        tau, _ = _scipy_kendalltau(x_valid, y_valid, method="auto", nan_policy="omit")  # type: ignore
        if tau is None or not np.isfinite(tau):
            return 0.0, 0
        total_pairs = n * (n - 1) // 2
        if total_pairs <= 0:
            return 0.0, 0
        return float(tau), total_pairs

    return _kendall_tau_python(x_valid, y_valid)


def _kendall_tau_python(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    n = int(x.size)
    if n <= 1:
        return 0.0, 0

    concordant = 0
    discordant = 0
    for i in range(n - 1):
        xi = float(x[i])
        yi = float(y[i])
        for j in range(i + 1, n):
            dx = xi - float(x[j])
            dy = yi - float(y[j])
            if dx == 0.0 and dy == 0.0:
                continue
            prod = dx * dy
            if prod > 0.0:
                concordant += 1
            elif prod < 0.0:
                discordant += 1

    used_pairs = concordant + discordant
    if used_pairs == 0:
        return 0.0, 0

    total_pairs = n * (n - 1) // 2
    counts_x = np.unique(x, return_counts=True)[1]
    counts_y = np.unique(y, return_counts=True)[1]
    tie_x_pairs = sum(int(c * (c - 1) // 2) for c in counts_x if c > 1)
    tie_y_pairs = sum(int(c * (c - 1) // 2) for c in counts_y if c > 1)
    den_x = total_pairs - tie_x_pairs
    den_y = total_pairs - tie_y_pairs
    denom = sqrt(max(den_x, 1) * max(den_y, 1))
    if denom <= 0.0:
        return 0.0, used_pairs
    tau = (concordant - discordant) / denom
    tau = max(-1.0, min(1.0, tau))
    return float(tau), used_pairs


def _kendall_tau_pvalue(tau: float, n: int) -> float:
    if n <= 1:
        return 1.0
    var = (4.0 * n + 10.0) / (9.0 * n * (n - 1))
    if var <= 0:
        return 1.0
    z = tau / sqrt(var)
    return float(max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z))))))


# ==================== HSIC helpers ====================

def _rbf_bandwidth(values: np.ndarray, bandwidth: float | str | None) -> float:
    """Return sigma^2 for RBF. 'median' => median heuristic on squared distances."""
    vector = _as_float32(values).reshape(-1, 1)
    if vector.size <= 1:
        return 1.0
    if bandwidth is None or bandwidth == "median":
        diffs = vector - vector.T
        sq_dists = diffs * diffs
        flattened = sq_dists[np.triu_indices_from(sq_dists, k=1)]
        positive = flattened[flattened > 0]
        if positive.size:
            sigma_sq = float(np.median(positive))
        else:
            sigma_sq = float(np.var(vector))
        if not np.isfinite(sigma_sq) or sigma_sq <= 1e-12:
            sigma_sq = 1.0
        return sigma_sq
    # numeric bandwidth interpreted as sigma^2 directly
    sigma_sq = float(bandwidth)
    if sigma_sq <= 1e-12:
        sigma_sq = 1.0
    return sigma_sq


def _rbf_kernel(values: np.ndarray, bandwidth: float | str | None) -> np.ndarray:
    """Full RBF Gram matrix (O(n^2) memory) – keep only for small n."""
    vector = _as_float32(values).reshape(-1, 1)
    if vector.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    diffs = vector - vector.T
    sq_dists = diffs * diffs
    sigma_sq = _rbf_bandwidth(values, bandwidth)
    gamma = 1.0 / (2.0 * sigma_sq)
    return np.exp(-gamma * sq_dists, dtype=np.float32)


def _center_kernel(kernel: np.ndarray) -> np.ndarray:
    """Center a symmetric Gram matrix (float32)."""
    if kernel.size == 0:
        return kernel
    K = kernel.astype(np.float32, copy=False)
    row_mean = K.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    col_mean = K.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    total_mean = np.float32(K.mean(dtype=np.float64))
    return K - row_mean - col_mean + total_mean


def _hsic_statistic_from_centered(Kc: np.ndarray, Lc: np.ndarray) -> float:
    n = float(Kc.shape[0])
    if n <= 1:
        return 0.0
    value = np.sum(Kc * Lc, dtype=np.float64) / ((n - 1.0) ** 2)
    return float(max(value, 0.0))


# ===== RFF-based HSIC (linear-time, memory O(nD)) =====

def _rff_features(values: np.ndarray, sigma_sq: float, D: int, rng: Generator) -> np.ndarray:
    """Random Fourier Features for 1D inputs. Returns (n, D) float32 matrix (centered not applied)."""
    x = _as_float32(values).reshape(-1, 1)
    if x.size == 0 or D <= 0:
        return np.empty((x.shape[0], 0), dtype=np.float32)
    # For k(x,x') = exp(-||x-x'||^2 / (2σ^2)), sample w ~ N(0, 1/σ^2), b ~ U[0, 2π)
    s = float(np.sqrt(max(sigma_sq, 1e-12)))
    W = rng.normal(loc=0.0, scale=1.0 / s, size=(1, D)).astype(np.float32)
    b = rng.uniform(0.0, 2.0 * pi, size=(D,)).astype(np.float32)
    Z = (x @ W).astype(np.float32)
    Z += b  # broadcast add
    # sqrt(2/D) * cos
    Z = np.cos(Z, dtype=np.float32)
    Z *= np.float32(np.sqrt(2.0 / max(D, 1)))
    return Z


def _hsic_rff_stat(phi_x: np.ndarray, phi_y: np.ndarray) -> float:
    """HSIC ≈ || Cov(phi_x, phi_y) ||_F^2; center each column then compute Frobenius norm of cross-cov."""
    if phi_x.size == 0 or phi_y.size == 0:
        return 0.0
    # Center columns (in-place safe copies)
    X = phi_x - phi_x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    Y = phi_y - phi_y.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    n = float(X.shape[0])
    if n <= 1:
        return 0.0
    C = (X.T @ Y) / (n - 1.0)
    val = float(np.sum(C * C, dtype=np.float64))
    return max(val, 0.0)


def _hsic_rff_test(
    x: np.ndarray,
    y: np.ndarray,
    cfg: Mapping[str, object],
    rng: Generator,
    *,
    precomputed_phi_y: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Permutation test on RFF-HSIC with small D (default 128)."""
    n = x.size
    if n < 8 or y.size < 8:
        return 0.0, 1.0
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0, 1.0

    D = int(cfg.get("rff_dim", 128))
    bw = cfg.get("bandwidth", "median")
    target_bw = cfg.get("target_bandwidth", None) or bw

    sigma_sq_x = _rbf_bandwidth(x, bw)
    phi_x = _rff_features(x, sigma_sq_x, D, rng)

    if precomputed_phi_y is not None and precomputed_phi_y.shape[0] == n:
        phi_y = precomputed_phi_y
    else:
        sigma_sq_y = _rbf_bandwidth(y, target_bw)
        phi_y = _rff_features(y, sigma_sq_y, D, rng)

    stat = _hsic_rff_stat(phi_x, phi_y)

    boot = cfg.get("bootstrap", {}) if isinstance(cfg, Mapping) else {}
    samples = int(boot.get("samples", 64))
    if samples <= 0:
        return stat, 1.0

    exceed = 0.0
    # Pre-center phi_x, phi_y once; permutation doesn't change column means
    Xc = phi_x - phi_x.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    Yc = phi_y - phi_y.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    for _ in range(samples):
        perm = rng.permutation(n)
        null = _hsic_rff_stat(Xc, Yc[perm])
        if null >= stat:
            exceed += 1.0
    pvalue = (exceed + 1.0) / (samples + 1.0)
    return float(stat), float(max(0.0, min(1.0, pvalue)))


# ==================== Distance correlation helpers ====================

def _distance_matrix(values: np.ndarray) -> np.ndarray:
    vector = _as_float32(values).reshape(-1, 1)
    return np.abs(vector - vector.T, dtype=np.float32)


def _center_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    M = matrix.astype(np.float32, copy=False)
    row_mean = M.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    col_mean = M.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    total_mean = np.float32(M.mean(dtype=np.float64))
    return M - row_mean - col_mean + total_mean


def _distance_correlation_statistic_from_centered(A: np.ndarray, B: np.ndarray) -> float:
    n = float(A.shape[0])
    if n <= 0:
        return 0.0
    cov = float(np.sum(A * B, dtype=np.float64) / (n * n))
    var_x = float(np.sum(A * A, dtype=np.float64) / (n * n))
    var_y = float(np.sum(B * B, dtype=np.float64) / (n * n))
    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    stat = cov / np.sqrt(var_x * var_y)
    return float(max(0.0, min(1.0, stat)))


def _distance_correlation_statistic(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    A = _center_distance_matrix(_distance_matrix(x))
    B = _center_distance_matrix(_distance_matrix(y))
    return _distance_correlation_statistic_from_centered(A, B)


# ===== Sampled dCor: operate on a small subset S (O(s^2) memory) =====

def _dcor_sampled_test(
    x: np.ndarray,
    y: np.ndarray,
    cfg: Mapping[str, object],
    rng: Generator,
    *,
    precomputed_BS: Optional[np.ndarray] = None,
    precomputed_idx: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Distance correlation on a subset of indices (size s), with permutation test."""
    n = x.size
    if n < 8 or y.size < 8:
        return 0.0, 1.0
    s = int(cfg.get("sample_size", 1024))
    s = max(32, min(s, n))
    if precomputed_idx is not None and precomputed_idx.size == s:
        idx = precomputed_idx
    else:
        idx = rng.choice(n, size=s, replace=False)
    xs = _as_float32(x[idx])
    if precomputed_BS is None:
        ys = _as_float32(y[idx])
        BS = _center_distance_matrix(_distance_matrix(ys))
    else:
        BS = precomputed_BS
    AS = _center_distance_matrix(_distance_matrix(xs))
    stat = _distance_correlation_statistic_from_centered(AS, BS)

    perm_cfg = cfg.get("permutation", {}) if isinstance(cfg, Mapping) else {}
    samples = int(cfg.get("permutations", perm_cfg.get("samples", 64)))
    if samples <= 0:
        return stat, 1.0
    exceed = 0.0
    for _ in range(samples):
        perm = rng.permutation(s)
        null = _distance_correlation_statistic_from_centered(AS, BS[perm][:, perm])
        if null >= stat:
            exceed += 1.0
    pvalue = (exceed + 1.0) / (samples + 1.0)
    return float(stat), float(max(0.0, min(1.0, pvalue)))


# ===== Original (small-n) permutation tests kept for backward-compat =====

def _hsic_statistic(x: np.ndarray, y: np.ndarray, cfg: Mapping[str, object]) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x_kernel = _rbf_kernel(x, cfg.get("bandwidth", "median"))
    y_bandwidth = cfg.get("target_bandwidth") or cfg.get("bandwidth", "median")
    y_kernel = _rbf_kernel(y, y_bandwidth)
    Kc = _center_kernel(x_kernel)
    Lc = _center_kernel(y_kernel)
    return _hsic_statistic_from_centered(Kc, Lc)


def _hsic_test_precomputed(
    Kc_x: np.ndarray,
    Lc_y: np.ndarray,
    cfg: Mapping[str, object],
    rng: Generator,
) -> Tuple[float, float]:
    n = Kc_x.shape[0]
    if n < 8:
        return 0.0, 1.0
    statistic = _hsic_statistic_from_centered(Kc_x, Lc_y)
    bootstrap_cfg = cfg.get("bootstrap", {}) if isinstance(cfg, Mapping) else {}
    samples = int(bootstrap_cfg.get("samples", 64))
    if samples <= 0:
        return statistic, 1.0
    block_size = int(bootstrap_cfg.get("block_size", max(8, n // 8 or 1)))
    method = str(bootstrap_cfg.get("method", "block")).lower()
    exceed = 0.0
    for _ in range(samples):
        permuted_idx = _block_permutation_indices(n, block_size, rng, method=method)
        Lc_perm = Lc_y[permuted_idx][:, permuted_idx]
        null_value = _hsic_statistic_from_centered(Kc_x, Lc_perm)
        if null_value >= statistic:
            exceed += 1.0
    pvalue = (exceed + 1.0) / (samples + 1.0)
    return statistic, float(max(0.0, min(1.0, pvalue)))


def _hsic_test(
    x: np.ndarray,
    y: np.ndarray,
    cfg: Mapping[str, object],
    rng: Generator,
) -> Tuple[float, float]:
    if x.size < 8 or y.size < 8:
        return 0.0, 1.0
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0, 1.0
    statistic = _hsic_statistic(x, y, cfg)
    bootstrap_cfg = cfg.get("bootstrap", {}) if isinstance(cfg, Mapping) else {}
    samples = int(bootstrap_cfg.get("samples", 64))
    if samples <= 0:
        return statistic, 1.0
    block_size = int(bootstrap_cfg.get("block_size", max(8, x.size // 8 or 1)))
    method = str(bootstrap_cfg.get("method", "block")).lower()
    exceed = 0.0
    for _ in range(samples):
        permuted_idx = _block_permutation_indices(x.size, block_size, rng, method=method)
        permuted_y = y[permuted_idx]
        null_value = _hsic_statistic(x, permuted_y, cfg)
        if null_value >= statistic:
            exceed += 1.0
    pvalue = (exceed + 1.0) / (samples + 1.0)
    return statistic, float(max(0.0, min(1.0, pvalue)))


def _distance_correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    cfg: Mapping[str, object],
    rng: Generator,
) -> Tuple[float, float]:
    if x.size < 8 or y.size < 8:
        return 0.0, 1.0
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0, 1.0
    statistic = _distance_correlation_statistic(x, y)
    perm_cfg = cfg.get("permutation", {}) if isinstance(cfg, Mapping) else {}
    samples = int(cfg.get("permutations", perm_cfg.get("samples", 64)))
    if samples <= 0:
        return statistic, 1.0
    block_size = int(cfg.get("block_size", perm_cfg.get("block_size", max(8, x.size // 8 or 1))))
    method = str(cfg.get("method", perm_cfg.get("method", "block"))).lower()
    exceed = 0.0
    for _ in range(samples):
        permuted_idx = _block_permutation_indices(x.size, block_size, rng, method=method)
        permuted_y = y[permuted_idx]
        null_value = _distance_correlation_statistic(x, permuted_y)
        if null_value >= statistic:
            exceed += 1.0
    pvalue = (exceed + 1.0) / (samples + 1.0)
    return statistic, float(max(0.0, min(1.0, pvalue)))


# ===== Distance correlation with precomputed target matrix =====

def _distance_correlation_test_precomputed(
    A_x_centered: np.ndarray,
    B_y_centered: np.ndarray,
    cfg: Mapping[str, object],
    rng: Generator,
) -> Tuple[float, float]:
    if A_x_centered.size == 0 or B_y_centered.size == 0:
        return 0.0, 1.0
    n = A_x_centered.shape[0]
    if n < 8:
        return 0.0, 1.0
    statistic = _distance_correlation_statistic_from_centered(A_x_centered, B_y_centered)
    perm_cfg = cfg.get("permutation", {}) if isinstance(cfg, Mapping) else {}
    samples = int(cfg.get("permutations", perm_cfg.get("samples", 64)))
    if samples <= 0:
        return statistic, 1.0
    block_size = int(cfg.get("block_size", perm_cfg.get("block_size", max(8, n // 8 or 1))))
    method = str(cfg.get("method", perm_cfg.get("method", "block"))).lower()
    exceed = 0.0
    for _ in range(samples):
        permuted_idx = _block_permutation_indices(n, block_size, rng, method=method)
        permuted_B = B_y_centered[permuted_idx][:, permuted_idx]
        null_value = _distance_correlation_statistic_from_centered(A_x_centered, permuted_B)
        if null_value >= statistic:
            exceed += 1.0
    pvalue = (exceed + 1.0) / (samples + 1.0)
    return statistic, float(max(0.0, min(1.0, pvalue)))


# ==================== P-value combination & covariance ====================

def _combine_simes(pvalues: Sequence[float]) -> float:
    ordered = sorted(pvalues)
    m = len(ordered)
    combined = min((m / (i + 1)) * p for i, p in enumerate(ordered))
    return float(min(1.0, combined))


def _combine_fisher(pvalues: Sequence[float]) -> float:
    stat = -2.0 * sum(np.log(p if p > 1e-16 else 1e-16) for p in pvalues)
    df = 2.0 * len(pvalues)
    return float(max(0.0, min(1.0, chi2_sf(stat, df))))


def _combine_pvalues(pvalues: Sequence[float], method: str) -> float:
    valid = [p for p in pvalues if 0.0 <= p <= 1.0]
    if not valid:
        return 1.0
    method = (method or "simes").lower()
    if method == "simes":
        return _combine_simes(valid)
    if method == "fisher":
        return _combine_fisher(valid)
    if method == "empirical_brown":
        raise RuntimeError("empirical_brown requires covariance context")
    raise ValueError(f"Unknown combine method: {method}")


def _empirical_brown(pvalues: Sequence[float], covariance: Optional[np.ndarray]) -> float:
    pvals = np.asarray(pvalues, dtype=float)
    k = pvals.size
    if k == 0:
        return 1.0
    if k == 1 or covariance is None:
        return _combine_simes(pvals)
    clipped = np.clip(pvals, 1e-16, 1.0)
    x = -2.0 * np.log(clipped)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (k, k):
        return _combine_simes(pvals)
    cov = np.nan_to_num(cov, nan=0.0)
    mu = 2.0 * k
    ones = np.ones(k)
    var = float(ones @ cov @ ones)
    if var <= 0:
        return _combine_simes(pvals)
    scale = var / (2.0 * mu)
    if scale <= 0:
        return _combine_simes(pvals)
    df = 2.0 * mu * mu / var
    stat = x.sum() / scale
    return float(max(0.0, min(1.0, chi2_sf(stat, df))))


def _estimate_empirical_covariance(
    per_feature_pvals: Mapping[str, Mapping[int, float]],
    test_count: int,
) -> Optional[np.ndarray]:
    if test_count == 0:
        return None
    rows = np.full((len(per_feature_pvals), test_count), np.nan, dtype=np.float32)
    valid_row = np.zeros(len(per_feature_pvals), dtype=bool)
    for row_idx, mapping in enumerate(per_feature_pvals.values()):
        if not mapping:
            continue
        for idx, value in mapping.items():
            if 0 <= idx < test_count:
                rows[row_idx, idx] = -2.0 * np.log(max(float(value), 1e-16))
        valid_row[row_idx] = not np.isnan(rows[row_idx]).all()

    if valid_row.sum() < 2:
        return None

    data = rows[valid_row]
    mask = ~np.isnan(data)
    if mask.sum() < 2:
        return None

    counts_per_test = mask.sum(axis=0)
    means = np.divide(
        np.nansum(data, axis=0, dtype=np.float64),
        np.maximum(counts_per_test, 1),
        out=np.zeros(test_count, dtype=np.float32),
        where=counts_per_test > 0,
    )
    centered = np.where(mask, data - means, 0.0).astype(np.float32, copy=False)
    counts = mask.T @ mask
    covariance = (centered.T @ centered).astype(np.float32, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = np.divide(
            covariance,
            np.maximum(counts - 1, 1),
            out=np.zeros_like(covariance, dtype=np.float32),
            where=counts > 1,
        )

    diag_idx = np.arange(test_count)
    diag_mask = counts[diag_idx, diag_idx] <= 1
    covariance[diag_idx[diag_mask], diag_idx[diag_mask]] = 4.0
    off_diag_mask = counts <= 1
    covariance[off_diag_mask & ~np.eye(test_count, dtype=bool)] = 0.0
    return covariance


# ==================== FDR ====================

def _apply_fdr(pvalues: Mapping[str, float], alpha: float, method: str) -> List[str]:
    if not pvalues:
        return []
    method = method.lower()
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    selected: List[str] = []
    if method == "bh":
        for rank, (feature, pval) in enumerate(items, start=1):
            threshold = alpha * rank / m
            if pval <= threshold:
                selected.append(feature)
        return selected
    if method == "by":
        harmonic = sum(1.0 / i for i in range(1, m + 1))
        for rank, (feature, pval) in enumerate(items, start=1):
            threshold = alpha * rank / (m * harmonic)
            if pval <= threshold:
                selected.append(feature)
        return selected
    raise ValueError(f"Unsupported FDR method: {method}")


# ==================== Lag helpers ====================

def _lagged_pairs(values: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    if lag <= 0:
        return np.empty(0), np.empty(0)
    if values.size <= lag:
        return np.empty(0), np.empty(0)
    x = values[:-lag]
    y = values[lag:]
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 8:
        return np.empty(0), np.empty(0)
    return x[mask].astype(np.float32, copy=False), y[mask].astype(np.float32, copy=False)


# ==================== Core test runner (with scalable backends) ====================

def _run_method_test(
    method_name: str,
    x: np.ndarray,
    y: np.ndarray,
    cfg: Mapping[str, object],
    rng: Generator,
    *,
    fold_ctx: Optional[Mapping[str, object]] = None,
    heavy_switch_n: int = 5000,
) -> Optional[Tuple[float, float, Dict[str, float]]]:
    """Dispatch a single dependence test for one feature.
    New scalable backends:
      - 'hsic_rff' (linear-time) and 'hsic' with backend='auto' → switch to RFF if n >= heavy_switch_n
      - 'dcor_sampled' (subset-based) and 'dcor' with backend='auto' → switch to sampled if n >= heavy_switch_n
    """
    sample_size = x.size
    if sample_size == 0 or y.size != sample_size:
        return None
    method = method_name.lower()
    method_rng = rng  # reuse provided RNG for determinism

    # ---- Spearman ----
    if method in {"spearman", "spearman_rho"}:
        y_ctx = None
        if fold_ctx is not None and "spearman" in fold_ctx:
            y_ctx = fold_ctx["spearman"]  # (y_centered_ranks, y_denom)
        rho = _spearman_corr(x, y, y_ctx=y_ctx)
        pval = _approx_spearman_pvalue(rho, sample_size)
        effect = abs(rho)
        delta = float(abs(cliffs_delta(x, y)))
        if bool(cfg.get("cliffs_delta", True)):
            effect = max(effect, delta)
        return float(effect), float(pval), {"cliffs_delta": delta}

    # ---- Kendall ----
    if method in {"kendall", "kendall_tau"}:
        tau, used_pairs = _kendall_tau(x, y)
        if used_pairs <= 0:
            return None
        pval = _kendall_tau_pvalue(tau, sample_size)
        return float(abs(tau)), float(pval), {}

    # ---- HSIC scalable ----
    if method.startswith("hsic"):
        backend = str(cfg.get("backend", "auto")).lower()
        if backend == "auto":
            backend = "rff" if sample_size >= heavy_switch_n else "matrix"
        if method in {"hsic_rff"} or backend == "rff":
            pre_phi = None
            if fold_ctx is not None and "hsic_rff" in fold_ctx:
                # key by (bw_key, D)
                D = int(cfg.get("rff_dim", 128))
                y_bw = cfg.get("target_bandwidth") or cfg.get("bandwidth", "median")
                bw_key = float(y_bw) if isinstance(y_bw, (int, float)) else str(y_bw)
                pre_phi = fold_ctx["hsic_rff"].get((bw_key, D))
            effect, pval = _hsic_rff_test(x, y, cfg, method_rng, precomputed_phi_y=pre_phi)
            return float(effect), float(pval), {}

        # matrix backend (small n)
        if fold_ctx is not None and "hsic" in fold_ctx:
            y_bw = cfg.get("target_bandwidth") or cfg.get("bandwidth", "median")
            bw_key = float(y_bw) if isinstance(y_bw, (int, float)) else str(y_bw)
            Lc_y = fold_ctx["hsic"].get(bw_key)
            if Lc_y is not None:
                Kc_x = _center_kernel(_rbf_kernel(x, cfg.get("bandwidth", "median")))
                effect, pval = _hsic_test_precomputed(Kc_x, Lc_y, cfg, method_rng)
                return float(effect), float(pval), {}
        effect, pval = _hsic_test(x, y, cfg, method_rng)
        return float(effect), float(pval), {}

    # ---- dCor scalable ----
    if method in {"dcor", "dcor_t", "distance_correlation"}:
        backend = str(cfg.get("backend", "auto")).lower()
        if backend == "auto":
            backend = "sampled" if sample_size >= heavy_switch_n else "matrix"
        if backend == "sampled" or method in {"dcor_sampled"}:
            pre_B = None
            pre_idx = None
            if fold_ctx is not None and "dcor_sampled" in fold_ctx:
                pre = fold_ctx["dcor_sampled"]
                pre_B = pre.get("B_sample")
                pre_idx = pre.get("idx")
            effect, pval = _dcor_sampled_test(x, y, cfg, method_rng, precomputed_BS=pre_B, precomputed_idx=pre_idx)
            return float(effect), float(pval), {}
        if fold_ctx is not None and "dcor" in fold_ctx:
            A_x = _center_distance_matrix(_distance_matrix(x))
            effect, pval = _distance_correlation_test_precomputed(A_x, fold_ctx["dcor"]["B"], cfg, method_rng)
            return float(effect), float(pval), {}
        effect, pval = _distance_correlation_test(x, y, cfg, method_rng)
        return float(effect), float(pval), {}

    return None


def _evaluate_screening_methods(
    series: np.ndarray,
    target: Optional[np.ndarray],
    methods: Mapping[str, Mapping[str, object] | object],
    rng: Generator,
    *,
    supervision: str,
    fold_ctx: Optional[Mapping[str, object]] = None,
    heavy_switch_n: int = 5000,
) -> Dict[str, Tuple[float, float, Dict[str, float]]]:
    results: Dict[str, Tuple[float, float, Dict[str, float]]] = {}
    values = _as_float32(series)
    supervised = supervision == "supervised" and target is not None
    if supervised:
        target_array = _as_float32(target)
        mask = np.isfinite(values) & np.isfinite(target_array)
        x = values[mask]
        y = target_array[mask]
        if x.size < 8 or y.size < 8:
            return results
    else:
        x = np.empty(0, dtype=np.float32)
        y = np.empty(0, dtype=np.float32)

    for name, raw_spec in methods.items():
        method_name = str(name).lower()
        if isinstance(raw_spec, Mapping):
            enabled = bool(raw_spec.get("enabled", True))
            cfg = {k: v for k, v in raw_spec.items() if k != "enabled"}
        else:
            enabled = bool(raw_spec)
            cfg = {}
        if not enabled:
            continue

        method_rng = default_rng(rng.integers(2**32))

        if supervised:
            outcome = _run_method_test(method_name, x, y, cfg, method_rng, fold_ctx=fold_ctx, heavy_switch_n=heavy_switch_n)
            if outcome is not None:
                results[method_name] = outcome
            continue

        # Unsupervised (auto_*)
        base_name = method_name
        if base_name.startswith("auto_"):
            base_name = base_name[5:]
        lags_raw = cfg.get("lags")
        if lags_raw is None:
            lag_value = cfg.get("lag", 1)
            if isinstance(lag_value, Iterable) and not isinstance(lag_value, (str, bytes)):
                lags = [int(l) for l in lag_value]
            else:
                lags = [int(lag_value or 1)]
        else:
            lags = [int(l) for l in lags_raw]
        lags = [lag for lag in lags if lag > 0]
        if not lags:
            lags = [1]

        lag_results: List[Tuple[int, float, float, Dict[str, float]]] = []
        for lag in lags:
            pair_x, pair_y = _lagged_pairs(values, lag)
            outcome = _run_method_test(base_name, pair_x, pair_y, cfg, method_rng, heavy_switch_n=heavy_switch_n)
            if outcome is None:
                continue
            effect, pval, extras = outcome
            lag_results.append((lag, effect, pval, extras))
        if not lag_results:
            continue

        combine_method = str(cfg.get("lag_combine", cfg.get("combine_method", "simes"))).lower()
        pvals = [pval for _, _, pval, _ in lag_results]
        if combine_method == "empirical_brown":
            combined_pval = _empirical_brown(pvals, None)
        else:
            combined_pval = _combine_pvalues(pvals, method=combine_method)
        best_lag, best_effect, _, best_extras = max(lag_results, key=lambda item: abs(item[1]))
        delta = max((abs(extra.get("cliffs_delta", 0.0)) for _, _, _, extra in lag_results), default=0.0)
        extras_out: Dict[str, float] = {"best_lag": float(best_lag)}
        if delta > 0.0:
            extras_out["cliffs_delta"] = float(delta)
        results[method_name] = (float(abs(best_effect)), float(combined_pval), extras_out)

    return results


# ==================== Top-level API ====================

def screening_out_of_fold(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: Optional[str],
    cross_splits: Sequence[CrossFitSplit],
    combine_cfg: Mapping[str, str | float],
    min_bars: int,
    *,
    methods: Optional[Mapping[str, Mapping[str, object]]] = None,
    random_state: Optional[int] = None,
    effect_min: float = 0.0,
    cliffs_delta_min: float = 0.0,
    supervision: str = "supervised",
    # ---- New scalability knobs (backward-compatible defaults) ----
    heavy_switch_n: int = 5000,
    hsic_default_backend: str = "auto",      # "auto"|"matrix"|"rff"
    dcor_default_backend: str = "auto",      # "auto"|"matrix"|"sampled"
    hsic_rff_dim: int = 128,
    dcor_sample_size: int = 1024,
    record_details: bool = True,
) -> ScreeningResult:
    """Out-of-fold feature screening with scalable dependence tests.

    Notes for scale:
      - For n >= heavy_switch_n, 'hsic' → RFF backend, 'dcor' → sampled backend by default.
      - RFF dimension (hsic_rff_dim) and dCor sample size (dcor_sample_size) control speed/accuracy tradeoff.
      - Memory is minimized by streaming features (no feature_matrix preallocation).
    """
    combine_method = str(combine_cfg.get("method", "simes")).lower()
    fdr_alpha = float(combine_cfg.get("alpha", 0.1))
    fdr_method = str(combine_cfg.get("fdr", "by")).lower()

    per_feature_effects: Dict[str, Dict[int, float]] = defaultdict(dict)
    per_feature_pvals: Dict[str, Dict[int, float]] = defaultdict(dict)
    details: List[ScreeningDetail] = []
    executed_methods: set[str] = set()
    test_positions: Dict[Tuple[str, str, str], int] = {}
    per_feature_deltas: Dict[str, Dict[int, float]] = defaultdict(dict)

    # Default method spec remains Spearman; scalable backends available by name/config.
    method_spec = methods or {"spearman": {"enabled": True}}
    rng = default_rng(random_state)

    supervision = supervision.lower()
    supervised_mode = supervision == "supervised"

    feature_list = [col for col in feature_columns if col in df.columns]

    # Target as contiguous float32 (view if possible)
    target_array = (
        _as_float32(df[target_column].to_numpy(copy=False))
        if supervised_mode and target_column and target_column in df.columns
        else None
    )

    for split in cross_splits:
        for fold_label, indices in (("A", split.inner_train_a), ("B", split.inner_train_b)):
            if indices.size < min_bars:
                continue

            # ---- Fold-level precomputations reused across features ----
            if supervised_mode and target_array is not None:
                target_values = _as_float32(target_array[indices])  # advanced indexing → copy; minimal size
                # Spearman target ranks context
                ry = _average_ranks(target_values)
                ry_centered = ry - ry.mean(dtype=np.float32)
                denom_y = float(np.sqrt(np.sum(ry_centered * ry_centered, dtype=np.float64)))

                # HSIC: precompute centered L (small-n matrix backend), and/or RFF phi_y (large-n backend)
                hsic_bw_keys: set = set()
                hsic_rff_keys: set = set()
                for mname, spec in method_spec.items():
                    mname = str(mname).lower()
                    if not mname.startswith("hsic"):
                        continue
                    if isinstance(spec, Mapping):
                        y_bw = spec.get("target_bandwidth") or spec.get("bandwidth", "median")
                        backend = str(spec.get("backend", "auto")).lower()
                        D = int(spec.get("rff_dim", hsic_rff_dim))
                    else:
                        y_bw = "median"
                        backend = "auto"
                        D = hsic_rff_dim
                    bw_key = float(y_bw) if isinstance(y_bw, (int, float)) else str(y_bw)
                    if backend in {"auto", "matrix"}:
                        hsic_bw_keys.add(bw_key)
                    if backend in {"auto", "rff"}:
                        hsic_rff_keys.add((bw_key, D))

                hsic_ctx: Dict[object, np.ndarray] = {}
                if len(target_values) < heavy_switch_n and hsic_bw_keys:
                    for key in hsic_bw_keys:
                        L = _rbf_kernel(target_values, key if key != "median" else "median")
                        hsic_ctx[key] = _center_kernel(L)

                hsic_rff_ctx: Dict[tuple, np.ndarray] = {}
                if hsic_rff_keys:
                    # Build phi_y for each (bw, D)
                    for (bw_key, D) in hsic_rff_keys:
                        bw = bw_key
                        sigma_sq = _rbf_bandwidth(target_values, "median" if isinstance(bw_key, str) else float(bw_key))
                        phi_y = _rff_features(target_values, sigma_sq, D, rng)
                        # Store centered phi_y to reuse across permutations
                        phi_yc = phi_y - phi_y.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
                        hsic_rff_ctx[(bw_key, D)] = phi_yc

                # dCor: precompute centered B for target (small-n) or sampled subset (large-n)
                dcor_ctx = {}
                if len(target_values) < heavy_switch_n:
                    By = _center_distance_matrix(_distance_matrix(target_values))
                    dcor_ctx["B"] = By
                else:
                    s = int(next((spec.get("sample_size", dcor_sample_size)
                                  for name, spec in method_spec.items()
                                  if str(name).lower() in {"dcor", "dcor_t", "distance_correlation", "dcor_sampled"}
                                  and isinstance(spec, Mapping)), dcor_sample_size))
                    s = max(32, min(s, len(target_values)))
                    idx_s = rng.choice(len(target_values), size=s, replace=False)
                    ys = target_values[idx_s]
                    B_sample = _center_distance_matrix(_distance_matrix(ys))
                    dcor_ctx["B_sample"] = B_sample
                    dcor_ctx["idx"] = idx_s

                # Bundle
                fold_ctx = {
                    "spearman": (ry_centered, denom_y),
                    "hsic": hsic_ctx,               # matrix backend (small-n)
                    "hsic_rff": hsic_rff_ctx,       # RFF backend (large-n)
                    "dcor": dcor_ctx if "B" in dcor_ctx else {},
                    "dcor_sampled": dcor_ctx if "B_sample" in dcor_ctx else {},
                }
            else:
                target_values = None
                fold_ctx = None

            if not feature_list:
                continue

            # ---- Stream features one-by-one to avoid large matrices in memory ----
            for feature in feature_list:
                series_full = _as_float32(df[feature].to_numpy(copy=False))
                series = series_full[indices]  # copy only fold-length vector

                if supervised_mode and target_values is None:
                    continue

                # Inject default backend choices if user did not set them
                adjusted_methods: Dict[str, Mapping[str, object]] = {}
                for mname, spec in (method_spec.items() if isinstance(method_spec, Mapping) else []):
                    mname_l = str(mname).lower()
                    if isinstance(spec, Mapping):
                        cfg = dict(spec)
                    else:
                        cfg = {"enabled": bool(spec)}
                    if mname_l.startswith("hsic") and "backend" not in cfg:
                        cfg["backend"] = hsic_default_backend
                    if mname_l in {"dcor", "dcor_t", "distance_correlation"} and "backend" not in cfg:
                        cfg["backend"] = dcor_default_backend
                    if mname_l.startswith("hsic") and "rff_dim" not in cfg:
                        cfg["rff_dim"] = hsic_rff_dim
                    if mname_l in {"dcor", "dcor_t", "distance_correlation", "dcor_sampled"} and "sample_size" not in cfg:
                        cfg["sample_size"] = dcor_sample_size
                    adjusted_methods[mname_l] = cfg

                results = _evaluate_screening_methods(
                    series,
                    target_values,
                    adjusted_methods,
                    rng,
                    supervision=supervision,
                    fold_ctx=fold_ctx,
                    heavy_switch_n=heavy_switch_n,
                )
                if not results:
                    continue

                for method_name, (effect, pval, extras) in results.items():
                    test_key = (split.outer.combo_id, fold_label, method_name)
                    test_idx = test_positions.setdefault(test_key, len(test_positions))
                    per_feature_effects[feature][test_idx] = effect
                    per_feature_pvals[feature][test_idx] = pval
                    if "cliffs_delta" in extras:
                        per_feature_deltas[feature][test_idx] = abs(extras["cliffs_delta"])
                    executed_methods.add(method_name)
                    if record_details:
                        details.append(
                            ScreeningDetail(
                                fold_id=f"{split.outer.combo_id}_{fold_label}",
                                feature=feature,
                                method=method_name,
                                effect=effect,
                                pvalue=pval,
                                extras={k: float(v) for k, v in extras.items()},
                            )
                        )

    test_count = len(test_positions)
    covariance = _estimate_empirical_covariance(per_feature_pvals, test_count)

    combined_pvalues: Dict[str, float] = {}
    effect_summary: Dict[str, float] = {}
    delta_summary: Dict[str, float] = {}
    for feature in feature_columns:
        mapping = per_feature_pvals.get(feature, {})
        if not mapping:
            combined_pvalues[feature] = 1.0
            effect_summary[feature] = 0.0
            delta_summary[feature] = 0.0
            continue
        ordered_indices = sorted(mapping.keys())
        pvals = [mapping[idx] for idx in ordered_indices]
        if combine_method == "empirical_brown":
            if covariance is not None:
                sub_cov = covariance[np.ix_(ordered_indices, ordered_indices)]
                combined = _empirical_brown(pvals, sub_cov)
            else:
                combined = _combine_simes(pvals)
        else:
            combined = _combine_pvalues(pvals, method=combine_method)
        combined_pvalues[feature] = combined
        effects_map = per_feature_effects.get(feature, {})
        effect_summary[feature] = float(max(effects_map.values())) if effects_map else 0.0
        delta_map = per_feature_deltas.get(feature, {})
        delta_summary[feature] = float(max(delta_map.values())) if delta_map else 0.0

    selected = _apply_fdr(combined_pvalues, alpha=fdr_alpha, method=fdr_method)
    if effect_min > 0.0 or cliffs_delta_min > 0.0:
        filtered: List[str] = []
        for feature in selected:
            if effect_min > 0.0 and effect_summary.get(feature, 0.0) < effect_min:
                continue
            if cliffs_delta_min > 0.0 and delta_summary.get(feature, 0.0) < cliffs_delta_min:
                continue
            filtered.append(feature)
        selected = filtered

    return ScreeningResult(
        pvalues=combined_pvalues,
        effects=effect_summary,
        cliffs_delta=delta_summary,
        details=details,
        passing_features=selected,
        methods=sorted(executed_methods),
    )


# ==================== Permutation helpers ====================

def _block_permutation_indices(
    length: int,
    block_size: int,
    rng: Generator,
    *,
    method: str,
) -> np.ndarray:
    if length <= 1:
        return np.arange(length, dtype=np.int32)
    block_size = max(1, min(int(block_size), length))
    method = method.lower()
    if method not in {"block", "wild", "permutation"}:
        method = "block"
    if method == "permutation" or block_size == 1 or block_size >= length:
        return rng.permutation(length).astype(np.int32, copy=False)

    block_starts = np.arange(0, length, block_size, dtype=np.int32)
    block_sizes = np.minimum(block_size, length - block_starts)
    base = np.arange(block_size, dtype=np.int32)
    blocks = block_starts[:, None] + base[None, :]
    mask = base[None, :] < block_sizes[:, None]
    order = rng.permutation(block_starts.size)
    ordered_blocks = blocks[order]
    ordered_mask = mask[order]
    flattened = ordered_blocks[ordered_mask]
    if flattened.size < length:
        remainder = np.setdiff1d(np.arange(length, dtype=np.int32), flattened, assume_unique=True)
        flattened = np.concatenate([flattened, remainder])
    return flattened[:length].astype(np.int32, copy=False)
