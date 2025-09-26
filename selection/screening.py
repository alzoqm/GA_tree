# /selection/screening.py
"""Stage 1 screening utilities (rank tests + FDR)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import erf, sqrt
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng

from .splits import CrossFitSplit
from .math_utils import chi2_sf
from .utils import cliffs_delta


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


def _average_ranks(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
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


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return 0.0
    rx = _average_ranks(np.asarray(x, dtype=np.float32))
    ry = _average_ranks(np.asarray(y, dtype=np.float32))
    x_centered = rx - rx.mean(dtype=np.float32)
    y_centered = ry - ry.mean(dtype=np.float32)
    denom = np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered))
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


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = x.size
    if n <= 1:
        return 0.0, 0
    concordant = discordant = 0
    for i in range(n - 1):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n):
            dx = xi - x[j]
            dy = yi - y[j]
            if dx == 0 and dy == 0:
                continue
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 0.0, 0
    num = concordant - discordant
    counts_x = np.unique(x, return_counts=True)[1]
    counts_y = np.unique(y, return_counts=True)[1]
    tie_x_pairs = sum(c * (c - 1) // 2 for c in counts_x if c > 1)
    tie_y_pairs = sum(c * (c - 1) // 2 for c in counts_y if c > 1)
    den_x = total_pairs - tie_x_pairs
    den_y = total_pairs - tie_y_pairs
    denom = sqrt(max(den_x, 1) * max(den_y, 1))
    if denom == 0:
        return 0.0, concordant + discordant
    tau = num / denom
    return float(max(-1.0, min(1.0, tau))), concordant + discordant


def _kendall_tau_pvalue(tau: float, n: int) -> float:
    if n <= 1:
        return 1.0
    var = (4.0 * n + 10.0) / (9.0 * n * (n - 1))
    if var <= 0:
        return 1.0
    z = tau / sqrt(var)
    return float(max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z))))))


def _rbf_kernel(values: np.ndarray, bandwidth: float | str | None) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1, 1)
    if vector.size == 0:
        return np.empty((0, 0), dtype=float)
    diffs = vector - vector.T
    sq_dists = diffs * diffs
    if bandwidth is None or bandwidth == "median":
        # Median heuristic on squared distances excluding zeros
        flattened = sq_dists[np.triu_indices_from(sq_dists, k=1)]
        positive = flattened[flattened > 0]
        if positive.size:
            sigma_sq = float(np.median(positive))
        else:
            sigma_sq = float(np.var(vector))
        if not np.isfinite(sigma_sq) or sigma_sq <= 1e-12:
            sigma_sq = 1.0
    else:
        sigma_sq = float(bandwidth)
        if sigma_sq <= 1e-12:
            sigma_sq = 1.0
    gamma = 1.0 / (2.0 * sigma_sq)
    return np.exp(-gamma * sq_dists)


def _center_kernel(kernel: np.ndarray) -> np.ndarray:
    if kernel.size == 0:
        return kernel
    n = kernel.shape[0]
    ones = np.ones((n, 1), dtype=float)
    H = np.eye(n) - ones @ ones.T / n
    return H @ kernel @ H


def _hsic_statistic(x: np.ndarray, y: np.ndarray, cfg: Mapping[str, object]) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x_kernel = _rbf_kernel(x, cfg.get("bandwidth", "median"))
    y_bandwidth = cfg.get("target_bandwidth")
    if y_bandwidth is None:
        y_bandwidth = cfg.get("bandwidth", "median")
    y_kernel = _rbf_kernel(y, y_bandwidth)
    Kc = _center_kernel(x_kernel)
    Lc = _center_kernel(y_kernel)
    n = float(x.size)
    if n <= 1:
        return 0.0
    value = np.sum(Kc * Lc) / ((n - 1.0) ** 2)
    return float(max(value, 0.0))


def _block_permutation_indices(
    length: int,
    block_size: int,
    rng: Generator,
    *,
    method: str,
) -> np.ndarray:
    if length <= 1:
        return np.arange(length)
    block_size = max(1, min(int(block_size), length))
    method = method.lower()
    if method not in {"block", "wild", "permutation"}:
        method = "block"
    if method == "permutation" or block_size == 1 or block_size >= length:
        return rng.permutation(length)

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
    return flattened[:length]


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
        resample_rng = default_rng(rng.integers(2**32))
        permuted_idx = _block_permutation_indices(x.size, block_size, resample_rng, method=method)
        permuted_y = y[permuted_idx]
        null_value = _hsic_statistic(x, permuted_y, cfg)
        if null_value >= statistic:
            exceed += 1.0
    pvalue = (exceed + 1.0) / (samples + 1.0)
    return statistic, float(max(0.0, min(1.0, pvalue)))


def _distance_matrix(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1, 1)
    return np.abs(vector - vector.T)


def _center_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    row_mean = matrix.mean(axis=1, keepdims=True)
    col_mean = matrix.mean(axis=0, keepdims=True)
    total_mean = matrix.mean()
    return matrix - row_mean - col_mean + total_mean


def _distance_correlation_statistic(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    A = _center_distance_matrix(_distance_matrix(x))
    B = _center_distance_matrix(_distance_matrix(y))
    n = float(x.size)
    if n <= 0:
        return 0.0
    cov = np.sum(A * B) / (n * n)
    var_x = np.sum(A * A) / (n * n)
    var_y = np.sum(B * B) / (n * n)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    stat = cov / np.sqrt(var_x * var_y)
    return float(max(0.0, min(1.0, stat)))


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
        resample_rng = default_rng(rng.integers(2**32))
        permuted_idx = _block_permutation_indices(x.size, block_size, resample_rng, method=method)
        permuted_y = y[permuted_idx]
        null_value = _distance_correlation_statistic(x, permuted_y)
        if null_value >= statistic:
            exceed += 1.0
    pvalue = (exceed + 1.0) / (samples + 1.0)
    return statistic, float(max(0.0, min(1.0, pvalue)))


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
    # Variance of the sum using covariance matrix
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
    return x[mask].astype(float), y[mask].astype(float)


def _run_method_test(
    method_name: str,
    x: np.ndarray,
    y: np.ndarray,
    cfg: Mapping[str, object],
    rng: Generator,
) -> Optional[Tuple[float, float, Dict[str, float]]]:
    sample_size = x.size
    if sample_size == 0 or y.size != sample_size:
        return None
    method = method_name.lower()
    method_rng = default_rng(rng.integers(2**32))
    if method in {"spearman", "spearman_rho"}:
        rho = _spearman_corr(x, y)
        pval = _approx_spearman_pvalue(rho, sample_size)
        effect = abs(rho)
        delta = float(abs(cliffs_delta(x, y)))
        if bool(cfg.get("cliffs_delta", True)):
            effect = max(effect, delta)
        return float(effect), float(pval), {"cliffs_delta": delta}
    if method in {"kendall", "kendall_tau"}:
        tau, used_pairs = _kendall_tau(x, y)
        if used_pairs <= 0:
            return None
        pval = _kendall_tau_pvalue(tau, sample_size)
        return float(abs(tau)), float(pval), {}
    if method.startswith("hsic"):
        effect, pval = _hsic_test(x, y, cfg, method_rng)
        return float(effect), float(pval), {}
    if method in {"dcor", "dcor_t", "distance_correlation"}:
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
) -> Dict[str, Tuple[float, float, Dict[str, float]]]:
    results: Dict[str, Tuple[float, float, Dict[str, float]]] = {}
    values = np.asarray(series, dtype=np.float32)
    supervised = supervision == "supervised" and target is not None
    if supervised:
        target_array = np.asarray(target, dtype=np.float32)
        mask = np.isfinite(values) & np.isfinite(target_array)
        x = values[mask]
        y = target_array[mask]
        if x.size < 8 or y.size < 8:
            return results
    else:
        x = np.empty(0)
        y = np.empty(0)
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
            outcome = _run_method_test(method_name, x, y, cfg, method_rng)
            if outcome is not None:
                results[method_name] = outcome
            continue

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
            outcome = _run_method_test(base_name, pair_x, pair_y, cfg, method_rng)
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
        best_lag, best_effect, _, best_extras = max(
            lag_results,
            key=lambda item: abs(item[1]),
        )
        delta = max((abs(extra.get("cliffs_delta", 0.0)) for _, _, _, extra in lag_results), default=0.0)
        extras_out: Dict[str, float] = {"best_lag": float(best_lag)}
        if delta > 0.0:
            extras_out["cliffs_delta"] = float(delta)
        results[method_name] = (float(abs(best_effect)), float(combined_pval), extras_out)
    return results


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
) -> ScreeningResult:
    combine_method = str(combine_cfg.get("method", "simes")).lower()
    fdr_alpha = float(combine_cfg.get("alpha", 0.1))
    fdr_method = str(combine_cfg.get("fdr", "by")).lower()

    per_feature_effects: Dict[str, Dict[int, float]] = defaultdict(dict)
    per_feature_pvals: Dict[str, Dict[int, float]] = defaultdict(dict)
    details: List[ScreeningDetail] = []
    executed_methods: set[str] = set()
    test_positions: Dict[Tuple[str, str, str], int] = {}
    per_feature_deltas: Dict[str, Dict[int, float]] = defaultdict(dict)

    method_spec = methods or {"spearman": {"enabled": True}}
    rng = default_rng(random_state)

    supervision = supervision.lower()
    supervised_mode = supervision == "supervised"

    feature_list = [col for col in feature_columns if col in df.columns]
    if feature_list:
        feature_matrix = df[feature_list].to_numpy(dtype=np.float32, copy=False)
    else:
        feature_matrix = np.empty((df.shape[0], 0), dtype=np.float32)
    target_array = (
        df[target_column].to_numpy(dtype=np.float32, copy=False)
        if supervised_mode and target_column and target_column in df.columns
        else None
    )

    for split in cross_splits:
        for fold_label, indices in (("A", split.inner_train_a), ("B", split.inner_train_b)):
            if indices.size < min_bars:
                continue

            if supervised_mode and target_array is not None:
                target_values = target_array[indices]
            else:
                target_values = None

            if not feature_list:
                continue

            fold_matrix = feature_matrix[indices]
            for col_idx, feature in enumerate(feature_list):
                series = fold_matrix[:, col_idx]
                if supervised_mode and target_values is None:
                    continue
                results = _evaluate_screening_methods(
                    series,
                    target_values,
                    method_spec,
                    rng,
                    supervision=supervision,
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
    # Zero out features not passing FDR to ensure downstream contracts know.
    return ScreeningResult(
        pvalues=combined_pvalues,
        effects=effect_summary,
        cliffs_delta=delta_summary,
        details=details,
        passing_features=selected,
        methods=sorted(executed_methods),
    )
