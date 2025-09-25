# /selection/selector.py
"""Representative feature selection and collinearity checks (Stage 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SelectionDiagnostics:
    vif: Dict[str, float]
    partial_r2: Dict[str, float]
    discarded_high_vif: List[str]
    discarded_high_corr: List[str]


@dataclass(frozen=True)
class StabilitySelectionConfig:
    """Configuration parameters for subsampling-based stability selection."""

    subsamples: int
    sample_fraction: float
    cutoff: float
    max_features: Optional[int]
    alpha_grid: Tuple[float, ...]
    random_state: Optional[int]
    max_iter: int
    tolerance: float


@dataclass
class StabilitySelectionOutcome:
    """Result of running stability selection on a single training frame."""

    frequencies: Dict[str, float]
    avg_magnitude: Dict[str, float]
    subsamples: int
    cutoff: float
    selected_features: List[str]
    support_counts: Dict[str, int]


@dataclass(frozen=True)
class KnockoffConfig:
    enabled: bool
    fdr: float
    max_features: Optional[int]
    shrinkage: float
    min_eigenvalue: float
    statistic: str
    random_state: Optional[int]
    generator: str
    fallback: Tuple[str, ...]


@dataclass
class KnockoffOutcome:
    statistics: Dict[str, float]
    threshold: float
    selected_features: List[str]
    discarded: List[str]
    fdr: float
    generator: str


def _soft_threshold(value: float, lam: float) -> float:
    if value > lam:
        return value - lam
    if value < -lam:
        return value + lam
    return 0.0


def _standardize_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized copy of matrix with column means and scales."""

    means = np.nanmean(matrix, axis=0)
    centered = matrix - means
    scales = np.nanstd(centered, axis=0)
    scales[scales == 0.0] = 1.0
    return centered / scales, means, scales


def _lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Simple coordinate descent solver for Lasso with standardized inputs."""

    n_samples, n_features = X.shape
    if n_features == 0:
        return np.zeros(0, dtype=float)
    beta = np.zeros(n_features, dtype=float)
    X_sq = np.sum(X * X, axis=0)
    for _ in range(max_iter):
        beta_prev = beta.copy()
        for j in range(n_features):
            if X_sq[j] <= 1e-12:
                beta[j] = 0.0
                continue
            residual = y - X @ beta + X[:, j] * beta[j]
            rho = float(np.dot(X[:, j], residual) / n_samples)
            beta[j] = _soft_threshold(rho, alpha) / (X_sq[j] / n_samples)
        if np.max(np.abs(beta - beta_prev)) < tol:
            break
    return beta


def _prepare_training_matrix(
    frame: pd.DataFrame,
    target_column: str,
    features: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    usable: List[str] = []
    columns: List[np.ndarray] = []
    for feature in features:
        if feature not in frame.columns:
            continue
        col = frame[feature].to_numpy(copy=False)
        if np.isnan(col).all():
            continue
        columns.append(col.astype(float))
        usable.append(feature)
    if not usable:
        return np.empty((0, 0)), np.empty(0), []
    X = np.column_stack(columns)
    y = frame[target_column].to_numpy(copy=False).astype(float)
    mask = np.isfinite(y)
    mask &= np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    return X, y, usable


def _prepare_feature_matrix(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    usable: List[str] = []
    columns: List[np.ndarray] = []
    for feature in features:
        if feature not in frame.columns:
            continue
        series = frame[feature].to_numpy(copy=False)
        if np.isnan(series).all():
            continue
        columns.append(series.astype(float))
        usable.append(feature)
    if not usable:
        return np.empty((0, 0)), []
    matrix = np.column_stack(columns)
    mask = np.all(np.isfinite(matrix), axis=1)
    matrix = matrix[mask]
    return matrix, usable


def _partial_correlation_matrix(matrix: np.ndarray, alpha: float) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((0, 0))
    n_samples = matrix.shape[0]
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    denom = max(n_samples - 1, 1)
    covariance = (centered.T @ centered) / denom
    covariance += np.eye(covariance.shape[0]) * float(max(alpha, 0.0))
    try:
        precision = np.linalg.pinv(covariance)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * 1e-6)
    diag = np.diag(precision)
    denom_matrix = np.sqrt(np.outer(diag, diag))
    denom_matrix[denom_matrix == 0.0] = 1.0
    partial = -precision / denom_matrix
    np.fill_diagonal(partial, 1.0)
    return partial


def _transform_knockoff_inputs(matrix: np.ndarray, method: str) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    method = str(method or "modelx").lower()
    if method == "copula":
        transformed = np.empty_like(matrix, dtype=float)
        denom = matrix.shape[0] + 1.0
        for col in range(matrix.shape[1]):
            series = pd.Series(matrix[:, col])
            ranks = series.rank(method="average").to_numpy(dtype=float)
            uniform = ranks / denom
            centered = uniform - 0.5
            std = centered.std()
            if std <= 1e-12:
                std = 1.0
            transformed[:, col] = centered / std
        return transformed
    if method == "robust":
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median), axis=0)
        mad[mad < 1e-8] = 1.0
        return (matrix - median) / mad
    if method == "tski":
        robust = _transform_knockoff_inputs(matrix, "robust")
        if robust.shape[0] <= 1:
            return robust
        lagged = np.vstack([robust[0], robust[:-1]])
        return 0.7 * robust + 0.3 * lagged
    return matrix


def stability_selection(
    frame: pd.DataFrame,
    target_column: str,
    features: Sequence[str],
    config: StabilitySelectionConfig,
) -> StabilitySelectionOutcome:
    """Run stability selection using subsampled Lasso fits."""

    X, y, usable = _prepare_training_matrix(frame, target_column, features)
    n_samples = X.shape[0]
    if n_samples == 0 or len(usable) == 0:
        return StabilitySelectionOutcome(
            frequencies={},
            avg_magnitude={},
            subsamples=0,
            cutoff=config.cutoff,
            selected_features=[],
            support_counts={},
        )

    subsamples = max(1, int(config.subsamples))
    sample_fraction = float(np.clip(config.sample_fraction, 0.1, 0.9))
    sample_size = max(16, int(round(n_samples * sample_fraction)))
    rng = np.random.default_rng(config.random_state)
    alpha_grid = tuple(float(a) for a in config.alpha_grid if float(a) > 0)
    if not alpha_grid:
        # Derive a geometric grid from alpha_max
        X_std, _, _ = _standardize_matrix(X)
        y_centered = y - y.mean()
        alpha_max = float(np.max(np.abs(X_std.T @ y_centered)) / max(n_samples, 1))
        alpha_min = alpha_max * 0.05 if alpha_max > 0 else 0.01
        alpha_grid = tuple(np.geomspace(alpha_max, alpha_min, num=5))

    support_counter: MutableMapping[str, int] = {name: 0 for name in usable}
    magnitude_accumulator: MutableMapping[str, float] = {name: 0.0 for name in usable}

    for draw_idx in range(subsamples):
        if n_samples <= sample_size:
            indices = np.arange(n_samples)
        else:
            indices = rng.choice(n_samples, size=sample_size, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
        X_std, _, _ = _standardize_matrix(X_sub)
        y_centered = y_sub - y_sub.mean()
        chosen_beta = None
        chosen_support: Optional[np.ndarray] = None
        chosen_alpha = None

        for alpha in alpha_grid:
            beta = _lasso_coordinate_descent(
                X_std,
                y_centered,
                alpha,
                max_iter=config.max_iter,
                tol=config.tolerance,
            )
            support = np.where(np.abs(beta) > 1e-6)[0]
            if support.size == 0:
                continue
            if config.max_features is not None and support.size > config.max_features:
                # Try a stronger penalty before accepting dense support
                continue
            chosen_beta = beta
            chosen_support = support
            chosen_alpha = alpha
            break

        if chosen_support is None:
            # Fall back to sparsest solution from last iteration (smallest alpha)
            beta = _lasso_coordinate_descent(
                X_std,
                y_centered,
                alpha_grid[-1],
                max_iter=config.max_iter,
                tol=config.tolerance,
            )
            support = np.where(np.abs(beta) > 1e-6)[0]
            if config.max_features is not None and support.size > config.max_features:
                magnitude_order = np.argsort(np.abs(beta))[::-1][: config.max_features]
                support = np.sort(magnitude_order)
            chosen_beta = beta
            chosen_support = support
            chosen_alpha = alpha_grid[-1]

        if chosen_beta is None or chosen_support is None:
            continue

        for idx in chosen_support:
            feature_name = usable[idx]
            support_counter[feature_name] += 1
            magnitude_accumulator[feature_name] += float(abs(chosen_beta[idx]))

    frequencies = {
        feature: counter / subsamples for feature, counter in support_counter.items()
    }
    avg_magnitude = {
        feature: (magnitude_accumulator[feature] / support_counter[feature])
        if support_counter[feature] > 0
        else 0.0
        for feature in usable
    }
    selected = [
        feature
        for feature, freq in frequencies.items()
        if freq >= config.cutoff
    ]
    selected.sort(key=lambda name: (frequencies[name], avg_magnitude.get(name, 0.0)), reverse=True)

    return StabilitySelectionOutcome(
        frequencies=frequencies,
        avg_magnitude=avg_magnitude,
        subsamples=subsamples,
        cutoff=config.cutoff,
        selected_features=selected,
        support_counts=dict(support_counter),
    )


def _compute_knockoff_statistics(
    X: np.ndarray,
    Xk: np.ndarray,
    y: np.ndarray,
    statistic: str,
) -> np.ndarray:
    y_centered = y - y.mean()
    if statistic == "correlation":
        orig = np.array([
            0.0 if X[:, j].std() < 1e-8 else abs(np.corrcoef(X[:, j], y_centered)[0, 1])
            for j in range(X.shape[1])
        ])
        knock = np.array([
            0.0 if Xk[:, j].std() < 1e-8 else abs(np.corrcoef(Xk[:, j], y_centered)[0, 1])
            for j in range(Xk.shape[1])
        ])
        return orig - knock
    raise ValueError(f"Unsupported knockoff statistic: {statistic}")


def _generate_modelx_knockoffs(
    X: np.ndarray,
    *,
    shrinkage: float,
    min_eigenvalue: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = X.shape
    if n_features == 0 or n_samples == 0:
        return X.copy(), X.copy()
    cov = np.cov(X, rowvar=False)
    cov += np.eye(n_features) * shrinkage
    eigvals = np.linalg.eigvalsh(cov)
    min_val = float(np.min(eigvals)) if eigvals.size else 1.0
    s_value = max(min(2.0 * min_val - 1e-6, 1.0), min_eigenvalue)
    S = np.eye(n_features) * s_value
    inv_cov = np.linalg.pinv(cov)
    middle = 2.0 * S - S @ inv_cov @ S
    jitter = 1e-6
    for _ in range(5):
        try:
            C = np.linalg.cholesky(middle + np.eye(n_features) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        C = np.linalg.cholesky(np.eye(n_features) * max(jitter, 1e-3))
    gaussian = rng.standard_normal(size=(n_samples, n_features)) @ C.T
    knockoffs = X - X @ inv_cov @ S + gaussian
    return X, knockoffs


def _knockoff_threshold(statistics: Dict[str, float], fdr: float) -> float:
    if not statistics:
        return float("inf")
    values = sorted({abs(v) for v in statistics.values() if abs(v) > 1e-10})
    if not values:
        return float("inf")
    for t in values:
        numerator = 1 + sum(1 for v in statistics.values() if v <= -t)
        denominator = max(1, sum(1 for v in statistics.values() if v >= t))
        if numerator / denominator <= fdr:
            return t
    return float("inf")


def modelx_knockoffs(
    frame: pd.DataFrame,
    target_column: str,
    features: Sequence[str],
    config: KnockoffConfig,
    *,
    strategy: Optional[str] = None,
) -> KnockoffOutcome:
    X, y, usable = _prepare_training_matrix(frame, target_column, features)
    if X.size == 0 or not usable:
        return KnockoffOutcome(
            statistics={},
            threshold=float("inf"),
            selected_features=[],
            discarded=list(features),
            fdr=config.fdr,
            generator=strategy or config.generator,
        )
    generator = str(strategy or config.generator or "modelx").lower()
    X_transformed = _transform_knockoff_inputs(X, generator)
    X_std, _, _ = _standardize_matrix(X_transformed)
    rng = np.random.default_rng(config.random_state)
    shrinkage = config.shrinkage
    min_eigen = config.min_eigenvalue
    if generator == "robust":
        shrinkage = max(shrinkage, 5e-3)
    elif generator == "tski":
        shrinkage = max(shrinkage, 1e-2)
    _, knockoffs = _generate_modelx_knockoffs(
        X_std,
        shrinkage=shrinkage,
        min_eigenvalue=min_eigen,
        rng=rng,
    )
    stats_array = _compute_knockoff_statistics(X_std, knockoffs, y, config.statistic)
    statistics = {usable[i]: float(stats_array[i]) for i in range(len(usable))}
    threshold = _knockoff_threshold(statistics, config.fdr)
    if np.isinf(threshold):
        ranked = sorted(statistics.items(), key=lambda kv: kv[1], reverse=True)
        limit = config.max_features or len(ranked)
        selected = [name for name, _ in ranked[:limit]]
    else:
        selected = [name for name, value in statistics.items() if value >= threshold]
        selected.sort(key=lambda name: statistics[name], reverse=True)
        if config.max_features is not None:
            selected = selected[: config.max_features]
    discarded = [name for name in features if name not in selected]
    return KnockoffOutcome(
        statistics=statistics,
        threshold=float(threshold),
        selected_features=selected,
        discarded=discarded,
        fdr=config.fdr,
        generator=generator,
    )
def _compute_vif(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.shape[1] == 0:
        return {}
    corr = frame.corr().fillna(0.0)
    try:
        inv_corr = np.linalg.pinv(corr.to_numpy())
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr.to_numpy() + np.eye(corr.shape[0]) * 1e-6)
    vif = {col: float(inv_corr[i, i]) for i, col in enumerate(corr.columns)}
    return vif


def _partial_r2(target: np.ndarray, features: pd.DataFrame, feature_name: str) -> float:
    if features.shape[1] <= 1:
        series = features[feature_name].to_numpy(copy=False)
        if series.std() < 1e-8:
            return 0.0
        corr = np.corrcoef(series, target)[0, 1]
        return float(corr**2)

    other_cols = [c for c in features.columns if c != feature_name]
    if not other_cols:
        return 0.0
    y = target
    x_feature = features[feature_name].to_numpy(copy=False)
    others = features[other_cols].to_numpy(copy=False)
    others = np.column_stack([np.ones(others.shape[0]), others])
    beta_y, *_ = np.linalg.lstsq(others, y, rcond=None)
    beta_x, *_ = np.linalg.lstsq(others, x_feature, rcond=None)
    resid_y = y - others @ beta_y
    resid_x = x_feature - others @ beta_x
    if resid_y.std() < 1e-8 or resid_x.std() < 1e-8:
        return 0.0
    corr = np.corrcoef(resid_x, resid_y)[0, 1]
    return float(corr**2)


def greedy_representative_selection(
    df: pd.DataFrame,
    target_column: str,
    candidate_features: Sequence[str],
    effect_scores: Dict[str, float],
    corr_cap: float = 0.9,
    max_features: int | None = None,
    partial_alpha: float = 0.01,
) -> Tuple[List[str], List[str]]:
    feature_matrix, usable = _prepare_feature_matrix(df, candidate_features)
    index_map = {name: idx for idx, name in enumerate(usable)}
    partial_matrix: Optional[np.ndarray]
    if usable:
        partial_matrix = _partial_correlation_matrix(feature_matrix, alpha=partial_alpha)
    else:
        partial_matrix = None

    sorted_candidates = sorted(
        (f for f in candidate_features if f in index_map),
        key=lambda name: effect_scores.get(name, 0.0),
        reverse=True,
    )
    selected: List[str] = []
    discarded: List[str] = []
    for feature in sorted_candidates:
        if max_features is not None and len(selected) >= max_features:
            break
        accept = True
        for kept in selected:
            if partial_matrix is not None:
                idx_feature = index_map.get(feature)
                idx_kept = index_map.get(kept)
                if idx_feature is None or idx_kept is None:
                    continue
                similarity = partial_matrix[idx_feature, idx_kept]
            else:
                corr_frame = df[[feature, kept]]
                similarity = corr_frame.corr().iloc[0, 1]
            if np.isnan(similarity):
                continue
            if abs(similarity) >= corr_cap:
                accept = False
                discarded.append(feature)
                break
        if accept:
            selected.append(feature)
    return selected, discarded


def enforce_vif_and_partial_r2(
    df: pd.DataFrame,
    target_column: str,
    features: Sequence[str],
    vif_threshold: float,
) -> tuple[List[str], SelectionDiagnostics]:
    working = list(features)
    discarded_vif: List[str] = []
    final_vif: Dict[str, float] = {}
    partial_scores: Dict[str, float] = {}

    while True:
        if not working:
            break
        frame = df[working]
        vif_scores = _compute_vif(frame)
        over_limit = [f for f, value in vif_scores.items() if value > vif_threshold]
        if not over_limit:
            final_vif = vif_scores
            break
        # Drop the feature with highest VIF
        drop_feature = max(over_limit, key=lambda f: vif_scores.get(f, 0.0))
        discarded_vif.append(drop_feature)
        working.remove(drop_feature)

    if working:
        final_vif = _compute_vif(df[working])
        target = df[target_column].to_numpy(copy=False)
        for feature in working:
            partial_scores[feature] = _partial_r2(target, df[working], feature)

    diagnostics = SelectionDiagnostics(
        vif=final_vif,
        partial_r2=partial_scores,
        discarded_high_vif=discarded_vif,
        discarded_high_corr=[f for f in features if f not in working and f not in discarded_vif],
    )
    return working, diagnostics
