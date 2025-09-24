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
) -> Tuple[List[str], List[str]]:
    sorted_candidates = sorted(
        (f for f in candidate_features if f in df.columns),
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
            corr = df[[feature, kept]].corr().iloc[0, 1]
            if abs(corr) >= corr_cap:
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
