# /selection/selector.py
"""Representative feature selection and collinearity checks (Stage 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

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
    """Configuration parameters for targetless stability selection."""

    subsamples: int
    sample_fraction: float
    cutoff: float
    max_features: Optional[int]
    random_state: Optional[int]


@dataclass
class StabilitySelectionOutcome:
    """Result diagnostics for unsupervised stability selection."""

    frequencies: Dict[str, float]
    subsamples: int
    cutoff: float
    selected_features: List[str]
    support_counts: Dict[str, int]


def _prepare_feature_matrix(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> tuple[np.ndarray, List[str]]:
    existing = [name for name in features if name in frame.columns]
    if not existing:
        return np.empty((0, 0)), []
    subset = frame[existing]
    non_nan = subset.notna().any(axis=0)
    subset = subset.loc[:, non_nan]
    usable = subset.columns.tolist()
    if not usable:
        return np.empty((0, 0)), []
    matrix = subset.to_numpy(dtype=float, copy=False)
    mask = np.all(np.isfinite(matrix), axis=1)
    if mask.any():
        matrix = matrix[mask]
    else:
        matrix = np.empty((0, len(usable)), dtype=float)
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


def greedy_representative_selection(
    df: pd.DataFrame,
    candidate_features: Sequence[str],
    effect_scores: Mapping[str, float] | None,
    *,
    corr_cap: float = 0.9,
    max_features: Optional[int] = None,
    partial_alpha: float = 0.01,
) -> tuple[List[str], List[str]]:
    """Select a representative, low-correlation subset of features."""

    scores = effect_scores or {}
    feature_matrix, usable = _prepare_feature_matrix(df, candidate_features)
    index_map = {name: idx for idx, name in enumerate(usable)}
    partial_matrix: Optional[np.ndarray]
    if usable:
        partial_matrix = _partial_correlation_matrix(feature_matrix, alpha=partial_alpha)
    else:
        partial_matrix = None

    sorted_candidates = sorted(
        (f for f in candidate_features if f in index_map),
        key=lambda name: (scores.get(name, 0.0), name),
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
                pair = df[[feature, kept]].dropna()
                if pair.shape[0] == 0:
                    continue
                similarity = pair.corr().iloc[0, 1]
            if np.isnan(similarity):
                continue
            if abs(similarity) >= corr_cap:
                accept = False
                discarded.append(feature)
                break
        if accept:
            selected.append(feature)
    return selected, discarded


def _compute_vif(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.shape[1] == 0:
        return {}
    values = frame.to_numpy(dtype=float, copy=False)
    mask = np.all(np.isfinite(values), axis=1)
    values = values[mask]
    if values.shape[0] <= 1:
        return {col: 1.0 for col in frame.columns}
    if values.shape[1] == 1:
        return {frame.columns[0]: 1.0}
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        inv_corr = np.linalg.pinv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr + np.eye(corr.shape[0]) * 1e-6)
    return {col: float(inv_corr[i, i]) for i, col in enumerate(frame.columns)}


def _target_partial_r2(target: np.ndarray, features: pd.DataFrame, feature_name: str) -> float:
    if features.shape[1] <= 1:
        series = features[feature_name].to_numpy(copy=False)
        if series.std() < 1e-8:
            return 0.0
        corr = np.corrcoef(series, target)[0, 1]
        return float(corr**2)

    other_cols = [c for c in features.columns if c != feature_name]
    if not other_cols:
        return 0.0
    x_feature = features[feature_name].to_numpy(copy=False)
    others = features[other_cols].to_numpy(copy=False)
    others = np.column_stack([np.ones(others.shape[0]), others])
    beta_y, *_ = np.linalg.lstsq(others, target, rcond=None)
    beta_x, *_ = np.linalg.lstsq(others, x_feature, rcond=None)
    resid_y = target - others @ beta_y
    resid_x = x_feature - others @ beta_x
    if resid_y.std() < 1e-8 or resid_x.std() < 1e-8:
        return 0.0
    corr = np.corrcoef(resid_x, resid_y)[0, 1]
    return float(corr**2)


def _self_r2(features: pd.DataFrame, feature_name: str) -> float:
    other_cols = [c for c in features.columns if c != feature_name]
    if not other_cols:
        return 0.0
    y = features[feature_name].to_numpy(copy=False).astype(float)
    X = features[other_cols].to_numpy(copy=False).astype(float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < max(8, X.shape[1] + 1):
        return 0.0
    y = y[mask]
    X = X[mask]
    X = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_total = float(np.sum((y - y.mean()) ** 2))
    if ss_total <= 1e-12:
        return 0.0
    ss_res = float(np.sum((y - pred) ** 2))
    value = 1.0 - ss_res / ss_total
    return float(max(0.0, min(1.0, value)))


def enforce_vif_and_partial_r2(
    df: pd.DataFrame,
    target_column: Optional[str],
    features: Sequence[str],
    vif_threshold: float,
    *,
    use_target_for_partial: bool = False,
) -> tuple[List[str], SelectionDiagnostics]:
    working = list(features)
    discarded_vif: List[str] = []
    final_vif: Dict[str, float] = {}
    partial_scores: Dict[str, float] = {}

    while working:
        frame = df[working]
        vif_scores = _compute_vif(frame)
        over_limit = [f for f, value in vif_scores.items() if value > vif_threshold]
        if not over_limit:
            final_vif = vif_scores
            break
        drop_feature = max(over_limit, key=lambda f: vif_scores.get(f, 0.0))
        discarded_vif.append(drop_feature)
        working.remove(drop_feature)

    if working:
        feature_frame = df[working]
        final_vif = _compute_vif(feature_frame)
        if use_target_for_partial and target_column and target_column in df.columns:
            target = df[target_column].to_numpy(copy=False).astype(float)
            mask = np.isfinite(target)
            target = target[mask]
            aligned = feature_frame.iloc[mask]
            for feature in working:
                partial_scores[feature] = _target_partial_r2(target, aligned, feature)
        else:
            for feature in working:
                partial_scores[feature] = _self_r2(feature_frame, feature)

    diagnostics = SelectionDiagnostics(
        vif=final_vif,
        partial_r2=partial_scores,
        discarded_high_vif=discarded_vif,
        discarded_high_corr=[f for f in features if f not in working and f not in discarded_vif],
    )
    return working, diagnostics


def unsupervised_stability_selection(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    effect_scores: Mapping[str, float] | None,
    subsamples: int,
    sample_fraction: float,
    cutoff: float,
    corr_cap: float,
    max_features: Optional[int],
    partial_alpha: float,
    random_state: Optional[int],
) -> StabilitySelectionOutcome:
    if not features or frame.empty:
        return StabilitySelectionOutcome(
            frequencies={},
            subsamples=0,
            cutoff=cutoff,
            selected_features=[],
            support_counts={},
        )

    subsamples = max(1, int(subsamples))
    sample_fraction = float(np.clip(sample_fraction, 0.1, 1.0))
    n_samples = frame.shape[0]
    sample_size = max(8, int(round(n_samples * sample_fraction)))
    sample_size = min(sample_size, n_samples)
    rng = np.random.default_rng(random_state)
    counts: Dict[str, int] = {name: 0 for name in features}

    for draw in range(subsamples):
        if sample_size >= n_samples:
            indices = np.arange(n_samples)
        else:
            indices = rng.choice(n_samples, size=sample_size, replace=False)
        subset = frame.iloc[indices]
        selected, _ = greedy_representative_selection(
            subset,
            features,
            effect_scores,
            corr_cap=corr_cap,
            max_features=max_features,
            partial_alpha=partial_alpha,
        )
        for name in selected:
            counts[name] = counts.get(name, 0) + 1

    frequencies = {
        name: counts.get(name, 0) / subsamples for name in features
    }
    sorted_features = sorted(
        features,
        key=lambda name: (frequencies.get(name, 0.0), (effect_scores or {}).get(name, 0.0)),
        reverse=True,
    )
    selected = [name for name in sorted_features if frequencies.get(name, 0.0) >= cutoff]
    limit = max_features or len(sorted_features)
    if not selected:
        selected = sorted_features[:limit]
    elif len(selected) > limit:
        selected = selected[:limit]

    return StabilitySelectionOutcome(
        frequencies=frequencies,
        subsamples=subsamples,
        cutoff=cutoff,
        selected_features=selected,
        support_counts=counts,
    )
