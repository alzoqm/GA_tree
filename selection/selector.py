"""Representative feature selection and collinearity checks (Stage 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SelectionDiagnostics:
    vif: Dict[str, float]
    partial_r2: Dict[str, float]
    discarded_high_vif: List[str]
    discarded_high_corr: List[str]


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
