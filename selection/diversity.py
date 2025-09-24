"""Diversity-aware selection utilities (Stage 2)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class DiversitySelectionResult:
    method: str
    selected: List[str]
    coverage: Dict[str, float]
    discarded: List[str]


def _family_key(feature: str) -> str:
    if "_" in feature:
        return feature.split("_", 1)[0]
    return feature


def _prepare_matrix(df: pd.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, List[str]]:
    usable: List[str] = []
    columns: List[np.ndarray] = []
    for feature in features:
        if feature not in df.columns:
            continue
        series = df[feature].to_numpy(copy=False)
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


def facility_location_diversity(
    df: pd.DataFrame,
    candidate_features: Sequence[str],
    effect_scores: Mapping[str, float],
    *,
    top_k: Optional[int],
    family_cap: Optional[int],
) -> DiversitySelectionResult:
    """Select a diverse subset of features via facility-location objective."""

    matrix, features = _prepare_matrix(df, candidate_features)
    if len(features) == 0 or matrix.shape[0] == 0:
        return DiversitySelectionResult(method="facility_location", selected=[], coverage={}, discarded=list(candidate_features))

    correlation = np.corrcoef(matrix, rowvar=False)
    if np.isscalar(correlation):
        similarity = np.array([[float(abs(correlation))]], dtype=float)
    else:
        similarity = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(similarity, 1.0)

    weights = np.array([abs(effect_scores.get(name, 0.0)) for name in features], dtype=float)
    if np.all(weights == 0.0):
        weights = np.ones_like(weights)

    limit = len(features) if top_k is None else max(1, min(int(top_k), len(features)))
    selected_indices: List[int] = []
    coverage = np.zeros(len(features), dtype=float)
    family_counts: Dict[str, int] = defaultdict(int)

    for _ in range(limit):
        best_idx = None
        best_gain = -np.inf
        baseline_gain = -np.inf
        for idx in range(len(features)):
            if idx in selected_indices:
                continue
            if family_cap is not None and family_counts[_family_key(features[idx])] >= family_cap:
                continue
            candidate_cover = np.maximum(coverage, similarity[:, idx])
            gain = float(np.sum(weights * (candidate_cover - coverage)))
            if gain > best_gain + 1e-12:
                best_idx = idx
                best_gain = gain
            baseline_gain = max(baseline_gain, gain)

        if best_idx is None:
            break
        # Allow selecting the first candidate even if gain is non-positive (ensures non-empty set)
        if best_gain <= 1e-10 and selected_indices:
            break
        selected_indices.append(best_idx)
        coverage = np.maximum(coverage, similarity[:, best_idx])
        if family_cap is not None:
            family_counts[_family_key(features[best_idx])] += 1

    if not selected_indices:
        # Fallback: pick top feature by weight respecting family cap
        sorted_candidates = sorted(
            range(len(features)),
            key=lambda idx: (weights[idx], similarity[idx].sum()),
            reverse=True,
        )
        for idx in sorted_candidates:
            if family_cap is None or family_counts[_family_key(features[idx])] < family_cap:
                selected_indices.append(idx)
                if family_cap is not None:
                    family_counts[_family_key(features[idx])] += 1
                coverage = similarity[:, idx]
                break

    selected = [features[idx] for idx in selected_indices]
    coverage_map = {features[i]: float(coverage[i]) for i in range(len(features))}
    discarded = [name for name in features if name not in selected]

    return DiversitySelectionResult(
        method="facility_location",
        selected=selected,
        coverage=coverage_map,
        discarded=discarded,
    )
