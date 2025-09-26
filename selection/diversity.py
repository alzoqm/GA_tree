# /selection/diversity.py
"""Diversity-aware selection utilities (Stage 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DiversitySelectionResult:
    method: str
    selected: List[str]
    coverage: Dict[str, float]
    discarded: List[str]
    similarity: str = "corr"
    diagnostics: Dict[str, float] = field(default_factory=dict)


def _family_key(feature: str) -> str:
    if "_" in feature:
        return feature.split("_", 1)[0]
    return feature


def _prepare_matrix(df: pd.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, List[str]]:
    existing = [name for name in features if name in df.columns]
    if not existing:
        return np.empty((0, 0)), []
    subset = df[existing]
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


def _partial_correlation_similarity(matrix: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((0, 0))
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    denom = max(centered.shape[0] - 1, 1)
    covariance = (centered.T @ centered) / denom
    covariance += np.eye(covariance.shape[0]) * jitter
    try:
        precision = np.linalg.pinv(covariance)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * jitter)
    diag = np.diag(precision)
    denom_matrix = np.sqrt(np.outer(diag, diag))
    denom_matrix[denom_matrix <= 1e-12] = 1.0
    partial = -precision / denom_matrix
    np.fill_diagonal(partial, 1.0)
    return np.nan_to_num(np.abs(partial), nan=0.0, posinf=0.0, neginf=0.0)


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

    similarity = _partial_correlation_similarity(matrix)
    if similarity.size == 0:
        similarity = np.eye(len(features), dtype=float)

    weights = np.array([abs(effect_scores.get(name, 0.0)) for name in features], dtype=float)
    if np.all(weights == 0.0):
        weights = np.ones_like(weights)

    limit = len(features) if top_k is None else max(1, min(int(top_k), len(features)))
    selected_mask = np.zeros(len(features), dtype=bool)
    selected_indices: List[int] = []
    coverage = np.zeros(len(features), dtype=float)
    if family_cap is not None:
        families = [_family_key(name) for name in features]
        unique_families = {family: idx for idx, family in enumerate(sorted(set(families)))}
        family_ids = np.array([unique_families[f] for f in families], dtype=np.int16)
        family_counts = np.zeros(len(unique_families), dtype=np.int32)
    else:
        family_ids = None
        family_counts = None

    for _ in range(limit):
        available = ~selected_mask
        if family_cap is not None and family_counts is not None and family_ids is not None:
            available &= family_counts[family_ids] < family_cap
        if not np.any(available):
            break
        diff = np.clip(similarity - coverage[:, None], a_min=0.0, a_max=None)
        gains = diff.T @ weights
        gains = np.asarray(gains, dtype=float)
        gains[~available] = -np.inf
        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]
        if not selected_indices and not np.isfinite(best_gain):
            best_gain = 0.0
        if best_gain <= 1e-10 and selected_indices:
            break
        selected_mask[best_idx] = True
        selected_indices.append(best_idx)
        coverage = np.maximum(coverage, similarity[:, best_idx])
        if family_cap is not None and family_counts is not None and family_ids is not None:
            family_counts[family_ids[best_idx]] += 1

    if not selected_indices:
        # Fallback: pick top feature by weight respecting family cap
        sorted_candidates = sorted(
            range(len(features)),
            key=lambda idx: (weights[idx], similarity[idx].sum()),
            reverse=True,
        )
        for idx in sorted_candidates:
            if family_cap is None:
                selected_indices.append(idx)
                coverage = similarity[:, idx]
                break
            if family_counts is not None and family_ids is not None:
                family_idx = family_ids[idx]
                if family_counts[family_idx] < family_cap:
                    selected_indices.append(idx)
                    family_counts[family_idx] += 1
                    coverage = similarity[:, idx]
                    break

    selected = [features[idx] for idx in selected_indices]
    coverage_map = {features[i]: float(coverage[i]) for i in range(len(features))}
    discarded = [name for name in features if name not in selected]

    diagnostics = {
        "similarity": "partial_correlation",
        "family_cap": float(family_cap) if family_cap is not None else 0.0,
        "top_k": float(limit),
    }

    return DiversitySelectionResult(
        method="facility_location",
        selected=selected,
        coverage=coverage_map,
        discarded=discarded,
        similarity="partial_correlation",
        diagnostics=diagnostics,
    )


def determinantal_point_process_diversity(
    df: pd.DataFrame,
    candidate_features: Sequence[str],
    effect_scores: Mapping[str, float],
    *,
    top_k: Optional[int],
    family_cap: Optional[int],
    lambda_scale: float = 1.0,
    jitter: float = 1e-6,
) -> DiversitySelectionResult:
    """Select a diverse subset using a k-DPP MAP approximation."""

    matrix, features = _prepare_matrix(df, candidate_features)
    if len(features) == 0 or matrix.shape[0] == 0:
        return DiversitySelectionResult(
            method="dpp",
            selected=[],
            coverage={},
            discarded=list(candidate_features),
        )

    similarity = _partial_correlation_similarity(matrix, jitter)
    if similarity.size == 0:
        similarity = np.eye(len(features), dtype=float)

    weights = np.array([abs(effect_scores.get(name, 0.0)) for name in features], dtype=float)
    if np.all(weights <= 0.0):
        weights = np.ones_like(weights)
    else:
        weights = weights / np.max(weights)

    kernel = np.exp(-lambda_scale * (1.0 - similarity))
    kernel = np.multiply(kernel, np.outer(weights, weights))
    kernel += np.eye(len(features)) * jitter

    limit = len(features) if top_k is None else max(1, min(int(top_k), len(features)))
    families = [_family_key(name) for name in features] if family_cap is not None else None
    if families is not None:
        unique_families = {family: idx for idx, family in enumerate(sorted(set(families)))}
        family_ids = np.array([unique_families[f] for f in families], dtype=np.int16)
        family_counts_arr = np.zeros(len(unique_families), dtype=np.int32)
    else:
        family_ids = None
        family_counts_arr = None

    selected_indices: List[int] = []
    candidates = list(range(len(features)))

    def _log_det(indices: List[int]) -> float:
        if not indices:
            return 0.0
        submatrix = kernel[np.ix_(indices, indices)]
        sign, logdet = np.linalg.slogdet(submatrix)
        if sign <= 0 or not np.isfinite(logdet):
            return -np.inf
        return float(logdet)

    try:
        while candidates and len(selected_indices) < limit:
            best_idx = None
            best_score = -np.inf
            for idx in candidates:
                if idx in selected_indices:
                    continue
                if family_cap is not None and family_counts_arr is not None and family_ids is not None:
                    if family_counts_arr[family_ids[idx]] >= family_cap:
                        continue
                score = _log_det(selected_indices + [idx])
                if score > best_score + 1e-12:
                    best_idx = idx
                    best_score = score
            if best_idx is None:
                break
            selected_indices.append(best_idx)
            if family_cap is not None and family_counts_arr is not None and family_ids is not None:
                family_counts_arr[family_ids[best_idx]] += 1
            candidates = [i for i in candidates if i != best_idx]
    except np.linalg.LinAlgError:
        selected_indices = []

    if not selected_indices:
        fallback = facility_location_diversity(
            df,
            candidate_features,
            effect_scores,
            top_k=top_k,
            family_cap=family_cap,
        )
        return DiversitySelectionResult(
            method="dpp_fallback_facility_location",
            selected=fallback.selected,
            coverage=fallback.coverage,
            discarded=fallback.discarded,
            similarity=fallback.similarity,
            diagnostics=fallback.diagnostics,
        )

    coverage_vector = np.zeros(len(features), dtype=float)
    for idx in selected_indices:
        coverage_vector = np.maximum(coverage_vector, similarity[:, idx])

    selected = [features[idx] for idx in selected_indices]
    coverage_map = {features[i]: float(coverage_vector[i]) for i in range(len(features))}
    discarded = [name for name in features if name not in selected]

    diagnostics = {
        "similarity": "partial_correlation",
        "lambda_scale": lambda_scale,
        "jitter": jitter,
        "family_cap": float(family_cap) if family_cap is not None else 0.0,
        "top_k": float(limit),
    }

    return DiversitySelectionResult(
        method="dpp",
        selected=selected,
        coverage=coverage_map,
        discarded=discarded,
        similarity="partial_correlation",
        diagnostics=diagnostics,
    )
