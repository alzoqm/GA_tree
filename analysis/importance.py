from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from fitness.simulator import FitnessSimulator


def quantile_binned_curve(feature_values: pd.Series,
                          returns_matrix: np.ndarray,
                          q: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Compute average step-returns per feature quantile bin.

    Args:
        feature_values: pd.Series aligned to OOS steps (length T)
        returns_matrix: np.ndarray with shape [T, B] (step returns for B individuals)
        q: number of quantile bins
    Returns:
        bin_centers: np.ndarray [q]
        mean_returns: np.ndarray [q] averaged across steps (and individuals via mean over B)
    """
    if len(feature_values) == 0 or returns_matrix.size == 0:
        return np.array([]), np.array([])
    T = returns_matrix.shape[0]
    feat = feature_values.iloc[:T].values.astype(float)
    # Per-step mean return across individuals
    step_mean = returns_matrix.mean(axis=1)
    qs = np.linspace(0, 1, q + 1)
    bins = np.quantile(feat, qs)
    # Avoid identical edges due to constant features
    bins = np.unique(bins)
    if len(bins) < 2:
        return np.array([feat.mean()]), np.array([step_mean.mean()])
    inds = np.digitize(feat, bins[1:-1], right=True)
    bin_values = []
    bin_means = []
    for bi in range(len(bins) - 1):
        m = inds == bi
        if np.any(m):
            bin_values.append((bins[bi] + bins[bi + 1]) / 2.0)
            bin_means.append(step_mean[m].mean())
    return np.array(bin_values), np.array(bin_means)


def permutation_importance(population,
                           data: pd.DataFrame,
                           start: int,
                           stop: int,
                           device: str,
                           evaluation_config: dict,
                           feature_names: List[str],
                           base_metric_index: int = 0) -> Dict[str, float]:
    """Lightweight permutation importance on OOS slice.

    Args:
        population: GATreePop with N individuals
        data: DataFrame with full features
        start, stop: OOS indices
        device: device string
        evaluation_config: evaluation config used for simulator
        feature_names: ordered list aligning to data columns used in simulator
        base_metric_index: which objective index to measure degradation on
    Returns:
        importance dict mapping feature->delta (positive drop means important)
    """
    sim = FitnessSimulator()
    # Baseline
    baseline = sim.simulate(population=population,
                            data=data,
                            all_feature_names=feature_names,
                            start_data_cnt=start,
                            stop_data_cnt=stop,
                            device=device,
                            evaluation_config=evaluation_config)
    # Mean across individuals for the chosen metric
    base_score = float(np.mean(baseline[:, base_metric_index])) if baseline.size else 0.0

    imp: Dict[str, float] = {}
    # Work on a copy per feature
    for fname in feature_names:
        tmp = data.copy(deep=True)
        # Shuffle only within OOS slice to avoid leakage
        col = tmp[fname].iloc[start:stop].values
        rng = np.random.default_rng(123)
        rng.shuffle(col)
        tmp[fname].iloc[start:stop] = col
        perturbed = sim.simulate(population=population,
                                 data=tmp,
                                 all_feature_names=feature_names,
                                 start_data_cnt=start,
                                 stop_data_cnt=stop,
                                 device=device,
                                 evaluation_config=evaluation_config)
        score = float(np.mean(perturbed[:, base_metric_index])) if perturbed.size else 0.0
        imp[fname] = base_score - score
    return imp

