"""Boolean event statistics and inference utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .math_utils import chi2_sf
from .utils import block_bootstrap_indices, cliffs_delta


@dataclass
class BooleanFeatureStats:
    feature: str
    counts: Dict[str, int]
    odds_ratio: float
    relative_risk: float
    g_statistic: float
    g_pvalue: float
    fisher_pvalue: float
    cliffs_delta: float
    bootstrap_ci: tuple[float, float]


def _fisher_exact(a: int, b: int, c: int, d: int) -> float:
    total = a + b + c + d
    if total == 0:
        return 1.0
    row1 = a + b
    col1 = a + c
    min_x = max(0, col1 - (total - row1))
    max_x = min(row1, col1)

    def hypergeom(x: int) -> float:
        return math.comb(row1, x) * math.comb(total - row1, col1 - x) / math.comb(total, col1)

    observed = hypergeom(a)
    p_value = 0.0
    for x in range(min_x, max_x + 1):
        prob = hypergeom(x)
        if prob <= observed + 1e-15:
            p_value += prob
    return min(1.0, p_value)


def _g_test(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    total = a + b + c + d
    if total == 0:
        return 0.0, 1.0
    expected = [
        (a + b) * (a + c) / total,
        (a + b) * (b + d) / total,
        (c + d) * (a + c) / total,
        (c + d) * (b + d) / total,
    ]
    observed = [a, b, c, d]
    g = 0.0
    for obs, exp in zip(observed, expected):
        if obs <= 0 or exp <= 0:
            continue
        g += 2.0 * obs * math.log(obs / exp)
    pval = chi2_sf(g, 1.0)
    return g, min(1.0, pval)


def _odds_ratio(a: int, b: int, c: int, d: int) -> float:
    return (a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5))


def _relative_risk(a: int, b: int, c: int, d: int) -> float:
    exposed = a + b
    unexposed = c + d
    if exposed == 0 or unexposed == 0:
        return 1.0
    return ((a + 0.5) / (exposed + 1.0)) / ((c + 0.5) / (unexposed + 1.0))


def compute_boolean_feature_stats(
    df: pd.DataFrame,
    target: pd.Series,
    features: Iterable[str],
    *,
    bootstrap: Mapping[str, int] | None = None,
    event_threshold: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, BooleanFeatureStats]:
    """Compute boolean-event statistics for the provided feature set."""

    if rng is None:
        rng = np.random.default_rng()
    params = bootstrap or {}
    samples = int(params.get("samples", 200))
    block_size = int(params.get("block_size", 32))
    alpha = float(params.get("alpha", 0.05))
    stats: Dict[str, BooleanFeatureStats] = {}

    indicator = (target > event_threshold).astype(int)
    for feature in features:
        aligned = df[[feature]].join(indicator.rename("target_indicator"), how="inner")
        values = aligned[feature].fillna(0).astype(int)
        targets = aligned["target_indicator"].astype(int)
        a = int(((values == 1) & (targets == 1)).sum())
        b = int(((values == 1) & (targets == 0)).sum())
        c = int(((values == 0) & (targets == 1)).sum())
        d = int(((values == 0) & (targets == 0)).sum())
        g_stat, g_pval = _g_test(a, b, c, d)
        fisher_p = _fisher_exact(a, b, c, d)
        or_value = _odds_ratio(a, b, c, d)
        rr_value = _relative_risk(a, b, c, d)

        aligned_target = target.loc[aligned.index]
        target_pos = aligned_target[values == 1]
        target_neg = aligned_target[values == 0]
        delta = cliffs_delta(target_pos.to_numpy(), target_neg.to_numpy())

        ci = (0.0, 0.0)
        if samples > 0 and aligned.shape[0] > 0:
            boot_stats = []
            values_np = values.to_numpy()
            target_np = aligned_target.to_numpy()
            for _ in range(samples):
                idx = block_bootstrap_indices(aligned.shape[0], block_size, rng)
                sample_bool = values_np[idx]
                sample_target = target_np[idx]
                pos = sample_target[sample_bool == 1]
                neg = sample_target[sample_bool == 0]
                if pos.size == 0 or neg.size == 0:
                    boot_stats.append(0.0)
                else:
                    boot_stats.append(float(pos.mean() - neg.mean()))
            boot_stats.sort()
            low_idx = max(0, int((alpha / 2) * samples) - 1)
            high_idx = min(samples - 1, int((1 - alpha / 2) * samples) - 1)
            ci = (boot_stats[low_idx], boot_stats[high_idx])

        stats[feature] = BooleanFeatureStats(
            feature=feature,
            counts={"a": a, "b": b, "c": c, "d": d},
            odds_ratio=or_value,
            relative_risk=rr_value,
            g_statistic=g_stat,
            g_pvalue=g_pval,
            fisher_pvalue=fisher_p,
            cliffs_delta=delta,
            bootstrap_ci=ci,
        )
    return stats
