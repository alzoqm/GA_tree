"""Range calibration and boolean handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class CalibrationResult:
    feature_num: Dict[str, tuple[float, float]]
    feature_bool: List[str]
    feature_comparison_map: Dict[str, List[str]]
    diagnostics: Dict[str, float]


def calibrate_ranges(
    df: pd.DataFrame,
    features: Sequence[str],
    quantiles: tuple[float, float] = (0.005, 0.995),
    mad_factor: float | None = 1.4826,
) -> CalibrationResult:
    feature_num: Dict[str, tuple[float, float]] = {}
    feature_bool: List[str] = []
    diagnostics: Dict[str, float] = {}

    q_low, q_high = quantiles

    for feature in features:
        series = df[feature].dropna()
        if series.empty:
            continue
        if pd.api.types.is_bool_dtype(series) or set(series.unique()).issubset({0, 1}):
            feature_bool.append(feature)
            continue
        low = float(series.quantile(q_low))
        high = float(series.quantile(q_high))
        if mad_factor is not None and mad_factor > 0:
            mad = float(np.median(np.abs(series - series.median())))
            diagnostics[f"mad_{feature}"] = mad
            margin = mad_factor * mad
        else:
            margin = 0.0
        feature_num[feature] = (low - margin, high + margin)

    return CalibrationResult(
        feature_num=feature_num,
        feature_bool=feature_bool,
        feature_comparison_map={},
        diagnostics=diagnostics,
    )
