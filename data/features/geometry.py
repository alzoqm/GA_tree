# /data/features/geometry.py
"""Shape-analysis feature generators."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .utils import (
    EPS,
    format_col_name,
    _compute_close_shape_features,
    _prepare_prototype_library,
)


def calculate_close_geometry_features(
    df: pd.DataFrame,
    window: int,
    resample_points: int = 16,
    prototype_library: Mapping[str, Sequence[float]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add geometric descriptors for close-price trajectories."""
    if 'Close' not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column.")
    if window < 3:
        raise ValueError("window must be >= 3 to describe shape geometry.")
    if resample_points < 2:
        raise ValueError("resample_points must be >= 2.")

    proto_vectors = _prepare_prototype_library(prototype_library, resample_points)

    base_metrics = [
        'SlopeDeg',
        'Area',
        'Runup',
        'Drawdown',
        'Symmetry',
    ]

    column_map = {
        metric: format_col_name('CloseGeom', window, metric) for metric in base_metrics
    }
    for name in proto_vectors:
        metric = f'ProtoSim_{name}'
        column_map[metric] = format_col_name('CloseGeom', window, metric)

    arrays = {col: np.full(df.shape[0], np.nan, dtype=np.float64) for col in column_map.values()}
    close_values = df['Close'].astype(np.float64).values

    for end_idx in range(window - 1, close_values.size):
        segment = close_values[end_idx - window + 1:end_idx + 1]
        feature_values = _compute_close_shape_features(segment, resample_points, proto_vectors)
        for metric_name, value in feature_values.items():
            col = column_map.get(metric_name)
            if col is None:
                continue
            arrays[col][end_idx] = value

    for col_name, values in arrays.items():
        df[col_name] = values

    return df, list(arrays.keys())


__all__ = ['calculate_close_geometry_features']
