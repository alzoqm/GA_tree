# /data/features/utils.py
"""Utility helpers shared across TA-Lib feature modules."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

PERCENT_SCALE = 100.0
EPS = 1e-9


def format_col_name(base_name: str, window: int | None = None, extra_str: str | None = None) -> str:
    """Create consistent column names used across feature modules."""
    name = base_name
    if window:
        name = f"{name}_{window}"
    if extra_str:
        name = f"{name}_{extra_str}"
    return name


def scale_by_reference(
    values: pd.Series | np.ndarray,
    reference: pd.Series | np.ndarray,
    *,
    multiplier: float = 1.0,
    eps: float = EPS,
) -> pd.Series | np.ndarray:
    """Scale a value series relative to the provided reference magnitude."""
    if isinstance(reference, pd.Series):
        denom = reference.abs().clip(lower=eps)
    else:
        denom = np.clip(np.abs(reference), eps, None)
    return (values / denom) * multiplier


def get_ohlcv_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract OHLCV arrays formatted for TA-Lib functions."""
    return (
        df['Open'].values,
        df['High'].values,
        df['Low'].values,
        df['Close'].values,
        df['Volume'].astype(float).values,
    )


def _compute_rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling linear-regression slope."""
    if window is None or window < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)

    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom <= EPS:
        denom = EPS

    def _slope(values: np.ndarray) -> float:
        y = values.astype(np.float64)
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        return cov / denom

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def _slope_to_degrees(slope_series: pd.Series) -> pd.Series:
    """Convert slope values to degrees."""
    return np.degrees(np.arctan(slope_series))


def _resample_series(values: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample an array to a fixed target length."""
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if values.size == 0:
        return np.zeros(target_len, dtype=np.float64)
    if values.size == target_len:
        return values.astype(np.float64, copy=True)

    x_old = np.linspace(0.0, 1.0, num=values.size, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
    return np.interp(x_new, x_old, values.astype(np.float64))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute z-score within each rolling window."""
    if window < 1:
        raise ValueError("window must be >= 1")
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    return (series - mean) / (std + EPS)


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Compute percentile rank of the most recent sample inside the window."""
    if window < 1:
        raise ValueError("window must be >= 1")

    def _percentile(values: np.ndarray) -> float:
        last = values[-1]
        return float(np.mean(values <= last))

    return series.rolling(window=window, min_periods=window).apply(_percentile, raw=True)


def _prepare_prototype_library(
    prototype_library: Mapping[str, Sequence[float]] | None,
    target_len: int,
) -> dict[str, np.ndarray]:
    """Normalise template series for shape-similarity comparisons."""
    if not prototype_library:
        return {}

    normalized: dict[str, np.ndarray] = {}
    for name, series in prototype_library.items():
        arr = np.asarray(series, dtype=np.float64).ravel()
        if arr.size < 2:
            continue
        resampled = _resample_series(arr, target_len)
        base = resampled[0]
        scaled = (resampled - base) / (np.abs(base) + EPS)
        scaled -= scaled.mean()
        norm = np.linalg.norm(scaled)
        if norm <= EPS:
            continue
        normalized[name] = scaled / norm
    return normalized


def _compute_close_shape_features(
    segment: np.ndarray,
    resample_points: int,
    proto_vectors: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Derive geometric features from a single closing-price segment."""
    values = segment.astype(np.float64)
    base = values[0]
    norm_series = (values - base) / (np.abs(base) + EPS)
    length = norm_series.size

    x_idx = np.arange(length, dtype=np.float64)
    x_mean = x_idx.mean()
    y_mean = norm_series.mean()
    denom = np.sum((x_idx - x_mean) ** 2) + EPS
    slope = np.sum((x_idx - x_mean) * (norm_series - y_mean)) / denom

    mid = length // 2
    if mid >= 2:
        front = norm_series[:mid]
        back = norm_series[mid:]
    else:
        front = norm_series
        back = norm_series

    def _segment_slope(values_slice: np.ndarray) -> float:
        if values_slice.size < 2:
            return 0.0
        idx = np.arange(values_slice.size, dtype=np.float64)
        idx_mean = idx.mean()
        val_mean = values_slice.mean()
        denom_local = np.sum((idx - idx_mean) ** 2) + EPS
        return np.sum((idx - idx_mean) * (values_slice - val_mean)) / denom_local

    front_slope = _segment_slope(front)
    back_slope = _segment_slope(back)

    line = np.linspace(norm_series[0], norm_series[-1], num=length, dtype=np.float64)
    deviation = norm_series - line
    amplitude = np.max(norm_series) - np.min(norm_series)
    amp_safe = amplitude + EPS

    diff = np.diff(norm_series)
    accel = np.diff(norm_series, n=2) if length >= 3 else np.array([], dtype=np.float64)

    swing_changes = 0.0
    if diff.size >= 2:
        signs = np.sign(diff)
        swing_changes = np.sum(signs[1:] * signs[:-1] < 0)
        swing_changes /= max(diff.size - 1, 1)

    symmetry = 0.0
    half = length // 2
    if half > 0:
        first_half = norm_series[:half]
        second_half = norm_series[-half:][::-1]
        symmetry = np.mean(np.abs(first_half - second_half)) / amp_safe

    resampled = _resample_series(norm_series, resample_points)
    resampled -= resampled.mean()
    resampled_norm = np.linalg.norm(resampled)
    if resampled_norm > EPS:
        resampled /= resampled_norm
    else:
        resampled = np.zeros_like(resampled)

    features: dict[str, float] = {
        'SlopeDeg': float(np.degrees(np.arctan(slope))),
        'FrontSlopeDeg': float(np.degrees(np.arctan(front_slope))),
        'BackSlopeDeg': float(np.degrees(np.arctan(back_slope))),
        'SlopeDiffDeg': float(np.degrees(np.arctan(back_slope - front_slope))),
        'EndReturn': float(norm_series[-1]),
        'Area': float(np.mean(np.abs(deviation)) / amp_safe),
        'MaxDeviation': float(np.max(np.abs(deviation)) / amp_safe),
        'Runup': float(np.max(norm_series)),
        'Drawdown': float(np.abs(np.min(norm_series))),
        'PeakPos': float(np.argmax(norm_series) / max(length - 1, 1)),
        'TroughPos': float(np.argmin(norm_series) / max(length - 1, 1)),
        'Symmetry': float(symmetry),
        'ShapeVol': float(np.std(norm_series)),
        'ShapeVelStd': float(np.std(diff) if diff.size > 0 else 0.0),
        'ShapeAccelStd': float(np.std(accel) if accel.size > 0 else 0.0),
        'Curvature': float(np.mean(np.abs(accel)) if accel.size > 0 else 0.0),
        'SwingRate': float(swing_changes),
    }

    for name, proto_vec in proto_vectors.items():
        features[f'ProtoSim_{name}'] = float(np.dot(resampled, proto_vec))

    return features


__all__ = [
    'PERCENT_SCALE',
    'EPS',
    'format_col_name',
    'scale_by_reference',
    'get_ohlcv_arrays',
    '_compute_rolling_slope',
    '_slope_to_degrees',
    '_resample_series',
    '_rolling_zscore',
    '_rolling_percentile_rank',
    '_prepare_prototype_library',
    '_compute_close_shape_features',
]
