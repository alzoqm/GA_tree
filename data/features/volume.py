"""Volume-oriented feature generators."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import talib

from .utils import (
    EPS,
    PERCENT_SCALE,
    format_col_name,
    get_ohlcv_arrays,
    scale_by_reference,
    _compute_close_shape_features,
    _compute_rolling_slope,
    _prepare_prototype_library,
    _rolling_percentile_rank,
    _rolling_zscore,
    _slope_to_degrees,
)

try:  # 패키지 실행 시
    from ..models.constants import (
        DEFAULT_VOLUME_WINDOWS,
        DEFAULT_VOLUME_SHAPE_WINDOWS,
        VOLUME_DRY_Z_THRESHOLD,
        VOLUME_SPIKE_Z_THRESHOLD,
    )
except Exception:  # 스크립트 단독 실행 보호
    DEFAULT_VOLUME_WINDOWS = (5, 14, 30, 60, 120)
    DEFAULT_VOLUME_SHAPE_WINDOWS = tuple(sorted({w for w in DEFAULT_VOLUME_WINDOWS if w >= 8}))
    VOLUME_SPIKE_Z_THRESHOLD = 2.0
    VOLUME_DRY_Z_THRESHOLD = -1.0


def calculate_volume_features(
    df: pd.DataFrame,
    windows: Sequence[int] | None = None,
    shape_windows: Sequence[int] | None = None,
    shape_resample_points: int = 16,
    volume_prototype_library: Mapping[str, Sequence[float]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if 'Volume' not in df.columns:
        raise KeyError("DataFrame must contain 'Volume' column.")

    if windows is None:
        windows = DEFAULT_VOLUME_WINDOWS
    if shape_windows is None:
        shape_windows = DEFAULT_VOLUME_SHAPE_WINDOWS

    new_cols: list[str] = []
    volume_series = df['Volume'].astype(np.float64)

    log_col = 'Volume_Log'
    df[log_col] = np.log1p(volume_series)
    new_cols.append(log_col)

    prev_volume = volume_series.shift(1)
    change_pct_col = 'Volume_ChangePct'
    change_pct = np.where(
        np.abs(prev_volume) > EPS,
        (volume_series - prev_volume) / (prev_volume + EPS) * PERCENT_SCALE,
        0.0,
    )
    df[change_pct_col] = change_pct
    new_cols.append(change_pct_col)

    number_trades_col = 'Number of trades'
    if number_trades_col in df.columns:
        volume_per_trade_col = 'Volume_PerTrade'
        df[volume_per_trade_col] = volume_series / (df[number_trades_col].astype(np.float64) + EPS)
        new_cols.append(volume_per_trade_col)

    quote_volume_col = 'Quote asset volume'
    if quote_volume_col in df.columns:
        quote_volume_series = df[quote_volume_col].astype(np.float64)
        quote_ratio_col = 'Volume_QuoteRatio'
        df[quote_ratio_col] = volume_series / (quote_volume_series + EPS)
        new_cols.append(quote_ratio_col)

        if number_trades_col in df.columns:
            quote_per_trade_col = 'QuoteVolume_PerTrade'
            df[quote_per_trade_col] = quote_volume_series / (df[number_trades_col].astype(np.float64) + EPS)
            new_cols.append(quote_per_trade_col)

    taker_buy_base_col = 'Taker buy base asset volume'
    taker_buy_quote_col = 'Taker buy quote asset volume'

    if taker_buy_base_col in df.columns:
        taker_buy_base_series = df[taker_buy_base_col].astype(np.float64)
        participation_col = 'TakerBuyBase_Ratio'
        imbalance_col = 'TakerBuyBase_Imbalance'
        df[participation_col] = taker_buy_base_series / (volume_series + EPS)
        df[imbalance_col] = (2.0 * taker_buy_base_series - volume_series) / (volume_series + EPS)
        new_cols.extend([participation_col, imbalance_col])

    if taker_buy_quote_col in df.columns:
        taker_buy_quote_series = df[taker_buy_quote_col].astype(np.float64)
        participation_col = 'TakerBuyQuote_Ratio'
        imbalance_col = 'TakerBuyQuote_Imbalance'
        reference = df.get(quote_volume_col, volume_series).astype(np.float64)
        df[participation_col] = taker_buy_quote_series / (reference + EPS)
        df[imbalance_col] = (2.0 * taker_buy_quote_series - reference) / (reference + EPS)
        new_cols.extend([participation_col, imbalance_col])

    ema_cache: dict[int, pd.Series] = {}

    for window in windows:
        if window < 2:
            continue

        mean = volume_series.rolling(window=window, min_periods=window).mean()
        median = volume_series.rolling(window=window, min_periods=window).median()
        std = volume_series.rolling(window=window, min_periods=window).std()
        zscore = _rolling_zscore(volume_series, window)
        percentile = _rolling_percentile_rank(volume_series, window)

        ema = ema_cache.get(window)
        if ema is None:
            ema = volume_series.ewm(span=window, adjust=False).mean()
            ema_cache[window] = ema

        ratio_mean_col = format_col_name('Volume', window, 'ToMean')
        ratio_median_col = format_col_name('Volume', window, 'ToMedian')
        zscore_col = format_col_name('Volume', window, 'Z')
        slope_col = format_col_name('Volume', window, 'SlopeDeg')
        percentile_col = format_col_name('Volume', window, 'PctRank')
        std_ratio_col = format_col_name('Volume', window, 'StdRatio')
        ema_ratio_col = format_col_name('Volume', window, 'ToEma')
        spike_flag_col = format_col_name('Volume', window, 'SpikeFlag')
        dry_flag_col = format_col_name('Volume', window, 'DryFlag')

        df[ratio_mean_col] = volume_series / (mean + EPS)
        df[ratio_median_col] = volume_series / (median + EPS)
        df[zscore_col] = zscore
        df[slope_col] = _slope_to_degrees(_compute_rolling_slope(df['Volume'], window))
        df[percentile_col] = percentile
        df[std_ratio_col] = std / (mean + EPS)
        df[ema_ratio_col] = volume_series / (ema + EPS)
        df[spike_flag_col] = zscore > VOLUME_SPIKE_Z_THRESHOLD
        df[dry_flag_col] = zscore < VOLUME_DRY_Z_THRESHOLD

        new_cols.extend(
            [
                ratio_mean_col,
                ratio_median_col,
                zscore_col,
                slope_col,
                percentile_col,
                std_ratio_col,
                ema_ratio_col,
                spike_flag_col,
                dry_flag_col,
            ]
        )

    geometry_windows = tuple(shape_windows) if shape_windows else ()
    for window in geometry_windows:
        if window < 3:
            continue
        df, geom_cols = calculate_volume_geometry_features(
            df,
            window=window,
            resample_points=shape_resample_points,
            prototype_library=volume_prototype_library,
        )
        new_cols.extend(geom_cols)

    return df, new_cols


def calculate_volume_geometry_features(
    df: pd.DataFrame,
    window: int,
    resample_points: int = 16,
    prototype_library: Mapping[str, Sequence[float]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if 'Volume' not in df.columns:
        raise KeyError("DataFrame must contain 'Volume' column.")
    if window < 3:
        raise ValueError("window must be >= 3.")
    if resample_points < 2:
        raise ValueError("resample_points must be >= 2.")

    proto_vectors = _prepare_prototype_library(prototype_library, resample_points)

    base_metrics = [
        'SlopeDeg',
        'FrontSlopeDeg',
        'BackSlopeDeg',
        'SlopeDiffDeg',
        'EndReturn',
        'Area',
        'MaxDeviation',
        'Runup',
        'Drawdown',
        'PeakPos',
        'TroughPos',
        'Symmetry',
        'ShapeVol',
        'ShapeVelStd',
        'ShapeAccelStd',
        'Curvature',
        'SwingRate',
    ]

    column_map = {
        metric: format_col_name('VolumeGeom', window, metric) for metric in base_metrics
    }
    for name in proto_vectors:
        metric = f'ProtoSim_{name}'
        column_map[metric] = format_col_name('VolumeGeom', window, metric)

    arrays = {col: np.full(df.shape[0], np.nan, dtype=np.float64) for col in column_map.values()}
    volume_values = df['Volume'].astype(np.float64).values

    for end_idx in range(window - 1, volume_values.size):
        segment = volume_values[end_idx - window + 1 : end_idx + 1]
        feature_values = _compute_close_shape_features(segment, resample_points, proto_vectors)
        for metric_name, value in feature_values.items():
            col = column_map.get(metric_name)
            if col is None:
                continue
            arrays[col][end_idx] = value

    for col_name, values in arrays.items():
        df[col_name] = values

    return df, list(arrays.keys())


def calculate_obv(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    obv_normalization_window = 20
    col_name = 'OBV'
    _, _, _, close_price, volume = get_ohlcv_arrays(df)
    obv_values = talib.OBV(close_price, volume)
    obv_series = pd.Series(obv_values, index=df.index)
    volume_sum = df['Volume'].rolling(window=obv_normalization_window, min_periods=1).sum()
    df[col_name] = scale_by_reference(obv_series, volume_sum)
    return df, [col_name]


def calculate_cmf(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    col_name = format_col_name('CMF', window)
    close = df['Close']
    low = df['Low']
    high = df['High']
    volume = df['Volume']

    mf_multiplier = ((close - low) - (high - close)) / (high - low + EPS)
    mf_volume = mf_multiplier * volume

    df[col_name] = mf_volume.rolling(window=window).sum() / (
        volume.rolling(window=window).sum() + EPS
    )
    return df, [col_name]


def calculate_ad(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    col_name = 'AD'
    _, high_price, low_price, close_price, volume = get_ohlcv_arrays(df)
    df[col_name] = talib.AD(high_price, low_price, close_price, volume)
    return df, [col_name]


def calculate_adosc(
    df: pd.DataFrame,
    fast_window: int = 3,
    slow_window: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    col_name = format_col_name('ADOSC', f"{fast_window}_{slow_window}")
    _, high_price, low_price, close_price, volume = get_ohlcv_arrays(df)
    adosc_values = talib.ADOSC(
        high_price,
        low_price,
        close_price,
        volume,
        fastperiod=fast_window,
        slowperiod=slow_window,
    )
    adosc_series = pd.Series(adosc_values, index=df.index)
    avg_volume = df['Volume'].rolling(window=slow_window, min_periods=1).mean()
    df[col_name] = scale_by_reference(adosc_series, avg_volume, multiplier=PERCENT_SCALE)
    return df, [col_name]


__all__ = [
    'DEFAULT_VOLUME_WINDOWS',
    'DEFAULT_VOLUME_SHAPE_WINDOWS',
    'VOLUME_DRY_Z_THRESHOLD',
    'VOLUME_SPIKE_Z_THRESHOLD',
    'calculate_volume_features',
    'calculate_volume_geometry_features',
    'calculate_obv',
    'calculate_cmf',
    'calculate_ad',
    'calculate_adosc',
]
