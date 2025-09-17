"""Volatility and band-based feature generators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from .utils import (
    EPS,
    PERCENT_SCALE,
    format_col_name,
    get_ohlcv_arrays,
    scale_by_reference,
    _compute_rolling_slope,
    _slope_to_degrees,
)


def calculate_bollinger_bands(
    df: pd.DataFrame,
    window: int,
    num_std: int = 2,
) -> tuple[pd.DataFrame, list[str]]:
    upper_col = format_col_name('BB_Upper', f"{window}_{num_std}")
    mid_col = format_col_name('BB_Mid', f"{window}_{num_std}")
    lower_col = format_col_name('BB_Lower', f"{window}_{num_std}")
    width_col = format_col_name('BB_Width', f"{window}_{num_std}")

    _, _, _, close_price, _ = get_ohlcv_arrays(df)

    upper, middle, lower = talib.BBANDS(
        close_price,
        timeperiod=window,
        nbdevup=num_std,
        nbdevdn=num_std,
        matype=0,
    )

    df[upper_col] = upper
    df[mid_col] = middle
    df[lower_col] = lower
    df[width_col] = (upper - lower) / (middle + EPS)

    return df, [upper_col, mid_col, lower_col, width_col]


def calculate_bollinger_geometry(
    df: pd.DataFrame,
    window: int,
    num_std: int = 2,
) -> tuple[pd.DataFrame, list[str]]:
    upper_col = format_col_name('BB_Upper', f"{window}_{num_std}")
    mid_col = format_col_name('BB_Mid', f"{window}_{num_std}")
    lower_col = format_col_name('BB_Lower', f"{window}_{num_std}")

    required_cols = [upper_col, mid_col, lower_col, 'Close']
    if not all(col in df.columns for col in required_cols):
        print(
            "Warning: Missing Bollinger band columns for geometry calculation:"
            f" {required_cols}"
        )
        return df, []

    width_actual = df[upper_col] - df[lower_col]
    mid_abs = (
        df[mid_col]
        .rolling(window=window, min_periods=window)
        .mean()
        .abs()
        .clip(lower=EPS)
    )

    area_col = format_col_name('BB_Geom_AreaNorm', f"{window}_{num_std}")
    band_area = width_actual.rolling(window=window, min_periods=window).mean() * window
    df[area_col] = band_area / mid_abs

    upper_angle_col = format_col_name('BB_Geom_UpperSlopeDeg', f"{window}_{num_std}")
    lower_angle_col = format_col_name('BB_Geom_LowerSlopeDeg', f"{window}_{num_std}")
    mid_angle_col = format_col_name('BB_Geom_MidSlopeDeg', f"{window}_{num_std}")
    slope_diff_col = format_col_name('BB_Geom_SlopeDiffDeg', f"{window}_{num_std}")

    upper_angle = _slope_to_degrees(_compute_rolling_slope(df[upper_col], window))
    lower_angle = _slope_to_degrees(_compute_rolling_slope(df[lower_col], window))
    mid_angle = _slope_to_degrees(_compute_rolling_slope(df[mid_col], window))

    df[upper_angle_col] = upper_angle
    df[lower_angle_col] = lower_angle
    df[mid_angle_col] = mid_angle
    df[slope_diff_col] = upper_angle - lower_angle

    width_vol_col = format_col_name('BB_Geom_WidthVol', f"{window}_{num_std}")
    width_velocity_col = format_col_name('BB_Geom_WidthVelocity', f"{window}_{num_std}")
    price_pos_col = format_col_name('BB_Geom_ClosePos', f"{window}_{num_std}")
    upper_touch_col = format_col_name('BB_Geom_UpperTouchRate', f"{window}_{num_std}")
    lower_touch_col = format_col_name('BB_Geom_LowerTouchRate', f"{window}_{num_std}")

    width_vol = width_actual.rolling(window=window, min_periods=window).std()
    width_velocity = width_actual.diff().rolling(window=window, min_periods=window).mean()

    df[width_vol_col] = width_vol / mid_abs
    df[width_velocity_col] = width_velocity / mid_abs

    close_position = (df['Close'] - df[lower_col]) / (width_actual + EPS)
    df[price_pos_col] = close_position.clip(lower=0.0, upper=1.0)

    df[upper_touch_col] = (
        (df['Close'] >= df[upper_col]).rolling(window=window, min_periods=window).mean()
    )
    df[lower_touch_col] = (
        (df['Close'] <= df[lower_col]).rolling(window=window, min_periods=window).mean()
    )

    new_cols = [
        area_col,
        upper_angle_col,
        lower_angle_col,
        mid_angle_col,
        slope_diff_col,
        width_vol_col,
        width_velocity_col,
        price_pos_col,
        upper_touch_col,
        lower_touch_col,
    ]

    return df, new_cols


def calculate_atr(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    col_name = format_col_name('ATR', window)
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    atr_values = talib.ATR(high_price, low_price, close_price, timeperiod=window)
    atr_series = pd.Series(atr_values, index=df.index)
    close_series = df['Close'].abs().clip(lower=EPS)
    df[col_name] = scale_by_reference(atr_series, close_series, multiplier=PERCENT_SCALE)
    return df, [col_name]


def calculate_natr(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    col_name = format_col_name('NATR', window)
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.NATR(high_price, low_price, close_price, timeperiod=window)
    return df, [col_name]


def calculate_atr_moving_average(
    df: pd.DataFrame,
    atr_window: int,
    ma_window: int,
) -> tuple[pd.DataFrame, list[str]]:
    atr_col = format_col_name('ATR', atr_window)
    new_col = format_col_name('ATR', f"{atr_window}_MA_{ma_window}")

    if atr_col not in df.columns:
        print(
            f"Warning: ATR column '{atr_col}' not found. Skipping ATR moving average calculation."
        )
        return df, []

    df[new_col] = df[atr_col].rolling(window=ma_window, min_periods=1).mean()

    return df, [new_col]


__all__ = [
    'calculate_bollinger_bands',
    'calculate_bollinger_geometry',
    'calculate_atr',
    'calculate_natr',
    'calculate_atr_moving_average',
]
