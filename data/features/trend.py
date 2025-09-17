"""Trend-oriented feature generators."""
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


def calculate_ma(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    """Simple moving average via TA-Lib."""
    col_name = format_col_name('SMA', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.SMA(close_price, timeperiod=window)
    return df, [col_name]


def calculate_ema(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    """Exponential moving average via TA-Lib."""
    col_name = format_col_name('EMA', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.EMA(close_price, timeperiod=window)
    return df, [col_name]


def calculate_vwma(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    """Volume-weighted moving average computed with pandas primitives."""
    col_name = format_col_name('VWMA', window)
    price_volume = df['Close'] * df['Volume']
    sum_price_volume = price_volume.rolling(window=window).sum()
    sum_volume = df['Volume'].rolling(window=window).sum()
    df[col_name] = sum_price_volume / (sum_volume + EPS)
    return df, [col_name]


def calculate_macd(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    signal_window: int,
) -> tuple[pd.DataFrame, list[str]]:
    """MACD, signal, and histogram scaled to price magnitude."""
    extra = f"{short_window}_{long_window}"
    macd_col = format_col_name('MACD', extra)
    signal_col = format_col_name('MACD_Signal', f"{extra}_{signal_window}")
    hist_col = format_col_name('MACD_Hist', f"{extra}_{signal_window}")

    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    macd, signal, hist = talib.MACD(
        close_price,
        fastperiod=short_window,
        slowperiod=long_window,
        signalperiod=signal_window,
    )

    close_series = df['Close'].abs().clip(lower=EPS)
    macd_series = pd.Series(macd, index=df.index)
    signal_series = pd.Series(signal, index=df.index)
    hist_series = pd.Series(hist, index=df.index)

    df[macd_col] = scale_by_reference(macd_series, close_series, multiplier=PERCENT_SCALE)
    df[signal_col] = scale_by_reference(signal_series, close_series, multiplier=PERCENT_SCALE)
    df[hist_col] = scale_by_reference(hist_series, close_series, multiplier=PERCENT_SCALE)

    return df, [macd_col, signal_col, hist_col]


def calculate_adx(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    """Average Directional Index plus +/- DI."""
    adx_col = format_col_name('ADX', window)
    plus_di_col = format_col_name('DI_plus', window)
    minus_di_col = format_col_name('DI_minus', window)

    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)

    df[adx_col] = talib.ADX(high_price, low_price, close_price, timeperiod=window)
    df[plus_di_col] = talib.PLUS_DI(high_price, low_price, close_price, timeperiod=window)
    df[minus_di_col] = talib.MINUS_DI(high_price, low_price, close_price, timeperiod=window)

    return df, [adx_col, plus_di_col, minus_di_col]


def calculate_ichimoku(
    df: pd.DataFrame,
    short_window: int = 9,
    mid_window: int = 26,
    long_window: int = 52,
) -> tuple[pd.DataFrame, list[str]]:
    """Ichimoku cloud components using rolling highs/lows."""
    high = df['High']
    low = df['Low']
    close = df['Close']

    tenkan_sen_col = format_col_name('Ichimoku_Tenkan', short_window)
    kijun_sen_col = format_col_name('Ichimoku_Kijun', mid_window)
    senkou_a_col = format_col_name('Ichimoku_SenkouA', f"{short_window}_{mid_window}")
    senkou_b_col = format_col_name('Ichimoku_SenkouB', long_window)
    chikou_ratio_col = format_col_name('Ichimoku_Chikou_Ratio', mid_window)

    df[tenkan_sen_col] = (
        high.rolling(window=short_window).max() + low.rolling(window=short_window).min()
    ) / 2
    df[kijun_sen_col] = (
        high.rolling(window=mid_window).max() + low.rolling(window=mid_window).min()
    ) / 2
    df[senkou_a_col] = ((df[tenkan_sen_col] + df[kijun_sen_col]) / 2).shift(mid_window)
    df[senkou_b_col] = (
        (high.rolling(window=long_window).max() + low.rolling(window=long_window).min()) / 2
    ).shift(mid_window)

    past_close = close.shift(mid_window)
    df[chikou_ratio_col] = close / (past_close + EPS)

    return df, [tenkan_sen_col, kijun_sen_col, senkou_a_col, senkou_b_col, chikou_ratio_col]


def calculate_ichimoku_geometry(
    df: pd.DataFrame,
    short_window: int = 9,
    mid_window: int = 26,
    long_window: int = 52,
) -> tuple[pd.DataFrame, list[str]]:
    """Geometric descriptors derived from Ichimoku components."""
    tenkan_col = format_col_name('Ichimoku_Tenkan', short_window)
    kijun_col = format_col_name('Ichimoku_Kijun', mid_window)
    senkou_a_col = format_col_name('Ichimoku_SenkouA', f"{short_window}_{mid_window}")
    senkou_b_col = format_col_name('Ichimoku_SenkouB', long_window)
    chikou_col = format_col_name('Ichimoku_Chikou_Ratio', mid_window)

    required_cols = [tenkan_col, kijun_col, senkou_a_col, senkou_b_col, 'Close']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing Ichimoku columns for geometry calculation: {required_cols}")
        return df, []

    thickness = (df[senkou_a_col] - df[senkou_b_col]).abs()
    price_scale = (
        df['Close']
        .rolling(window=mid_window, min_periods=mid_window)
        .mean()
        .abs()
        .clip(lower=EPS)
    )

    area_col = format_col_name(
        'Ichimoku_Geom_CloudAreaNorm', f"{short_window}_{mid_window}_{long_window}"
    )
    cloud_area = thickness.rolling(window=mid_window, min_periods=mid_window).mean() * mid_window
    df[area_col] = cloud_area / price_scale

    senkou_a_slope_col = format_col_name(
        'Ichimoku_Geom_SenkouA_SlopeDeg', f"{short_window}_{mid_window}_{long_window}"
    )
    senkou_b_slope_col = format_col_name(
        'Ichimoku_Geom_SenkouB_SlopeDeg', f"{short_window}_{mid_window}_{long_window}"
    )
    senkou_diff_col = format_col_name(
        'Ichimoku_Geom_Senkou_SlopeDiffDeg', f"{short_window}_{mid_window}_{long_window}"
    )

    senkou_a_angle = _slope_to_degrees(_compute_rolling_slope(df[senkou_a_col], mid_window))
    senkou_b_angle = _slope_to_degrees(_compute_rolling_slope(df[senkou_b_col], mid_window))

    df[senkou_a_slope_col] = senkou_a_angle
    df[senkou_b_slope_col] = senkou_b_angle
    df[senkou_diff_col] = senkou_a_angle - senkou_b_angle

    thickness_vol_col = format_col_name(
        'Ichimoku_Geom_CloudThicknessVol', f"{short_window}_{mid_window}_{long_window}"
    )
    cloud_velocity_col = format_col_name(
        'Ichimoku_Geom_CloudVelocity', f"{short_window}_{mid_window}_{long_window}"
    )
    close_pos_col = format_col_name(
        'Ichimoku_Geom_ClosePos', f"{short_window}_{mid_window}_{long_window}"
    )
    tenkan_slope_col = format_col_name(
        'Ichimoku_Geom_Tenkan_SlopeDeg', f"{short_window}_{mid_window}_{long_window}"
    )
    kijun_slope_col = format_col_name(
        'Ichimoku_Geom_Kijun_SlopeDeg', f"{short_window}_{mid_window}_{long_window}"
    )
    tenkan_above_col = format_col_name(
        'Ichimoku_Geom_TenkanAboveRate', f"{short_window}_{mid_window}_{long_window}"
    )
    chikou_slope_col = format_col_name(
        'Ichimoku_Geom_Chikou_SlopeDeg', f"{short_window}_{mid_window}_{long_window}"
    )

    thickness_vol = thickness.rolling(window=mid_window, min_periods=mid_window).std()
    thickness_velocity = thickness.diff().rolling(window=mid_window, min_periods=mid_window).mean()

    df[thickness_vol_col] = thickness_vol / price_scale
    df[cloud_velocity_col] = thickness_velocity / price_scale

    cloud_mid = (df[senkou_a_col] + df[senkou_b_col]) / 2
    close_position = (df['Close'] - cloud_mid) / (thickness + EPS)
    df[close_pos_col] = close_position.clip(lower=-3.0, upper=3.0)

    df[tenkan_slope_col] = _slope_to_degrees(_compute_rolling_slope(df[tenkan_col], short_window))
    df[kijun_slope_col] = _slope_to_degrees(_compute_rolling_slope(df[kijun_col], mid_window))
    df[tenkan_above_col] = (
        (df[tenkan_col] > df[kijun_col]).rolling(window=mid_window, min_periods=mid_window).mean()
    )

    new_cols = [
        area_col,
        senkou_a_slope_col,
        senkou_b_slope_col,
        senkou_diff_col,
        thickness_vol_col,
        cloud_velocity_col,
        close_pos_col,
        tenkan_slope_col,
        kijun_slope_col,
        tenkan_above_col,
    ]

    if chikou_col in df.columns:
        df[chikou_slope_col] = _slope_to_degrees(_compute_rolling_slope(df[chikou_col], mid_window))
        new_cols.append(chikou_slope_col)

    return df, new_cols


def calculate_dema(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    """Double exponential moving average via TA-Lib."""
    col_name = format_col_name('DEMA', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.DEMA(close_price, timeperiod=window)
    return df, [col_name]


def calculate_tema(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    """Triple exponential moving average via TA-Lib."""
    col_name = format_col_name('TEMA', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.TEMA(close_price, timeperiod=window)
    return df, [col_name]


def calculate_sar(
    df: pd.DataFrame,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> tuple[pd.DataFrame, list[str]]:
    """Parabolic SAR indicator."""
    col_name = format_col_name('SAR', extra_str=f"{acceleration}_{maximum}")
    _, high_price, low_price, _, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.SAR(
        high_price,
        low_price,
        acceleration=acceleration,
        maximum=maximum,
    )
    return df, [col_name]


def calculate_trima(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, list[str]]:
    """Triangular moving average via TA-Lib."""
    col_name = format_col_name('TRIMA', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.TRIMA(close_price, timeperiod=window)
    return df, [col_name]


__all__ = [
    'calculate_ma',
    'calculate_ema',
    'calculate_vwma',
    'calculate_macd',
    'calculate_adx',
    'calculate_ichimoku',
    'calculate_ichimoku_geometry',
    'calculate_dema',
    'calculate_tema',
    'calculate_sar',
    'calculate_trima',
]
