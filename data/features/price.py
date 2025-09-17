"""Price-based feature generators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    PERCENT_SCALE,
    EPS,
    format_col_name,
    scale_by_reference,
    get_ohlcv_arrays,
)


def calculate_price_change_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute price-delta and candle-shape basics."""
    new_cols: list[str] = []

    df['close_change_pct'] = df['Close'].pct_change() * 100
    new_cols.append('close_change_pct')

    open_price, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)

    body_size_raw = np.abs(open_price - close_price)
    upper_wick_raw = high_price - np.maximum(open_price, close_price)
    lower_wick_raw = np.minimum(open_price, close_price) - low_price
    total_range_raw = high_price - low_price

    df['body_size'] = body_size_raw

    close_magnitude = df['Close'].abs().clip(lower=EPS)
    df['upper_wick_size'] = scale_by_reference(
        pd.Series(upper_wick_raw, index=df.index),
        close_magnitude,
        multiplier=PERCENT_SCALE,
    )
    df['lower_wick_size'] = scale_by_reference(
        pd.Series(lower_wick_raw, index=df.index),
        close_magnitude,
        multiplier=PERCENT_SCALE,
    )
    df['total_range'] = scale_by_reference(
        pd.Series(total_range_raw, index=df.index),
        close_magnitude,
        multiplier=PERCENT_SCALE,
    )
    new_cols.extend(['body_size', 'upper_wick_size', 'lower_wick_size', 'total_range'])

    df['body_to_range_ratio'] = body_size_raw / (total_range_raw + EPS)
    new_cols.append('body_to_range_ratio')

    df.fillna({'close_change_pct': 0}, inplace=True)
    return df, new_cols


__all__ = ['calculate_price_change_features']
