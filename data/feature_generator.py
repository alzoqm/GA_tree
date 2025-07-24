# feature_generator.py

import pandas as pd
import numpy as np
from numba import njit

# ==============================================================================
# 유틸리티 함수
# ==============================================================================
def format_col_name(base_name, window=None, extra_str=None):
    """일관된 컬럼명 생성을 위한 헬퍼 함수"""
    name = base_name
    if window:
        name = f"{name}_{window}"
    if extra_str:
        name = f"{name}_{extra_str}"
    return name

# ==============================================================================
# 1. 기본 OHLCV 및 캔들 기반 피처
# ==============================================================================
def calculate_price_change_features(df):
    """가격 변화 및 캔들 몸통/꼬리 관련 기본 피처를 계산합니다."""
    new_cols = []

    # 가격 변화율 (백분율)
    df['close_change_pct'] = df['Close'].pct_change() * 100
    new_cols.append('close_change_pct')

    # 캔들 크기
    df['body_size'] = abs(df['Open'] - df['Close'])
    df['upper_wick_size'] = df['High'] - np.maximum(df['Open'], df['Close'])
    df['lower_wick_size'] = np.minimum(df['Open'], df['Close']) - df['Low']
    df['total_range'] = df['High'] - df['Low']
    new_cols.extend(['body_size', 'upper_wick_size', 'lower_wick_size', 'total_range'])

    # 캔들 크기 비율 (분모가 0이 되는 것을 방지)
    df['body_to_range_ratio'] = df['body_size'] / (df['total_range'] + 1e-9)
    new_cols.append('body_to_range_ratio')

    df.fillna({'close_change_pct': 0}, inplace=True)
    return df, new_cols

# ==============================================================================
# 2. 추세 (Trend) 지표
# ==============================================================================
def calculate_ma(df, window):
    """단순 이동평균 (SMA)을 계산합니다."""
    col_name = format_col_name('SMA', window)
    df[col_name] = df['Close'].rolling(window=window).mean()
    return df, [col_name]

def calculate_ema(df, window):
    """지수 이동평균 (EMA)을 계산합니다."""
    col_name = format_col_name('EMA', window)
    df[col_name] = df['Close'].ewm(span=window, adjust=False).mean()
    return df, [col_name]

def calculate_vwma(df, window):
    """VWMA (거래량 가중 이동 평균)를 계산합니다."""
    col_name = format_col_name('VWMA', window)
    
    price_volume = df['Close'] * df['Volume']
    sum_price_volume = price_volume.rolling(window=window).sum()
    sum_volume = df['Volume'].rolling(window=window).sum()
    
    df[col_name] = sum_price_volume / (sum_volume + 1e-9)
    
    return df, [col_name]


def calculate_macd(df, short_window, long_window, signal_window):
    """MACD, Signal Line, Histogram을 계산합니다."""
    extra = f"{short_window}_{long_window}"
    macd_col = format_col_name('MACD', extra)
    signal_col = format_col_name('MACD_Signal', f"{extra}_{signal_window}")
    hist_col = format_col_name('MACD_Hist', f"{extra}_{signal_window}")

    ema_short = df['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = df['Close'].ewm(span=long_window, adjust=False).mean()

    df[macd_col] = ema_short - ema_long
    df[signal_col] = df[macd_col].ewm(span=signal_window, adjust=False).mean()
    df[hist_col] = df[macd_col] - df[signal_col]

    return df, [macd_col, signal_col, hist_col]



def calculate_adx(df, window):
    """ADX, +DI, -DI를 계산합니다."""
    adx_col = format_col_name('ADX', window)
    plus_di_col = format_col_name('DI_plus', window)
    minus_di_col = format_col_name('DI_minus', window)

    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = low.diff().mul(-1)

    plus_dm[plus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < 0] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    df[plus_di_col] = 100 * plus_dm.ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-9)
    df[minus_di_col] = 100 * minus_dm.ewm(alpha=1/window, adjust=False).mean() / (atr + 1e-9)
    
    dx = 100 * abs(df[plus_di_col] - df[minus_di_col]) / (df[plus_di_col] + df[minus_di_col] + 1e-9)
    df[adx_col] = dx.ewm(alpha=1/window, adjust=False).mean()

    return df, [adx_col, plus_di_col, minus_di_col]

def calculate_ichimoku(df, short_window=9, mid_window=26, long_window=52):
    """일목균형표 (Ichimoku Cloud) 지표들을 계산합니다."""
    high = df['High']
    low = df['Low']
    close = df['Close']

    tenkan_sen_col = format_col_name('Ichimoku_Tenkan', short_window)
    kijun_sen_col = format_col_name('Ichimoku_Kijun', mid_window)
    senkou_a_col = format_col_name('Ichimoku_SenkouA', f"{short_window}_{mid_window}")
    senkou_b_col = format_col_name('Ichimoku_SenkouB', long_window)
    chikou_col = format_col_name('Ichimoku_Chikou')

    # 전환선 (Tenkan-sen)
    df[tenkan_sen_col] = (high.rolling(window=short_window).max() + low.rolling(window=short_window).min()) / 2
    # 기준선 (Kijun-sen)
    df[kijun_sen_col] = (high.rolling(window=mid_window).max() + low.rolling(window=mid_window).min()) / 2
    # 선행스팬 A (Senkou Span A)
    df[senkou_a_col] = ((df[tenkan_sen_col] + df[kijun_sen_col]) / 2).shift(mid_window)
    # 선행스팬 B (Senkou Span B)
    df[senkou_b_col] = ((high.rolling(window=long_window).max() + low.rolling(window=long_window).min()) / 2).shift(mid_window)
    # 후행스팬 (Chikou Span)
    df[chikou_col] = close.shift(-mid_window)

    return df, [tenkan_sen_col, kijun_sen_col, senkou_a_col, senkou_b_col, chikou_col]


# ==============================================================================
# 3. 모멘텀 (Momentum) 지표
# ==============================================================================
def calculate_rsi(df, window):
    """RSI (상대강도지수)를 원본 값(0-100)으로 계산합니다."""
    col_name = format_col_name('RSI', window)
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0).ewm(alpha=1/window, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df[col_name] = 100 - (100 / (1 + rs))
    return df, [col_name]

def calculate_stochastic_oscillator(df, k_window, d_window):
    """Stochastic Oscillator (%K, %D)를 원본 값으로 계산합니다."""
    extra = f"{k_window}_{d_window}"
    k_col = format_col_name('%K', extra)
    d_col = format_col_name('%D', extra)

    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    
    df[k_col] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-9)
    df[d_col] = df[k_col].rolling(window=d_window).mean()
    return df, [k_col, d_col]

def calculate_williams_r(df, window):
    """Williams %R을 원본 값(-100 to 0)으로 계산합니다."""
    col_name = format_col_name('Williams_R', window)
    high_max = df['High'].rolling(window=window).max()
    low_min = df['Low'].rolling(window=window).min()
    df[col_name] = -100 * (high_max - df['Close']) / (high_max - low_min + 1e-9)
    return df, [col_name]

def calculate_cci(df, window, constant=0.015):
    """CCI (Commodity Channel Index)를 계산합니다."""
    col_name = format_col_name('CCI', f"{window}_{constant}")
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(window=window).mean()
    mean_deviation = abs(typical_price - sma_tp).rolling(window=window).mean()
    
    df[col_name] = (typical_price - sma_tp) / (constant * mean_deviation + 1e-9)
    
    return df, [col_name]

# ==============================================================================
# 4. 변동성 (Volatility) 지표
# ==============================================================================
def calculate_bollinger_bands(df, window, num_std=2):
    """볼린저 밴드 (상단, 중간, 하단) 및 밴드폭을 계산합니다."""
    upper_col = format_col_name('BB_Upper', f"{window}_{num_std}")
    mid_col = format_col_name('BB_Mid', f"{window}_{num_std}")
    lower_col = format_col_name('BB_Lower', f"{window}_{num_std}")
    width_col = format_col_name('BB_Width', f"{window}_{num_std}")
    
    ma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    
    df[mid_col] = ma
    df[upper_col] = ma + (std * num_std)
    df[lower_col] = ma - (std * num_std)
    df[width_col] = (df[upper_col] - df[lower_col]) / (ma + 1e-9)
    
    return df, [upper_col, mid_col, lower_col, width_col]

def calculate_atr(df, window):
    """ATR (평균 실제 범위)을 계산합니다."""
    col_name = format_col_name('ATR', window)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[col_name] = tr.ewm(span=window, adjust=False).mean()
    return df, [col_name]

# ==============================================================================
# 5. 거래량 (Volume) 지표
# ==============================================================================
def calculate_obv(df):
    """OBV (On-Balance Volume)를 계산합니다."""
    col_name = 'OBV'
    direction = np.sign(df['Close'].diff()).fillna(0)
    df[col_name] = (df['Volume'] * direction).cumsum()
    return df, [col_name]


def calculate_cmf(df, window):
    """CMF (Chaikin Money Flow)를 계산합니다."""
    col_name = format_col_name('CMF', window)
    close = df['Close']
    low = df['Low']
    high = df['High']
    volume = df['Volume']
    
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-9)
    mf_volume = mf_multiplier * volume
    
    df[col_name] = mf_volume.rolling(window=window).sum() / (volume.rolling(window=window).sum() + 1e-9)
    return df, [col_name]

# ==============================================================================
# 6. 시간 기반 피처
# ==============================================================================
def cyclic_encode_fn(df, timestamp_col, cycle):
    """시간 정보를 순환형 피처로 변환합니다. (예: 요일, 시간)"""
    new_columns = []
    if cycle == 'minute_of_day':
        minutes = df[timestamp_col].dt.hour * 60 + df[timestamp_col].dt.minute
        period = 24 * 60
        sin_col = 'minute_of_day_sin'
        cos_col = 'minute_of_day_cos'
        df[sin_col] = np.sin(2 * np.pi * minutes / period)
        df[cos_col] = np.cos(2 * np.pi * minutes / period)
        new_columns.extend([sin_col, cos_col])
    elif cycle == 'day_of_week':
        days = df[timestamp_col].dt.dayofweek
        period = 7
        sin_col = 'day_of_week_sin'
        cos_col = 'day_of_week_cos'
        df[sin_col] = np.sin(2 * np.pi * days / period)
        df[cos_col] = np.cos(2 * np.pi * days / period)
        new_columns.extend([sin_col, cos_col])
    else:
        raise ValueError("Invalid cycle option.")
    return df, new_columns


# ==============================================================================
# 7. 지지 (Support) 및 저항 (Resistance) 지표
# ==============================================================================
def calculate_support_resistance(df, window):
    """이동 평균 및 피벗 포인트를 사용하여 지지 및 저항 수준을 계산합니다."""
    new_cols = []

    # 이동 평균 기반 지지/저항
    support_ma_col = format_col_name('Support_MA', window)
    resistance_ma_col = format_col_name('Resistance_MA', window)
    df[support_ma_col] = df['Low'].rolling(window=window).min()
    df[resistance_ma_col] = df['High'].rolling(window=window).max()
    new_cols.extend([support_ma_col, resistance_ma_col])

    # 피벗 포인트 기반 지지/저항 (일일 기준)
    # 전일 데이터가 필요하므로 shift(1) 사용
    high_prev = df['High'].shift(1)
    low_prev = df['Low'].shift(1)
    close_prev = df['Close'].shift(1)

    pivot_point_col = 'Pivot_Point'
    support1_col = 'Support1_Pivot'
    support2_col = 'Support2_Pivot'
    resistance1_col = 'Resistance1_Pivot'
    resistance2_col = 'Resistance2_Pivot'

    df[pivot_point_col] = (high_prev + low_prev + close_prev) / 3
    df[support1_col] = (2 * df[pivot_point_col]) - high_prev
    df[resistance1_col] = (2 * df[pivot_point_col]) - low_prev
    df[support2_col] = df[pivot_point_col] - (high_prev - low_prev)
    df[resistance2_col] = df[pivot_point_col] + (high_prev - low_prev)
    new_cols.extend([pivot_point_col, support1_col, support2_col, resistance1_col, resistance2_col])
    
    return df, new_cols