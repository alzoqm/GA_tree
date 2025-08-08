import pandas as pd
import numpy as np
import talib

# ==============================================================================
# 유틸리티 함수 (기존과 동일)
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
# 데이터 준비 (TA-Lib 함수에 맞게 numpy 배열 추출)
# ==============================================================================
def get_ohlcv_arrays(df):
    """TA-Lib 함수 입력을 위해 데이터프레임에서 numpy 배열을 추출합니다."""
    return (
        df['Open'].values,
        df['High'].values,
        df['Low'].values,
        df['Close'].values,
        df['Volume'].astype(float).values  # Volume은 float 타입이어야 합니다.
    )

# ==============================================================================
# 1. 기본 OHLCV 및 캔들 기반 피처
# ==============================================================================
def calculate_price_change_features(df):
    """
    가격 변화 및 캔들 몸통/꼬리 관련 기본 피처를 계산합니다.
    - 참고: 이 기능들은 기본적인 산술 연산이므로 TA-Lib에 해당 기능이 없습니다.
            `CDLREALBODY`가 몸통 크기를 계산하지만, 다른 지표들과의 일관성을 위해
            기존 numpy 연산을 유지합니다.
    """
    new_cols = []

    # 가격 변화율 (백분율) - pandas의 pct_change가 효율적
    df['close_change_pct'] = df['Close'].pct_change() * 100
    new_cols.append('close_change_pct')

    # 캔들 크기
    open_price, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    
    # TA-Lib 함수로 캔들 몸통 계산
    df['body_size'] = np.abs(open_price - close_price)
    
    df['upper_wick_size'] = high_price - np.maximum(open_price, close_price)
    df['lower_wick_size'] = np.minimum(open_price, close_price) - low_price
    df['total_range'] = high_price - low_price
    new_cols.extend(['body_size', 'upper_wick_size', 'lower_wick_size', 'total_range'])

    # 캔들 크기 비율 (분모가 0이 되는 것을 방지)
    df['body_to_range_ratio'] = df['body_size'] / (df['total_range'] + 1e-9)
    new_cols.append('body_to_range_ratio')

    df.fillna({'close_change_pct': 0}, inplace=True)
    return df, new_cols

# ==============================================================================
# 2. 추세 (Trend) 지표 (TA-Lib 적용)
# ==============================================================================
def calculate_ma(df, window):
    """단순 이동평균 (SMA)을 TA-Lib으로 계산합니다."""
    col_name = format_col_name('SMA', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.SMA(close_price, timeperiod=window)
    return df, [col_name]

def calculate_ema(df, window):
    """지수 이동평균 (EMA)을 TA-Lib으로 계산합니다."""
    col_name = format_col_name('EMA', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.EMA(close_price, timeperiod=window)
    return df, [col_name]

def calculate_vwma(df, window):
    """
    VWMA (거래량 가중 이동 평균)를 계산합니다.
    - 참고: TA-Lib는 VWMA를 직접 지원하지 않으므로 기존 pandas 구현을 유지합니다.
    """
    col_name = format_col_name('VWMA', window)
    price_volume = df['Close'] * df['Volume']
    sum_price_volume = price_volume.rolling(window=window).sum()
    sum_volume = df['Volume'].rolling(window=window).sum()
    df[col_name] = sum_price_volume / (sum_volume + 1e-9)
    return df, [col_name]

def calculate_macd(df, short_window, long_window, signal_window):
    """MACD, Signal Line, Histogram을 TA-Lib으로 계산합니다."""
    extra = f"{short_window}_{long_window}"
    macd_col = format_col_name('MACD', extra)
    signal_col = format_col_name('MACD_Signal', f"{extra}_{signal_window}")
    hist_col = format_col_name('MACD_Hist', f"{extra}_{signal_window}")
    
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    
    macd, signal, hist = talib.MACD(
        close_price,
        fastperiod=short_window,
        slowperiod=long_window,
        signalperiod=signal_window
    )
    df[macd_col] = macd
    df[signal_col] = signal
    df[hist_col] = hist
    
    return df, [macd_col, signal_col, hist_col]

def calculate_adx(df, window):
    """ADX, +DI, -DI를 TA-Lib으로 계산합니다."""
    adx_col = format_col_name('ADX', window)
    plus_di_col = format_col_name('DI_plus', window)
    minus_di_col = format_col_name('DI_minus', window)
    
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)

    df[adx_col] = talib.ADX(high_price, low_price, close_price, timeperiod=window)
    df[plus_di_col] = talib.PLUS_DI(high_price, low_price, close_price, timeperiod=window)
    df[minus_di_col] = talib.MINUS_DI(high_price, low_price, close_price, timeperiod=window)

    return df, [adx_col, plus_di_col, minus_di_col]

def calculate_ichimoku(df, short_window=9, mid_window=26, long_window=52):
    """
    일목균형표 (Ichimoku Cloud) 지표들을 계산합니다.
    - 참고: TA-Lib는 일목균형표를 지원하지 않으므로 기존 pandas 구현을 유지합니다.
    - [수정] Chikou Span의 미래 데이터 참조 버그를 수정했습니다.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    tenkan_sen_col = format_col_name('Ichimoku_Tenkan', short_window)
    kijun_sen_col = format_col_name('Ichimoku_Kijun', mid_window)
    senkou_a_col = format_col_name('Ichimoku_SenkouA', f"{short_window}_{mid_window}")
    senkou_b_col = format_col_name('Ichimoku_SenkouB', long_window)
    
    # --- [수정] 미래 데이터 참조 버그 수정 ---
    # 원인: close.shift(-mid_window)는 미래 데이터를 참조하는 심각한 Lookahead Bias를 유발합니다.
    # 해결: Chikou Span의 본래 목적인 '현재 가격과 과거 가격의 비교'를 위해, 
    #       '현재 종가 / 과거 종가' 비율을 새로운 피처로 정의합니다. 이는 과거 데이터만 참조합니다.
    chikou_ratio_col = format_col_name('Ichimoku_Chikou_Ratio', mid_window)
    # --- [수정] 끝 ---

    df[tenkan_sen_col] = (high.rolling(window=short_window).max() + low.rolling(window=short_window).min()) / 2
    df[kijun_sen_col] = (high.rolling(window=mid_window).max() + low.rolling(window=mid_window).min()) / 2
    df[senkou_a_col] = ((df[tenkan_sen_col] + df[kijun_sen_col]) / 2).shift(mid_window)
    df[senkou_b_col] = ((high.rolling(window=long_window).max() + low.rolling(window=long_window).min()) / 2).shift(mid_window)
    
    # --- [수정] 실제 계산 로직 변경 ---
    past_close = close.shift(mid_window) # 양수 값을 사용하여 과거 데이터를 가져옵니다.
    df[chikou_ratio_col] = close / (past_close + 1e-9) # 0으로 나누기 방지
    # --- [수정] 끝 ---

    # 반환되는 컬럼 리스트에 수정된 컬럼명 포함
    return df, [tenkan_sen_col, kijun_sen_col, senkou_a_col, senkou_b_col, chikou_ratio_col]

def calculate_dema(df, window):
    """DEMA (이중 지수 이동평균)를 계산합니다."""
    col_name = format_col_name('DEMA', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.DEMA(close_price, timeperiod=window)
    return df, [col_name]

def calculate_tema(df, window):
    """TEMA (삼중 지수 이동평균)를 계산합니다."""
    col_name = format_col_name('TEMA', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.TEMA(close_price, timeperiod=window)
    return df, [col_name]

def calculate_sar(df, acceleration=0.02, maximum=0.2):
    """Parabolic SAR을 계산합니다."""
    col_name = format_col_name('SAR', extra_str=f"{acceleration}_{maximum}")
    _, high_price, low_price, _, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.SAR(high_price, low_price, acceleration=acceleration, maximum=maximum)
    return df, [col_name]
    
def calculate_trima(df, window):
    """TRIMA (삼각 이동평균)을 계산합니다."""
    col_name = format_col_name('TRIMA', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.TRIMA(close_price, timeperiod=window)
    return df, [col_name]

# ==============================================================================
# 3. 모멘텀 (Momentum) 지표 (TA-Lib 적용)
# ==============================================================================
def calculate_rsi(df, window):
    """RSI (상대강도지수)를 TA-Lib으로 계산합니다."""
    col_name = format_col_name('RSI', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.RSI(close_price, timeperiod=window)
    return df, [col_name]

def calculate_stochastic_oscillator(df, k_window, d_window):
    """Stochastic Oscillator (%K, %D)를 TA-Lib으로 계산합니다."""
    extra = f"{k_window}_{d_window}"
    k_col = format_col_name('%K', extra)
    d_col = format_col_name('%D', extra)
    
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    
    slowk, slowd = talib.STOCH(
        high_price, low_price, close_price,
        fastk_period=k_window,
        slowk_period=d_window,
        slowk_matype=0, # SMA
        slowd_period=d_window,
        slowd_matype=0  # SMA
    )
    df[k_col] = slowk
    df[d_col] = slowd
    return df, [k_col, d_col]

def calculate_williams_r(df, window):
    """Williams %R을 TA-Lib으로 계산합니다."""
    col_name = format_col_name('Williams_R', window)
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.WILLR(high_price, low_price, close_price, timeperiod=window)
    return df, [col_name]

def calculate_cci(df, window):
    """CCI (Commodity Channel Index)를 TA-Lib으로 계산합니다."""
    col_name = format_col_name('CCI', window)
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.CCI(high_price, low_price, close_price, timeperiod=window)
    return df, [col_name]

def calculate_stochrsi(df, window=14, k_window=5, d_window=3):
    """Stochastic RSI를 계산합니다."""
    extra = f"{window}_{k_window}_{d_window}"
    k_col = format_col_name('STOCHRSI_K', extra)
    d_col = format_col_name('STOCHRSI_D', extra)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    k, d = talib.STOCHRSI(close_price, timeperiod=window, fastk_period=k_window, fastd_period=d_window, fastd_matype=0)
    df[k_col] = k
    df[d_col] = d
    return df, [k_col, d_col]

def calculate_ppo(df, fast_window=12, slow_window=26):
    """PPO (Percentage Price Oscillator)를 계산합니다."""
    col_name = format_col_name('PPO', f"{fast_window}_{slow_window}")
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.PPO(close_price, fastperiod=fast_window, slowperiod=slow_window, matype=0)
    return df, [col_name]

def calculate_roc(df, window):
    """ROC (Rate of Change)를 계산합니다."""
    col_name = format_col_name('ROC', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.ROC(close_price, timeperiod=window)
    return df, [col_name]
    
def calculate_ultosc(df, window1=7, window2=14, window3=28):
    """Ultimate Oscillator를 계산합니다."""
    col_name = format_col_name('ULTOSC', f"{window1}_{window2}_{window3}")
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.ULTOSC(high_price, low_price, close_price, timeperiod1=window1, timeperiod2=window2, timeperiod3=window3)
    return df, [col_name]

def calculate_mom(df, window):
    """MOM (Momentum)을 계산합니다."""
    col_name = format_col_name('MOM', window)
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.MOM(close_price, timeperiod=window)
    return df, [col_name]

# ==============================================================================
# 4. 변동성 (Volatility) 지표 (TA-Lib 적용)
# ==============================================================================
def calculate_bollinger_bands(df, window, num_std=2):
    """볼린저 밴드 및 밴드폭을 TA-Lib으로 계산합니다."""
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
        matype=0 # SMA
    )
    
    df[upper_col] = upper
    df[mid_col] = middle
    df[lower_col] = lower
    df[width_col] = (upper - lower) / (middle + 1e-9)
    
    return df, [upper_col, mid_col, lower_col, width_col]

def calculate_atr(df, window):
    """ATR (평균 실제 범위)을 TA-Lib으로 계산합니다."""
    col_name = format_col_name('ATR', window)
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.ATR(high_price, low_price, close_price, timeperiod=window)
    return df, [col_name]

def calculate_natr(df, window):
    """NATR (Normalized Average True Range)를 계산합니다."""
    col_name = format_col_name('NATR', window)
    _, high_price, low_price, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.NATR(high_price, low_price, close_price, timeperiod=window)
    return df, [col_name]

# ==============================================================================
# 5. 거래량 (Volume) 지표 (TA-Lib 적용)
# ==============================================================================
def calculate_obv(df):
    """OBV (On-Balance Volume)를 TA-Lib으로 계산합니다."""
    col_name = 'OBV'
    _, _, _, close_price, volume = get_ohlcv_arrays(df)
    df[col_name] = talib.OBV(close_price, volume)
    return df, [col_name]

def calculate_cmf(df, window):
    """
    CMF (Chaikin Money Flow)를 계산합니다.
    - 참고: TA-Lib는 CMF를 직접 지원하지 않으므로 기존 pandas 구현을 유지합니다.
            TA-Lib의 ADOSC (Chaikin A/D Oscillator)와는 다른 지표입니다.
    """
    col_name = format_col_name('CMF', window)
    close = df['Close']
    low = df['Low']
    high = df['High']
    volume = df['Volume']
    
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-9)
    mf_volume = mf_multiplier * volume
    
    df[col_name] = mf_volume.rolling(window=window).sum() / (volume.rolling(window=window).sum() + 1e-9)
    return df, [col_name]

def calculate_ad(df):
    """Chaikin A/D Line을 계산합니다."""
    col_name = 'AD'
    _, high_price, low_price, close_price, volume = get_ohlcv_arrays(df)
    df[col_name] = talib.AD(high_price, low_price, close_price, volume)
    return df, [col_name]

def calculate_adosc(df, fast_window=3, slow_window=10):
    """Chaikin A/D Oscillator를 계산합니다."""
    col_name = format_col_name('ADOSC', f"{fast_window}_{slow_window}")
    _, high_price, low_price, close_price, volume = get_ohlcv_arrays(df)
    df[col_name] = talib.ADOSC(high_price, low_price, close_price, volume, fastperiod=fast_window, slowperiod=slow_window)
    return df, [col_name]

def calculate_ht_dcperiod(df):
    """Hilbert Transform - Dominant Cycle Period를 계산합니다."""
    col_name = 'HT_DCPERIOD'
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.HT_DCPERIOD(close_price)
    return df, [col_name]

def calculate_ht_trendmode(df):
    """Hilbert Transform - Trend vs Cycle Mode를 계산합니다."""
    col_name = 'HT_TRENDMODE'
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[col_name] = talib.HT_TRENDMODE(close_price)
    return df, [col_name]

def calculate_ht_sine(df):
    """Hilbert Transform - SineWave (Sine and Lead Sine)를 계산합니다."""
    sine_col = 'HT_SINE'
    leadsine_col = 'HT_LEADSINE'
    _, _, _, close_price, _ = get_ohlcv_arrays(df)
    df[sine_col], df[leadsine_col] = talib.HT_SINE(close_price)
    return df, [sine_col, leadsine_col]
    

# ==============================================================================
# 6. 시간 기반 피처 (TA-Lib과 무관)
# ==============================================================================
def cyclic_encode_fn(df, timestamp_col, cycle):
    """
    시간 정보를 순환형 피처로 변환합니다. (기존과 동일)
    - 참고: 이 기능은 TA-Lib과 관련이 없습니다.
    """
    # (기존 코드와 동일하게 유지)
    # ...
    return df, [] # 실제 구현에서는 new_columns 반환

# ==============================================================================
# 7. 지지 (Support) 및 저항 (Resistance) 지표
# ==============================================================================
def calculate_support_resistance(df, window):
    """
    지지 및 저항 수준을 계산합니다.
    - 이동평균 기반 지지/저항은 TA-Lib의 MIN/MAX 함수로 대체합니다.
    - 피벗 포인트는 TA-Lib에서 지원하지 않으므로 기존 구현을 유지합니다.
    - [수정] 피벗 포인트 컬럼명에 window 값을 포함하여 중복을 방지합니다.
    """
    new_cols = []
    _, high_price, low_price, _, _ = get_ohlcv_arrays(df)

    # TA-Lib MIN/MAX 함수를 사용한 지지/저항
    support_ma_col = format_col_name('Support_MA', window)
    resistance_ma_col = format_col_name('Resistance_MA', window)
    df[support_ma_col] = talib.MIN(low_price, timeperiod=window)
    df[resistance_ma_col] = talib.MAX(high_price, timeperiod=window)
    new_cols.extend([support_ma_col, resistance_ma_col])

    # 피벗 포인트 (기존 pandas 구현 유지)
    high_prev = df['High'].shift(1)
    low_prev = df['Low'].shift(1)
    close_prev = df['Close'].shift(1)
    
    # --- 수정된 부분 시작 ---
    # 피벗 포인트 컬럼명에 window를 추가하여 고유성을 보장
    pivot_point_col = format_col_name('Pivot_Point', window)
    support1_col = format_col_name('Support1_Pivot', window)
    support2_col = format_col_name('Support2_Pivot', window)
    resistance1_col = format_col_name('Resistance1_Pivot', window)
    resistance2_col = format_col_name('Resistance2_Pivot', window)
    # --- 수정된 부분 끝 ---

    df[pivot_point_col] = (high_prev + low_prev + close_prev) / 3
    df[support1_col] = (2 * df[pivot_point_col]) - high_prev
    df[resistance1_col] = (2 * df[pivot_point_col]) - low_prev
    df[support2_col] = df[pivot_point_col] - (high_prev - low_prev)
    df[resistance2_col] = df[pivot_point_col] + (high_prev - low_prev)
    new_cols.extend([pivot_point_col, support1_col, support2_col, resistance1_col, resistance2_col])
    
    return df, new_cols
    
# ==============================================================================
# 8. 캔들스틱 패턴 인식 (TA-Lib 적용)
# ==============================================================================
def calculate_all_candlestick_patterns(df):
    """
    모든 캔들스틱 패턴을 TA-Lib으로 계산하고, 결과를 True/False로 저장합니다.
    전수 조사를 통해 확인된 모든 양방향성 패턴(강세/약세)을 별도 컬럼으로 분리합니다.
    """
    open_p, high_p, low_p, close_p, _ = get_ohlcv_arrays(df)
    
    # --- 1. 단방향성 패턴 처리 (기존 + 신규) ---
    # (결과가 0이 아니면 True, 0이면 False로 변환)
    df['InvertedHammers'] = talib.CDLINVERTEDHAMMER(open_p, high_p, low_p, close_p) != 0
    df['Hammers'] = talib.CDLHAMMER(open_p, high_p, low_p, close_p) != 0
    df['HangingMen'] = talib.CDLHANGINGMAN(open_p, high_p, low_p, close_p) != 0
    df['DarkCloudCovers'] = talib.CDLDARKCLOUDCOVER(open_p, high_p, low_p, close_p) != 0
    df['Dojis'] = talib.CDLDOJI(open_p, high_p, low_p, close_p) != 0
    df['DragonflyDojis'] = talib.CDLDRAGONFLYDOJI(open_p, high_p, low_p, close_p) != 0
    df['GravestoneDojis'] = talib.CDLGRAVESTONEDOJI(open_p, high_p, low_p, close_p) != 0
    df['MorningStars'] = talib.CDLMORNINGSTAR(open_p, high_p, low_p, close_p) != 0
    df['MorningStarDojis'] = talib.CDLMORNINGDOJISTAR(open_p, high_p, low_p, close_p) != 0
    df['PiercingPatterns'] = talib.CDLPIERCING(open_p, high_p, low_p, close_p) != 0
    df['ShootingStars'] = talib.CDLSHOOTINGSTAR(open_p, high_p, low_p, close_p) != 0
    
    # [신규 추가된 단방향성 패턴]
    df['3BlackCrows'] = talib.CDL3BLACKCROWS(open_p, high_p, low_p, close_p) != 0
    df['3WhiteSoldiers'] = talib.CDL3WHITESOLDIERS(open_p, high_p, low_p, close_p) != 0
    df['3StarsInSouth'] = talib.CDL3STARSINSOUTH(open_p, high_p, low_p, close_p) != 0

    # --- 2. 양방향성 패턴 처리 (기존 + 신규) ---
    # (강세/약세를 별도의 Boolean 컬럼으로 분리)
    
    # Harami Pattern
    harami = talib.CDLHARAMI(open_p, high_p, low_p, close_p)
    df['BullishHarami'] = harami > 0
    df['BearishHarami'] = harami < 0
    
    # Engulfing Pattern
    engulfing = talib.CDLENGULFING(open_p, high_p, low_p, close_p)
    df['BullishEngulfing'] = engulfing > 0
    df['BearishEngulfing'] = engulfing < 0
    
    # Doji Star Pattern
    doji_star = talib.CDLDOJISTAR(open_p, high_p, low_p, close_p)
    df['BullishDojiStar'] = doji_star > 0
    df['BearishDojiStar'] = doji_star < 0

    # [신규 추가된 양방향성 패턴]
    # Tasuki Gap
    tasuki_gap = talib.CDLTASUKIGAP(open_p, high_p, low_p, close_p)
    df['BullishTasukiGap'] = tasuki_gap > 0
    df['BearishTasukiGap'] = tasuki_gap < 0

    # X-Side Gap 3 Methods
    xside_gap = talib.CDLXSIDEGAP3METHODS(open_p, high_p, low_p, close_p)
    df['BullishXSideGap3Methods'] = xside_gap > 0
    df['BearishXSideGap3Methods'] = xside_gap < 0

    # Spinning Top
    spinning_top = talib.CDLSPINNINGTOP(open_p, high_p, low_p, close_p)
    df['BullishSpinningTop'] = spinning_top > 0
    df['BearishSpinningTop'] = spinning_top < 0

    # Rise/Fall 3 Methods
    rise_fall = talib.CDLRISEFALL3METHODS(open_p, high_p, low_p, close_p)
    df['BullishRise3Methods'] = rise_fall > 0
    df['BearishFall3Methods'] = rise_fall < 0

    # 생성된 모든 캔들 컬럼의 최종 목록
    candle_cols = [
        # 단방향성 패턴
        'InvertedHammers', 'Hammers', 'HangingMen', 'DarkCloudCovers', 'Dojis', 
        'DragonflyDojis', 'GravestoneDojis', 'MorningStars', 
        'MorningStarDojis', 'PiercingPatterns', 'ShootingStars',
        '3BlackCrows', '3WhiteSoldiers', '3StarsInSouth',
        
        # 양방향성 패턴 (분리됨)
        'BullishHarami', 'BearishHarami', 
        'BullishEngulfing', 'BearishEngulfing',
        'BullishDojiStar', 'BearishDojiStar',
        'BullishTasukiGap', 'BearishTasukiGap',
        'BullishXSideGap3Methods', 'BearishXSideGap3Methods',
        'BullishSpinningTop', 'BearishSpinningTop',
        'BullishRise3Methods', 'BearishFall3Methods'
    ]
    
    return df, candle_cols


# ==============================================================================
# 9. 신규 복합 특성 (Composite & Ratio Features)
# 아래 함수들은 기존에 계산된 지표들을 조합하여 새로운 특성을 생성합니다.
# YAML 설정 시, 이 함수들이 필요한 기본 지표들(RSI, ADX 등)보다 나중에 호출되도록 순서를 조정해야 합니다.
# ==============================================================================

def calculate_rsi_adx_ratio(df: pd.DataFrame, rsi_window: int, adx_window: int) -> tuple[pd.DataFrame, list[str]]:
    """
    RSI와 ADX의 비율을 계산하여 모멘텀과 추세의 상대적 강도를 파악합니다.

    Args:
        df (pd.DataFrame): 'RSI_{rsi_window}'와 'ADX_{adx_window}' 컬럼이 포함된 데이터프레임.
        rsi_window (int): RSI 계산에 사용된 윈도우.
        adx_window (int): ADX 계산에 사용된 윈도우.

    Returns:
        tuple[pd.DataFrame, list[str]]: 특성이 추가된 데이터프레임과 새로운 컬럼명 리스트.
    """
    rsi_col = format_col_name('RSI', rsi_window)
    adx_col = format_col_name('ADX', adx_window)
    new_col = format_col_name('RSI_ADX_Ratio', f"{rsi_window}_{adx_window}")

    if rsi_col in df and adx_col in df:
        df[new_col] = df[rsi_col] / (df[adx_col] + 1e-9)
        return df, [new_col]
    else:
        # 필요한 컬럼이 없는 경우 경고 메시지를 출력하고 아무 작업도 하지 않을 수 있습니다.
        # print(f"Warning: Required columns '{rsi_col}' or '{adx_col}' not found for RSI/ADX Ratio.")
        return df, []

def calculate_macd_atr_ratio(df: pd.DataFrame, short_window: int, long_window: int, signal_window: int, atr_window: int) -> tuple[pd.DataFrame, list[str]]:
    """
    MACD를 ATR로 정규화하여 변동성 대비 추세 신호의 강도를 측정합니다.

    Args:
        df (pd.DataFrame): MACD와 ATR 컬럼이 포함된 데이터프레임.
        short_window (int): MACD 단기 윈도우.
        long_window (int): MACD 장기 윈도우.
        signal_window (int): MACD 시그널 윈도우.
        atr_window (int): ATR 윈도우.

    Returns:
        tuple[pd.DataFrame, list[str]]: 특성이 추가된 데이터프레임과 새로운 컬럼명 리스트.
    """
    macd_col = format_col_name('MACD', f"{short_window}_{long_window}")
    atr_col = format_col_name('ATR', atr_window)
    new_col = format_col_name('MACD_ATR_Ratio', f"{short_window}_{long_window}_{atr_window}")

    if macd_col in df and atr_col in df:
        df[new_col] = df[macd_col] / (df[atr_col] + 1e-9)
        return df, [new_col]
    else:
        return df, []

def calculate_short_long_ratio(df: pd.DataFrame, indicator_name: str, short_window: int, long_window: int) -> tuple[pd.DataFrame, list[str]]:
    """
    단기 지표와 장기 지표의 비율을 계산합니다. (예: 단기/장기 RSI 비율)

    Args:
        df (pd.DataFrame): 지표 컬럼들이 포함된 데이터프레임.
        indicator_name (str): 비율을 계산할 지표의 기본 이름 (예: 'RSI', 'BB_Width').
        short_window (int): 단기 윈도우.
        long_window (int): 장기 윈도우.

    Returns:
        tuple[pd.DataFrame, list[str]]: 특성이 추가된 데이터프레임과 새로운 컬럼명 리스트.
    """
    # BB_Width는 파라미터가 2개이므로 특별 처리
    if indicator_name == 'BB_Width':
        # 이 예제에서는 num_std=2로 고정한다고 가정합니다. 필요 시 파라미터 추가 가능.
        short_col = format_col_name(indicator_name, f"{short_window}_2")
        long_col = format_col_name(indicator_name, f"{long_window}_2")
    else:
        short_col = format_col_name(indicator_name, short_window)
        long_col = format_col_name(indicator_name, long_window)
    
    new_col = format_col_name(f"{indicator_name}_Ratio", f"{short_window}_{long_window}")

    if short_col in df and long_col in df:
        df[new_col] = df[short_col] / (df[long_col] + 1e-9)
        return df, [new_col]
    else:
        return df, []

def calculate_indicator_roc(df: pd.DataFrame, indicator_name: str, indicator_window: int, roc_period: int) -> tuple[pd.DataFrame, list[str]]:
    """
    주어진 지표의 변화율(Rate of Change)을 계산합니다.

    Args:
        df (pd.DataFrame): 지표 컬럼이 포함된 데이터프레임.
        indicator_name (str): 변화율을 계산할 지표의 이름 (예: 'RSI', 'ATR').
        indicator_window (int): 해당 지표 계산에 사용된 윈도우.
        roc_period (int): 변화율을 계산할 기간.

    Returns:
        tuple[pd.DataFrame, list[str]]: 특성이 추가된 데이터프레임과 새로운 컬럼명 리스트.
    """
    indicator_col = format_col_name(indicator_name, indicator_window)
    new_col = format_col_name(f"{indicator_name}_ROC", f"{indicator_window}_{roc_period}")
    
    if indicator_col in df:
        shifted_indicator = df[indicator_col].shift(roc_period)
        df[new_col] = (df[indicator_col] - shifted_indicator) / (abs(shifted_indicator) + 1e-9) * 100
        return df, [new_col]
    else:
        return df, []

def calculate_volume_spike(df: pd.DataFrame, window: int, factor: float) -> tuple[pd.DataFrame, list[str]]:
    """
    거래량이 이동평균 대비 특정 배수 이상으로 급증했는지 여부를 판단합니다.

    Args:
        df (pd.DataFrame): 'Volume' 컬럼이 포함된 데이터프레임.
        window (int): 거래량 이동평균을 계산할 윈도우.
        factor (float): 급증을 판단할 배수 (예: 2.0).

    Returns:
        tuple[pd.DataFrame, list[str]]: 특성이 추가된 데이터프레임과 새로운 컬럼명 리스트.
    """
    new_col = format_col_name('Volume_Spike', f"{window}_{factor}")
    avg_volume = df['Volume'].rolling(window=window).mean()
    df[new_col] = df['Volume'] > (avg_volume * factor)
    return df, [new_col]

def calculate_consecutive_candles(df: pd.DataFrame, period: int, direction: str) -> tuple[pd.DataFrame, list[str]]:
    """
    연속적인 양봉 또는 음봉 캔들의 출현 여부를 판단합니다.

    Args:
        df (pd.DataFrame): 'Open', 'Close' 컬럼이 포함된 데이터프레임.
        period (int): 연속으로 간주할 캔들의 수 (예: 3).
        direction (str): 'bullish' 또는 'bearish'.

    Returns:
        tuple[pd.DataFrame, list[str]]: 특성이 추가된 데이터프레임과 새로운 컬럼명 리스트.
    """
    if direction == 'bullish':
        is_positive = df['Close'] > df['Open']
        new_col = format_col_name('Consecutive_Bullish', period)
    elif direction == 'bearish':
        is_positive = df['Close'] < df['Open']
        new_col = format_col_name('Consecutive_Bearish', period)
    else:
        return df, []

    consecutive_count = is_positive.rolling(window=period).sum()
    df[new_col] = (consecutive_count == period)
    return df, [new_col]

def calculate_confirmation_signal(df: pd.DataFrame, rsi_window: int, adx_window: int, macd_short: int, macd_long: int, rsi_threshold: int, adx_threshold: int, direction: str) -> tuple[pd.DataFrame, list[str]]:
    """
    여러 지표를 조합하여 강세 또는 약세 신호의 확인(Confirmation) 여부를 판단합니다.

    Args:
        df (pd.DataFrame): 필요한 지표 컬럼들이 포함된 데이터프레임.
        rsi_window (int): RSI 윈도우.
        adx_window (int): ADX 윈도우.
        macd_short (int): MACD 단기 윈도우.
        macd_long (int): MACD 장기 윈도우.
        rsi_threshold (int): RSI 기준값.
        adx_threshold (int): ADX 기준값.
        direction (str): 'bullish' 또는 'bearish'.

    Returns:
        tuple[pd.DataFrame, list[str]]: 특성이 추가된 데이터프레임과 새로운 컬럼명 리스트.
    """
    rsi_col = format_col_name('RSI', rsi_window)
    adx_col = format_col_name('ADX', adx_window)
    macd_col = format_col_name('MACD', f"{macd_short}_{macd_long}")

    if not all(col in df for col in [rsi_col, adx_col, macd_col]):
        return df, []

    base_cond = df[adx_col] > adx_threshold

    if direction == 'bullish':
        cond = base_cond & (df[rsi_col] > rsi_threshold) & (df[macd_col] > 0)
        new_col = format_col_name('Bullish_Confirmation', f"{rsi_window}_{adx_window}")
    elif direction == 'bearish':
        cond = base_cond & (df[rsi_col] < rsi_threshold) & (df[macd_col] < 0)
        new_col = format_col_name('Bearish_Confirmation', f"{rsi_window}_{adx_window}")
    else:
        return df, []
        
    df[new_col] = cond
    return df, [new_col]

# ==============================================================================
# 메인 실행 부분 (예시)
# ==============================================================================
if __name__ == '__main__':
    # 가상 데이터프레임 생성
    data = {
        'Open': np.random.uniform(95, 105, 200),
        'High': np.random.uniform(105, 110, 200),
        'Low': np.random.uniform(90, 95, 200),
        'Close': np.random.uniform(98, 108, 200),
        'Volume': np.random.uniform(100000, 500000, 200)
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    
    # --- TA-Lib으로 변환된 함수들 호출 ---
    
    # 1. 추세 (Trend) 지표
    df, price_col = calculate_price_change_features(df)
    df, ma_col = calculate_ma(df, window=14)
    df, ema_col = calculate_ema(df, window=24)
    df, vwma_col = calculate_vwma(df, window=5) # TA-Lib 미지원
    df, macd_cols = calculate_macd(df, short_window=12, long_window=26, signal_window=9)
    df, adx_cols = calculate_adx(df, window=14)
    df, ichimoku_cols = calculate_ichimoku(df) # TA-Lib 미지원

    # 2. 모멘텀 (Momentum) 지표
    df, rsi_col = calculate_rsi(df, window=14)
    df, stoch_cols = calculate_stochastic_oscillator(df, k_window=14, d_window=3)
    df, williams_r_col = calculate_williams_r(df, window=14)
    df, cci_col = calculate_cci(df, window=20)

    # 3. 변동성 (Volatility) 지표
    df, bb_cols = calculate_bollinger_bands(df, window=20, num_std=2)
    df, atr_col = calculate_atr(df, window=14)

    # 4. 거래량 (Volume) 지표
    df, obv_col = calculate_obv(df)
    df, cmf_col = calculate_cmf(df, window=20) # TA-Lib 미지원

    # 5. 지지 (Support) 및 저항 (Resistance) 지표
    df, sr_cols = calculate_support_resistance(df, window=14)
    
    # 6. 캔들스틱 패턴
    df, candle_cols = calculate_all_candlestick_patterns(df)

    # 결측치 처리 (TA-Lib 계산 시 초반 데이터는 NaN이 됨)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print("DataFrame shape after adding TA-Lib features:", df.shape)
    print("\nAdded Columns (sample):")
    print(df[['Close', 'SMA_14', 'RSI_14', 'MACD_12_26', 'ADX_14', 'BB_Upper_20_2', 'Hammers']].tail())
