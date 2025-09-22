# /data/merge_dataset.py
# === 수정된 data/merge_dataset.py 파일의 전체 코드 ===

import gc
import re

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
import yaml
try:
    import data.ta_lib_feature_generator as talib_feat
except Exception:
    # Lazy/optional import to allow using classification helpers without TA-Lib
    talib_feat = None
from typing import List, Dict, Any, Tuple


def optimize_dataframe_memory(df: pd.DataFrame, convert_bool: bool = True) -> pd.DataFrame:
    """Downcast numeric dtypes (and optional bools) to reduce memory footprint."""
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    int_cols = df.select_dtypes(include=["int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"]).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    if convert_bool:
        bool_cols = df.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            df[col] = df[col].astype(np.uint8)

    return df

# ==============================================================================
#           [수정] 피처 분류 템플릿 제거
# ==============================================================================
# FEATURE_NUM_TEMPLATE, PRICE_AND_TREND_FEATURES_TEMPLATE, FEATURE_BOOL_TEMPLATE
# 변수들이 여기서 삭제되었습니다. 이제 이 정보는 YAML 설정 파일에서 가져옵니다.


# ==============================================================================
#                      기존 함수 (수정 없음)
# ==============================================================================

def generate_multi_timeframe_features(
    df: pd.DataFrame,
    timestamp_col: str,
    target_timeframes: list,
    feature_params: dict
) -> Tuple[pd.DataFrame, List[str]]:
    """
    서로 다른 시간 단위의 기술적 분석 지표를 생성하고 기준 데이터프레임에 통합합니다.
    (이전 코드와 동일하며, 수정되지 않았습니다)
    """
    # ... (내부 코드는 변경 없음) ...
    # --- 0. 입력 유효성 검사 및 초기 설정 ---
    if not target_timeframes:
        raise ValueError("`target_timeframes` 리스트는 비어 있을 수 없습니다.")
        
    if timestamp_col not in df.columns:
        raise ValueError(f"'{timestamp_col}' 컬럼이 데이터프레임에 존재하지 않습니다.")

    source_df = df.copy()
    
    source_df[timestamp_col] = pd.to_datetime(source_df[timestamp_col])
    source_df.set_index(timestamp_col, inplace=True)
    
    all_new_columns = []

    # talib 기반 기능이 필요하므로 모듈 체크
    if talib_feat is None:
        raise ImportError("TA-Lib 기반 피처 생성을 위해 'data.ta_lib_feature_generator' 임포트가 필요합니다. TA-Lib 설치를 확인하세요.")

    # --- 1. 기준 시간 단위 결정 및 기준 데이터프레임 생성 ---
    time_deltas = [pd.to_timedelta(tf.replace('m', 'T').replace('h','H').replace('d','D')) for tf in target_timeframes]
    base_timeframe = target_timeframes[np.argmin(time_deltas)]
    
    print(f"기준 시간 단위가 '{base_timeframe}'으로 설정되었습니다.")

    agg_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'fundingRate': 'last',
        'fundingRate_est': 'last',
        'markPrice_est': 'last',
        'markLow_est': 'min',
        'markHigh_est': 'max',
    }
    # 시간 단위 문자열 변환 (e.g., '1h' -> '1H')
    resample_freq = base_timeframe.replace('m', 'T').replace('h','H').replace('d','D')
    base_df = source_df.resample(resample_freq).agg(agg_rules)
    # Drop only if essential OHLCV are missing; allow estimated columns to be NaN
    base_df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    base_df = optimize_dataframe_memory(base_df)

    # --- 2. 시간 단위별 피처 계산 및 통합 ---
    features_by_timeframe = {}

    for timeframe in sorted(list(set(target_timeframes)), key=lambda x: pd.to_timedelta(x.replace('m', 'T').replace('h','H').replace('d','D'))):
        if timeframe not in feature_params:
            print(f"'{timeframe}'에 대한 피처 설정이 없어 건너뜁니다.")
            continue
            
        print(f"--- '{timeframe}' 시간 단위 피처 계산 시작 ---")

        resample_freq = timeframe.replace('m', 'T').replace('h','H').replace('d','D')
        df_resampled = source_df.resample(resample_freq).agg(agg_rules)
        df_resampled = df_resampled.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        # 누적 의존 피처를 전달하기 위한 작업용 DataFrame
        df_work = df_resampled.copy()

        for func_name, params_list in feature_params[timeframe].items():
            try:
                feature_function = getattr(talib_feat, func_name)
            except AttributeError:
                print(f"경고: '{func_name}' 함수를 찾을 수 없습니다. 건너뜁니다.")
                continue

            for params in params_list:
                # 의존 피처가 필요한 함수(예: ATR -> ATR-MA)를 위해 누적 DataFrame을 전달
                # 주의: feature_function은 df_work를 인플레이스 변경할 수 있음
                original_df = df_work
                df_with_feature, new_cols = feature_function(df_work, **params)

                # in-place insert가 계속 누적되면 DataFrame이 조각날 수 있으므로,
                # 동일 객체를 그대로 돌려받고 새 컬럼이 추가된 경우에는 사본으로 치환한다.
                if new_cols and df_with_feature is original_df:
                    df_work = df_with_feature.copy()
                else:
                    df_work = df_with_feature

                # 새로 생성된 컬럼들만 별도 복사 후 타임프레임 접미사 부여
                if new_cols:
                    renamed_cols = [f"{col}_{timeframe}" for col in new_cols]
                    renamed_df = df_with_feature[new_cols].copy()
                    renamed_df.columns = renamed_cols
                else:
                    renamed_cols = []
                    renamed_df = pd.DataFrame(index=df_with_feature.index)

                # 타임프레임별 피처 모음에 병합
                if renamed_cols:
                    if timeframe not in features_by_timeframe:
                        optimized_df = optimize_dataframe_memory(renamed_df[renamed_cols])
                        features_by_timeframe[timeframe] = optimized_df
                    else:
                        joined = features_by_timeframe[timeframe].join(renamed_df[renamed_cols])
                        features_by_timeframe[timeframe] = optimize_dataframe_memory(joined)

                all_new_columns.extend(renamed_cols)
                print(f"  - {func_name} ({params}): {len(renamed_cols)}개 컬럼 생성 완료")

    # --- 3. 피처 병합 ---
    print("\n--- 모든 피처를 기준 데이터프레임에 병합합니다 ---")
    final_df = base_df.copy()

    sorted_timeframes = sorted(features_by_timeframe.keys(), key=lambda x: pd.to_timedelta(x.replace('m', 'T').replace('h','H').replace('d','D')))

    for timeframe in sorted_timeframes:
        feature_df = features_by_timeframe[timeframe]
        if timeframe == base_timeframe:
            final_df = final_df.join(feature_df)
            print(f"'{timeframe}' 피처가 직접 통합되었습니다.")
        else:
            final_df = pd.merge_asof(
                left=final_df,
                right=feature_df,
                left_index=True,
                right_index=True,
                direction='backward'
            )
            print(f"'{timeframe}' 피처가 'merge_asof'로 통합되었습니다.")

        final_df = optimize_dataframe_memory(final_df)
        gc.collect()

    # At final stage, keep rows where essential OHLCV exist; allow feature gaps and est columns
    final_df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    final_df.reset_index(inplace=True)
    final_df = optimize_dataframe_memory(final_df, convert_bool=False)

    print("\n최종 피처 생성 및 통합이 완료되었습니다.")
    return final_df, sorted(list(set(all_new_columns)))

# ==============================================================================
#           [수정] 모델 설정을 동적으로 생성하는 헬퍼 함수
# ==============================================================================

def _generate_model_config_from_features(
    all_generated_cols: List[str],
    classification_config: Dict[str, Any],
    feature_frame: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """
    생성된 전체 피처 목록과 YAML에서 로드한 '분류 설정'을 받아
    모델 초기화에 필요한 설정 딕셔너리들을 생성합니다.

    추가로 feature_frame을 전달하면 YAML에서 정의되지 않은 피처들에 대해
    데이터 기반의 자동 분류(불리언/숫자형)와 범위 추정을 수행합니다.

    개선 사항:
    - 접두어 startswith 매칭을 경계 기반 정규식으로 대체: ^key(?:_|$)
    - 컬럼명 끝의 타임프레임 접미사(_5m/_1h/_4h/_1d 등)는 매칭 시에만 제거
    - 교차 카테고리 키 중복/다중 매칭 경고 출력
    """
    print("\n--- YAML 설정을 기반으로 모델 config를 동적으로 생성합니다 ---")

    feature_num_template = classification_config.get('feature_num', {})
    feature_comparison_template = classification_config.get('feature_comparison', [])
    feature_bool_template = classification_config.get('feature_bool', [])

    # Helper: strip trailing timeframe suffix like _5m, _1h, _4h, _1d (matching only)
    tf_suffix_re = re.compile(r"^(.*)_(\d+[mhd])$")
    def strip_tf(name: str) -> str:
        m = tf_suffix_re.match(name)
        return m.group(1) if m else name

    # Build boundary-aware patterns and sort by key length desc
    def make_patterns(keys):
        sorted_keys = sorted(keys, key=len, reverse=True)
        return [(k, re.compile(rf"^{re.escape(k)}(?:_|$)")) for k in sorted_keys]

    pat_bool = make_patterns(feature_bool_template)
    pat_cmp = make_patterns(feature_comparison_template)
    pat_num = make_patterns(list(feature_num_template.keys()))

    fallback_quantiles = classification_config.get('fallback_quantiles', [0.5, 99.5])
    if not isinstance(fallback_quantiles, (list, tuple)) or len(fallback_quantiles) != 2:
        fallback_quantiles = [0.5, 99.5]
    fallback_range = classification_config.get('fallback_feature_num_range', [-1.0, 1.0])
    if not isinstance(fallback_range, (list, tuple)) or len(fallback_range) != 2:
        fallback_range = [-1.0, 1.0]

    bool_registry: set[str] = set()

    # Cross-category key overlap check (warn only)
    set_bool = set(feature_bool_template)
    set_cmp = set(feature_comparison_template)
    set_num = set(feature_num_template.keys())
    overlaps = {
        'bool&cmp': sorted(list(set_bool & set_cmp)),
        'bool&num': sorted(list(set_bool & set_num)),
        'cmp&num': sorted(list(set_cmp & set_num)),
    }
    for tag, ov in overlaps.items():
        if ov:
            print(f"경고: 분류 키 중복 감지({tag}): {ov}")

    final_feature_num: Dict[str, Tuple[float, float]] = {}
    final_feature_bool: List[str] = []
    comparison_feature_list: List[str] = []
    unclassified_cols: List[str] = []
    auto_classified_num: Dict[str, Tuple[float, float]] = {}
    auto_classified_bool: List[str] = []

    def match_keys(name: str, patterns):
        base = strip_tf(name)
        return [k for (k, p) in patterns if p.match(base)]

    for col_name in all_generated_cols:
        matches_bool = match_keys(col_name, pat_bool)
        matches_cmp = match_keys(col_name, pat_cmp)
        matches_num = match_keys(col_name, pat_num)

        # Multi-category conflict detection (warn only)
        matched_categories = [c for c, arr in (
            ('bool', matches_bool), ('cmp', matches_cmp), ('num', matches_num)
        ) if arr]
        if len(matched_categories) > 1:
            print(f"경고: 다중 카테고리 매칭 감지: '{col_name}' -> {matched_categories}")

        # Priority: bool > cmp > num (backward compatible)
        if matches_bool:
            if col_name not in bool_registry:
                final_feature_bool.append(col_name)
                bool_registry.add(col_name)
            continue
        if matches_cmp:
            comparison_feature_list.append(col_name)
            continue
        if matches_num:
            key = matches_num[0]
            final_feature_num[col_name] = feature_num_template[key]
            continue

        # Attempt automatic classification for unmatched columns
        handled = False
        if feature_frame is not None and col_name in feature_frame.columns:
            series = feature_frame[col_name]

            def _looks_like_bool(s: pd.Series) -> bool:
                if ptypes.is_bool_dtype(s):
                    return True
                finite = pd.to_numeric(s, errors='coerce')
                finite = finite.replace([np.inf, -np.inf], np.nan).dropna()
                if finite.empty:
                    return False
                unique_vals = set(finite.unique())
                return unique_vals.issubset({0.0, 1.0})

            if _looks_like_bool(series):
                if col_name not in bool_registry:
                    final_feature_bool.append(col_name)
                    bool_registry.add(col_name)
                    auto_classified_bool.append(col_name)
                handled = True
            else:
                numeric_series = pd.to_numeric(series, errors='coerce')
                numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan).dropna()
                if not numeric_series.empty:
                    low_q, high_q = sorted(fallback_quantiles)
                    lower = float(np.nanpercentile(numeric_series, low_q))
                    upper = float(np.nanpercentile(numeric_series, high_q))
                    if lower == upper:
                        margin = abs(lower) if lower != 0 else 1.0
                        lower -= margin
                        upper += margin
                    final_feature_num[col_name] = (lower, upper)
                    auto_classified_num[col_name] = (lower, upper)
                else:
                    final_feature_num[col_name] = (float(fallback_range[0]), float(fallback_range[1]))
                    auto_classified_num[col_name] = (float(fallback_range[0]), float(fallback_range[1]))
                handled = True

        if handled:
            continue

        # No match after automated handling
        unclassified_cols.append(col_name)

    # 5. 최종 feature_comparison_map 생성 (모든 비교 피처 간 pairwise 비교)
    comparison_feature_list = list(dict.fromkeys(comparison_feature_list))

    final_feature_comparison_map = {
        feat: [other for other in comparison_feature_list if other != feat]
        for feat in comparison_feature_list
    }

    print(f"  - 숫자 비교(feature_num) 타입 피처: {len(final_feature_num)}개")
    print(f"  - 피처간 비교(feature_comparison) 타입 피처: {len(final_feature_comparison_map)}개")
    print(f"  - 불리언(feature_bool) 타입 피처: {len(final_feature_bool)}개")

    if auto_classified_num:
        sample_items = list(auto_classified_num.items())[:3]
        print(
            "  - 자동 분류된 숫자형 피처 {}개 (예시: {})".format(
                len(auto_classified_num), sample_items
            )
        )

    if auto_classified_bool:
        print(
            "  - 자동 분류된 불리언 피처 {}개 (예시: {})".format(
                len(auto_classified_bool), auto_classified_bool[:5]
            )
        )

    if unclassified_cols:
        print(f"  - 경고: 분류되지 않은 피처 {len(unclassified_cols)}개 발견: {unclassified_cols}")

    return {
        'feature_num': final_feature_num,
        'feature_comparison_map': final_feature_comparison_map,
        'feature_bool': final_feature_bool,
    }


# ==============================================================================
#            메인 래퍼 함수 및 테스트 실행 부분 (변경 없음)
# ==============================================================================
def run_feature_generation_from_yaml(
    df: pd.DataFrame,
    timestamp_col: str,
    target_timeframes: list,
    yaml_config_path: str,
    classification_config: Dict[str, Any]
) -> Tuple[pd.DataFrame | None, Dict[str, Any] | None]:
    """
    YAML 설정 파일을 기반으로 Multi-Timeframe 피처 생성을 실행하고,
    모델에 필요한 설정 딕셔너리까지 함께 반환하는 래퍼 함수입니다.
    """
    # ... (내부 코드는 변경 없음) ...
    # 1. YAML 설정 파일 로드
    print(f"'{yaml_config_path}' 에서 피처 생성 설정을 로드합니다.")
    try:
        with open(yaml_config_path, 'r', encoding='utf-8') as file:
            full_config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"오류: 설정 파일 '{yaml_config_path}'을(를) 찾을 수 없습니다.")
        return None, None
    except Exception as e:
        print(f"오류: YAML 파일을 읽는 중 에러 발생 - {e}")
        return None, None
        
    # 2. 요청된 타임프레임에 대한 파라미터만 필터링
    feature_generation_params = {}
    valid_timeframes = []
    for tf in target_timeframes:
        if tf in full_config:
            feature_generation_params[tf] = full_config[tf]
            valid_timeframes.append(tf)
        else:
            print(f"경고: '{tf}'에 대한 설정이 YAML 파일에 없습니다. 이 타임프레임은 건너뜁니다.")
    
    if not feature_generation_params:
        print("오류: 유효한 타임프레임 설정이 하나도 없습니다. 프로세스를 중단합니다.")
        return df, {}

    # 3. 메인 피처 생성 함수 호출
    print("\nYAML 설정에 기반하여 Multi-Timeframe 피처 생성을 시작합니다.")
    final_dataframe, added_cols = generate_multi_timeframe_features(
        df=df,
        timestamp_col=timestamp_col,
        target_timeframes=valid_timeframes,
        feature_params=feature_generation_params
    )
    
    # 4. 생성된 피처 목록으로 모델 설정(config) 생성
    model_config = _generate_model_config_from_features(
        added_cols,
        classification_config,
        feature_frame=final_dataframe,
    )

    return final_dataframe, model_config


if __name__ == '__main__':
    # 1. 가상 데이터 생성 (1분봉, 15일치 데이터)
    print("1. 가상 1분봉 데이터 생성...")
    time_index = pd.date_range(start='2024-07-15 00:00', periods=15 * 24 * 60, freq='T')
    data_size = len(time_index)
    data = {
        'Open time': time_index,
        'Open': np.random.uniform(-0.1, 0.1, data_size).cumsum() + 2000,
        'Close': np.random.uniform(-0.1, 0.1, data_size).cumsum() + 2000,
        'Volume': np.random.uniform(10, 100, data_size)
    }
    data['High'] = np.maximum(data['Open'], data['Close']) + np.random.uniform(0, 0.2, data_size)
    data['Low'] = np.minimum(data['Open'], data['Close']) - np.random.uniform(0, 0.2, data_size)
    
    source_df = pd.DataFrame(data)
    print(f"생성된 원본 데이터 Shape: {source_df.shape}")

    # 2. YAML 파일 생성 (테스트용)
    yaml_content = """
5m:
  calculate_ema:
    - {window: 12}
  calculate_rsi:
    - {window: 14}
  calculate_stochastic_oscillator:
    - {k_window: 14, d_window: 3}
15m:
  calculate_ma:
    - {window: 20}
    - {window: 50}
  calculate_cci:
    - {window: 20}
1h:
  calculate_macd:
    - {short_window: 12, long_window: 26, signal_window: 9}
  calculate_bollinger_bands:
    - {window: 20, num_std: 2}
  calculate_adx:
    - {window: 14}
4h:
  calculate_ma:
    - {window: 120}
  calculate_atr:
    - {window: 14}
  calculate_obv:
    - {}
1d:
  calculate_support_resistance:
    - {window: 14}
  calculate_all_candlestick_patterns:
    - {}
"""
    yaml_path = 'test_feature_config.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # [신규] 테스트를 위한 가상의 classification_config 딕셔너리 생성
    # 실제 실행 시에는 main.py에서 Config/experiment_config.yaml로부터 로드됩니다.
    mock_classification_config = {
        'feature_num': {
            'close_change_pct': (-10, 10),
            'body_size': (0, 100),
            'upper_wick_size': (0, 200),
            'lower_wick_size': (0, 200),
            'total_range': (0, 200),
            'body_to_range_ratio': (0, 1),
            'RSI': (0, 100),
            '%K': (0, 100),
            '%D': (0, 100),
            'Williams_R': (-100, 0),
            'CCI': (-250, 250),
            'MACD': (-50, 50),
            'ADX': (0, 100),
            'ATR': (0, 200),
        },
        'feature_comparison': [
            'Open', 'High', 'Low', 'Close', 'SMA', 'EMA', 'BB_Upper', 'BB_Mid', 'BB_Lower',
            'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Support_MA', 'Resistance_MA'
        ],
        'feature_bool': [
            'InvertedHammers', 'Hammers', 'HangingMen', 'DarkCloudCovers', 'Dojis',
            'MorningStars', 'ShootingStars', 'BullishHarami', 'BearishHarami'
        ]
    }


    # 3. [수정된] Multi-Timeframe 피처 생성 함수 호출
    print("\n2. Multi-Timeframe 피처 및 모델 설정 생성 시작...")
    target_timeframes_list = ['5m', '15m', '1h', '4h', '1d']
    
    # [수정] 함수의 반환값이 (DataFrame, Dict)으로 변경되고, 새 인자가 추가됨
    final_dataframe, model_config = run_feature_generation_from_yaml(
        df=source_df,
        timestamp_col='Open time',
        target_timeframes=target_timeframes_list,
        yaml_config_path=yaml_path,
        classification_config=mock_classification_config # 가상 설정 전달
    )
    
    # 4. 결과 확인
    print("\n3. 최종 결과 확인...")
    if final_dataframe is not None and model_config is not None:
        print(f"최종 데이터프레임 Shape: {final_dataframe.shape}")
        
        # 생성된 model_config 내용 확인
        print("\n--- 생성된 Model Config 미리보기 ---")
        
        # Feature_num
        fn_keys = list(model_config['feature_num'].keys())
        print(f"\n[Feature_num] (총 {len(fn_keys)}개)")
        print(f"  - 예시: {dict(list(model_config['feature_num'].items())[:3])}")
        
        # Feature_comparison_map
        fcm_keys = list(model_config['feature_comparison_map'].keys())
        print(f"\n[Feature_comparison_map] (총 {len(fcm_keys)}개)")
        if fcm_keys:
            print(f"  - 예시 (키): '{fcm_keys[0]}' -> 비교 대상 {len(model_config['feature_comparison_map'][fcm_keys[0]])}개")

        # Feature_bool
        fb_keys = model_config['feature_bool']
        print(f"\n[Feature_bool] (총 {len(fb_keys)}개)")
        print(f"  - 예시: {fb_keys[:5]}")

        print("\n이제 이 'model_config' 딕셔너리를 GATreePop 생성자에 바로 전달할 수 있습니다.")
        print("예: GATreePop(..., **model_config)")
