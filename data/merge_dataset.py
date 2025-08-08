# === 수정된 data/merge_dataset.py 파일의 전체 코드 ===

import pandas as pd
import numpy as np
import yaml
import data.ta_lib_feature_generator as talib_feat
from typing import List, Dict, Any, Tuple

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

    # --- 1. 기준 시간 단위 결정 및 기준 데이터프레임 생성 ---
    time_deltas = [pd.to_timedelta(tf.replace('m', 'T').replace('h','H').replace('d','D')) for tf in target_timeframes]
    base_timeframe = target_timeframes[np.argmin(time_deltas)]
    
    print(f"기준 시간 단위가 '{base_timeframe}'으로 설정되었습니다.")

    agg_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    # 시간 단위 문자열 변환 (e.g., '1h' -> '1H')
    resample_freq = base_timeframe.replace('m', 'T').replace('h','H').replace('d','D')
    base_df = source_df.resample(resample_freq).agg(agg_rules)
    base_df.dropna(inplace=True)

    # --- 2. 시간 단위별 피처 계산 및 통합 ---
    features_by_timeframe = {}

    for timeframe in sorted(list(set(target_timeframes)), key=lambda x: pd.to_timedelta(x.replace('m', 'T').replace('h','H').replace('d','D'))):
        if timeframe not in feature_params:
            print(f"'{timeframe}'에 대한 피처 설정이 없어 건너뜁니다.")
            continue
            
        print(f"--- '{timeframe}' 시간 단위 피처 계산 시작 ---")

        resample_freq = timeframe.replace('m', 'T').replace('h','H').replace('d','D')
        df_resampled = source_df.resample(resample_freq).agg(agg_rules).dropna()
        
        for func_name, params_list in feature_params[timeframe].items():
            try:
                feature_function = getattr(talib_feat, func_name)
            except AttributeError:
                print(f"경고: '{func_name}' 함수를 찾을 수 없습니다. 건너뜁니다.")
                continue

            for params in params_list:
                df_with_feature, new_cols = feature_function(df_resampled.copy(), **params)
                
                rename_dict = {col: f"{col}_{timeframe}" for col in new_cols}
                df_with_feature.rename(columns=rename_dict, inplace=True)
                
                renamed_cols = list(rename_dict.values())
                
                if timeframe not in features_by_timeframe:
                    features_by_timeframe[timeframe] = df_with_feature[renamed_cols]
                else:
                    features_by_timeframe[timeframe] = features_by_timeframe[timeframe].join(df_with_feature[renamed_cols])

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

    final_df.dropna(inplace=True)
    final_df.reset_index(inplace=True)

    print("\n최종 피처 생성 및 통합이 완료되었습니다.")
    return final_df, sorted(list(set(all_new_columns)))

# ==============================================================================
#           [수정] 모델 설정을 동적으로 생성하는 헬퍼 함수
# ==============================================================================

def _generate_model_config_from_features(
    all_generated_cols: List[str],
    classification_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    생성된 전체 피처 목록과 YAML에서 로드한 '분류 설정'을 받아
    모델 초기화에 필요한 설정 딕셔너리들을 생성합니다.
    [수정] `startswith`로 인한 오분류를 막기 위해 템플릿을 이름 길이순으로 정렬합니다.
    """
    print("\n--- YAML 설정을 기반으로 모델 config를 동적으로 생성합니다 ---")
    
    feature_num_template = classification_config.get('feature_num', {})
    feature_comparison_template = classification_config.get('feature_comparison', [])
    feature_bool_template = classification_config.get('feature_bool', [])
    
    # --- [수정] 가장 긴 이름부터 매칭되도록 템플릿 키를 정렬 (오분류 방지) ---
    # 예: 'SMA_Ratio'가 'SMA'보다 먼저 검사되도록 하여 정확한 타입으로 분류
    sorted_bool_keys = sorted(feature_bool_template, key=len, reverse=True)
    sorted_comparison_keys = sorted(feature_comparison_template, key=len, reverse=True)
    sorted_num_keys = sorted(feature_num_template.keys(), key=len, reverse=True)
    # --- [수정] 끝 ---
    
    final_feature_num = {}
    final_feature_bool = []
    comparison_feature_list = []
    unclassified_cols = []

    for col_name in all_generated_cols:
        classified = False
        
        # 1. 불리언 타입 피처 분류 (정렬된 키 사용)
        for base_name in sorted_bool_keys:
            if col_name.startswith(base_name):
                final_feature_bool.append(col_name)
                classified = True
                break
        if classified: continue

        # 2. 피처-피처 비교 타입 피처 분류 (정렬된 키 사용)
        for base_name in sorted_comparison_keys:
            if col_name.startswith(base_name):
                comparison_feature_list.append(col_name)
                classified = True
                break
        if classified: continue

        # 3. 피처-숫자 비교 타입 피처 분류 (정렬된 키 사용)
        for base_name in sorted_num_keys:
            if col_name.startswith(base_name):
                final_feature_num[col_name] = feature_num_template[base_name]
                classified = True
                break
        if classified: continue
        
        if not classified:
            unclassified_cols.append(col_name)

    # 5. 최종 feature_comparison_map 생성
    final_feature_comparison_map = {
        feat: [other for other in comparison_feature_list if other != feat]
        for feat in comparison_feature_list
    }
    
    print(f"  - 숫자 비교(feature_num) 타입 피처: {len(final_feature_num)}개")
    print(f"  - 피처간 비교(feature_comparison) 타입 피처: {len(final_feature_comparison_map)}개")
    print(f"  - 불리언(feature_bool) 타입 피처: {len(final_feature_bool)}개")
    
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
    model_config = _generate_model_config_from_features(added_cols, classification_config)

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
    # 실제 실행 시에는 main.py에서 experiment_config.yaml로부터 로드됩니다.
    mock_classification_config = {
        'feature_num': {
            'close_change_pct': (-10, 10), 'body_size': (0, 100), 'upper_wick_size': (0, 100),
            'lower_wick_size': (0, 100), 'total_range': (0, 200), 'body_to_range_ratio': (0, 1),
            'RSI': (0, 100), '%K': (0, 100), '%D': (0, 100), 'Williams_R': (-100, 0),
            'CCI': (-250, 250), 'MACD': (-100, 100), 'ADX': (0, 100), 'ATR': (0, 100)
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