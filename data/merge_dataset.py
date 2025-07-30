import pandas as pd
import numpy as np
import yaml
import data.ta_lib_feature_generator as talib_feat

def generate_multi_timeframe_features(
    df: pd.DataFrame,
    timestamp_col: str,
    target_timeframes: list,
    feature_params: dict
):
    """
    서로 다른 시간 단위의 기술적 분석 지표를 생성하고 기준 데이터프레임에 통합합니다.
    (이전 답변과 동일한 함수 코드)
    """
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
    time_deltas = [pd.to_timedelta(tf.replace('m', 'T')) for tf in target_timeframes]
    base_timeframe = target_timeframes[np.argmin(time_deltas)]
    
    print(f"기준 시간 단위가 '{base_timeframe}'으로 설정되었습니다.")

    agg_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    base_df = source_df.resample(base_timeframe.replace('m', 'T')).agg(agg_rules)
    base_df.dropna(inplace=True)

    # --- 2. 시간 단위별 피처 계산 및 통합 ---
    features_by_timeframe = {}

    for timeframe in sorted(list(set(target_timeframes)), key=lambda x: pd.to_timedelta(x.replace('m', 'T'))):
        if timeframe not in feature_params:
            print(f"'{timeframe}'에 대한 피처 설정이 없어 건너뜁니다.")
            continue
            
        print(f"--- '{timeframe}' 시간 단위 피처 계산 시작 ---")

        df_resampled = source_df.resample(timeframe.replace('m', 'T')).agg(agg_rules).dropna()
        
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

    sorted_timeframes = sorted(features_by_timeframe.keys(), key=lambda x: pd.to_timedelta(x.replace('m', 'T')))

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


def run_feature_generation_from_yaml(
    df: pd.DataFrame,
    timestamp_col: str,
    target_timeframes: list,
    yaml_config_path: str='/feature_config.yaml'
):
    """
    YAML 설정 파일을 기반으로 Multi-Timeframe 피처 생성을 실행하는 래퍼 함수입니다.

    이 함수는 YAML 파일 경로를 받아 내부적으로 파라미터를 파싱하고,
    기존의 `generate_multi_timeframe_features` 함수를 호출하여 피처 생성을 수행합니다.

    Args:
        df (pd.DataFrame): 'Open', 'High', 'Low', 'Close', 'Volume' 및 타임스탬프 컬럼을 포함하는 원본 데이터프레임.
        timestamp_col (str): 데이터프레임 내의 타임스탬프 컬럼명.
        target_timeframes (list): 피처를 생성할 시간 단위 목록 (예: ['5m', '1h', '1d']).
                                  이 목록에 있는 타임프레임에 대한 설정만 YAML 파일에서 사용됩니다.
        yaml_config_path (str): 피처 생성 파라미터를 정의한 .yaml 파일의 경로.

    Returns:
        pd.DataFrame: 모든 시간 단위의 피처가 통합된 최종 데이터프레임.
                      오류 발생 시 None을 반환합니다.
        list: 새로 추가된 모든 피처 컬럼명의 리스트.
              오류 발생 시 None을 반환합니다.
    """
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
    # YAML 파일에 정의된 키와 target_timeframes를 비교하여 해당 설정만 추출합니다.
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
        return df, []

    # 3. 메인 피처 생성 함수 호출
    print("\nYAML 설정에 기반하여 Multi-Timeframe 피처 생성을 시작합니다.")
    final_dataframe, added_cols = generate_multi_timeframe_features(
        df=df,
        timestamp_col=timestamp_col,
        target_timeframes=valid_timeframes, # 유효한 타임프레임만 전달
        feature_params=feature_generation_params
    )

    return final_dataframe, added_cols

# ==============================================================================
#                      복잡하고 풍부한 함수 실행 예시
# ==============================================================================
if __name__ == '__main__':
    # 1. 가상 데이터 생성 (1분봉, 15일치 데이터)
    print("1. 가상 1분봉 데이터 생성...")
    time_index = pd.to_datetime(pd.date_range(start='2024-07-15 00:00', periods=15*24*60*200, freq='1T'))
    data_size = len(time_index)
    data = {
        'Timestamp': time_index,
        'Open': np.random.uniform(-0.5, 0.5, data_size).cumsum() + 2000,
        'Close': np.random.uniform(-0.5, 0.5, data_size).cumsum() + 2000,
        'Volume': np.random.uniform(10, 100, data_size)
    }
    data['High'] = np.maximum(data['Open'], data['Close']) + np.random.uniform(0, 2, data_size)
    data['Low'] = np.minimum(data['Open'], data['Close']) - np.random.uniform(0, 2, data_size)
    
    source_df = pd.DataFrame(data)
    print(f"생성된 원본 데이터 Shape: {source_df.shape}")

    # 2. 복잡한 피처 생성 규칙 정의
    feature_generation_params = {
        # -- 단기(5분봉) 지표: 빠른 반응성 지표들 --
        '5m': {
            'calculate_ema': [{'window': 12}],
            'calculate_rsi': [{'window': 14}],
            'calculate_stochastic_oscillator': [{'k_window': 14, 'd_window': 3}],
        },
        # -- 중단기(15분봉) 지표 --
        '15m': {
            'calculate_ma': [{'window': 20}, {'window': 50}], # 두 개의 다른 SMA 계산
            'calculate_cci': [{'window': 20}],
        },
        # -- 중기(1시간봉) 지표: 추세 및 변동성 중심 --
        '1h': {
            'calculate_macd': [{'short_window': 12, 'long_window': 26, 'signal_window': 9}],
            'calculate_bollinger_bands': [{'window': 20, 'num_std': 2}],
            'calculate_adx': [{'window': 14}],
        },
        # -- 장기(4시간봉) 지표: 긴 호흡의 추세 및 거래량 --
        '4h': {
            'calculate_ma': [{'window': 120}], # 긴 기간의 SMA
            'calculate_atr': [{'window': 14}],
            'calculate_obv': [{}], # 인자가 없는 함수는 빈 dict 전달
        },
        # -- 초장기(일봉) 지표: 지지/저항 및 종합 패턴 --
        '1d': {
            'calculate_support_resistance': [{'window': 14}],
            'calculate_all_candlestick_patterns': [{}], # 모든 캔들 패턴 생성
        }
    }

    # 3. Multi-Timeframe 피처 생성 함수 호출
    print("\n2. Multi-Timeframe 피처 생성 시작...")
    # target_timeframes_list = ['5m', '15m', '1h', '4h', '1d']
    target_timeframes_list = ['5m', '15m', '1h']
    final_dataframe, added_cols = generate_multi_timeframe_features(
        df=df,
        timestamp_col='Close time',
        target_timeframes=target_timeframes_list,
        feature_params=feature_generation_params
    )
    
    # 4. 결과 확인
    print("\n3. 최종 결과 확인...")
    print(f"최종 데이터프레임 Shape: {final_dataframe.shape}")
    print(f"총 {len(added_cols)}개의 피처가 추가되었습니다.")
    # 추가된 컬럼 중 일부만 출력
    print("추가된 컬럼 목록 (일부):", added_cols[:5], "...", added_cols[-5:])
    
    # 각 시간 단위별 대표 피처들을 선정하여 병합 결과 확인
    print("\n각 시간 단위별 대표 피처 병합 결과 샘플 (마지막 15개 행):")
    display_cols = [
        'Timestamp',
        'Close',
        'RSI_14_5m',                   # 5분봉 대표
        'SMA_50_15m',                  # 15분봉 대표
        'MACD_12_26_9_1h',             # 1시간봉 대표
        # 'ATR_14_4h',                   # 4시간봉 대표
        # 'Support_MA_14_1d',            # 일봉 대표
        # 'Hammers_1d'                   # 일봉 캔들 패턴 대표
    ]
    # display_cols에 있는 컬럼만 필터링
    display_cols_exist = [col for col in display_cols if col in final_dataframe.columns]
    
    # 소수점 2자리까지만 표시하도록 설정
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    print(final_dataframe[display_cols_exist].tail(15).to_string())