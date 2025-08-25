# data/README.md

GA Tree 학습/평가를 위한 데이터 수집과 피처 엔지니어링 유틸리티가 들어 있습니다.

## 파일 구성

- `data_download.py`: Binance 데이터 수집 유틸리티
  - `fetch_historical_data(symbol, interval, days_to_fetch)`: `.env`의 API 키를 사용하여 OHLCV(Kline) 데이터를 로드합니다. 타임스탬프/가격/거래량/거래건수 등으로 구성된 `pandas.DataFrame` 반환.
  - `fetch_funding_rate_history(symbol, days_to_fetch)`: 선물 펀딩 비율 이력을 로드합니다. `fundingTime`, `fundingRate` 컬럼 반환.
  - 스크립트 실행 예시(메인): Kline과 펀딩 데이터를 가져와 시간 기준 `merge_asof`로 병합 후, 샘플 출력 및 CSV 저장.

- `ta_lib_feature_generator.py`: 기술적 지표 피처 생성(TA‑Lib + pandas)
  - 입력 헬퍼: `get_ohlcv_arrays(df)`로 TA‑Lib 입력용 numpy 배열 추출.
  - 가격/캔들 기본: `calculate_price_change_features`(종가 변화율, 몸통/꼬리/범위 및 비율).
  - 추세: `calculate_ma`(SMA), `calculate_ema`(EMA), `calculate_vwma`(pandas), `calculate_macd`, `calculate_adx`, `calculate_ichimoku`(pandas; Chikou는 과거만 참조), `calculate_dema`, `calculate_tema`, `calculate_sar`, `calculate_trima`.
  - 모멘텀: `calculate_rsi`, `calculate_stochastic_oscillator`(%K/%D), `calculate_williams_r`, `calculate_cci`, `calculate_stochrsi`, `calculate_ppo`, `calculate_roc`, `calculate_ultosc`, `calculate_mom`.
  - 변동성: `calculate_bollinger_bands`(상/중/하단 및 폭), `calculate_atr`, `calculate_natr`.
  - 거래량: `calculate_obv`, `calculate_cmf`(pandas), `calculate_ad`, `calculate_adosc`, Hilbert 변환(`calculate_ht_*`).
  - 캔들 패턴: TA‑Lib CDL 패턴들을 불리언 컬럼으로 생성(해머, 도지, 엔걸핑, 3백병/3까마귀 등), 강세/약세를 분리 제공.
  - 복합 피처: `calculate_rsi_adx_ratio`, `calculate_macd_atr_ratio`, `calculate_short_long_ratio`, `calculate_indicator_roc`, `calculate_volume_spike`, `calculate_consecutive_candles`, `calculate_confirmation_signal` 등 파생 비율/신호.
  - 컬럼명 규칙: `format_col_name`로 일관된 네이밍을 사용하며, 모든 함수는 `(df, new_columns)` 형태로 반환.

- `merge_dataset.py`: 멀티 타임프레임 피처 결합 + 모델 설정 도출
  - `generate_multi_timeframe_features(df, timestamp_col, target_timeframes, feature_params)`:
    - 원본 OHLCV(+옵션 `fundingRate`)를 기준 타임프레임으로 리샘플하고, 각 타임프레임의 피처를 계산하여 `merge_asof`로 정렬/병합.
  - `_generate_model_config_from_features(all_generated_cols, classification_config)`:
    - YAML 기반 분류 템플릿으로 생성된 피처명을 분류하여 `feature_num`(숫자 비교 범위), `feature_comparison_map`(피처-피처 비교), `feature_bool`(불리언 피처)로 구성된 설정을 반환.
  - `run_feature_generation_from_yaml(df, timestamp_col, target_timeframes, yaml_config_path, classification_config)`:
    - 타임프레임별 피처 정의를 YAML에서 읽어 위의 생성기 호출 후, `(final_dataframe, model_config)`을 반환(모델 초기화에 바로 사용 가능).

- `feature_config.yaml`: 예시 YAML 구성. 각 타임프레임(`5m`, `15m`, `1h`, `4h`, `1d` 등) 키 아래에 호출할 피처 함수명과 파라미터를 명시. 함수명은 `ta_lib_feature_generator.py`의 함수와 매핑됩니다.

## 참고 사항

- TA‑Lib, pandas, numpy가 필요합니다.
- Binance API 사용 시, 본 폴더의 상위 경로에 `.env`(키: `BINANCE_API_KEY`, `BINANCE_API_SECRET`)가 있어야 합니다.
- 병합 시 과거 데이터만 참조하도록 `merge_asof`를 사용하여 룩어헤드 바이어스를 줄였습니다.

## 예시

### 1) 설치 및 환경 변수

```bash
pip install -r requirements.txt
# 또는 필요한 최소 패키지
pip install pandas numpy python-binance python-dotenv ta-lib

# .env (프로젝트 루트에 위치)
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```

### 2) Kline + 펀딩 비율 병합 샘플

```python
from data.data_download import fetch_historical_data, fetch_funding_rate_history
import pandas as pd

symbol = 'BTCUSDT'
interval = '1h'
days = 30

klines = fetch_historical_data(symbol, interval, days)
funding = fetch_funding_rate_history(symbol, days)

klines = klines.sort_values('Open time')
funding = funding.sort_values('fundingTime')
merged = pd.merge_asof(
    klines, funding, left_on='Open time', right_on='fundingTime', direction='backward'
)
merged.to_csv('merged.csv', index=False)
```

### 3) 멀티타임프레임 피처 생성 + 모델 설정 도출

```python
import yaml
from data.merge_dataset import run_feature_generation_from_yaml

# 예시 입력 데이터프레임: merged (위에서 생성)
yaml_config_path = 'data/feature_config.yaml'

# experiment_config.yaml 등에서 불러온 분류 템플릿(dict)
# 실제 키 구조는 프로젝트 설정에 맞게 조정하세요.
with open('experiment_config.yaml', 'r', encoding='utf-8') as f:
    exp_cfg = yaml.safe_load(f)
classification_config = exp_cfg.get('model', {}).get('classification', {
    'feature_num': {}, 'feature_comparison': [], 'feature_bool': []
})

final_df, model_config = run_feature_generation_from_yaml(
    df=merged,
    timestamp_col='Open time',
    target_timeframes=['5m', '15m', '1h', '4h', '1d'],
    yaml_config_path=yaml_config_path,
    classification_config=classification_config
)

all_feature_names = list(model_config['feature_num'].keys()) \
                    + list(model_config['feature_comparison_map'].keys()) \
                    + model_config['feature_bool']
print(final_df.shape, len(all_feature_names))
```
