import os
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv
import logging

# ==============================
# 로그 설정
# ==============================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_historical_data(symbol: str, interval: str, days_to_fetch: int) -> pd.DataFrame | None:
    """
    Binance에서 지정된 기간만큼의 과거 K-line(봉) 데이터를 가져옵니다.

    Args:
        symbol (str): 데이터를 가져올 코인 심볼 (예: 'BTCUSDT')
        interval (str): 데이터 간격 문자열 (예: '1m', '5m', '1h', '4h', '1d', '1w').
        days_to_fetch (int): 오늘부터 과거 몇 일까지의 데이터를 가져올지 지정

    Returns:
        pd.DataFrame | None: 성공 시 데이터프레임, 실패 시 None을 반환합니다.
    """
    # .env 파일에서 환경 변수(API 키) 로드
    # 이 스크립트는 data/ 폴더 안에 있으므로, .env 파일은 상위 폴더에 위치합니다.
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        logging.error("API 키가 .env 파일에 설정되지 않았습니다.")
        print("오류: API 키가 .env 파일에 설정되지 않았습니다.")
        return None

    # Binance 클라이언트 초기화
    client = Client(api_key, api_secret)
    logging.info("Binance 클라이언트 초기화 완료.")

    # 입력된 interval 문자열을 Binance API 상수로 매핑
    interval_mapping = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1M': Client.KLINE_INTERVAL_1MONTH,
    }

    binance_interval = interval_mapping.get(interval)
    if binance_interval is None:
        valid_intervals = ", ".join(f"'{k}'" for k in interval_mapping.keys())
        error_msg = f"'{interval}'은(는) 유효하지 않은 간격입니다. 사용 가능한 간격: {valid_intervals}"
        logging.error(error_msg)
        print(f"오류: {error_msg}")
        return None

    # 데이터 조회 시작 시점 계산
    start_date = (datetime.utcnow() - timedelta(days=days_to_fetch)).strftime('%d %b, %Y')
    logging.info(f"{symbol} 데이터 다운로드를 시작합니다. 시작일: {start_date}, 간격: {interval} ({binance_interval})")

    try:
        # get_historical_klines API 호출
        klines = client.get_historical_klines(symbol, binance_interval, start_date)
        logging.info(f"데이터 다운로드 완료. 총 {len(klines)}개의 봉 데이터를 가져왔습니다.")

        # 데이터프레임 컬럼명 정의
        cols = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Close time', 'Quote asset volume', 'Number of trades', 
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]
        
        # 데이터프레임 생성 및 데이터 타입 변환
        df = pd.DataFrame(klines, columns=cols)
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

        # 숫자형 데이터로 변환
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 
                        'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        logging.info("데이터프레임 생성 및 전처리 완료.")
        return df

    except Exception as e:
        logging.error(f"데이터 다운로드 중 오류 발생: {e}")
        print(f"데이터 다운로드 중 오류가 발생했습니다: {e}")
        return None


# ==============================
# 메인 실행 부분 (테스트용)
# ==============================
if __name__ == "__main__":
    # 필요한 라이브러리 설치 안내
    try:
        import dotenv
    except ImportError:
        print("실행에 필요한 라이브러리가 설치되지 않았습니다.")
        print("pip install python-binance pandas python-dotenv")
        exit()

    # --- 함수 사용 예시 ---
    SYMBOL = 'BTCUSDT'
    INTERVAL = Client.KLINE_INTERVAL_1MINUTE
    DAYS = 365*5  # 과거 130일치 데이터

    historical_df = fetch_historical_data(SYMBOL, INTERVAL, DAYS)

    if historical_df is not None:
        print(f"성공적으로 {SYMBOL} 데이터를 가져왔습니다.")
        print("데이터 정보:")
        historical_df.info()
        print("\n데이터 앞부분 (5줄):")
        print(historical_df.head())
        print("\n데이터 뒷부분 (5줄):")
        print(historical_df.tail())

        # 다운로드한 데이터를 CSV 파일로 저장 (선택 사항)
        save_path = os.path.join(os.path.dirname(__file__), f'{SYMBOL}_{INTERVAL}_{DAYS}d_data.csv')
        historical_df.to_csv(save_path, index=False)
        print(f"\n데이터를 다음 경로에 저장했습니다: {save_path}")