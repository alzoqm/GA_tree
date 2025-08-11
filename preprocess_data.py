# preprocess_data.py
import os
import sys
import yaml
import pandas as pd
import logging
import numpy as np

# --- 프로젝트 모듈 임포트 ---
# 이 스크립트가 프로젝트 루트에서 실행된다고 가정합니다.
from data.data_download import fetch_historical_data, fetch_funding_rate_history
from data.merge_dataset import run_feature_generation_from_yaml

# ==============================================================================
#                      환경 설정 및 유틸리티 함수
# ==============================================================================

# --- 기본 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def load_config(path="experiment_config.yaml"):
    """YAML 설정 파일을 로드합니다."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"FATAL: Configuration file not found at '{path}'")
        sys.exit(1)
    except Exception as e:
        logging.error(f"FATAL: Error parsing YAML file: {e}")
        sys.exit(1)

def run_preprocessing():
    """
    데이터 다운로드, 피처 생성 및 저장을 위한 전체 전처리 파이프라인을 실행합니다.
    """
    logging.info("==============================================")
    logging.info("=== Starting Data Preprocessing Pipeline ===")
    logging.info("==============================================")
    
    # --- 설정 로드 ---
    config = load_config()
    env_cfg = config['environment']
    data_cfg = config['data']
    classification_cfg = config['feature_classification']

    # --- 최종 전처리된 데이터가 있는지 확인 ---
    final_features_path = os.path.join(env_cfg['data_dir'], data_cfg['final_features_path'])
    model_config_path = os.path.join(env_cfg['data_dir'], data_cfg['model_config_path'])

    if os.path.exists(final_features_path) and os.path.exists(model_config_path):
        logging.info(f"Pre-processed data already exists at '{final_features_path}'.")
        logging.info(f"Model config already exists at '{model_config_path}'.")
        logging.info("Skipping preprocessing.")
        
        # 간단한 정보 출력
        try:
            df_check = pd.read_csv(final_features_path, nrows=5)
            logging.info(f"Checked existing data shape (first 5 rows): {df_check.shape}")
            logging.info("\nPreprocessing is already complete. You can now run main.py for training.")
        except Exception as e:
            logging.warning(f"Could not check existing data file. Error: {e}")
            
        return

    # --- 최종 데이터가 없으면 전처리 시작 ---
    logging.info("Pre-processed data not found. Starting data download and feature generation pipeline...")
    
    # --- 1. 원본 데이터 로드 또는 다운로드 ---
    data_csv_path = os.path.join(env_cfg['data_dir'], data_cfg['merged_csv_path'])

    if not os.path.exists(data_csv_path):
        logging.warning(f"Raw merged data file not found at '{data_csv_path}'. Starting download...")
        
        kline_df = fetch_historical_data(data_cfg['symbol'], data_cfg['interval'], data_cfg['days_to_fetch'])
        if kline_df is None:
            logging.error("Failed to download K-line data. Exiting.")
            return
        
        funding_df = fetch_funding_rate_history(data_cfg['symbol'], data_cfg['days_to_fetch'])
        if funding_df is None:
            logging.warning("Failed to download funding rate data. Proceeding without it.")
            merged_df = kline_df
        else:
            kline_df = kline_df.sort_values('Open time')
            funding_df = funding_df.sort_values('fundingTime')
            merged_df = pd.merge_asof(
                kline_df, funding_df,
                left_on='Open time', right_on='fundingTime', direction='backward'
            )
        
        # 데이터 디렉터리가 없으면 생성
        os.makedirs(env_cfg['data_dir'], exist_ok=True)
        merged_df.to_csv(data_csv_path, index=False)
        logging.info(f"Downloaded and merged raw data saved to '{data_csv_path}'")
        source_df = merged_df
    else:
        logging.info(f"Loading raw merged data from '{data_csv_path}'")
        source_df = pd.read_csv(data_csv_path, parse_dates=['Open time'])

    # --- 2. Multi-Timeframe 피처 생성 ---
    logging.info("Generating multi-timeframe features from YAML config...")
    final_df, model_config = run_feature_generation_from_yaml(
        df=source_df,
        timestamp_col='Open time',
        target_timeframes=data_cfg['feature_engineering']['target_timeframes'],
        yaml_config_path=data_cfg['feature_engineering']['config_yaml_path'],
        classification_config=classification_cfg
    )

    if final_df is None or model_config is None:
        logging.error("Feature generation failed. Exiting.")
        return

    # --- 3. 생성된 최종 데이터 및 설정 저장 ---
    logging.info(f"Saving generated feature data to '{final_features_path}'...")
    try:
        final_df.to_csv(final_features_path, index=False)
        logging.info("Feature data saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save feature data: {e}")
        return
    
    logging.info(f"Saving generated model config to '{model_config_path}'...")
    try:
        with open(model_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f, default_flow_style=False, allow_unicode=True)
        logging.info("Model config saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save model config: {e}")
        return

    logging.info("\nPreprocessing pipeline finished successfully!")
    logging.info(f"Final data and config are ready at '{env_cfg['data_dir']}/'.")

if __name__ == "__main__":
    run_preprocessing()