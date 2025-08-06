# main.py
import os
import sys
import torch
import pandas as pd
import logging
import random
import numpy as np

# --- 0. 모듈 임포트 ---
# 데이터 처리 모듈
from data.data_download import fetch_historical_data, fetch_funding_rate_history
from data.merge_dataset import run_feature_generation_from_yaml

# 모델 및 GA 핵심 모듈
from models.model import GATreePop
from evolution import Evolution

# GA 연산자 모듈
from evolution.Selection import TournamentSelection
from evolution.Crossover import ChainCrossover, SubtreeCrossover, NodeCrossover, RootBranchCrossover
from evolution.Mutation import (
    ChainMutation, AddNodeMutation, DeleteNodeMutation, NodeParamMutation,
    ReinitializeNodeMutation, AddSubtreeMutation, DeleteSubtreeMutation
)

# 학습 및 평가 모듈
from training.trading_env import generation_valid, generation_test, calculate_fitness

# ==============================================================================
#                      Phase 0: 환경 설정 및 사전 준비
# ==============================================================================

# --- 기본 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- 재현성을 위한 시드 고정 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# --- 학습 설정을 담는 클래스 ---
class Config:
    """학습에 필요한 모든 하이퍼파라미터와 설정을 관리합니다."""
    # 파일 및 경로 설정
    DATA_DIR = "data"
    OUTPUT_DIR = "results"
    DATA_CSV_PATH = os.path.join(DATA_DIR, "BTCUSDT_1h_365d_merged.csv")
    FEATURE_CONFIG_YAML_PATH = os.path.join(DATA_DIR, "feature_config.yaml")
    BEST_POPULATION_PATH = os.path.join(OUTPUT_DIR, "best_population.pth")
    GENERATION_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

    # 데이터 관련 설정
    SYMBOL = "BTCUSDT"
    INTERVAL = "1h"
    DAYS_TO_FETCH = 365
    TARGET_TIMEFRAMES = ['5m', '30m', '1h', '4h', '1d']

    # GA 하이퍼파라미터
    POP_SIZE = 512
    MAX_NODES = 2048
    MAX_DEPTH = 100
    MAX_CHILDREN = 50
    GEN_LOOP = 200        # 총 학습 세대 수
    PARENT_SIZE = 128     # 교배 풀(Mating Pool) 크기
    ELITE_SIZE = 8        # 다음 세대로 보존될 엘리트 개체 수
    BEST_SIZE = 32        # 최종까지 보관할 누적 베스트 개체 수
    TOURNAMENT_K_SIZE = 5

    # 교차 및 변이 연산자 가중치/확률
    CROSSOVER_WEIGHTS = [0.6, 0.2, 0.2] # Subtree, Node, RootBranch
    MUTATION_PROBS = {
        'add_node': 0.1, 'delete_node': 0.1,
        'add_subtree': 0.05, 'delete_subtree': 0.05,
        'reinitialize_node': 0.05, 'node_param': 0.2
    }

    # 학습 제어 파라미터
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    TRAIN_RATIO = 0.8  # 학습 데이터 비율
    WARMING_STEP = 5   # 엘리트 누적 시작 전 초기 세대 수


def setup_environment(config):
    """학습 환경을 설정하고 필요한 디렉토리를 생성합니다."""
    logging.info("Setting up environment...")
    # CUDA 확장 모듈 확인
    try:
        import gatree_cuda
        logging.info("'gatree_cuda' module found. Running on GPU is enabled.")
    except ImportError:
        logging.error("FATAL: 'gatree_cuda' module not found.")
        logging.error("Please compile the C++/CUDA extension first by running:")
        logging.error(">>> python setup.py build_ext --inplace")
        sys.exit(1)

    # 출력 디렉토리 생성
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.GENERATION_CHECKPOINT_DIR, exist_ok=True)
    logging.info(f"Output directory created at: {config.OUTPUT_DIR}")


def main():
    """전체 학습 파이프라인을 실행하는 메인 함수"""
    config = Config()
    setup_environment(config)
    
    # ==========================================================================
    #             Phase 1: 데이터 준비 및 피처 생성
    # ==========================================================================
    logging.info("--- Phase 1: Data Preparation & Feature Generation ---")

    if not os.path.exists(config.DATA_CSV_PATH):
        logging.warning(f"Data file not found at '{config.DATA_CSV_PATH}'. Starting download...")
        # 1. K-line 데이터 다운로드
        kline_df = fetch_historical_data(config.SYMBOL, config.INTERVAL, config.DAYS_TO_FETCH)
        if kline_df is None:
            logging.error("Failed to download K-line data. Exiting.")
            return
        
        # 2. 펀딩 비율 데이터 다운로드
        funding_df = fetch_funding_rate_history(config.SYMBOL, config.DAYS_TO_FETCH)
        if funding_df is None:
            logging.warning("Failed to download funding rate data. Proceeding without it.")
            merged_df = kline_df
        else:
            # 3. 데이터 병합
            kline_df = kline_df.sort_values('Open time')
            funding_df = funding_df.sort_values('fundingTime')
            merged_df = pd.merge_asof(
                kline_df, funding_df,
                left_on='Open time', right_on='fundingTime', direction='backward'
            )
        
        # 4. 저장
        merged_df.to_csv(config.DATA_CSV_PATH, index=False)
        logging.info(f"Downloaded and merged data saved to '{config.DATA_CSV_PATH}'")
        source_df = merged_df
    else:
        logging.info(f"Loading data from '{config.DATA_CSV_PATH}'")
        source_df = pd.read_csv(config.DATA_CSV_PATH, parse_dates=['Open time'])

    # 5. Multi-Timeframe 피처 생성
    logging.info("Generating multi-timeframe features from YAML config...")
    final_df, model_config = run_feature_generation_from_yaml(
        df=source_df,
        timestamp_col='Open time',
        target_timeframes=config.TARGET_TIMEFRAMES,
        yaml_config_path=config.FEATURE_CONFIG_YAML_PATH
    )

    if final_df is None:
        logging.error("Feature generation failed. Exiting.")
        return

    # 데이터프레임 인덱스를 타임스탬프로 설정
    final_df.set_index('Open time', inplace=True)
    logging.info(f"Feature generation complete. Final data shape: {final_df.shape}")

    # ==========================================================================
    #             Phase 2: 유전 알고리즘(GA) 집단 초기화
    # ==========================================================================
    logging.info("--- Phase 2: GA Population Initialization ---")

    # model_config에 all_features 리스트 추가 (변이 연산자에서 사용)
    model_config['all_features'] = list(model_config['feature_num'].keys()) + \
                                   list(model_config['feature_comparison_map'].keys()) + \
                                   model_config['feature_bool']

    population = GATreePop(
        pop_size=config.POP_SIZE,
        max_nodes=config.MAX_NODES,
        max_depth=config.MAX_DEPTH,
        max_children=config.MAX_CHILDREN,
        **model_config
    )
    
    logging.info("Creating initial random population...")
    population.make_population()
    logging.info("Population initialized successfully.")

    # ==========================================================================
    #             Phase 3: 진화(Evolution) 엔진 설정
    # ==========================================================================
    logging.info("--- Phase 3: Evolution Engine Setup ---")

    # 1. 선택 연산자
    selection_op = TournamentSelection(k=config.TOURNAMENT_K_SIZE)

    # 2. 교차 연산자
    crossover_ops = [
        SubtreeCrossover(rate=1.0, max_nodes=config.MAX_NODES, max_depth=config.MAX_DEPTH, mode='context'),
        NodeCrossover(rate=1.0, mode='context'),
        RootBranchCrossover(rate=1.0, max_nodes=config.MAX_NODES)
    ]
    crossover_op = ChainCrossover(crossovers=crossover_ops, weights=config.CROSSOVER_WEIGHTS)

    # 3. 변이 연산자 (필요한 설정값을 담은 딕셔너리 생성)
    mutation_config = {
        'max_depth': config.MAX_DEPTH,
        'max_children': config.MAX_CHILDREN,
        **model_config
    }
    mutation_ops = [
        AddNodeMutation(prob=config.MUTATION_PROBS['add_node'], config=mutation_config),
        DeleteNodeMutation(prob=config.MUTATION_PROBS['delete_node'], config=mutation_config),
        AddSubtreeMutation(prob=config.MUTATION_PROBS['add_subtree'], config=mutation_config),
        DeleteSubtreeMutation(prob=config.MUTATION_PROBS['delete_subtree'], config=mutation_config),
        ReinitializeNodeMutation(prob=config.MUTATION_PROBS['reinitialize_node'], config=mutation_config),
        NodeParamMutation(prob=config.MUTATION_PROBS['node_param'], config=mutation_config)
    ]
    mutation_op = ChainMutation(mutations=mutation_ops)

    # 4. 진화 엔진
    evolution_engine = Evolution(
        population=population,
        selection=selection_op,
        crossover=crossover_op,
        mutation=mutation_op,
        parent_size=config.PARENT_SIZE,
        num_elites=config.ELITE_SIZE
    )
    logging.info("Evolution engine configured.")

    # ==========================================================================
    #             Phase 4: 학습 루프 실행
    # ==========================================================================
    logging.info("--- Phase 4: Training Loop Execution ---")

    # 학습/검증 데이터 기간 설정
    total_data_len = len(final_df)
    skip_data_cnt = 0  # 초기 데이터는 피처 계산으로 불안정할 수 있으므로 건너뛸 수 있음
    valid_skip_data_cnt = int(total_data_len * config.TRAIN_RATIO)
    
    logging.info(f"Total data points: {total_data_len}")
    logging.info(f"Training period: index {skip_data_cnt} to {valid_skip_data_cnt}")

    # generation_valid 함수는 내부적으로 체크포인트를 저장하므로,
    # 여기서는 최종 결과만 받습니다.
    best_chromosomes, best_profit = generation_valid(
        data_1m=final_df,
        evolution=evolution_engine,
        skip_data_cnt=skip_data_cnt,
        valid_skip_data_cnt=valid_skip_data_cnt,
        chromosomes_size=config.POP_SIZE,
        gen_loop=config.GEN_LOOP,
        best_size=config.BEST_SIZE,
        elite_size=config.ELITE_SIZE,
        device=config.DEVICE,
        warming_step=config.WARMING_STEP
    )
    
    logging.info("Training loop finished.")

    # ==========================================================================
    #             Phase 5: 결과 저장 및 테스트
    # ==========================================================================
    logging.info("--- Phase 5: Saving Best Population and Final Evaluation ---")

    if best_chromosomes is None or len(best_chromosomes) == 0:
        logging.warning("No best chromosomes were found during training. Skipping final save and test.")
    else:
        # 1. 최적 개체 집단 저장
        logging.info(f"Saving the best {len(best_chromosomes)} chromosomes to '{config.BEST_POPULATION_PATH}'")
        
        # 새로운 GATreePop 객체에 최고 성능의 개체들만 담아서 저장
        best_population = GATreePop(
            pop_size=len(best_chromosomes),
            max_nodes=config.MAX_NODES,
            max_depth=config.MAX_DEPTH,
            max_children=config.MAX_CHILDREN,
            **model_config
        )
        best_population.population_tensor.copy_(best_chromosomes)
        best_population.set_next_idx() # 각 트리의 next_idx 재설정
        best_population.initialized = True
        
        # GATreePop.save()는 내부적으로 딕셔너리를 만들어 torch.save를 호출
        best_population.save(config.BEST_POPULATION_PATH)
        logging.info("Best population saved successfully.")
        
        # 2. (선택) 저장된 최적 집단으로 테스트 기간 성능 평가
        logging.info("--- Starting final test with the best population ---")
        test_start_cnt = valid_skip_data_cnt
        test_end_cnt = total_data_len
        
        logging.info(f"Test period: index {test_start_cnt} to {test_end_cnt}")
        
        test_metrics = generation_test(
            data_1m=final_df,
            population=best_population,
            skip_data_cnt=0, # 테스트 함수 내부에서는 사용되지 않음
            start_data_cnt=test_start_cnt,
            end_data_cnt=test_end_cnt,
            device=config.DEVICE
        )
        
        test_fitness = calculate_fitness(test_metrics)
        
        logging.info("\n========== FINAL TEST RESULTS ==========")
        logging.info(f"Evaluated on {len(best_population.population_tensor)} best individuals.")
        for i in range(len(test_metrics)):
            metrics = test_metrics[i]
            fitness = test_fitness[i]
            logging.info(
                f"Individual {i:02d} | "
                f"Fitness: {fitness:.4f} | "
                f"Mean Return: {metrics[0]:.4f} | "
                f"Profit Factor: {metrics[1]:.2f} | "
                f"Win Rate: {metrics[2]:.2f} | "
                f"Max DD: {metrics[3]:.4f} | "
                f"Compound Value: {metrics[4]:.2f}"
            )
        logging.info("========================================")

    logging.info("Main script finished.")

if __name__ == "__main__":
    main()