# main.py
import os
import sys
import yaml  # YAML 로드를 위해 추가
import torch
import pandas as pd
import logging
import random
import numpy as np
import torch.multiprocessing as mp  # 멀티프로세싱 임포트

# --- 프로젝트 모듈 임포트 ---
from models.model import GATreePop
from evolution import Evolution
from evolution.Selection import TournamentSelection, RouletteSelection
from evolution.Crossover import ChainCrossover, SubtreeCrossover, NodeCrossover, RootBranchCrossover
from evolution.Mutation import (
    ChainMutation, AddNodeMutation, DeleteNodeMutation, NodeParamMutation,
    ReinitializeNodeMutation, AddSubtreeMutation, DeleteSubtreeMutation
)
from training.trading_env import generation_valid, generation_test, calculate_fitness

try:
    import gatree_cuda_compat as gatree_cuda
except ImportError:
    print("="*60)
    print(">>> Warning: 'gatree_cuda' module not found.")
    print(">>> Build the CUDA extension first:")
    print(">>> python setup.py build_ext --inplace")
    print("="*60)
    gatree_cuda = None

# ==============================================================================
#                      Phase 0: 환경 설정 및 사전 준비
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


def set_seed(seed):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_environment(output_dir):
    """학습 환경을 설정하고 필요한 디렉토리를 생성합니다."""
    logging.info("Setting up environment...")
    try:
        import gatree_cuda_compat as gatree_cuda
        logging.info("'gatree_cuda' module found. Running on GPU is enabled.")
    except ImportError:
        logging.error("FATAL: 'gatree_cuda' module not found.")
        logging.error("Please compile the C++/CUDA extension first by running:")
        logging.error(">>> python setup.py build_ext --inplace")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    logging.info(f"Output directory created at: {output_dir}")


def main():
    """전체 학습 파이프라인을 실행하는 메인 함수"""
    # --- 설정 로드 ---
    config = load_config()

    # --- 환경 설정 ---
    env_cfg = config['environment']
    data_cfg = config['data']
    ga_cfg = config['ga_core']
    op_cfg = config['operators']
    eval_cfg = config['evaluation']
    classification_cfg = config['feature_classification']

    set_seed(env_cfg['seed'])
    setup_environment(env_cfg['output_dir'])

    # ==========================================================================
    #             Phase 1: 데이터 준비 및 피처 생성 (단순 검사)
    # ==========================================================================
    logging.info("--- Phase 1: Data Preparation & Feature Check ---")

    final_features_path = os.path.join(env_cfg['data_dir'], data_cfg['final_features_path'])
    model_config_path = os.path.join(env_cfg['data_dir'], data_cfg['model_config_path'])

    # --- 최종 전처리된 데이터가 있는지 확인 ---
    if not (os.path.exists(final_features_path) and os.path.exists(model_config_path)):
        logging.error("Preprocessed data not found.")
        logging.error("Please run 'preprocess_data.py' first to generate the data.")
        sys.exit(1)

    logging.info(f"Loading pre-processed data from '{final_features_path}'")
    final_df = pd.read_csv(final_features_path, parse_dates=['Open time'])

    logging.info(f"Loading model config from '{model_config_path}'")
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    if final_df.empty or not model_config:
        logging.error("Loaded data or config is empty. Please regenerate by deleting files and re-running preprocess_data.py.")
        return

    # --- 후처리 (공통) ---
    final_df.set_index('Open time', inplace=True)
    logging.info(f"Data preparation complete. Final data shape: {final_df.shape}")

    # ==========================================================================
    #             Phase 2: 유전 알고리즘(GA) 집단 초기화
    # ==========================================================================
    logging.info("--- Phase 2: GA Population Initialization ---")

    # Collect all unique features including RHS features from feature_comparison_map
    all_features_set = set()
    all_features_set.update(model_config['feature_num'].keys())
    all_features_set.update(model_config['feature_comparison_map'].keys())
    # Add all the values from feature_comparison_map (the RHS features)
    for feature_list in model_config['feature_comparison_map'].values():
        all_features_set.update(feature_list)
    all_features_set.update(model_config['feature_bool'])
    
    model_config['all_features'] = list(all_features_set)

    population = GATreePop(
        pop_size=ga_cfg['population']['size'],
        max_nodes=ga_cfg['population']['max_nodes'],
        max_depth=ga_cfg['population']['max_depth'],
        max_children=ga_cfg['population']['max_children'],
        **model_config
    )

    logging.info("Creating initial random population...")
    num_processes = os.cpu_count() or 1
    population.make_population(num_processes=num_processes, device=env_cfg['device'], init_mode='cuda')
    # Validate trees after CUDA population init (if available)
    try:
        if gatree_cuda is not None and str(env_cfg['device']).startswith('cuda'):
            gatree_cuda.validate_trees(population.population_tensor.to(env_cfg['device']).contiguous())
    except Exception as e:
        raise RuntimeError(f"validate_trees failed after CUDA init: {e}")
    logging.info("Population initialized successfully.")

    # ==========================================================================
    #             Phase 3: 진화(Evolution) 엔진 설정
    # ==========================================================================
    logging.info("--- Phase 3: Evolution Engine Setup ---")

    # 1. 선택 연산자
    if op_cfg['selection']['type'] == 'TournamentSelection':
        selection_op = TournamentSelection(k=op_cfg['selection']['tournament_k_size'])
    elif op_cfg['selection']['type'] == 'RouletteSelection':
        selection_op = RouletteSelection()
    else:
        raise ValueError(f"Unknown selection type: {op_cfg['selection']['type']}")

    # 2. 교차 연산자
    crossover_ops = [
        SubtreeCrossover(
            max_nodes=ga_cfg['population']['max_nodes'],
            max_depth=ga_cfg['population']['max_depth'],
            mode=op_cfg['crossover']['subtree_crossover']['mode']
        ),
        NodeCrossover(mode=op_cfg['crossover']['node_crossover']['mode']),
        RootBranchCrossover(max_nodes=ga_cfg['population']['max_nodes'])
    ]
    crossover_op = ChainCrossover(crossovers=crossover_ops, weights=op_cfg['crossover']['weights'])

    # 3. 변이 연산자
    mutation_base_config = {
        'max_depth': ga_cfg['population']['max_depth'],
        'max_children': ga_cfg['population']['max_children'],
        **model_config
    }

    mut_probs = op_cfg['mutation']['probabilities']
    mut_params = op_cfg['mutation']

    mutation_ops = [
        AddNodeMutation(prob=mut_probs['add_node'], config=mutation_base_config, **mut_params['add_node_mutation']),
        DeleteNodeMutation(prob=mut_probs['delete_node'], config=mutation_base_config, **mut_params['delete_node_mutation']),
        AddSubtreeMutation(prob=mut_probs['add_subtree'], config=mutation_base_config, **mut_params['add_subtree_mutation']),
        DeleteSubtreeMutation(prob=mut_probs['delete_subtree'], config=mutation_base_config),
        ReinitializeNodeMutation(prob=mut_probs['reinitialize_node'], config=mutation_base_config),
        NodeParamMutation(prob=mut_probs['node_param'], config=mutation_base_config, **mut_params['node_param_mutation'])
    ]
    mutation_op = ChainMutation(mutations=mutation_ops)

    # 4. 진화 엔진
    evolution_engine = Evolution(
        population=population,
        selection=selection_op,
        crossover=crossover_op,
        mutation=mutation_op,
        parent_size=min(ga_cfg['evolution_loop']['parent_pool_size'], 50),  # Reduced for testing
        num_elites=min(ga_cfg['evolution_loop']['elite_size'], 10)  # Reduced for testing
    )
    logging.info("Evolution engine configured.")

    # ==========================================================================
    #             Phase 4: 학습 루프 실행
    # ==========================================================================
    logging.info("--- Phase 4: Training Loop Execution ---")

    total_data_len = len(final_df)
    skip_data_cnt = 0
    valid_skip_data_cnt = int(total_data_len * data_cfg['train_ratio'])
    valid_skip_data_cnt = 3

    logging.info(f"Total data points: {total_data_len}")
    logging.info(f"Training period: index {skip_data_cnt} to {valid_skip_data_cnt}")

    best_chromosomes, best_profit = generation_valid(
        data_1m=final_df,
        evolution=evolution_engine,
        skip_data_cnt=skip_data_cnt,
        valid_skip_data_cnt=valid_skip_data_cnt,
        chromosomes_size=100,  # Match the reduced population size
        gen_loop=ga_cfg['evolution_loop']['generations'],
        best_size=ga_cfg['evolution_loop']['best_chromosome_pool_size'],
        elite_size=min(ga_cfg['evolution_loop']['elite_size'], 10),  # Match the reduced elite size
        device=env_cfg['device'],
        warming_step=ga_cfg['evolution_loop']['warming_steps'],
        evaluation_config=eval_cfg,
        output_dir=env_cfg['output_dir']
    )

    logging.info("Training loop finished.")

    # ==========================================================================
    #             Phase 5: 결과 저장 및 테스트
    # ==========================================================================
    logging.info("--- Phase 5: Saving Best Population and Final Evaluation ---")

    best_population_path = os.path.join(env_cfg['output_dir'], "best_population.pth")

    if best_chromosomes is None or len(best_chromosomes) == 0:
        logging.warning("No best chromosomes were found during training. Skipping final save and test.")
    else:
        logging.info(f"Saving the best {len(best_chromosomes)} chromosomes to '{best_population_path}'")

        best_population = GATreePop(
            pop_size=len(best_chromosomes),
            max_nodes=ga_cfg['population']['max_nodes'],
            max_depth=ga_cfg['population']['max_depth'],
            max_children=ga_cfg['population']['max_children'],
            **model_config
        )
        best_population.population_tensor.copy_(best_chromosomes)
        best_population.set_next_idx()
        best_population.initialized = True
        best_population.save(best_population_path)
        logging.info("Best population saved successfully.")

        logging.info("--- Starting final test with the best population ---")
        test_start_cnt = valid_skip_data_cnt
        test_end_cnt = total_data_len

        logging.info(f"Test period: index {test_start_cnt} to {test_end_cnt}")

        test_metrics = generation_test(
            data_1m=final_df,
            population=best_population,
            start_data_cnt=test_start_cnt,
            end_data_cnt=test_end_cnt,
            device=env_cfg['device'],
            evaluation_config=eval_cfg
        )

        test_fitness = calculate_fitness(test_metrics, weights=eval_cfg['fitness_function']['weights'])

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
