# main.py
import os
import sys
import yaml  # YAML 로드를 위해 추가
from tqdm import tqdm
import torch
import pandas as pd
import logging
import random
import numpy as np
import json
from datetime import datetime
from typing import List

# --- 프로젝트 모듈 임포트 ---
from models.model import GATreePop
from evolution.evolution import Evolution
from evolution.Selection import PPTSSelection
from evolution.Crossover import ChainCrossover, SubtreeCrossover, NodeCrossover, RootBranchCrossover
from evolution.Mutation import (
    ChainMutation, AddNodeMutation, DeleteNodeMutation, NodeParamMutation,
    ReinitializeNodeMutation, AddSubtreeMutation, DeleteSubtreeMutation
)
from training.runner import generation_training, generation_test
from training.wfo import WalkForwardOptimizer as _WFO
from tools import wandb_logger

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

    # --- Initialize Weights & Biases (optional) ---
    wandb_cfg = config.get('wandb', {})
    ts = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    run_name = f"{data_cfg.get('symbol','SYM')}-{data_cfg.get('interval','int')}-{env_cfg.get('seed',0)}-{ts}"
    try:
        wandb_logger.init_run(wandb_cfg, run_name, env_cfg['output_dir'], full_config=config)
    except Exception:
        pass

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

    logging.info("Creating initial random population using GPU...")
    population.make_population(device=env_cfg['device'])
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

    # 1. 선택 연산자 (PPTS 전용)
    if op_cfg['selection']['type'] == 'PPTSSelection':
        chunk_size = op_cfg['selection'].get('chunk_size', 8192)
        selection_op = PPTSSelection(chunk_size=chunk_size)
        logging.info(f"Using PPTS (Multi-Objective) Selection with chunk_size={chunk_size}")
    else:
        # Only PPTS is supported in the unified system
        logging.warning(f"Unknown selection type: {op_cfg['selection']['type']}. Using PPTSSelection instead.")
        chunk_size = 8192
        selection_op = PPTSSelection(chunk_size=chunk_size)
        logging.info(f"Using PPTS (Multi-Objective) Selection with chunk_size={chunk_size}")

    # 2. 교차 연산자
    crossover_ops = [
        SubtreeCrossover(
            max_nodes=ga_cfg['population']['max_nodes'],
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

    # 4. 진화 엔진 (통합 버전)
    evolution_engine = Evolution(
        population=population,
        selection=selection_op,
        crossover=crossover_op,
        mutation=mutation_op,
        parent_size=ga_cfg['evolution_loop']['parent_pool_size'],  # Reduced for testing
        num_elites=ga_cfg['evolution_loop']['elite_size'] # Reduced for testing
    )
    logging.info("Evolution engine configured.")

    # ==========================================================================
    #             Phase 4: Walk-Forward Optimization (WFO) 학습 루프 실행
    # ==========================================================================
    logging.info("--- Phase 4: Walk-Forward Optimization Training Loop ---")

    total_data_len = len(final_df)
    
    # WFO 전용 모드 강제
    wfo_cfg = config.get('walk_forward', {'enabled': False})
    if not wfo_cfg.get('enabled', False):
        logging.error("Walk-Forward Optimization mode is required but disabled in config.")
        logging.error("Please set walk_forward.enabled: true in experiment_config.yaml")
        sys.exit(1)

    # === Walk-Forward Optimization Mode ===
    logging.info("=== Walk-Forward Optimization Mode Enabled ===")

    # --- Encapsulated WFO execution (refactored) ---

    wfo_optimizer = _WFO(wfo_cfg, eval_cfg)
    best_chromosomes, best_metrics, windows = wfo_optimizer.run(
        final_df=final_df,
        evolution_engine=evolution_engine,
        model_config=model_config,
        ga_cfg=ga_cfg,
        env_cfg=env_cfg,
        wfo_cfg=wfo_cfg,
        wandb_cfg=wandb_cfg,
        output_dir=env_cfg['output_dir'],
    )
    if not windows:
        logging.error("No valid WFO windows could be generated. Check your configuration.")
        logging.error(f"Data length: {total_data_len}, train_bars: {wfo_cfg['train_bars']}, oos_bars: {wfo_cfg['oos_bars']}")
        return
    logging.info(f"Generated {len(windows)} WFO windows from {total_data_len} data points (refactored WFO)")

    # --- Weights & Biases: log final summary and artifacts ---
    try:
        final_step = len(windows) if 'windows' in locals() else 0
        agg_names = eval_cfg.get('oos_aggregate', {}).get('objectives', [])
        agg_min = eval_cfg.get('oos_aggregate', {}).get('minimize', [True] * len(agg_names))
        if isinstance(best_metrics, np.ndarray) and best_metrics.size and len(agg_names) > 0:
            # Natural orientation for logging
            top1 = best_metrics[0]
            topk_avg = np.mean(best_metrics, axis=0)
            payload = {"final/windows": int(len(windows)),
                        "final/candidates": int(len(best_metrics))}
            for j, name in enumerate(agg_names):
                val_top1 = float(-top1[j] if not agg_min[j] else top1[j])
                val_avg = float(-topk_avg[j] if not agg_min[j] else topk_avg[j])
                payload[f"final/top1/{name}"] = val_top1
                payload[f"final/topK/avg/{name}"] = val_avg
            wandb_logger.log(payload, step=final_step)
        # Artifacts
        artifacts = []
        best_population_path = os.path.join(env_cfg['output_dir'], "best_population.pth")
        topk_path = os.path.join(env_cfg['output_dir'], "best_population_topK.pth")
        wfo_results_path = os.path.join(env_cfg['output_dir'], 'wfo_results.json')
        for p in [best_population_path, topk_path, wfo_results_path]:
            if os.path.exists(p):
                artifacts.append(p)
        if artifacts and bool(wandb_cfg.get('log_artifacts', True)):
            wandb_logger.log_artifacts(artifacts, name=f"{run_name}-models", type_="model")
            # Also add results json under results
            wandb_logger.log_artifacts([wfo_results_path], name=f"{run_name}-results", type_="results")
        # Analysis images, if generated by tools/analyze_best.py
        analysis_dir = os.path.join(env_cfg['output_dir'], 'analysis')
        if os.path.isdir(analysis_dir):
            for fname in [
                'feature_usage.png',
                'comparison_type_mix.png',
                'feature_interaction.png',
                'permutation_importance.png',
            ]:
                fpath = os.path.join(analysis_dir, fname)
                if os.path.exists(fpath):
                    wandb_logger.log_image(f"analysis/{fname}", fpath, step=final_step)
    except Exception:
        pass
        

    logging.info("Training loop finished.")

    # ==========================================================================
    #             Phase 5: 결과 저장 및 최종 평가
    # ==========================================================================
    logging.info("--- Phase 5: Saving Best Population and Final Evaluation ---")

    best_population_path = os.path.join(env_cfg['output_dir'], "best_population.pth")

    if best_chromosomes is None or (hasattr(best_chromosomes, '__len__') and len(best_chromosomes) == 0):
        logging.warning("No best chromosomes were found during training. Skipping final save and test.")
    else:
        # === WFO Mode: Save aggregated results and best population ===
        logging.info("=== WFO Mode: Saving aggregated results & best model ===")
        
        # Save WFO results summary
        wfo_results_path = os.path.join(env_cfg['output_dir'], "wfo_results.json")
        wfo_results = {
            'num_windows': len(windows) if 'windows' in locals() else 0,
            'num_candidates': len(wfo_optimizer.candidate_archives) if 'wfo_optimizer' in locals() else 0,
            'aggregated_objectives': best_metrics.tolist() if isinstance(best_metrics, np.ndarray) else best_metrics,
            'objective_names': eval_cfg.get('oos_aggregate', {}).get('objectives', []),
            'minimize_directions': eval_cfg.get('oos_aggregate', {}).get('minimize', []),
            'wfo_config': wfo_cfg
        }
        
        with open(wfo_results_path, 'w') as f:
            json.dump(wfo_results, f, indent=2)
        
        logging.info(f"WFO results saved to '{wfo_results_path}'")

        # Save the top-1 and Top-K best models if reconstructed
        if best_chromosomes is not None and len(best_chromosomes) > 0:
            # Top-1
            top1 = best_chromosomes[:1]
            logging.info(f"Saving top-1 WFO model to '{best_population_path}'")
            best_population = GATreePop(
                pop_size=1,
                max_nodes=ga_cfg['population']['max_nodes'],
                max_depth=ga_cfg['population']['max_depth'],
                max_children=ga_cfg['population']['max_children'],
                **model_config
            )
            best_population.population_tensor[:1].copy_(top1)
            best_population.set_next_idx()
            best_population.initialized = True
            best_population.save(best_population_path)
            logging.info("WFO best population saved successfully.")

            # Top-K (all reconstructed)
            topk_path = os.path.join(env_cfg['output_dir'], "best_population_topK.pth")
            logging.info(f"Saving Top-K WFO models to '{topk_path}'")
            topk_population = GATreePop(
                pop_size=len(best_chromosomes),
                max_nodes=ga_cfg['population']['max_nodes'],
                max_depth=ga_cfg['population']['max_depth'],
                max_children=ga_cfg['population']['max_children'],
                **model_config
            )
            topk_population.population_tensor.copy_(best_chromosomes)
            topk_population.set_next_idx()
            topk_population.initialized = True
            topk_population.save(topk_path)
            logging.info("WFO Top-K population saved successfully.")
        else:
            logging.warning("Could not reconstruct best chromosomes for saving in WFO mode.")
        
        # Display final summary
        if 'wfo_optimizer' in locals() and wfo_optimizer.candidate_archives:
            logging.info("\n========== WFO FINAL SUMMARY ==========")
            logging.info(f"Total windows processed: {len(windows)}")
            logging.info(f"Total candidates evaluated: {len(wfo_optimizer.candidate_archives)}")
            
            if isinstance(best_metrics, np.ndarray) and len(best_metrics) > 0:
                agg_obj_names = eval_cfg.get('oos_aggregate', {}).get('objectives', [])
                agg_minimize = eval_cfg.get('oos_aggregate', {}).get('minimize', [])
                
                logging.info("Top 5 candidates by aggregated OOS performance:")
                for i in range(min(5, len(best_metrics))):
                    objectives = best_metrics[i]
                    logging.info(f"Candidate {i+1:2d} | ", end="")
                    for obj_name, obj_val, minimize in zip(agg_obj_names, objectives, agg_minimize):
                        # Display in natural form: flip sign for maximize metrics
                        disp_val = obj_val if minimize else -obj_val
                        direction = "↓" if minimize else "↑"
                        logging.info(f"{obj_name}: {disp_val:.4f} {direction} | ", end="")
                    logging.info("")
        
        logging.info("============================================")
        
    logging.info("Main script finished.")


if __name__ == "__main__":
    main()
