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

    if True:
        # Import WFO utilities
        from training.wfo import WalkForwardOptimizer
        
        # Initialize WFO coordinator
        wfo_optimizer = WalkForwardOptimizer(wfo_cfg, eval_cfg)
        
        # Generate WFO windows
        windows = wfo_optimizer.generate_windows(total_data_len)
        logging.info(f"Generated {len(windows)} WFO windows from {total_data_len} data points")
        
        if len(windows) == 0:
            logging.error("No valid WFO windows could be generated. Check your configuration.")
            logging.error(f"Data length: {total_data_len}, train_bars: {wfo_cfg['train_bars']}, oos_bars: {wfo_cfg['oos_bars']}")
            return
        
        # Using in-place population reuse across windows to avoid overhead
        # We keep the same GATreePop instance bound to the evolution engine.
        
        # WFO window loop with progress
        for window_idx, (train_start, train_end, oos_start, oos_end) in enumerate(tqdm(windows, desc="WFO Windows")):
            logging.info(f"\n=== WFO Window {window_idx + 1}/{len(windows)} ===")
            logging.info(f"IS period: index {train_start} to {train_end} ({train_end - train_start} bars)")
            logging.info(f"OOS period: index {oos_start} to {oos_end} ({oos_end - oos_start} bars)")
            
            # Create window-specific checkpoint directory
            window_checkpoint_dir = os.path.join(env_cfg['output_dir'], 'checkpoints', f'wf_window_{window_idx}')
            os.makedirs(window_checkpoint_dir, exist_ok=True)
            
            # Reset population only when requested; otherwise reuse in-place
            if window_idx > 0 and wfo_cfg.get('reset_population_each_window', False):
                population.make_population(device=env_cfg['device'])
                logging.info("Population reset for new window")
            
            # === In-Sample Training ===
            logging.info(f"Starting IS training for window {window_idx + 1}...")
            
            # Get IS evaluator and objectives config
            is_evaluator = wfo_optimizer.get_is_evaluator()
            is_objectives_config = eval_cfg.get('is_objectives', eval_cfg.get('objectives'))
            
            # Prepare window-specific evaluation config
            window_eval_config = {**eval_cfg, 'objectives': is_objectives_config}
            if 'min_trades_per_window' in wfo_cfg:
                window_eval_config['min_trades_per_window'] = wfo_cfg['min_trades_per_window']
            
            window_best_chromosomes, window_best_objectives = generation_training(
                data_1m=final_df,
                evolution=evolution_engine,
                skip_data_cnt=train_start,
                valid_skip_data_cnt=train_end,
                chromosomes_size=ga_cfg['population']['size'],
                gen_loop=ga_cfg['evolution_loop']['generations'],
                best_size=ga_cfg['evolution_loop']['best_chromosome_pool_size'],
                elite_size=ga_cfg['evolution_loop']['elite_size'],
                device=env_cfg['device'],
                warming_step=ga_cfg['evolution_loop']['warming_steps'],
                evaluation_config=window_eval_config,
                output_dir=window_checkpoint_dir,
                wandb_config=wandb_cfg,
            )
            
            if window_best_chromosomes is None or len(window_best_chromosomes) == 0:
                logging.warning(f"No valid chromosomes found in window {window_idx + 1}. Skipping OOS evaluation.")
                continue
            
            logging.info(f"IS training completed. Found {len(window_best_chromosomes)} candidate solutions.")
            
            # === Out-of-Sample Evaluation ===
            logging.info(f"Starting OOS evaluation for window {window_idx + 1}...")
            
            # Create population with best candidates for OOS evaluation
            oos_population = GATreePop(
                pop_size=len(window_best_chromosomes),
                max_nodes=ga_cfg['population']['max_nodes'],
                max_depth=ga_cfg['population']['max_depth'],
                max_children=ga_cfg['population']['max_children'],
                **model_config
            )
            oos_population.population_tensor.copy_(window_best_chromosomes)
            oos_population.set_next_idx()
            oos_population.initialized = True
            
            # Get OOS evaluator and objectives config
            oos_evaluator = wfo_optimizer.get_oos_evaluator()
            oos_objectives_config = eval_cfg.get('oos_objectives', eval_cfg.get('objectives'))
            
            # Prepare OOS-specific evaluation config
            oos_eval_config = {**eval_cfg, 'objectives': oos_objectives_config}
            if 'min_trades_per_window' in wfo_cfg:
                oos_eval_config['min_trades_per_window'] = wfo_cfg['min_trades_per_window']
            
            # Evaluate candidates on OOS data
            oos_objectives, oos_summary = generation_test(
                data_1m=final_df,
                population=oos_population,
                start_data_cnt=oos_start,
                end_data_cnt=oos_end,
                device=env_cfg['device'],
                evaluation_config=oos_eval_config
            )
            
            logging.info(f"OOS evaluation completed for {len(oos_objectives)} candidates.")
            
            # Store OOS results for each candidate (and the corresponding chromosome)
            for candidate_idx, oos_obj in enumerate(oos_objectives):
                candidate_id = f"w{window_idx}_c{candidate_idx}"  # Unique candidate ID
                # Track chromosome tensor for later reconstruction of best individuals
                candidate_chrom = window_best_chromosomes[candidate_idx]
                wfo_optimizer.record_window_result(window_idx, candidate_id, oos_obj, chromosome=candidate_chrom)
            
            # Log window summary
            logging.info("=== Window OOS Summary ===")
            for obj_name, stats in oos_summary.get('statistics', {}).items():
                if stats['valid_count'] > 0:
                    direction = "↓" if stats['minimize'] else "↑"
                    logging.info(f"  {obj_name}: {stats['mean']:.4f}±{stats['std']:.4f} "
                               f"[{stats['min']:.4f}, {stats['max']:.4f}] {direction}")

            # --- Weights & Biases logging per window ---
            try:
                def _natural_values(arr: np.ndarray, minimize_flags: List[bool]) -> np.ndarray:
                    arr = np.array(arr, dtype=float)
                    out = arr.copy()
                    for j, minimize in enumerate(minimize_flags):
                        if not minimize:
                            out[:, j] = -out[:, j]
                    return out

                def _valid_mask(v: np.ndarray) -> np.ndarray:
                    return (~np.isin(v, [-1e9, 1e9])) & np.isfinite(v)

                # IS stats (from window_best_objectives)
                is_cfg = eval_cfg.get('is_objectives', eval_cfg.get('objectives', {}))
                is_names = is_cfg.get('objectives', [])
                is_min_flags = is_cfg.get('minimize', [True] * len(is_names))
                if window_best_objectives is not None and len(window_best_objectives) > 0:
                    is_vals_nat = _natural_values(window_best_objectives, is_min_flags)
                    log_payload = {"window/index": window_idx}
                    for j, name in enumerate(is_names):
                        col = is_vals_nat[:, j]
                        m = _valid_mask(col)
                        if m.any():
                            log_payload[f"is/{name}/mean"] = float(np.mean(col[m]))
                            log_payload[f"is/{name}/std"] = float(np.std(col[m]))
                            log_payload[f"is/{name}/p05"] = float(np.percentile(col[m], 5))
                            log_payload[f"is/{name}/p95"] = float(np.percentile(col[m], 95))
                            log_payload[f"is/{name}/valid_count"] = int(np.sum(m))
                        else:
                            log_payload[f"is/{name}/valid_count"] = 0
                    # OOS stats
                    oos_cfg_names = oos_objectives_config.get('objectives', [])
                    oos_min_flags = oos_objectives_config.get('minimize', [True] * len(oos_cfg_names))
                    if oos_objectives is not None and len(oos_objectives) > 0:
                        oos_vals_nat = _natural_values(oos_objectives, oos_min_flags)
                        for j, name in enumerate(oos_cfg_names):
                            col = oos_vals_nat[:, j]
                            m = _valid_mask(col)
                            if m.any():
                                log_payload[f"oos/{name}/mean"] = float(np.mean(col[m]))
                                log_payload[f"oos/{name}/std"] = float(np.std(col[m]))
                                log_payload[f"oos/{name}/p05"] = float(np.percentile(col[m], 5))
                                log_payload[f"oos/{name}/p95"] = float(np.percentile(col[m], 95))
                                log_payload[f"oos/{name}/valid_count"] = int(np.sum(m))
                            else:
                                log_payload[f"oos/{name}/valid_count"] = 0
                        # Log candidate table (Top-K)
                        topk = int(wandb_cfg.get('log_tables_topk', 200))
                        K = min(topk, oos_objectives.shape[0])
                        if K > 0:
                            table_cols = ["candidate_id"] + oos_cfg_names
                            rows = []
                            for k in range(K):
                                cid = f"w{window_idx}_c{k}"
                                vals = [float(v) for v in oos_vals_nat[k, :len(oos_cfg_names)].tolist()]
                                rows.append([cid] + vals)
                            wandb_logger.log_table("tables/oos_candidates", table_cols, rows, step=window_idx)
                    wandb_logger.log(log_payload, step=window_idx)
            except Exception:
                pass
            
            # In-place elite carryover for next window (if not resetting)
            if not wfo_cfg.get('reset_population_each_window', False):
                elite_size = min(ga_cfg['evolution_loop']['elite_size'], len(window_best_chromosomes))
                # Option: refresh diversity by reinitializing full population, then overwrite elites
                population.make_population(device=env_cfg['device'])
                population.population_tensor[:elite_size].copy_(window_best_chromosomes[:elite_size])
                population.set_next_idx()
                population.initialized = True
                logging.info(f"Prepared next window population in-place with {elite_size} elites + random rest")
        
        # === Final Aggregation and Ranking ===
        logging.info("\n=== Final WFO Aggregation and Ranking ===")
        
        if not wfo_optimizer.candidate_archives:
            logging.error("No candidate results available for final ranking.")
            best_chromosomes, best_metrics = None, None
        else:
            # Get final rankings based on aggregated OOS objectives
            ranked_candidate_ids, aggregated_objectives = wfo_optimizer.get_final_rankings(device=env_cfg['device'])
            
            logging.info(f"Ranked {len(ranked_candidate_ids)} candidates based on aggregated OOS performance.")
            
            # Log aggregated results
            agg_obj_names = eval_cfg.get('oos_aggregate', {}).get('objectives', [])
            agg_minimize = eval_cfg.get('oos_aggregate', {}).get('minimize', [])
            
            logging.info("=== Aggregated OOS Objective Rankings ===")
            for i, candidate_id in enumerate(ranked_candidate_ids[:10]):  # Show top 10
                objectives = aggregated_objectives[i]
                logging.info(f"Rank {i+1:2d} | {candidate_id} | ", end="")
                for j, (obj_name, obj_val, minimize) in enumerate(zip(agg_obj_names, objectives, agg_minimize)):
                    direction = "↓" if minimize else "↑"
                    logging.info(f"{obj_name}: {obj_val:.4f} {direction} | ", end="")
                logging.info("")
            
            # Prepare final best chromosomes (reconstruct from stored candidate tensors)
            top_k = min(len(aggregated_objectives), ga_cfg['evolution_loop']['best_chromosome_pool_size'])
            reconstructed = []
            for rank_idx in range(top_k):
                cid = ranked_candidate_ids[rank_idx]
                chrom = wfo_optimizer.get_candidate_chromosome(cid)
                if chrom is not None:
                    reconstructed.append(chrom)
            if reconstructed:
                best_chromosomes = torch.stack(reconstructed, dim=0)
            else:
                best_chromosomes = None
            best_metrics = aggregated_objectives[:top_k]
            
        logging.info("Walk-Forward Optimization completed.")

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
        
    # (No Traditional Single-Split Mode)

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
