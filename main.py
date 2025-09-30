# /main.py
import os
import sys
import argparse
import yaml  # YAML 로드를 위해 추가
from tqdm import tqdm
import torch
import pandas as pd
import logging
import random
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional


DEFAULT_CONFIG_PATH = os.path.join("Config", "experiment_config.yaml")


def _construct_python_tuple(loader, node):
    """Allow YAML !!python/tuple tags when using SafeLoader."""
    return tuple(loader.construct_sequence(node))


yaml.SafeLoader.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    _construct_python_tuple,
)

# --- 프로젝트 모듈 임포트 ---
from models.model import GATreePop
from evolution.evolution import Evolution
from evolution.Selection import PPTSSelection
from evolution.Crossover import ChainCrossover, SubtreeCrossover, NodeCrossover, RootBranchCrossover
from evolution.Mutation import (
    ChainMutation, AddNodeMutation, DeleteNodeMutation, NodeParamMutation,
    ReinitializeNodeMutation, AddSubtreeMutation, DeleteSubtreeMutation
)
from training.runner import generation_training, generation_test, GenerationRunner
from training.wfo import WalkForwardOptimizer as _WFO
from training.cpcv_manager import CPCVManager
from analysis import wandb_logger
from utils.checkpoint_manager import CheckpointManager, detect_resumable_training

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


def load_config(path=DEFAULT_CONFIG_PATH):
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


def parse_arguments():
    """Parse command line arguments for resumption support."""
    parser = argparse.ArgumentParser(description="GA Tree Training with Resumption Support")
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume training from the latest checkpoint if available"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=DEFAULT_CONFIG_PATH,
        help="Path to the experiment configuration file"
    )
    parser.add_argument(
        "--clear-checkpoints", 
        action="store_true", 
        help="Clear all checkpoints and start fresh (use with caution!)"
    )
    return parser.parse_args()


class TrainingPipeline:
    """Class-based wrapper to run the GA Tree training pipeline."""

    def run(self):
        return _main_impl()


def _main_impl():
    """전체 학습 파이프라인을 실행하는 메인 함수"""
    # --- Parse command line arguments ---
    args = parse_arguments()
    
    # --- 설정 로드 ---
    config = load_config(args.config)

    # --- 환경 설정 ---
    env_cfg = config['environment']
    data_cfg = config['data']
    ga_cfg = config['ga_core']
    op_cfg = config['operators']
    eval_cfg = config['evaluation']
    classification_cfg = config['feature_classification']
    training_strategy_cfg = config.get('training_strategy', {'mode': 'wfo'})
    cpcv_cfg = config.get('cpcv', {})
    fitness_objectives_cfg = config.get('fitness_objectives', {})
    robustness_cfg = config.get('robustness_filter', {})

    set_seed(env_cfg['seed'])
    setup_environment(env_cfg['output_dir'])

    # --- Initialize Checkpoint Manager ---
    checkpoint_manager = CheckpointManager(env_cfg['output_dir'], config)
    
    # Handle checkpoint clearing if requested
    if args.clear_checkpoints:
        logging.info("Clearing all checkpoints...")
        checkpoint_manager.clear_checkpoints()
    
    # Check for resumable training
    resume_run_id = None
    resumption_cfg = config.get('resumption', {})
    if args.resume or resumption_cfg.get('auto_resume', False):
        if checkpoint_manager.is_resumable():
            main_checkpoint = checkpoint_manager.load_main_checkpoint()
            resume_run_id = main_checkpoint.get('wandb_run_id') if main_checkpoint else None
            if resume_run_id:
                logging.info(f"Found resumable training with wandb run ID: {resume_run_id}")
            else:
                logging.info("Found resumable training (no wandb run ID)")
                
            # Clean up old checkpoints if requested
            if resumption_cfg.get('cleanup_old_checkpoints', True):
                keep_n = resumption_cfg.get('keep_last_n_generations', 5)
                checkpoint_manager.cleanup_old_checkpoints(keep_n)
        elif args.resume:
            logging.warning("Resume requested but no valid checkpoint found. Starting fresh.")
    
    # --- Initialize Weights & Biases (optional) ---
    wandb_cfg = config.get('wandb', {})
    ts = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    run_name = f"{data_cfg.get('symbol','SYM')}-{data_cfg.get('interval','int')}-{env_cfg.get('seed',0)}-{ts}"
    try:
        wandb_logger.init_run(wandb_cfg, run_name, env_cfg['output_dir'], full_config=config, resume_run_id=resume_run_id)
        # Update checkpoint with current wandb run ID
        current_run_id = wandb_logger.get_run_id()
        if current_run_id:
            checkpoint_manager.save_main_checkpoint(
                current_window=0,
                total_windows=0,
                completed_generations=0,
                wandb_run_id=current_run_id,
            )
    except Exception:
        pass

    # ==========================================================================
    #             Phase 1: 데이터 준비 및 피처 생성 (단순 검사)
    # ==========================================================================
    logging.info("--- Phase 1: Data Preparation & Feature Check ---")

    final_features_path = os.path.join(env_cfg['data_dir'], data_cfg['final_features_path'])
    base, ext = os.path.splitext(final_features_path)
    final_features_pkl = base + '.pkl'
    model_config_path = os.path.join(env_cfg['data_dir'], data_cfg['model_config_path'])

    # --- 최종 전처리된 데이터가 있는지 확인 ---
    if not ((os.path.exists(final_features_pkl) or os.path.exists(final_features_path)) and os.path.exists(model_config_path)):
        logging.error("Preprocessed data not found.")
        logging.error("Please run 'preprocess_data.py' first to generate the data.")
        sys.exit(1)

    # Prefer pickle if available (smaller and faster)
    if os.path.exists(final_features_pkl):
        logging.info(f"Loading pre-processed data from '{final_features_pkl}' (pickle)")
        final_df = pd.read_pickle(final_features_pkl)
    else:
        logging.info(f"Loading pre-processed data from '{final_features_path}' (csv)")
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

    def to_natural_space(metrics: np.ndarray | None, minimize_flags: List[bool]) -> np.ndarray | None:
        if metrics is None or not isinstance(metrics, np.ndarray):
            return metrics
        natural = metrics.copy()
        num_cols = natural.shape[1] if natural.ndim == 2 else natural.shape[0]
        for idx in range(num_cols):
            if idx < len(minimize_flags) and not minimize_flags[idx]:
                natural[..., idx] = -natural[..., idx]
        return natural

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
        RootBranchCrossover(max_nodes=ga_cfg['population']['max_nodes']),
        NodeCrossover(mode=op_cfg['crossover']['node_crossover']['mode']),
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

    # Ensure structural mutations receive the correct max_nodes matching the population
    _max_nodes = ga_cfg['population']['max_nodes']
    mutation_ops = [
        AddNodeMutation(
            prob=mut_probs['add_node'],
            config=mutation_base_config,
            max_nodes=_max_nodes,
            **mut_params['add_node_mutation']
        ),
        DeleteNodeMutation(
            prob=mut_probs['delete_node'],
            config=mutation_base_config,
            max_nodes=_max_nodes,
            **mut_params['delete_node_mutation']
        ),
        AddSubtreeMutation(
            prob=mut_probs['add_subtree'],
            config=mutation_base_config,
            max_nodes=_max_nodes,
            **mut_params['add_subtree_mutation']
        ),
        DeleteSubtreeMutation(
            prob=mut_probs['delete_subtree'],
            config=mutation_base_config,
            max_nodes=_max_nodes,
        ),
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
    #             Phase 4: 전략 학습 루프 실행 (WFO / CPCV)
    # ==========================================================================
    logging.info("--- Phase 4: Strategy Training Loop ---")

    total_data_len = len(final_df)
    strategy_mode = str(training_strategy_cfg.get('mode', 'wfo')).lower()
    best_chromosomes = None
    best_metrics = None
    strategy_context: Dict[str, object] = {'mode': strategy_mode}
    wfo_optimizer = None

    if strategy_mode == 'cpcv':
        logging.info("=== CPCV Mode Enabled ===")
        cpcv_manager = CPCVManager(
            training_strategy_cfg=training_strategy_cfg,
            cpcv_cfg=cpcv_cfg,
            evaluation_cfg=eval_cfg,
            fitness_cfg=fitness_objectives_cfg,
            robustness_cfg=robustness_cfg,
        )
        best_chromosomes, best_metrics, combo_ids = cpcv_manager.run(
            final_df=final_df,
            evolution_engine=evolution_engine,
            model_config=model_config,
            ga_cfg=ga_cfg,
            env_cfg=env_cfg,
            wandb_cfg=wandb_cfg,
            output_dir=env_cfg['output_dir'],
            checkpoint_manager=checkpoint_manager,
        )
        if best_chromosomes is None or best_metrics is None:
            logging.error("CPCV training did not produce valid results. Aborting.")
            return
        strategy_context.update({
            'combo_ids': combo_ids,
            'summary': getattr(cpcv_manager, 'last_summary', {}),
            'objective_names': fitness_objectives_cfg.get('cpcv_objectives', {}).get('names', []),
            'objective_minimize': fitness_objectives_cfg.get('cpcv_objectives', {}).get('minimize', []),
            'robustness': getattr(cpcv_manager, 'last_robustness', []),
        })

        holdout_obj = getattr(cpcv_manager, 'holdout_objectives', None)
        if getattr(cpcv_manager, 'holdout_length', 0) > 0 and isinstance(holdout_obj, np.ndarray) and holdout_obj.size:
            holdout_summary = getattr(cpcv_manager, 'holdout_summary', {}) or {}
            holdout_indices = getattr(cpcv_manager, 'holdout_indices', None)
            holdout_timestamps = getattr(cpcv_manager, 'holdout_timestamps', (None, None)) or (None, None)

            ts_serialized: List[Optional[str]] = []
            for ts in holdout_timestamps:
                if isinstance(ts, (pd.Timestamp, datetime)):
                    ts_serialized.append(ts.isoformat())
                elif ts is None:
                    ts_serialized.append(None)
                else:
                    ts_serialized.append(str(ts))

            holdout_payload = {
                'length': cpcv_manager.holdout_length,
                'indices': list(holdout_indices) if holdout_indices else None,
                'timestamps': ts_serialized,
                'objective_names': strategy_context.get('objective_names', []),
                'objectives': holdout_obj.tolist(),
                'summary': {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in holdout_summary.items()
                },
            }
            strategy_context['holdout'] = holdout_payload

            try:
                holdout_natural = to_natural_space(holdout_obj, strategy_context.get('objective_minimize', []))
                if isinstance(holdout_natural, np.ndarray) and holdout_natural.size:
                    top1_holdout = holdout_natural[0]
                    logging.info("Holdout top-1 metrics: %s", {
                        name: float(top1_holdout[idx])
                        for idx, name in enumerate(strategy_context.get('objective_names', []))
                    })
            except Exception:
                pass

        logging.info("CPCV mode completed with %d combinations.", len(combo_ids))
    else:
        logging.info("=== Walk-Forward Optimization Mode Enabled ===")
        wfo_cfg = config.get('walk_forward', {'enabled': False})
        if not wfo_cfg.get('enabled', False):
            logging.error("Walk-Forward Optimization mode is required but disabled in config.")
            logging.error(
                f"Please set walk_forward.enabled: true in {DEFAULT_CONFIG_PATH} or switch "
                "training_strategy.mode to 'cpcv'."
            )
            sys.exit(1)

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
            checkpoint_manager=checkpoint_manager,
        )
        if not windows:
            logging.error("No valid WFO windows could be generated. Check your configuration.")
            logging.error(
                "Data length: %s, train_bars: %s, oos_bars: %s",
                total_data_len,
                wfo_cfg.get('train_bars'),
                wfo_cfg.get('oos_bars'),
            )
            return
        logging.info("Generated %d WFO windows from %d data points (refactored WFO)", len(windows), total_data_len)
        strategy_context.update({
            'windows': windows,
            'wfo_cfg': wfo_cfg,
            'optimizer': wfo_optimizer,
            'objective_names': eval_cfg.get('oos_aggregate', {}).get('objectives', []),
            'objective_minimize': eval_cfg.get('oos_aggregate', {}).get('minimize', []),
        })
        if bool(eval_cfg.get('oos_aggregate', {}).get('include_dsr', False)):
            agg_names = strategy_context.get('objective_names', [])
            agg_minimize = strategy_context.get('objective_minimize', [])
            if isinstance(best_metrics, np.ndarray) and best_metrics.shape[1] > len(agg_names):
                dsr_name = 'oos_dsr'
                dsr_minimize = bool(eval_cfg.get('oos_aggregate', {}).get('dsr_minimize', False))
                strategy_context['objective_names'] = list(agg_names) + [dsr_name]
                strategy_context['objective_minimize'] = list(agg_minimize) + [dsr_minimize]

    # --- Weights & Biases: log final summary ---
    try:
        metric_names = strategy_context.get('objective_names', [])
        minimize_flags = strategy_context.get('objective_minimize', [])
        final_step = 0
        if strategy_mode == 'wfo':
            final_step = len(strategy_context.get('windows', []))
        elif strategy_mode == 'cpcv':
            final_step = len(strategy_context.get('combo_ids', []))

        natural_metrics = to_natural_space(best_metrics, minimize_flags)
        if isinstance(natural_metrics, np.ndarray) and natural_metrics.size and metric_names:
            top1 = natural_metrics[0]
            topk_avg = np.mean(natural_metrics, axis=0)
            payload = {
                "final/units": final_step,
                "final/candidates": int(len(best_metrics)),
                "final/mode": strategy_mode,
            }
            for j, name in enumerate(metric_names):
                payload[f"final/top1/{name}"] = float(top1[j])
                payload[f"final/topK/avg/{name}"] = float(topk_avg[j])
                if j < len(minimize_flags):
                    payload[f"final/minimize/{name}"] = bool(minimize_flags[j])
            wandb_logger.log(payload, step=final_step)
    except Exception:
        pass
        

    logging.info("Training loop finished.")

    # ==========================================================================
    #             Phase 5: 결과 저장 및 최종 평가
    # ==========================================================================
    logging.info("--- Phase 5: Saving Best Population and Final Evaluation ---")

    best_population_path = os.path.join(env_cfg['output_dir'], "best_population.pth")
    topk_path = os.path.join(env_cfg['output_dir'], "best_population_topK.pth")

    saved_artifacts: List[str] = []

    if best_chromosomes is None or (hasattr(best_chromosomes, '__len__') and len(best_chromosomes) == 0):
        logging.warning("No best chromosomes were found during training. Skipping final save and test.")
    else:
        cpu_best = best_chromosomes.detach().to('cpu') if isinstance(best_chromosomes, torch.Tensor) else torch.as_tensor(best_chromosomes)

    # Save Top-1 model
    top1_population = GATreePop(
        pop_size=1,
        max_nodes=ga_cfg['population']['max_nodes'],
        max_depth=ga_cfg['population']['max_depth'],
        max_children=ga_cfg['population']['max_children'],
        **model_config
    )
    top1_population.population_tensor[:1].copy_(cpu_best[:1])
    top1_population.set_next_idx()
    top1_population.initialized = True
    top1_population.save(best_population_path)
    saved_artifacts.append(best_population_path)
    logging.info("Saved top-1 model to %s", best_population_path)

    # Save Top-K bundle
    topk_population = GATreePop(
        pop_size=len(cpu_best),
        max_nodes=ga_cfg['population']['max_nodes'],
        max_depth=ga_cfg['population']['max_depth'],
        max_children=ga_cfg['population']['max_children'],
        **model_config
    )
    topk_population.population_tensor.copy_(cpu_best)
    topk_population.set_next_idx()
    topk_population.initialized = True
    topk_population.save(topk_path)
    saved_artifacts.append(topk_path)
    logging.info("Saved Top-K models to %s", topk_path)

    # Mode-specific summaries
    if strategy_mode == 'wfo' and wfo_optimizer is not None:
        windows = strategy_context.get('windows', [])
        wfo_cfg = strategy_context.get('wfo_cfg', {})
        wfo_results_path = os.path.join(env_cfg['output_dir'], "wfo_results.json")
        natural_metrics = to_natural_space(best_metrics, strategy_context.get('objective_minimize', []))
        wfo_results = {
            'mode': 'wfo',
            'num_windows': len(windows),
            'num_candidates': len(getattr(wfo_optimizer, 'candidate_archives', {})),
            'aggregated_objectives': natural_metrics.tolist() if isinstance(natural_metrics, np.ndarray) else natural_metrics,
            'objective_names': strategy_context.get('objective_names', []),
            'minimize_directions': strategy_context.get('objective_minimize', []),
            'wfo_config': wfo_cfg,
        }

        try:
            from analysis.stat_tests import window_level_dsr

            top1_id = None
            if cpu_best.numel() > 0:
                top1_chrom = cpu_best[0]
                for cid, chrom in getattr(wfo_optimizer, 'candidate_chromosomes', {}).items():
                    try:
                        if torch.equal(chrom, top1_chrom):
                            top1_id = cid
                            break
                    except Exception:
                        continue

            dsr_per_window = []
            oos_obj_cfg = eval_cfg.get('oos_objectives', eval_cfg.get('objectives', {}))
            oos_names = oos_obj_cfg.get('objectives', []) if isinstance(oos_obj_cfg, dict) else []
            sharpe_idx = oos_names.index('sharpe_ratio') if 'sharpe_ratio' in oos_names else None

            if top1_id is not None and sharpe_idx is not None:
                n_obs = int(wfo_cfg.get('oos_bars', 1))
                lookback_period = int(oos_obj_cfg.get('lookback_period', 252)) if isinstance(oos_obj_cfg, dict) else 252
                scale = np.sqrt(max(1, lookback_period))
                dsr_sr_bm = float(eval_cfg.get('oos_aggregate', {}).get('dsr_sr_benchmark', 0.0))

                for w_idx, _ in enumerate(windows):
                    sr_all = []
                    sr_selected = None
                    for cid, win_map in wfo_optimizer.candidate_archives.items():
                        if w_idx in win_map:
                            val = win_map[w_idx][sharpe_idx]
                            sr_all.append(float(val) / scale)
                            if cid == top1_id:
                                sr_selected = float(val) / scale
                    if sr_selected is None or len(sr_all) == 0:
                        dsr_per_window.append(float('nan'))
                    else:
                        dsr_val = window_level_dsr(
                            sr_selected=float(sr_selected),
                            n_obs=n_obs,
                            sr_all_candidates=sr_all,
                            sr_benchmark=dsr_sr_bm,
                        )
                        dsr_per_window.append(float(dsr_val))

                dsr_valid = [x for x in dsr_per_window if x == x]
                if dsr_valid:
                    wfo_results['dsr_summary'] = {
                        'mean': float(np.mean(dsr_valid)),
                        'median': float(np.median(dsr_valid)),
                        'min': float(np.min(dsr_valid)),
                        'num_windows': int(len(dsr_valid)),
                    }
                wfo_results['dsr_per_window'] = dsr_per_window
        except Exception:
            pass

        with open(wfo_results_path, 'w', encoding='utf-8') as f:
            json.dump(wfo_results, f, indent=2)
        saved_artifacts.append(wfo_results_path)
        logging.info("WFO results saved to %s", wfo_results_path)

        if getattr(wfo_optimizer, 'candidate_archives', None):
            logging.info("\n========== WFO FINAL SUMMARY ==========")
            logging.info("Total windows processed: %d", len(windows))
            logging.info("Total candidates evaluated: %d", len(wfo_optimizer.candidate_archives))
            if isinstance(natural_metrics, np.ndarray) and len(natural_metrics) > 0:
                agg_obj_names = list(strategy_context.get('objective_names', []))
                agg_minimize = list(strategy_context.get('objective_minimize', []))
                logging.info("Top 5 candidates by aggregated OOS performance:")
                for i in range(min(5, len(natural_metrics))):
                    objectives = natural_metrics[i]
                    line = f"Candidate {i+1:2d} | "
                    for obj_name, obj_val, minimize in zip(agg_obj_names, objectives, agg_minimize):
                        direction = "↓" if minimize else "↑"
                        line += f"{obj_name}: {obj_val:.4f} {direction} | "
                    logging.info(line)
            logging.info("============================================")

    elif strategy_mode == 'cpcv':
        cpcv_results_path = os.path.join(env_cfg['output_dir'], "cpcv_results.json")
        combo_ids = strategy_context.get('combo_ids', [])
        summary = strategy_context.get('summary', {})
        summary_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in summary.items()}
        robustness_reports = strategy_context.get('robustness', [])
        robustness_serializable = [
            {
                'pbo': report.pbo,
                'dsr': report.dsr,
                'spa_pvalue': report.spa_pvalue,
                'min_trl_ok': report.min_trl_ok,
                'bootstrap_ci': {k: list(v) for k, v in report.bootstrap_ci.items()},
                'notes': report.notes,
            }
            for report in robustness_reports
        ]

        natural_metrics = to_natural_space(best_metrics, strategy_context.get('objective_minimize', []))
        cpcv_results = {
            'mode': 'cpcv',
            'num_combinations': len(combo_ids),
            'combo_ids': combo_ids,
            'objective_names': strategy_context.get('objective_names', []),
            'minimize_directions': strategy_context.get('objective_minimize', []),
            'aggregated_objectives': natural_metrics.tolist() if isinstance(natural_metrics, np.ndarray) else natural_metrics,
            'summary': summary_serializable,
            'cpcv_config': cpcv_cfg,
            'robustness': robustness_serializable,
            'holdout': strategy_context.get('holdout'),
        }
        with open(cpcv_results_path, 'w', encoding='utf-8') as f:
            json.dump(cpcv_results, f, indent=2)
        saved_artifacts.append(cpcv_results_path)
        logging.info("CPCV results saved to %s", cpcv_results_path)

        if isinstance(natural_metrics, np.ndarray) and len(natural_metrics) > 0:
            metric_names = strategy_context.get('objective_names', [])
            minimize_flags = strategy_context.get('objective_minimize', [])
            logging.info("\n========== CPCV FINAL SUMMARY ==========")
            logging.info("Total combinations evaluated: %d", len(combo_ids))
            for i in range(min(5, len(natural_metrics))):
                objectives = natural_metrics[i]
                line = f"Candidate {i+1:2d} | "
                for obj_name, obj_val, minimize in zip(metric_names, objectives, minimize_flags):
                    direction = "↓" if minimize else "↑"
                    line += f"{obj_name}: {obj_val:.4f} {direction} | "
                logging.info(line)
            logging.info("============================================")

    # Log artifacts after saving
    try:
        if saved_artifacts and bool(wandb_cfg.get('log_artifacts', True)):
            wandb_logger.log_artifacts(saved_artifacts, name=f"{run_name}-artifacts", type_="model")
    except Exception:
        pass

    logging.info("Main script finished.")


if __name__ == "__main__":
    TrainingPipeline().run()
