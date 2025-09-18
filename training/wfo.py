# /training/wfo.py

import numpy as np
import torch
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict, Any

from .multi_objective_fitness import MultiObjectiveFitnessEvaluator, OBJECTIVE_CLAMP_MIN, OBJECTIVE_CLAMP_MAX
from analysis.stat_tests import window_level_dsr
from utils.checkpoint_manager import CheckpointManager
from .pruner import PopulationPruner


def aggregate_oos_objectives(objectives_per_window: np.ndarray, aggregate_config: dict) -> np.ndarray:
    if objectives_per_window.size == 0:
        return np.array([])

    methods = aggregate_config.get('methods', ['mean'] * objectives_per_window.shape[1])
    thresholds = aggregate_config.get('thresholds', {})
    base_minimize = aggregate_config.get('base_minimize', [False] * objectives_per_window.shape[1])

    aggregated = []

    for i, method in enumerate(methods):
        if i < objectives_per_window.shape[1]:
            obj_values = objectives_per_window[:, i]
            valid_mask = ~np.isin(obj_values, [-1e9, 1e9]) & ~np.isnan(obj_values) & ~np.isinf(obj_values)

            if not np.any(valid_mask):
                aggregated.append(1e9 if base_minimize[i] else -1e9)
                continue

            valid_values = np.clip(
                obj_values[valid_mask].astype(np.float64, copy=False),
                OBJECTIVE_CLAMP_MIN,
                OBJECTIVE_CLAMP_MAX,
            )

            if method == 'mean':
                result = np.mean(valid_values)
            elif method == 'median':
                result = np.median(valid_values)
            elif method == 'min':
                result = np.min(valid_values)
            elif method == 'max':
                result = np.max(valid_values)
            elif method == 'std':
                result = np.std(valid_values, dtype=np.float64)
            elif method.startswith('p'):
                percentile = float(method[1:])
                result = np.percentile(valid_values, percentile)
            else:
                result = np.mean(valid_values)

            aggregated.append(result)
        else:
            if method == 'hit_rate':
                result = calculate_consistency_rate(objectives_per_window, thresholds)
                aggregated.append(result)
            else:
                if i < len(base_minimize):
                    aggregated.append(1e9 if base_minimize[i] else -1e9)
                else:
                    aggregated.append(0.0)

    return np.array(aggregated)


def calculate_consistency_rate(objectives_per_window: np.ndarray, thresholds: dict) -> float:
    if objectives_per_window.size == 0:
        return 0.0

    num_windows = objectives_per_window.shape[0]
    meets_criteria = np.ones(num_windows, dtype=bool)

    min_return = thresholds.get('min_return', 0.0)
    if objectives_per_window.shape[1] > 0:
        return_values = objectives_per_window[:, 0]
        valid_mask = ~np.isin(return_values, [-1e9, 1e9]) & ~np.isnan(return_values)
        meets_criteria &= (return_values >= min_return) & valid_mask

    max_dd = thresholds.get('max_dd', 0.3)
    if objectives_per_window.shape[1] > 1:
        dd_values = objectives_per_window[:, 1]
        valid_mask = ~np.isin(dd_values, [-1e9, 1e9]) & ~np.isnan(dd_values)
        meets_criteria &= (dd_values <= max_dd) & valid_mask

    return float(np.mean(meets_criteria))


def generate_wfo_windows(
    total_length: int,
    train_bars: int,
    oos_bars: int,
    step_bars: int,
    warmup_bars: int = 0,
    embargo_bars: int = 0,
    scheme: str = 'sliding',
) -> List[Tuple[int, int, int, int]]:
    windows = []
    start_pos = warmup_bars

    while True:
        if scheme == 'expanding':
            train_start = warmup_bars
        else:
            train_start = start_pos

        train_end = start_pos + train_bars
        oos_start = train_end + embargo_bars
        oos_end = oos_start + oos_bars

        if oos_end > total_length:
            break

        windows.append((train_start, train_end, oos_start, oos_end))
        start_pos += step_bars

    return windows


class WalkForwardOptimizer:
    """WFO coordinator managing rolling IS/OOS evaluation and aggregation."""

    def __init__(self, wfo_config: dict, evaluation_config: dict):
        self.wfo_config = wfo_config
        self.evaluation_config = evaluation_config
        self._validate_config()

        self.is_evaluator: Optional[MultiObjectiveFitnessEvaluator] = None
        self.oos_evaluator: Optional[MultiObjectiveFitnessEvaluator] = None
        self.aggregate_config = evaluation_config.get('oos_aggregate', {})

        self.window_results = []
        self.candidate_archives: Dict[int, Dict[int, np.ndarray]] = {}
        self.candidate_chromosomes: Dict[int, torch.Tensor] = {}

    def _validate_config(self):
        required_keys = ['train_bars', 'oos_bars', 'step_bars']
        for key in required_keys:
            if key not in self.wfo_config:
                raise ValueError(f"Missing required WFO config key: {key}")
        if self.wfo_config['train_bars'] <= 0:
            raise ValueError("train_bars must be positive")
        if self.wfo_config['oos_bars'] <= 0:
            raise ValueError("oos_bars must be positive")
        if self.wfo_config['step_bars'] <= 0:
            raise ValueError("step_bars must be positive")

    def get_is_evaluator(self) -> MultiObjectiveFitnessEvaluator:
        if self.is_evaluator is None:
            is_config = self.evaluation_config.get('is_objectives', self.evaluation_config.get('objectives'))
            self.is_evaluator = MultiObjectiveFitnessEvaluator(is_config)
        return self.is_evaluator

    def get_oos_evaluator(self) -> MultiObjectiveFitnessEvaluator:
        if self.oos_evaluator is None:
            oos_config = self.evaluation_config.get('oos_objectives', self.evaluation_config.get('objectives'))
            self.oos_evaluator = MultiObjectiveFitnessEvaluator(oos_config)
        return self.oos_evaluator

    def generate_windows(self, total_length: int) -> List[Tuple[int, int, int, int]]:
        return generate_wfo_windows(
            total_length=total_length,
            train_bars=self.wfo_config['train_bars'],
            oos_bars=self.wfo_config['oos_bars'],
            step_bars=self.wfo_config['step_bars'],
            warmup_bars=self.wfo_config.get('warmup_bars', 0),
            embargo_bars=self.wfo_config.get('embargo_bars', 0),
            scheme=self.wfo_config.get('scheme', 'sliding'),
        )

    def record_window_result(
        self,
        window_idx: int,
        candidate_id: int,
        oos_objectives: np.ndarray,
        chromosome: Optional[torch.Tensor] = None,
    ):
        if candidate_id not in self.candidate_archives:
            self.candidate_archives[candidate_id] = {}
        self.candidate_archives[candidate_id][window_idx] = oos_objectives.copy()

        if chromosome is not None and candidate_id not in self.candidate_chromosomes:
            try:
                self.candidate_chromosomes[candidate_id] = chromosome.detach().to('cpu').clone()
            except Exception:
                pass

    def get_candidate_chromosome(self, candidate_id: int) -> Optional[torch.Tensor]:
        return self.candidate_chromosomes.get(candidate_id, None)

    def aggregate_candidate_objectives(self, candidate_id: int) -> np.ndarray:
        if candidate_id not in self.candidate_archives:
            raise ValueError(f"No results found for candidate {candidate_id}")
        window_results = self.candidate_archives[candidate_id]
        full_window_objectives = np.array(list(window_results.values()))

        # Reorder/select base OOS columns to align with aggregate objectives naming (oos_*)
        # Map oos_aggregate.objectives like ['oos_mean_return','oos_max_drawdown','oos_psr', 'oos_consistency']
        # to base OOS names ['mean_return','max_drawdown','psr'] so that column i corresponds to aggregate objective i.
        oos_obj_cfg = self.evaluation_config.get('oos_objectives', self.evaluation_config.get('objectives', {}))
        oos_names = oos_obj_cfg.get('objectives', []) if isinstance(oos_obj_cfg, dict) else []
        agg_obj_names = list(self.aggregate_config.get('objectives', []))

        selected_cols = []
        for agg_name in agg_obj_names:
            if not isinstance(agg_name, str):
                continue
            if agg_name.startswith('oos_') and agg_name != 'oos_consistency':
                base = agg_name[len('oos_'):]
                if base in oos_names:
                    idx = oos_names.index(base)
                    selected_cols.append(idx)
                else:
                    # Base metric not present; skip (will be handled as missing by aggregator methods)
                    pass
            else:
                # Special objectives like 'oos_consistency' are handled by aggregator methods
                pass

        if selected_cols:
            window_objectives = full_window_objectives[:, selected_cols]
        else:
            window_objectives = full_window_objectives

        # Base aggregation of requested OOS objectives
        aggregated = aggregate_oos_objectives(window_objectives, self.aggregate_config)

        # Optional: include DSR as additional aggregated objective
        include_dsr = bool(self.aggregate_config.get('include_dsr', False)) and ('sharpe_ratio' in oos_names)
        if include_dsr:
            try:
                sharpe_idx = oos_names.index('sharpe_ratio')
                lookback_period = int(oos_obj_cfg.get('lookback_period', 252)) if isinstance(oos_obj_cfg, dict) else 252
                scale = float(np.sqrt(max(1, lookback_period)))
                n_obs = int(self.wfo_config.get('oos_bars', 1))
                dsr_sr_bm = float(self.aggregate_config.get('dsr_sr_benchmark', 0.0))

                dsr_per_window: list[float] = []
                for w_idx in sorted(window_results.keys()):
                    sr_all: list[float] = []
                    sr_selected: Optional[float] = None
                    for cid, win_map in self.candidate_archives.items():
                        if w_idx in win_map and len(win_map[w_idx]) > sharpe_idx:
                            val = float(win_map[w_idx][sharpe_idx])
                            sr = val / scale  # convert to per-period SR
                            sr_all.append(sr)
                            if cid == candidate_id:
                                sr_selected = sr
                    if sr_selected is None or len(sr_all) == 0:
                        continue
                    dsr_val = window_level_dsr(
                        sr_selected=float(sr_selected),
                        n_obs=n_obs,
                        sr_all_candidates=sr_all,
                        sr_benchmark=dsr_sr_bm,
                    )
                    dsr_per_window.append(float(dsr_val))

                if dsr_per_window:
                    method = str(self.aggregate_config.get('dsr_method', 'mean')).lower()
                    if method == 'median':
                        agg_dsr = float(np.median(dsr_per_window))
                    elif method == 'min':
                        agg_dsr = float(np.min(dsr_per_window))
                    elif method == 'max':
                        agg_dsr = float(np.max(dsr_per_window))
                    elif method.startswith('p'):
                        try:
                            percentile = float(method[1:])
                            agg_dsr = float(np.percentile(dsr_per_window, percentile))
                        except Exception:
                            agg_dsr = float(np.mean(dsr_per_window))
                    else:
                        agg_dsr = float(np.mean(dsr_per_window))
                else:
                    dsr_minimize = bool(self.aggregate_config.get('dsr_minimize', False))
                    agg_dsr = (1e9 if dsr_minimize else -1e9)

                aggregated = np.append(aggregated, agg_dsr)
            except Exception:
                # On failure, keep base aggregated only
                pass

        return aggregated

    def get_final_rankings(self, device: str | torch.device | None = None) -> Tuple[List[int], np.ndarray]:
        if not self.candidate_archives:
            return [], np.array([])

        candidate_ids = list(self.candidate_archives.keys())
        aggregated_matrix = []
        for candidate_id in candidate_ids:
            agg_obj = self.aggregate_candidate_objectives(candidate_id)
            aggregated_matrix.append(agg_obj)
        aggregated_matrix = np.array(aggregated_matrix)

        # Convert to minimization space based on config
        minimize_flags = list(self.aggregate_config.get('minimize', []))
        include_dsr = bool(self.aggregate_config.get('include_dsr', False))
        if include_dsr and len(minimize_flags) < aggregated_matrix.shape[1]:
            dsr_minimize = bool(self.aggregate_config.get('dsr_minimize', False))
            minimize_flags = minimize_flags + [dsr_minimize]
        if len(minimize_flags) < aggregated_matrix.shape[1]:
            minimize_flags = minimize_flags + [False] * (aggregated_matrix.shape[1] - len(minimize_flags))

        agg_for_min = aggregated_matrix.copy()
        for j in range(agg_for_min.shape[1]):
            if not minimize_flags[j]:
                agg_for_min[:, j] = -agg_for_min[:, j]

        agg_evaluator = MultiObjectiveFitnessEvaluator({'objectives': ['agg']*agg_for_min.shape[1], 'minimize': [True]*agg_for_min.shape[1]})
        ranks, distances = agg_evaluator.pareto_rank(agg_for_min, device=device)
        sort_indices = np.lexsort((-distances, ranks))
        ranked_candidates = [candidate_ids[i] for i in sort_indices]
        return ranked_candidates, aggregated_matrix[sort_indices]

    def save_wfo_state(self, checkpoint_manager: CheckpointManager, current_window: int, total_windows: int) -> None:
        """Save WFO state for resumption."""
        wfo_state = {
            "candidate_archives": {
                str(k): {str(win_k): win_v.tolist() for win_k, win_v in v.items()}
                for k, v in self.candidate_archives.items()
            },
            "candidate_chromosomes_keys": list(self.candidate_chromosomes.keys()),
            "window_results": self.window_results,
            "current_window": current_window,
            "total_windows": total_windows,
        }
        
        try:
            checkpoint_manager.save_main_checkpoint(
                current_window=current_window,
                total_windows=total_windows,
                completed_generations=0,  # Will be updated by generation training
                wfo_state=wfo_state,
            )
            
            # Save candidate chromosomes separately
            for candidate_id, chromosome in self.candidate_chromosomes.items():
                checkpoint_path = checkpoint_manager.checkpoint_dir / f"candidate_{candidate_id}.pth"
                torch.save(chromosome, checkpoint_path)
                
        except Exception as e:
            logging.error(f"Failed to save WFO state: {e}")

    def load_wfo_state(self, checkpoint_manager: CheckpointManager) -> Optional[Tuple[int, int]]:
        """Load WFO state for resumption. Returns (current_window, total_windows) if successful."""
        try:
            main_checkpoint = checkpoint_manager.load_main_checkpoint()
            if not main_checkpoint:
                return None
            
            wfo_state = main_checkpoint.get("wfo_state", {})
            if not wfo_state:
                return None
            
            # Restore candidate archives
            self.candidate_archives = {}
            for cand_id_str, win_dict in wfo_state.get("candidate_archives", {}).items():
                cand_id = int(cand_id_str)
                self.candidate_archives[cand_id] = {}
                for win_idx_str, objectives_list in win_dict.items():
                    win_idx = int(win_idx_str)
                    self.candidate_archives[cand_id][win_idx] = np.array(objectives_list)
            
            # Restore candidate chromosomes
            self.candidate_chromosomes = {}
            for candidate_id in wfo_state.get("candidate_chromosomes_keys", []):
                checkpoint_path = checkpoint_manager.checkpoint_dir / f"candidate_{candidate_id}.pth"
                if checkpoint_path.exists():
                    self.candidate_chromosomes[candidate_id] = torch.load(checkpoint_path, weights_only=False)
            
            # Restore other state
            self.window_results = wfo_state.get("window_results", [])
            
            current_window = wfo_state.get("current_window", 0)
            total_windows = wfo_state.get("total_windows", 0)
            
            logging.info(f"Restored WFO state: window {current_window}/{total_windows}, "
                        f"{len(self.candidate_archives)} candidates")
            
            return current_window, total_windows
            
        except Exception as e:
            logging.error(f"Failed to load WFO state: {e}")
            return None

    def should_skip_window(self, window_idx: int, checkpoint_manager: CheckpointManager) -> bool:
        """Check if a window should be skipped during resumption."""
        completed_windows = checkpoint_manager.get_completed_windows()
        return window_idx in completed_windows

    # --- High-level orchestration moved from main.py ---
    def run(
        self,
        final_df: pd.DataFrame,
        evolution_engine: Any,
        model_config: dict,
        ga_cfg: dict,
        env_cfg: dict,
        wfo_cfg: dict,
        wandb_cfg: Optional[dict] = None,
        output_dir: Optional[str] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> Tuple[Optional['torch.Tensor'], Optional[np.ndarray], List[Tuple[int, int, int, int]]]:
        """Execute Walk-Forward Optimization and return best chromosomes and metrics.

        Returns (best_chromosomes, best_metrics, windows).
        """
        import os
        import numpy as np
        import torch
        from tqdm import tqdm
        from training.runner import generation_training, generation_test
        from models.model import GATreePop

        total_data_len = len(final_df)
        windows = self.generate_windows(total_data_len)
        if len(windows) == 0:
            return None, None, []

        population = evolution_engine.population
        
        # Handle resumption if checkpoint manager is provided
        start_window_idx = 0
        latest_generation_checkpoint = None
        if checkpoint_manager is not None:
            resumption_state = self.load_wfo_state(checkpoint_manager)
            if resumption_state is not None:
                current_window, total_windows = resumption_state
                if current_window < len(windows):
                    start_window_idx = current_window
                    logging.info(f"Resuming WFO from window {start_window_idx}")
                else:
                    logging.info("All windows already completed, starting final aggregation")
                    start_window_idx = len(windows)
            else:
                logging.info("No valid checkpoint found, starting from beginning")
            latest_generation_checkpoint = checkpoint_manager.find_latest_checkpoint()

        for window_idx, (train_start, train_end, oos_start, oos_end) in enumerate(tqdm(windows[start_window_idx:], desc="WFO Windows", initial=start_window_idx, total=len(windows)), start=start_window_idx):
            # Per-window checkpoint directory
            window_checkpoint_dir = os.path.join(output_dir or env_cfg['output_dir'], 'checkpoints', f'wf_window_{window_idx}')
            os.makedirs(window_checkpoint_dir, exist_ok=True)

            population = evolution_engine.population
            window_start_gen = 0
            resume_archive = None
            resume_early_stopping_state = None

            if checkpoint_manager is not None and latest_generation_checkpoint is not None:
                latest_window, latest_generation = latest_generation_checkpoint
                if latest_window == window_idx and latest_generation >= 0:
                    from training.runner import GenerationRunner

                    restored = GenerationRunner.load_generation_checkpoint(
                        checkpoint_manager=checkpoint_manager,
                        window_idx=window_idx,
                        generation=latest_generation,
                        config=ga_cfg,
                    )
                    if restored is not None:
                        restored_population, restored_best_chromosomes, restored_best_objectives, evolution_state = restored

                        try:
                            restored_population.move_to(env_cfg['device'])
                        except Exception:
                            restored_population.population_tensor = restored_population.population_tensor.to(env_cfg['device'])
                        restored_population.initialized = True
                        restored_population.set_next_idx()
                        evolution_engine.population = restored_population
                        population = evolution_engine.population

                        resume_best_node_stats = None
                        if isinstance(evolution_state, dict):
                            resume_best_node_stats = evolution_state.get('best_node_stats')
                            resume_early_stopping_state = evolution_state.get('early_stopping_state')
                            if 'generation_stats' in evolution_state and hasattr(evolution_engine, 'generation_stats'):
                                evolution_engine.generation_stats = evolution_state['generation_stats']

                        if restored_best_chromosomes is not None:
                            restored_best_chromosomes = restored_best_chromosomes.to(env_cfg['device'])
                        if isinstance(restored_best_objectives, torch.Tensor):
                            restored_best_objectives = restored_best_objectives.cpu().numpy()

                        if restored_best_chromosomes is not None and restored_best_objectives is not None:
                            resume_archive = (restored_best_chromosomes, restored_best_objectives, resume_best_node_stats)

                        window_start_gen = min(latest_generation + 1, ga_cfg['evolution_loop']['generations'])
                        logging.info(f"Resuming window {window_idx} from generation {window_start_gen}")
                    latest_generation_checkpoint = None

            # Reset population if requested
            if window_idx > 0 and wfo_cfg.get('reset_population_each_window', False):
                if not (window_idx == start_window_idx and window_start_gen > 0):
                    population.make_population(device=env_cfg['device'])

            # In-Sample training
            is_objectives_config = self.evaluation_config.get('is_objectives', self.evaluation_config.get('objectives'))
            window_eval_config = {**self.evaluation_config, 'objectives': is_objectives_config}
            if 'min_trades_per_window' in wfo_cfg:
                window_eval_config['min_trades_per_window'] = wfo_cfg['min_trades_per_window']

            window_best_chromosomes, window_best_objectives, window_best_node_stats = generation_training(
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
                checkpoint_manager=checkpoint_manager,
                window_idx=window_idx,
                best_archive=resume_archive,
                start_gen=window_start_gen,
                early_stopping_config=ga_cfg['evolution_loop'].get('early_stopping'),
                early_stopping_state=resume_early_stopping_state,
            )

            if window_best_chromosomes is None or len(window_best_chromosomes) == 0:
                # Skip OOS if nothing valid
                continue

            # Optional CUDA pruning before OOS
            pruning_cfg = self.evaluation_config.get('pruning', {}) if isinstance(self.evaluation_config, dict) else {}
            pruner = PopulationPruner(pruning_cfg)
            if pruner.is_enabled():
                from models.model import GATreePop
                # Build pruning population from best
                prune_population = GATreePop(
                    pop_size=len(window_best_chromosomes),
                    max_nodes=ga_cfg['population']['max_nodes'],
                    max_depth=ga_cfg['population']['max_depth'],
                    max_children=ga_cfg['population']['max_children'],
                    **model_config
                )
                prune_population.population_tensor = window_best_chromosomes.detach().clone().to(env_cfg['device'])
                prune_population.initialized = True
                prune_population.set_next_idx()

                # Use archived node stats from IS; no re-simulation
                stats = window_best_node_stats if isinstance(window_best_node_stats, dict) else None
                if stats is None:
                    logging.warning("Pruning enabled but no node stats found; skipping pruning for this window.")
                else:
                    pruner.prune_in_place(prune_population, stats)
                    window_best_chromosomes = prune_population.population_tensor.detach().clone().to('cpu')

            # OOS evaluation
            oos_objectives_config = self.evaluation_config.get('oos_objectives', self.evaluation_config.get('objectives'))
            oos_eval_config = {**self.evaluation_config, 'objectives': oos_objectives_config}
            if 'min_trades_per_window' in wfo_cfg:
                oos_eval_config['min_trades_per_window'] = wfo_cfg['min_trades_per_window']

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

            oos_objectives, _ = generation_test(
                data_1m=final_df,
                population=oos_population,
                start_data_cnt=oos_start,
                end_data_cnt=oos_end,
                device=env_cfg['device'],
                evaluation_config=oos_eval_config,
            )

            # Record results and chromosomes
            for candidate_idx, oos_obj in enumerate(oos_objectives):
                candidate_id = f"w{window_idx}_c{candidate_idx}"
                candidate_chrom = window_best_chromosomes[candidate_idx]
                self.record_window_result(window_idx, candidate_id, oos_obj, chromosome=candidate_chrom)

            # Prepare next window population in-place if not resetting
            if not wfo_cfg.get('reset_population_each_window', False):
                elite_size = min(ga_cfg['evolution_loop']['elite_size'], len(window_best_chromosomes))
                population.make_population(device=env_cfg['device'])
                population.population_tensor[:elite_size].copy_(window_best_chromosomes[:elite_size])
                population.set_next_idx()
                population.initialized = True
            
            # Save WFO checkpoint after completing each window
            if checkpoint_manager is not None:
                self.save_wfo_state(checkpoint_manager, window_idx + 1, len(windows))

        # Final aggregation
        if not self.candidate_archives:
            return None, None, windows

        ranked_candidate_ids, aggregated_objectives = self.get_final_rankings(device=env_cfg['device'])

        # Reconstruct best chromosomes
        top_k = min(len(aggregated_objectives), ga_cfg['evolution_loop']['best_chromosome_pool_size'])
        reconstructed: List[torch.Tensor] = []
        for rank_idx in range(top_k):
            cid = ranked_candidate_ids[rank_idx]
            chrom = self.get_candidate_chromosome(cid)
            if chrom is not None:
                reconstructed.append(chrom)
        best_chromosomes = torch.stack(reconstructed, dim=0) if reconstructed else None
        best_metrics = aggregated_objectives[:top_k] if aggregated_objectives.size else None

        return best_chromosomes, best_metrics, windows
