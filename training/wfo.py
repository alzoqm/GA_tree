# training/wfo.py

import numpy as np
import torch
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

from .multi_objective_fitness import MultiObjectiveFitnessEvaluator


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

            valid_values = obj_values[valid_mask]

            if method == 'mean':
                result = np.mean(valid_values)
            elif method == 'median':
                result = np.median(valid_values)
            elif method == 'min':
                result = np.min(valid_values)
            elif method == 'max':
                result = np.max(valid_values)
            elif method == 'std':
                result = np.std(valid_values)
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
        window_objectives = np.array(list(window_results.values()))
        return aggregate_oos_objectives(window_objectives, self.aggregate_config)

    def get_final_rankings(self, device: str | torch.device | None = None) -> Tuple[List[int], np.ndarray]:
        if not self.candidate_archives:
            return [], np.array([])

        candidate_ids = list(self.candidate_archives.keys())
        aggregated_matrix = []
        for candidate_id in candidate_ids:
            agg_obj = self.aggregate_candidate_objectives(candidate_id)
            aggregated_matrix.append(agg_obj)
        aggregated_matrix = np.array(aggregated_matrix)

        agg_evaluator = MultiObjectiveFitnessEvaluator(self.aggregate_config)
        ranks, distances = agg_evaluator.pareto_rank(aggregated_matrix, device=device)
        sort_indices = np.lexsort((-distances, ranks))
        ranked_candidates = [candidate_ids[i] for i in sort_indices]
        return ranked_candidates, aggregated_matrix[sort_indices]

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

        for window_idx, (train_start, train_end, oos_start, oos_end) in enumerate(tqdm(windows, desc="WFO Windows")):
            # Per-window checkpoint directory
            window_checkpoint_dir = os.path.join(output_dir or env_cfg['output_dir'], 'checkpoints', f'wf_window_{window_idx}')
            os.makedirs(window_checkpoint_dir, exist_ok=True)

            # Reset population if requested
            if window_idx > 0 and wfo_cfg.get('reset_population_each_window', False):
                population.make_population(device=env_cfg['device'])

            # In-Sample training
            is_objectives_config = self.evaluation_config.get('is_objectives', self.evaluation_config.get('objectives'))
            window_eval_config = {**self.evaluation_config, 'objectives': is_objectives_config}
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
                # Skip OOS if nothing valid
                continue

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
