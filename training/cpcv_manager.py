# /training/cpcv_manager.py
"""Coordinator for CPCV-based training mode."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import torch
import pandas as pd

from models.model import GATreePop
from training.cpcv import CPCVScheduler
from training.runner import generation_training, GenerationRunner
from fitness.cpcv_objectives import CPCVObjectiveAggregator, CPCVAggregationResult
from analysis.robustness import run_robustness_checks, RobustnessReport
from training.pruner import PopulationPruner
from training.simulator import FitnessSimulator


class CPCVManager:
    """Run CPCV training and aggregation for GA Tree."""

    def __init__(
        self,
        training_strategy_cfg: dict,
        cpcv_cfg: dict,
        evaluation_cfg: dict,
        fitness_cfg: dict,
        robustness_cfg: dict,
    ) -> None:
        self.training_strategy_cfg = training_strategy_cfg
        self.cpcv_cfg = cpcv_cfg
        self.evaluation_cfg = evaluation_cfg
        self.fitness_cfg = fitness_cfg
        self.robustness_cfg = robustness_cfg

        self.scheduler: Optional[CPCVScheduler] = None
        self.last_summary: Optional[Dict[str, np.ndarray]] = None
        self.last_combo_ids: List[int] = []
        self.last_raw_oos = None
        self.last_raw_returns = None
        self.last_robustness: List[RobustnessReport] = []

        # Holdout evaluation state
        self.holdout_length: int = 0
        self.holdout_summary: Dict[str, np.ndarray] = {}
        self.holdout_objectives: Optional[np.ndarray] = None
        self.holdout_returns = None
        self.holdout_indices: Optional[Tuple[int, int]] = None
        self.holdout_timestamps: Optional[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = None

    def _build_scheduler(self, total_length: int) -> CPCVScheduler:
        folds = int(self.cpcv_cfg.get('folds', 10))
        test_folds = int(self.cpcv_cfg.get('test_folds', 2))
        purge = self.cpcv_cfg.get('purge_bars')
        embargo = self.cpcv_cfg.get('embargo_bars')
        fallback = int(self.training_strategy_cfg.get('max_hold_period_bars', 0))
        if purge is None:
            purge = fallback
        if embargo is None:
            embargo = fallback
        scheduler = CPCVScheduler(
            total_length=total_length,
            folds=folds,
            test_folds=test_folds,
            purge_bars=purge,
            embargo_bars=embargo,
            shuffle_seed=self.cpcv_cfg.get('shuffle_seed'),
            max_combinations=self.cpcv_cfg.get('max_combinations'),
            min_bars_per_split=int(self.cpcv_cfg.get('min_trades_per_split', 32)),
            parallel_chunks=self.cpcv_cfg.get('parallel_chunks'),
        )
        return scheduler

    def _resolve_holdout_length(self, total_length: int) -> int:
        """Determine how many trailing bars to reserve as holdout."""

        if total_length <= 0:
            return 0

        cfg_raw = self.cpcv_cfg.get('holdout', {})
        holdout_cfg = cfg_raw if isinstance(cfg_raw, dict) else {}

        bars = holdout_cfg.get('bars', self.cpcv_cfg.get('holdout_bars'))
        ratio = holdout_cfg.get('ratio', self.cpcv_cfg.get('holdout_ratio'))
        min_bars = holdout_cfg.get('min_bars', self.cpcv_cfg.get('holdout_min_bars', 0))
        max_bars = holdout_cfg.get('max_bars', self.cpcv_cfg.get('holdout_max_bars'))

        length = 0
        holdout_requested = False
        if bars is not None:
            try:
                length = max(0, int(bars))
            except (TypeError, ValueError):
                length = 0
            else:
                holdout_requested = length > 0
        elif ratio is not None:
            try:
                ratio_f = float(ratio)
            except (TypeError, ValueError):
                ratio_f = 0.0
            if ratio_f > 0.0:
                length = int(math.ceil(total_length * min(1.0, max(0.0, ratio_f))))
                holdout_requested = True

        if not holdout_requested:
            length = 0
        elif min_bars:
            try:
                length = max(length, int(min_bars))
            except (TypeError, ValueError):
                pass

        if max_bars is not None:
            try:
                length = min(length, int(max_bars))
            except (TypeError, ValueError):
                pass

        min_training = max(2 * int(self.cpcv_cfg.get('min_trades_per_split', 32)), 64)
        if length >= total_length - min_training:
            # Not enough data left for training; disable holdout.
            logging.warning(
                "Requested holdout of %s bars leaves insufficient data (%s) for CPCV. Skipping holdout.",
                length,
                total_length - length,
            )
            return 0

        return min(length, total_length)

    def _build_eval_config(self, scheduler: CPCVScheduler, cpcv_objectives: dict) -> dict:
        eval_cfg = dict(self.evaluation_cfg)
        eval_cfg['mode'] = 'cpcv'
        eval_cfg['cpcv_scheduler'] = scheduler
        eval_cfg['is_objectives'] = self.evaluation_cfg.get('is_objectives', self.evaluation_cfg.get('objectives', {}))
        eval_cfg['oos_objectives'] = self.evaluation_cfg.get('oos_objectives', self.evaluation_cfg.get('objectives', {}))
        eval_cfg['min_trades_per_split'] = int(self.cpcv_cfg.get('min_trades_per_split', 32))
        eval_cfg['robustness_filter'] = self.robustness_cfg
        eval_cfg['cpcv_objectives'] = cpcv_objectives
        if 'evaluation_cache_dir' not in eval_cfg and self.cpcv_cfg.get('evaluation_cache_dir'):
            eval_cfg['evaluation_cache_dir'] = self.cpcv_cfg.get('evaluation_cache_dir')
        return eval_cfg

    def run(
        self,
        final_df,
        evolution_engine,
        model_config: dict,
        ga_cfg: dict,
        env_cfg: dict,
        wandb_cfg: Optional[dict] = None,
        output_dir: Optional[str] = None,
        checkpoint_manager=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], List[int]]:
        total_length = len(final_df)
        if total_length < 2:
            logging.error("Dataset too small for CPCV training.")
            return None, None, []

        holdout_len = self._resolve_holdout_length(total_length)
        if holdout_len > 0:
            train_df = final_df.iloc[:-holdout_len].copy()
            holdout_df = final_df.iloc[-holdout_len:].copy()
        else:
            train_df = final_df
            holdout_df = final_df.iloc[0:0]

        train_length = len(train_df)
        if train_length < 2:
            logging.error(
                "Training dataset too small after holdout split (train=%s, holdout=%s).",
                train_length,
                holdout_len,
            )
            return None, None, []

        self.holdout_length = holdout_len
        if holdout_len > 0 and not holdout_df.empty:
            self.holdout_indices = (train_length, total_length - 1)
            start_ts = holdout_df.index[0] if len(holdout_df.index) else None
            end_ts = holdout_df.index[-1] if len(holdout_df.index) else None
            self.holdout_timestamps = (start_ts, end_ts)
        else:
            self.holdout_indices = None
            self.holdout_timestamps = None

        scheduler = self._build_scheduler(train_length)
        self.scheduler = scheduler

        cpcv_objectives_cfg = self.fitness_cfg.get('cpcv_objectives')
        if not cpcv_objectives_cfg:
            raise ValueError("Missing 'cpcv_objectives' configuration for CPCV mode")

        # Ensure MultiObjectiveFitnessEvaluator receives the expected keys
        cpcv_objectives = dict(cpcv_objectives_cfg)
        if 'objectives' not in cpcv_objectives and 'names' in cpcv_objectives:
            cpcv_objectives['objectives'] = list(cpcv_objectives['names'])
        if not cpcv_objectives:
            raise ValueError("CPCV objectives configuration is empty")
        eval_cfg = self._build_eval_config(scheduler, cpcv_objectives)

        start_generation = 0
        resume_archive = None
        resume_early_stopping_state = None

        if checkpoint_manager is not None:
            latest_generation = checkpoint_manager.find_latest_cpcv_generation()
            if latest_generation is not None and latest_generation >= 0:
                restored = GenerationRunner.load_generation_checkpoint(
                    checkpoint_manager=checkpoint_manager,
                    window_idx=None,
                    generation=latest_generation,
                    config=ga_cfg,
                    mode='cpcv',
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

                    start_generation = min(latest_generation + 1, ga_cfg['evolution_loop']['generations'])
                    logging.info(
                        "Resuming CPCV training from generation %s", start_generation
                    )

        best_chromosomes, best_objectives, best_node_stats = generation_training(
            data_1m=train_df,
            evolution=evolution_engine,
            skip_data_cnt=0,
            valid_skip_data_cnt=train_length,
            chromosomes_size=ga_cfg['population']['size'],
            gen_loop=ga_cfg['evolution_loop']['generations'],
            best_size=ga_cfg['evolution_loop']['best_chromosome_pool_size'],
            elite_size=ga_cfg['evolution_loop']['elite_size'],
            device=env_cfg['device'],
            warming_step=ga_cfg['evolution_loop']['warming_steps'],
            evaluation_config=eval_cfg,
            output_dir=output_dir or env_cfg['output_dir'],
            wandb_config=wandb_cfg,
            checkpoint_manager=checkpoint_manager,
            objectives_config=cpcv_objectives,
            early_stopping_config=ga_cfg['evolution_loop'].get('early_stopping'),
            best_archive=resume_archive,
            start_gen=start_generation,
            early_stopping_state=resume_early_stopping_state,
        )

        if best_chromosomes is None or len(best_chromosomes) == 0:
            logging.warning("CPCV training produced no valid best chromosomes.")
            return None, None, []

        best_population = GATreePop(
            pop_size=len(best_chromosomes),
            max_nodes=ga_cfg['population']['max_nodes'],
            max_depth=ga_cfg['population']['max_depth'],
            max_children=ga_cfg['population']['max_children'],
            **model_config
        )
        cpu_best = best_chromosomes.detach().to('cpu') if isinstance(best_chromosomes, torch.Tensor) else best_chromosomes
        best_population.population_tensor[:len(cpu_best)].copy_(cpu_best)
        best_population.initialized = True
        best_population.set_next_idx()
        best_population.move_to(env_cfg['device'])

        aggregation_eval_cfg = dict(eval_cfg)
        pruning_cfg = aggregation_eval_cfg.get('pruning', {}) if isinstance(aggregation_eval_cfg, dict) else {}
        if isinstance(pruning_cfg, dict) and pruning_cfg.get('enabled', False):
            aggregation_eval_cfg['collect_node_stats'] = True
        else:
            aggregation_eval_cfg['collect_node_stats'] = False
        aggregation_eval_cfg['current_generation'] = 'aggregation'

        aggregator = CPCVObjectiveAggregator(
            scheduler=scheduler,
            evaluation_config=aggregation_eval_cfg,
            cpcv_objectives=cpcv_objectives,
            is_objectives=eval_cfg.get('is_objectives', {}),
            oos_objectives=eval_cfg.get('oos_objectives', {}),
            device=env_cfg['device'],
            min_trades_per_split=int(self.cpcv_cfg.get('min_trades_per_split', 32)),
            parallel_chunks=self.cpcv_cfg.get('parallel_chunks'),
            reuse_population=bool(self.cpcv_cfg.get('reuse_population', True)),
            cache_dir=self.cpcv_cfg.get('evaluation_cache_dir'),
        )
        result = aggregator.evaluate(
            population=best_population,
            data=train_df,
            all_feature_names=model_config.get('all_features', []),
        )

        best_chromosomes = best_population.population_tensor.detach().clone()
        pruner = PopulationPruner(pruning_cfg)
        if pruner.is_enabled():
            stats_for_prune = result.node_stats
            if isinstance(stats_for_prune, dict) and stats_for_prune:
                try:
                    pruner.prune_in_place(best_population, stats_for_prune)
                    best_population.set_next_idx()
                    best_population.initialized = True
                    best_chromosomes = best_population.population_tensor.detach().clone()
                    result = aggregator.evaluate(
                        population=best_population,
                        data=train_df,
                        all_feature_names=model_config.get('all_features', []),
                    )
                except Exception as exc:
                    logging.warning("CPCV pruning failed: %s", exc)
            else:
                logging.warning("Pruning enabled but CPCV node statistics were unavailable; skipping pruning.")

        if self.holdout_length > 0 and len(holdout_df) > 1:
            holdout_eval_cfg = dict(aggregation_eval_cfg)
            holdout_eval_cfg.pop('cpcv_scheduler', None)
            holdout_eval_cfg.pop('cpcv_objectives', None)
            holdout_eval_cfg['mode'] = 'holdout'
            holdout_eval_cfg['objectives'] = aggregator.oos_objectives
            holdout_eval_cfg['multi_objectives'] = aggregator.oos_objectives

            holdout_min_trades = int(self.cpcv_cfg.get('holdout_min_trades', self.cpcv_cfg.get('min_trades_per_split', 32)))

            try:
                holdout_evaluation = aggregator._evaluate_segment_group(
                    simulator=FitnessSimulator(),
                    population=best_population,
                    segments=[holdout_df],
                    eval_cfg=holdout_eval_cfg,
                    objective_cfg=aggregator.oos_objectives,
                    all_feature_names=model_config.get('all_features', []),
                    min_trades=holdout_min_trades,
                    collect_returns=True,
                    collect_node_stats=aggregator.want_node_stats,
                )
            except Exception as exc:
                logging.warning("Holdout evaluation failed: %s", exc)
                holdout_evaluation = None

            if holdout_evaluation is not None:
                natural_metrics, _, holdout_returns, _, _ = holdout_evaluation
                oos_stack = natural_metrics[np.newaxis, ...]
                train_stack = np.zeros((0, best_population.pop_size, 0), dtype=np.float32)
                returns_stack = [holdout_returns] if holdout_returns.size else []

                holdout_summary = aggregator._aggregate_metrics(
                    oos_stack=oos_stack,
                    train_stack=train_stack,
                    returns_stack=returns_stack,
                    population=best_population,
                )
                holdout_objectives = aggregator._build_objective_matrix(holdout_summary, best_population.pop_size)

                self.holdout_summary = holdout_summary
                self.holdout_objectives = holdout_objectives
                self.holdout_returns = holdout_returns if holdout_returns.size else None
            else:
                logging.warning("Holdout segment did not produce valid metrics (length=%s).", len(holdout_df))
                self.holdout_summary = {}
                self.holdout_objectives = None
                self.holdout_returns = None
        else:
            self.holdout_summary = {}
            self.holdout_objectives = None
            self.holdout_returns = None

        if self.holdout_objectives is not None and self.holdout_objectives.size:
            start_idx, end_idx = self.holdout_indices if self.holdout_indices else (None, None)
            logging.info(
                "Holdout evaluation completed on %s bars (indices %s-%s).",
                self.holdout_length,
                start_idx,
                end_idx,
            )

        self.last_summary = result.summary
        self.last_combo_ids = result.combo_ids
        self.last_raw_oos = result.raw_oos_metrics
        self.last_raw_returns = result.raw_returns
        try:
            oos_obj_cfg = eval_cfg.get('oos_objectives', {})
            oos_names = oos_obj_cfg.get('objectives', [])
            oos_minimize = oos_obj_cfg.get('minimize', [])
            self.last_robustness = run_robustness_checks(
                raw_returns=self.last_raw_returns,
                raw_oos_metrics=self.last_raw_oos,
                oos_metric_names=oos_names,
                config=self.robustness_cfg,
                minimize_flags=oos_minimize,
                metrics_are_minimized=True,
            )
        except Exception as exc:
            logging.warning("Robustness checks failed: %s", exc)
            self.last_robustness = []

        return best_chromosomes, result.objectives, result.combo_ids
