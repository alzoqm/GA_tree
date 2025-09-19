# /training/cpcv_manager.py
"""Coordinator for CPCV-based training mode."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from models.model import GATreePop
from training.cpcv import CPCVScheduler
from training.runner import generation_training, GenerationRunner
from fitness.cpcv_objectives import CPCVObjectiveAggregator, CPCVAggregationResult
from analysis.robustness import run_robustness_checks, RobustnessReport
from training.pruner import PopulationPruner


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

    def _build_eval_config(self, scheduler: CPCVScheduler, cpcv_objectives: dict) -> dict:
        eval_cfg = dict(self.evaluation_cfg)
        eval_cfg['mode'] = 'cpcv'
        eval_cfg['cpcv_scheduler'] = scheduler
        eval_cfg['is_objectives'] = self.evaluation_cfg.get('is_objectives', self.evaluation_cfg.get('objectives', {}))
        eval_cfg['oos_objectives'] = self.evaluation_cfg.get('oos_objectives', self.evaluation_cfg.get('objectives', {}))
        eval_cfg['min_trades_per_split'] = int(self.cpcv_cfg.get('min_trades_per_split', 32))
        eval_cfg['robustness_filter'] = self.robustness_cfg
        eval_cfg['cpcv_objectives'] = cpcv_objectives
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

        scheduler = self._build_scheduler(total_length)
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
            data_1m=final_df,
            evolution=evolution_engine,
            skip_data_cnt=0,
            valid_skip_data_cnt=total_length,
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
            data=final_df,
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
                        data=final_df,
                        all_feature_names=model_config.get('all_features', []),
                    )
                except Exception as exc:
                    logging.warning("CPCV pruning failed: %s", exc)
            else:
                logging.warning("Pruning enabled but CPCV node statistics were unavailable; skipping pruning.")

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
