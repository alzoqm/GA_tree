# /training/runner.py

import os
import numpy as np
import torch
import pandas as pd
import logging
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm

from models.model import GATreePop
from training.predictor import build_adjacency_list_cuda
from training.simulator import FitnessSimulator
from utils.checkpoint_manager import CheckpointManager
from training.early_stopping import build_early_stopping_monitor


class GenerationRunner:
    """Class-based generation trainer/tester to manage evaluation and checkpoints."""

    def __init__(
        self,
        device: str,
        evaluation_config: dict,
        output_dir: str,
        checkpoint_manager: Optional[CheckpointManager] = None,
        wandb_config: Optional[Dict] = None,
        early_stopping_config: Optional[dict] = None,
    ) -> None:
        self.device = device
        self.evaluation_config = evaluation_config
        self.output_dir = output_dir
        self.checkpoint_manager = checkpoint_manager
        self.wandb_config = wandb_config or {}
        self.early_stopping = build_early_stopping_monitor(early_stopping_config)
        self._early_stopping_warning_emitted = False

    # ----- internal helpers (formerly module-level) -----
    def compute_generation_objectives(
        self,
        population: GATreePop,
        data_1m: pd.DataFrame,
        start_cnt: int,
        stop_cnt: int,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """Compute per-individual objectives for a generation.

        If pruning is enabled in the evaluation config, also return node-level
        attribution stats by running the simulator with return_node_stats=True.
        """
        from training.simulator import FitnessSimulator

        # Determine whether to collect node stats (for pruning)
        pruning_cfg = self.evaluation_config.get('pruning', {}) if isinstance(self.evaluation_config, dict) else {}
        want_node_stats = bool(pruning_cfg.get('enabled', False))

        simulator = FitnessSimulator()
        eval_cfg = dict(self.evaluation_config)
        if want_node_stats:
            eval_cfg = {**eval_cfg, 'return_node_stats': True}

        sim_ret = simulator.simulate(
            population=population,
            data=data_1m,
            all_feature_names=population.all_features,
            start_data_cnt=start_cnt,
            stop_data_cnt=stop_cnt,
            device=self.device,
            evaluation_config=eval_cfg,
        )

        if want_node_stats:
            # simulate returns (objectives, stats) when return_node_stats=True
            return sim_ret  # type: ignore[return-value]
        else:
            return sim_ret  # objectives only

    def update_pareto_archive(
        self,
        evaluator,
        population: GATreePop,
        train_objectives: np.ndarray,
        best_chromosomes: Optional[torch.Tensor],
        best_objectives: Optional[np.ndarray],
        best_size: int,
        current_node_stats: Optional[dict] = None,
        best_node_stats: Optional[dict] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[dict]]:
        return _update_pareto_archive(
            evaluator,
            population,
            train_objectives,
            best_chromosomes,
            best_objectives,
            best_size,
            current_node_stats=current_node_stats,
            best_node_stats=best_node_stats,
        )

    def log_generation(
        self,
        evaluator,
        evolution: Any,
        train_objectives: np.ndarray,
        best_objectives: Optional[np.ndarray],
        gen_idx: int,
        start_gen: int,
    ) -> None:
        return _log_generation(
            evaluator=evaluator,
            evolution=evolution,
            train_objectives=train_objectives,
            best_objectives=best_objectives,
            gen_idx=gen_idx,
            start_gen=start_gen,
            wandb_config=self.wandb_config,
        )

    def save_generation_checkpoint(
        self,
        gen_idx: int,
        population: GATreePop,
        best_chromosomes: Optional[torch.Tensor],
        best_objectives: Optional[np.ndarray],
        best_node_stats: Optional[Dict] ,
        generation_summary: Dict,
        objectives_config: dict,
        evolution: Any,
        window_idx: Optional[int] = None,
        early_stopping_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        return _save_generation_checkpoint(
            checkpoint_dir=checkpoint_dir,
            gen_idx=gen_idx,
            population=population,
            best_chromosomes=best_chromosomes,
            best_objectives=best_objectives,
            best_node_stats=best_node_stats,
            generation_summary=generation_summary,
            objectives_config=objectives_config,
            evolution=evolution,
            checkpoint_manager=self.checkpoint_manager,
            window_idx=window_idx,
            early_stopping_state=early_stopping_state,
        )

    def _update_early_stopping(self, generation: int, generation_summary: Dict[str, Any]) -> bool:
        """Update early stopping monitor with current generation statistics."""
        if self.early_stopping is None:
            return False

        statistics = generation_summary.get('statistics', {}) if isinstance(generation_summary, dict) else {}
        metric_name = self.early_stopping.metric
        metric_stats = statistics.get(metric_name)

        if metric_stats is None:
            if not self._early_stopping_warning_emitted:
                logging.warning(
                    "Early stopping metric '%s' not present in generation summary; monitoring suspended.",
                    metric_name,
                )
                self._early_stopping_warning_emitted = True
            return False

        minimize_flag = self.early_stopping.config.minimize
        if minimize_flag is None:
            minimize_flag = metric_stats.get('minimize')
        if minimize_flag is None and self.early_stopping.minimize is not None:
            minimize_flag = self.early_stopping.minimize

        value = None
        if metric_stats.get('valid_count', 0) > 0:
            value = metric_stats.get(self.early_stopping.stat)

        if minimize_flag is None and self.early_stopping.minimize is None:
            if not self._early_stopping_warning_emitted:
                logging.warning(
                    "Unable to infer optimization direction for early stopping metric '%s'.",
                    metric_name,
                )
                self._early_stopping_warning_emitted = True
            return False

        try:
            return self.early_stopping.update(
                value=value,
                generation=generation,
                minimize=minimize_flag,
            )
        except ValueError as exc:
            if not self._early_stopping_warning_emitted:
                logging.warning("Skipping early stopping update: %s", exc)
                self._early_stopping_warning_emitted = True
            return False

    @staticmethod
    def load_generation_checkpoint(
        checkpoint_manager: CheckpointManager,
        window_idx: int,
        generation: int,
        config: Dict[str, Any],
    ) -> Optional[Tuple[GATreePop, Optional[torch.Tensor], Optional[np.ndarray], Dict]]:
        return load_generation_checkpoint(checkpoint_manager, window_idx, generation, config)

    # ----- public API -----
    def train(
        self,
        data_1m: pd.DataFrame,
        evolution,
        skip_data_cnt: int,
        valid_skip_data_cnt: int,
        chromosomes_size: int,
        gen_loop: int,
        best_size: int,
        elite_size: int,
        warming_step: int,
        objectives_config: Optional[dict] = None,
        best_archive=None,
        start_gen: int = 0,
        window_idx: Optional[int] = None,
        early_stopping_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[dict]]:
        from .multi_objective_fitness import MultiObjectiveFitnessEvaluator

        if objectives_config is None:
            objectives_config = self.evaluation_config.get('multi_objectives', self.evaluation_config.get('objectives', {}))
        evaluator = MultiObjectiveFitnessEvaluator(objectives_config)

        if not isinstance(data_1m.index, pd.DatetimeIndex):
            data_1m.index = pd.to_datetime(data_1m.index)

        population = evolution.population

        # Pareto archive
        best_chromosomes: Optional[torch.Tensor] = None
        best_objectives: Optional[np.ndarray] = None
        best_node_stats: Optional[dict] = None
        if best_archive is not None:
            if isinstance(best_archive, tuple):
                if len(best_archive) >= 1:
                    best_chromosomes = best_archive[0]
                if len(best_archive) >= 2:
                    best_objectives = best_archive[1]
                if len(best_archive) >= 3:
                    best_node_stats = best_archive[2]
            else:
                best_chromosomes, best_objectives = best_archive

        if isinstance(best_chromosomes, torch.Tensor) and str(best_chromosomes.device) != str(self.device):
            best_chromosomes = best_chromosomes.to(self.device)
        if isinstance(best_objectives, torch.Tensor):
            best_objectives = best_objectives.detach().cpu().numpy()

        if self.early_stopping and early_stopping_state:
            try:
                self.early_stopping.load_state_dict(early_stopping_state)
            except Exception as exc:
                logging.warning(
                    "Failed to restore early stopping state: %s. Restarting patience counters.",
                    exc,
                )

        for gen_idx in tqdm(range(start_gen, gen_loop), desc="Generations"):
            print(f'Generation {gen_idx}:')

            # Build adjacency once to catch early structural errors
            adj_offsets, adj_indices = build_adjacency_list_cuda(population, strict_overflow=False)
            if adj_offsets is None:
                raise RuntimeError("Failed to build adjacency list on GPU.")
            del adj_offsets, adj_indices

            # Compute objectives; when pruning is enabled, also gather node stats
            comp_ret = self.compute_generation_objectives(
                population=population,
                data_1m=data_1m,
                start_cnt=skip_data_cnt,
                stop_cnt=valid_skip_data_cnt,
            )
            if isinstance(comp_ret, tuple) and len(comp_ret) == 2:
                train_objectives, current_node_stats = comp_ret
            else:
                train_objectives = comp_ret  # type: ignore[assignment]
                current_node_stats = None

            train_objectives_tensor = torch.from_numpy(train_objectives).to(self.device)

            if gen_idx >= warming_step:
                prev_size = 0 if best_objectives is None else len(best_objectives)
                best_chromosomes, best_objectives, best_node_stats = self.update_pareto_archive(
                    evaluator,
                    population,
                    train_objectives,
                    best_chromosomes,
                    best_objectives,
                    best_size,
                    current_node_stats=current_node_stats,
                    best_node_stats=best_node_stats,
                )
                new_size = 0 if best_objectives is None else len(best_objectives)
                if new_size != prev_size:
                    print(f"  Archive size: {new_size}")

            self.log_generation(
                evaluator=evaluator,
                evolution=evolution,
                train_objectives=train_objectives,
                best_objectives=best_objectives,
                gen_idx=gen_idx,
                start_gen=start_gen,
            )

            generation_summary = evaluator.get_objective_summary(train_objectives)

            stop_triggered = False
            if self.early_stopping is not None:
                stop_triggered = self._update_early_stopping(gen_idx, generation_summary)

            self.save_generation_checkpoint(
                gen_idx=gen_idx,
                population=population,
                best_chromosomes=best_chromosomes,
                best_objectives=best_objectives,
                best_node_stats=best_node_stats,
                generation_summary=generation_summary,
                objectives_config=objectives_config,
                evolution=evolution,
                window_idx=window_idx,
                early_stopping_state=self.early_stopping.state_dict() if self.early_stopping is not None else None,
            )

            if stop_triggered:
                logging.info(
                    "Early stopping triggered at generation %s. %s",
                    gen_idx,
                    self.early_stopping.describe() if self.early_stopping is not None else "",
                )
                break

            evolution.evolve(train_objectives_tensor)
            # 디바이스 전환 시 뷰 무효화를 방지하기 위해 안전한 API 사용
            population.move_to(self.device)
            population.reorganize_nodes()

        return best_chromosomes, best_objectives, best_node_stats

    def test(
        self,
        data_1m: pd.DataFrame,
        population: GATreePop,
        start_data_cnt: int,
        end_data_cnt: int,
    ) -> Tuple[np.ndarray, Dict]:
        from .multi_objective_fitness import MultiObjectiveFitnessEvaluator

        if not isinstance(data_1m.index, pd.DatetimeIndex):
            data_1m.index = pd.to_datetime(data_1m.index)

        all_feature_names = population.all_features
        simulator = FitnessSimulator()

        test_objectives = simulator.simulate(
            population=population,
            data=data_1m,
            all_feature_names=all_feature_names,
            start_data_cnt=start_data_cnt,
            stop_data_cnt=end_data_cnt,
            device=self.device,
            evaluation_config=self.evaluation_config,
        )

        objectives_config = self.evaluation_config.get('multi_objectives', self.evaluation_config.get('objectives', {}))
        evaluator = MultiObjectiveFitnessEvaluator(objectives_config)
        test_summary = evaluator.get_objective_summary(test_objectives)

        return test_objectives, test_summary


# --- Internal helpers to keep responsibilities separated ---
def _compute_generation_objectives(
    population: GATreePop,
    data_1m: pd.DataFrame,
    start_cnt: int,
    stop_cnt: int,
    device: str,
    evaluation_config: dict,
) -> np.ndarray:
    """Compute per-individual objectives for a generation."""
    simulator = FitnessSimulator()
    return simulator.simulate(
        population=population,
        data=data_1m,
        all_feature_names=population.all_features,
        start_data_cnt=start_cnt,
        stop_data_cnt=stop_cnt,
        device=device,
        evaluation_config=evaluation_config,
    )


def _slice_node_stats(stats: dict, mask_or_idx) -> dict:
    """Utility to slice node stats dict along batch dimension consistently.

    Assumes all tensor values (except possibly 'total_steps') have shape [B, N, ...].
    """
    if stats is None:
        return None  # type: ignore[return-value]
    out = {}
    for k, v in stats.items():
        if k == 'total_steps':
            out[k] = v
            continue
        try:
            out[k] = v[mask_or_idx]
        except Exception:
            out[k] = v
    return out


def _concat_node_stats(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
    if a is None:
        return b
    if b is None:
        return a
    out = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        if k == 'total_steps':
            # Prefer non-zero, else keep a's
            out[k] = a.get(k, None) if a.get(k, None) is not None else b.get(k, None)
            continue
        va = a.get(k, None)
        vb = b.get(k, None)
        if va is None:
            out[k] = vb
        elif vb is None:
            out[k] = va
        else:
            try:
                out[k] = torch.cat([va, vb], dim=0)
            except Exception:
                out[k] = va
    return out


def _select_k_node_stats(stats: Optional[dict], indices: np.ndarray) -> Optional[dict]:
    if stats is None:
        return None
    return _slice_node_stats(stats, indices)


def _update_pareto_archive(
    evaluator,
    population: GATreePop,
    train_objectives: np.ndarray,
    best_chromosomes: Optional[torch.Tensor],
    best_objectives: Optional[np.ndarray],
    best_size: int,
    current_node_stats: Optional[dict] = None,
    best_node_stats: Optional[dict] = None,
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[dict]]:
    """Update Pareto archive given current gen objectives and optionally node stats.

    Mirrors chromosome/objective selection on node stats so pruning can consume
    the best-archive-aligned stats without re-simulation.
    """
    valid_mask = ~np.any(np.isin(train_objectives, [-1e9, 1e9]), axis=1)
    if not np.any(valid_mask):
        return best_chromosomes, best_objectives, best_node_stats

    valid_objectives = train_objectives[valid_mask]
    valid_chromosomes = population.population_tensor[valid_mask].clone()
    current_node_stats = _slice_node_stats(current_node_stats, valid_mask) if current_node_stats is not None else None

    pareto_front_mask = evaluator._find_pareto_front(valid_objectives)
    if not np.any(pareto_front_mask):
        return best_chromosomes, best_objectives, best_node_stats

    pareto_objectives = valid_objectives[pareto_front_mask]
    pareto_chromosomes = valid_chromosomes[pareto_front_mask]
    pareto_node_stats = _slice_node_stats(current_node_stats, pareto_front_mask) if current_node_stats is not None else None

    if best_chromosomes is None:
        best_objectives = pareto_objectives.copy()
        best_chromosomes = pareto_chromosomes.clone()
        best_node_stats = pareto_node_stats
    else:
        combined_objectives = np.vstack([best_objectives, pareto_objectives])
        combined_chromosomes = torch.cat([best_chromosomes, pareto_chromosomes])
        combined_node_stats = _concat_node_stats(best_node_stats, pareto_node_stats)

        combined_pareto_mask = evaluator._find_pareto_front(combined_objectives)
        best_objectives = combined_objectives[combined_pareto_mask]
        best_chromosomes = combined_chromosomes[combined_pareto_mask]
        best_node_stats = _slice_node_stats(combined_node_stats, combined_pareto_mask) if combined_node_stats is not None else None

        if len(best_objectives) > best_size:
            distances = evaluator._calculate_crowding_distances(best_objectives)
            selected_indices = np.argsort(distances)[-best_size:]
            best_objectives = best_objectives[selected_indices]
            best_chromosomes = best_chromosomes[selected_indices]
            best_node_stats = _select_k_node_stats(best_node_stats, selected_indices)

    return best_chromosomes, best_objectives, best_node_stats


def _log_generation(
    evaluator,
    evolution: Any,
    train_objectives: np.ndarray,
    best_objectives: Optional[np.ndarray],
    gen_idx: int,
    start_gen: int,
    wandb_config: Optional[Dict] = None,
) -> None:
    """Log generation statistics (console + optional wandb)."""
    generation_summary = evaluator.get_objective_summary(train_objectives)
    print(f"  Valid individuals: {generation_summary['statistics'].get('mean_return', {}).get('valid_count', 0)}")
    for obj_name, stats in generation_summary['statistics'].items():
        if stats['valid_count'] and stats['mean'] is not None:
            direction = "↓" if stats['minimize'] else "↑"
            print(
                f"    {obj_name}: {stats['mean']:.4f}±{stats['std']:.4f} "
                f"[{stats['min']:.4f}, {stats['max']:.4f}] {direction}"
            )

    # wandb logging (best-effort)
    try:
        from analysis import wandb_logger
        wb = wandb_config or {}
        if bool(wb.get('gen_log', True)) and ((gen_idx - start_gen) % int(wb.get('gen_log_interval', 1)) == 0):
            obj_names = evaluator.objective_names
            minimize_flags = evaluator.minimize_objectives
            arr = np.array(train_objectives, dtype=float)
            valid_mask_rows = ~np.any(np.isin(arr, [-1e9, 1e9]) | ~np.isfinite(arr), axis=1)
            payload: Dict[str, float | int] = {"gen/index": int(gen_idx)}
            payload["gen/valid_count"] = int(np.sum(valid_mask_rows))
            pareto_mask = evaluator._find_pareto_front(arr)
            payload["gen/pareto/front_size"] = int(np.sum(pareto_mask))
            archive_size = len(best_objectives) if best_objectives is not None else 0
            payload["gen/archive/size"] = int(archive_size)
            try:
                if np.any(pareto_mask):
                    front_obj = arr[pareto_mask]
                    distances = evaluator._calculate_crowding_distances(front_obj)
                    if distances.size:
                        payload["gen/diversity/crowding_mean"] = float(np.mean(distances[np.isfinite(distances)]))
                        payload["gen/diversity/crowding_p95"] = float(np.percentile(distances[np.isfinite(distances)], 95))
            except Exception:
                pass
            try:
                evo_stats = evolution.get_generation_statistics() if hasattr(evolution, 'get_generation_statistics') else {}
                if isinstance(evo_stats, dict):
                    for k, v in evo_stats.items():
                        if isinstance(v, (int, float)):
                            payload[f"gen/evolution/{k}"] = float(v)
            except Exception:
                pass
            for j, name in enumerate(obj_names):
                col = arr[:, j].copy()
                if not minimize_flags[j]:
                    col = -col
                m = (~np.isin(col, [-1e9, 1e9])) & np.isfinite(col)
                if np.any(m):
                    payload[f"gen/is/{name}/mean"] = float(np.mean(col[m]))
                    payload[f"gen/is/{name}/std"] = float(np.std(col[m]))
                    payload[f"gen/is/{name}/min"] = float(np.min(col[m]))
                    payload[f"gen/is/{name}/max"] = float(np.max(col[m]))
                    payload[f"gen/is/{name}/p05"] = float(np.percentile(col[m], 5))
                    payload[f"gen/is/{name}/p95"] = float(np.percentile(col[m], 95))
                    payload[f"gen/is/{name}/valid_count"] = int(np.sum(m))
                    # Also log raw (non orientation-flipped) statistics so metrics like
                    # root_balance_penalty can be monitored directly in wandb.
                    raw_col = arr[:, j]
                    raw_mask = (~np.isin(raw_col, [-1e9, 1e9])) & np.isfinite(raw_col)
                    if np.any(raw_mask):
                        payload[f"gen/is_raw/{name}/mean"] = float(np.mean(raw_col[raw_mask]))
                        payload[f"gen/is_raw/{name}/min"] = float(np.min(raw_col[raw_mask]))
                        payload[f"gen/is_raw/{name}/max"] = float(np.max(raw_col[raw_mask]))
                else:
                    payload[f"gen/is/{name}/valid_count"] = 0
            wandb_logger.log(payload, step=int(gen_idx))
    except Exception:
        pass


def _save_generation_checkpoint(
    checkpoint_dir: str,
    gen_idx: int,
    population: GATreePop,
    best_chromosomes: Optional[torch.Tensor],
    best_objectives: Optional[np.ndarray],
    best_node_stats: Optional[Dict],
    generation_summary: Dict,
    objectives_config: dict,
    evolution: Any,
    checkpoint_manager: Optional[CheckpointManager] = None,
    window_idx: Optional[int] = None,
    early_stopping_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Save generation checkpoint with enhanced resumption support."""
    # Legacy checkpoint format for compatibility
    checkpoint_data = {
        "generation": gen_idx,
        "population_state_dict": population.population_tensor.cpu(),
        "best_objectives": best_objectives,
        "best_chromosomes": best_chromosomes.cpu() if best_chromosomes is not None else None,
        "best_node_stats": {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in best_node_stats.items()} if isinstance(best_node_stats, dict) else None,
        "generation_stats": generation_summary,
        "objectives_config": objectives_config,
        "evolution_stats": evolution.get_generation_statistics() if hasattr(evolution, 'get_generation_statistics') else {},
        "early_stopping_state": early_stopping_state,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'generation_{gen_idx}.pt')
    torch.save(checkpoint_data, checkpoint_path)
    
    # Enhanced checkpoint using CheckpointManager
    if checkpoint_manager is not None and window_idx is not None:
        try:
            # Prepare population state for serialization
            population_state = {
                "population_tensor": population.population_tensor.cpu(),
                "pop_size": population.pop_size,
                "max_nodes": population.max_nodes,
                "max_depth": population.max_depth,
                "max_children": population.max_children,
                "initialized": population.initialized,
                "feature_num": population.feature_num,
                "feature_comparison_map": population.feature_comparison_map,
                "feature_bool": population.feature_bool,
                "all_features": population.all_features,
            }
            
            # Prepare evolution state
            evolution_state = {
                "generation_stats": evolution.get_generation_statistics() if hasattr(evolution, 'get_generation_statistics') else {},
                "parent_size": getattr(evolution, 'parent_size', None),
                "num_elites": getattr(evolution, 'num_elites', None),
                "early_stopping_state": early_stopping_state,
            }

            checkpoint_manager.save_window_checkpoint(
                window_idx=window_idx,
                generation=gen_idx,
                population_state=population_state,
                best_chromosomes=best_chromosomes.cpu() if best_chromosomes is not None else None,
                best_objectives=best_objectives,
                best_node_stats={k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in best_node_stats.items()} if isinstance(best_node_stats, dict) else None,
                evolution_state=evolution_state,
                generation_stats=generation_summary,
                objectives_config=objectives_config,
                early_stopping_state=early_stopping_state,
            )
        except Exception as e:
            logging.warning(f"Failed to save enhanced checkpoint: {e}")


def load_generation_checkpoint(
    checkpoint_manager: CheckpointManager,
    window_idx: int,
    generation: int,
    config: Dict[str, Any]
) -> Optional[Tuple[GATreePop, Optional[torch.Tensor], Optional[np.ndarray], Dict]]:
    """Load generation checkpoint for resumption."""
    try:
        checkpoint_data = checkpoint_manager.load_window_checkpoint(window_idx, generation)
        if checkpoint_data is None:
            return None
        
        # Reconstruct population
        population_state = checkpoint_data.get("population_state", {})
        if not population_state:
            logging.error(f"No population state found in checkpoint window {window_idx}, gen {generation}")
            return None
        
        # Create new population instance
        population = GATreePop(
            pop_size=population_state["pop_size"],
            max_nodes=population_state["max_nodes"],
            max_depth=population_state["max_depth"],
            max_children=population_state["max_children"],
            feature_num=population_state["feature_num"],
            feature_comparison_map=population_state["feature_comparison_map"],
            feature_bool=population_state["feature_bool"],
            all_features=population_state["all_features"],
        )
        
        # Load population tensor
        population.population_tensor = population_state["population_tensor"]
        population.initialized = population_state.get("initialized", True)
        population.set_next_idx()
        
        # Load best chromosomes and objectives
        best_chromosomes = checkpoint_data.get("best_chromosomes")
        best_objectives = checkpoint_data.get("best_objectives")
        
        # Load evolution state and attach best_node_stats if present
        evolution_state = checkpoint_data.get("evolution_state", {})
        if 'best_node_stats' in checkpoint_data:
            try:
                evolution_state['best_node_stats'] = checkpoint_data['best_node_stats']
            except Exception:
                pass
        if 'early_stopping_state' in checkpoint_data:
            try:
                evolution_state['early_stopping_state'] = checkpoint_data['early_stopping_state']
            except Exception:
                pass

        logging.info(f"Successfully loaded checkpoint: window {window_idx}, generation {generation}")

        return population, best_chromosomes, best_objectives, evolution_state
        
    except Exception as e:
        logging.error(f"Failed to load generation checkpoint: {e}")
        return None


def generation_training(
    data_1m: pd.DataFrame,
    evolution,
    skip_data_cnt: int,
    valid_skip_data_cnt: int,
    chromosomes_size: int,
    gen_loop: int,
    best_size: int,
    elite_size: int,
    device: str,
    warming_step: int,
    evaluation_config: dict,
    output_dir: str,
    objectives_config: Optional[dict] = None,
    best_archive=None,
    start_gen: int = 0,
    wandb_config: Optional[Dict] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    window_idx: Optional[int] = None,
    early_stopping_config: Optional[dict] = None,
    early_stopping_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """
    Generation training loop using the decoupled FitnessSimulator.
    Returns Pareto-archive best chromosomes, objectives, and node stats (if pruning enabled).
    """
    # wrapper to keep compatibility while migrating to class-based API
    runner = GenerationRunner(
        device=device,
        evaluation_config=evaluation_config,
        output_dir=output_dir,
        checkpoint_manager=checkpoint_manager,
        wandb_config=wandb_config,
        early_stopping_config=early_stopping_config,
    )
    return runner.train(
        data_1m=data_1m,
        evolution=evolution,
        skip_data_cnt=skip_data_cnt,
        valid_skip_data_cnt=valid_skip_data_cnt,
        chromosomes_size=chromosomes_size,
        gen_loop=gen_loop,
        best_size=best_size,
        elite_size=elite_size,
        warming_step=warming_step,
        objectives_config=objectives_config,
        best_archive=best_archive,
        start_gen=start_gen,
        window_idx=window_idx,
        early_stopping_state=early_stopping_state,
    )


def generation_test(
    data_1m: pd.DataFrame,
    population: GATreePop,
    start_data_cnt: int,
    end_data_cnt: int,
    device: str,
    evaluation_config: dict,
) -> Tuple[np.ndarray, Dict]:
    """Evaluate best population on test split using simulator/evaluator."""
    # wrapper for class-based API
    runner = GenerationRunner(
        device=device,
        evaluation_config=evaluation_config,
        output_dir="",
    )
    return runner.test(
        data_1m=data_1m,
        population=population,
        start_data_cnt=start_data_cnt,
        end_data_cnt=end_data_cnt,
    )
