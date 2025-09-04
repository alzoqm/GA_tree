# training/runner.py

import os
import numpy as np
import torch
import pandas as pd
from typing import Optional, Tuple, Dict
from tqdm import tqdm

from models.model import GATreePop
from training.predictor import build_adjacency_list_cuda
from fitness.simulator import FitnessSimulator


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
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """
    Generation training loop using the decoupled FitnessSimulator.
    Returns Pareto-archive best chromosomes and objectives.
    """
    from .multi_objective_fitness import MultiObjectiveFitnessEvaluator

    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    if objectives_config is None:
        objectives_config = evaluation_config.get('multi_objectives', evaluation_config.get('objectives', {}))
    evaluator = MultiObjectiveFitnessEvaluator(objectives_config)

    if not isinstance(data_1m.index, pd.DatetimeIndex):
        data_1m.index = pd.to_datetime(data_1m.index)

    population = evolution.population
    all_feature_names = population.all_features
    simulator = FitnessSimulator()

    # Pareto archive
    if best_archive is None:
        best_chromosomes = None
        best_objectives = None
    else:
        best_chromosomes, best_objectives = best_archive

    for gen_idx in tqdm(range(start_gen, gen_loop), desc="Generations"):
        print(f'Generation {gen_idx}:')

        # Build adj to ensure early errors are caught
        adj_offsets, adj_indices = build_adjacency_list_cuda(population, strict_overflow=False)
        if adj_offsets is None:
            raise RuntimeError("Failed to build adjacency list on GPU.")
        del adj_offsets, adj_indices  # FitnessSimulator will rebuild as needed

        train_objectives = simulator.simulate(
            population=population,
            data=data_1m,
            all_feature_names=all_feature_names,
            start_data_cnt=skip_data_cnt,
            stop_data_cnt=valid_skip_data_cnt,
            device=device,
            evaluation_config=evaluation_config,
        )

        train_objectives_tensor = torch.from_numpy(train_objectives).to(device)

        if gen_idx >= warming_step:
            valid_mask = ~np.any(np.isin(train_objectives, [-1e9, 1e9]), axis=1)
            if np.any(valid_mask):
                valid_objectives = train_objectives[valid_mask]
                valid_chromosomes = population.population_tensor[valid_mask].clone()

                pareto_front_mask = evaluator._find_pareto_front(valid_objectives)
                if np.any(pareto_front_mask):
                    pareto_objectives = valid_objectives[pareto_front_mask]
                    pareto_chromosomes = valid_chromosomes[pareto_front_mask]

                    print(f"  Found {len(pareto_objectives)} Pareto optimal solutions")

                    if best_chromosomes is None:
                        best_objectives = pareto_objectives.copy()
                        best_chromosomes = pareto_chromosomes.clone()
                    else:
                        combined_objectives = np.vstack([best_objectives, pareto_objectives])
                        combined_chromosomes = torch.cat([best_chromosomes, pareto_chromosomes])

                        combined_pareto_mask = evaluator._find_pareto_front(combined_objectives)
                        best_objectives = combined_objectives[combined_pareto_mask]
                        best_chromosomes = combined_chromosomes[combined_pareto_mask]

                        if len(best_objectives) > best_size:
                            distances = evaluator._calculate_crowding_distances(best_objectives)
                            selected_indices = np.argsort(distances)[-best_size:]
                            best_objectives = best_objectives[selected_indices]
                            best_chromosomes = best_chromosomes[selected_indices]

                    print(f"  Archive size: {len(best_objectives)}")

        generation_summary = evaluator.get_objective_summary(train_objectives)
        print(f"  Valid individuals: {generation_summary['statistics'].get('mean_return', {}).get('valid_count', 0)}")
        for obj_name, stats in generation_summary['statistics'].items():
            if stats['valid_count'] and stats['mean'] is not None:
                direction = "↓" if stats['minimize'] else "↑"
                print(
                    f"    {obj_name}: {stats['mean']:.4f}±{stats['std']:.4f} "
                    f"[{stats['min']:.4f}, {stats['max']:.4f}] {direction}"
                )

        # --- wandb generation-level logging ---
        try:
            from tools import wandb_logger
            wb = wandb_config or {}
            if bool(wb.get('gen_log', True)) and ((gen_idx - start_gen) % int(wb.get('gen_log_interval', 1)) == 0):
                # Convert to natural orientation for logging
                obj_names = evaluator.objective_names
                minimize_flags = evaluator.minimize_objectives
                arr = np.array(train_objectives, dtype=float)
                # mask invalids
                valid_mask_rows = ~np.any(np.isin(arr, [-1e9, 1e9]) | ~np.isfinite(arr), axis=1)
                payload: Dict[str, float | int] = {"gen/index": int(gen_idx)}
                payload["gen/valid_count"] = int(np.sum(valid_mask_rows))
                # Pareto front size (current gen)
                pareto_mask = evaluator._find_pareto_front(arr)
                payload["gen/pareto/front_size"] = int(np.sum(pareto_mask))
                # Archive size
                archive_size = len(best_objectives) if best_archive is not None and best_objectives is not None else 0
                payload["gen/archive/size"] = int(archive_size)
                # Diversity via crowding distance on current Pareto front
                try:
                    if np.any(pareto_mask):
                        front_obj = arr[pareto_mask]
                        distances = evaluator._calculate_crowding_distances(front_obj)
                        if distances.size:
                            payload["gen/diversity/crowding_mean"] = float(np.mean(distances[np.isfinite(distances)]))
                            payload["gen/diversity/crowding_p95"] = float(np.percentile(distances[np.isfinite(distances)], 95))
                except Exception:
                    pass
                # Evolution operator stats if provided
                try:
                    evo_stats = evolution.get_generation_statistics() if hasattr(evolution, 'get_generation_statistics') else {}
                    if isinstance(evo_stats, dict):
                        for k, v in evo_stats.items():
                            # flatten simple scalars
                            if isinstance(v, (int, float)):
                                payload[f"gen/evolution/{k}"] = float(v)
                except Exception:
                    pass
                # Metric stats per objective (natural orientation)
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
                    else:
                        payload[f"gen/is/{name}/valid_count"] = 0
                wandb_logger.log(payload, step=int(gen_idx))
        except Exception:
            pass

        checkpoint_data = {
            "generation": gen_idx,
            "population_state_dict": population.population_tensor.cpu(),
            "best_objectives": best_objectives,
            "best_chromosomes": best_chromosomes.cpu() if best_chromosomes is not None else None,
            "generation_stats": generation_summary,
            "objectives_config": objectives_config,
            "evolution_stats": evolution.get_generation_statistics() if hasattr(evolution, 'get_generation_statistics') else {},
        }
        checkpoint_path = os.path.join(checkpoint_dir, f'generation_{gen_idx}.pt')
        torch.save(checkpoint_data, checkpoint_path)

        evolution.evolve(train_objectives_tensor)
        # 디바이스 전환 시 뷰 무효화를 방지하기 위해 안전한 API 사용
        population.move_to(device)
        population.reorganize_nodes()
        print()

    return best_chromosomes, best_objectives


def generation_test(
    data_1m: pd.DataFrame,
    population: GATreePop,
    start_data_cnt: int,
    end_data_cnt: int,
    device: str,
    evaluation_config: dict,
) -> Tuple[np.ndarray, Dict]:
    """Evaluate best population on test split using simulator/evaluator."""
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
        device=device,
        evaluation_config=evaluation_config,
    )

    objectives_config = evaluation_config.get('multi_objectives', evaluation_config.get('objectives', {}))
    evaluator = MultiObjectiveFitnessEvaluator(objectives_config)
    test_summary = evaluator.get_objective_summary(test_objectives)

    return test_objectives, test_summary
