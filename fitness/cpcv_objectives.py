# /fitness/cpcv_objectives.py
"""Objective aggregation helpers for CPCV mode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import erf
import pandas as pd
import torch
import math

from models.constants import COL_DEPTH, COL_NODE_TYPE, NODE_TYPE_UNUSED
from models.model import GATreePop
from training.cpcv import CPCVScheduler, CPCVSplit
from training.simulator import FitnessSimulator, _compute_root_branch_balance_penalty
from tqdm import tqdm


@dataclass
class CPCVAggregationResult:
    objectives: np.ndarray
    summary: Dict[str, np.ndarray]
    combo_ids: List[int]
    raw_oos_metrics: Optional[np.ndarray] = None  # Minimization-orientated [combos, pop, metrics]
    raw_train_metrics: Optional[np.ndarray] = None
    raw_returns: Optional[List[np.ndarray]] = None
    node_stats: Optional[Dict[str, Union[int, torch.Tensor]]] = None


_INVALID_SENTINEL = 1e9


def _nanmean_axis(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standalone nanmean implementation that avoids RuntimeWarnings."""

    sums = np.nansum(arr, axis=axis)
    counts = np.sum(~np.isnan(arr), axis=axis)
    mean = np.full_like(sums, np.nan, dtype=np.float64)
    valid = counts > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        mean[valid] = sums[valid] / counts[valid]
    return mean


def _reduce_with_nan(values: np.ndarray, reducer: Callable[[np.ndarray, int], np.ndarray], minimize: bool) -> np.ndarray:
    """Apply a reduction across combinations while ignoring sentinel markers."""

    if values.size == 0 or values.shape[0] == 0:
        fallback = _INVALID_SENTINEL if minimize else -_INVALID_SENTINEL
        length = values.shape[1] if values.ndim >= 2 else 0
        return np.full((length,), fallback, dtype=np.float32)

    cleaned = values.astype(np.float64, copy=True)
    invalid = ~np.isfinite(cleaned) | (np.abs(cleaned) >= _INVALID_SENTINEL)
    cleaned[invalid] = np.nan

    with np.errstate(all='ignore'):
        reduced = reducer(cleaned, axis=0)

    reduced = np.asarray(reduced, dtype=np.float64)
    all_invalid = np.all(np.isnan(cleaned), axis=0)
    fallback = _INVALID_SENTINEL if minimize else -_INVALID_SENTINEL

    reduced[all_invalid] = fallback
    reduced[np.isnan(reduced)] = fallback
    reduced[np.isposinf(reduced)] = fallback
    reduced[np.isneginf(reduced)] = fallback

    return reduced.astype(np.float32, copy=False)


def _compute_iqr_with_nan(values: np.ndarray) -> np.ndarray:
    """Compute IQR across combinations, treating sentinel values as missing."""

    if values.size == 0 or values.shape[0] == 0:
        length = values.shape[1] if values.ndim >= 2 else 0
        return np.full((length,), _INVALID_SENTINEL, dtype=np.float32)

    cleaned = values.astype(np.float64, copy=True)
    invalid = ~np.isfinite(cleaned) | (np.abs(cleaned) >= _INVALID_SENTINEL)
    cleaned[invalid] = np.nan

    with np.errstate(all='ignore'):
        q3 = np.nanpercentile(cleaned, 75, axis=0)
        q1 = np.nanpercentile(cleaned, 25, axis=0)

    iqr = np.asarray(q3 - q1, dtype=np.float64)
    all_invalid = np.all(np.isnan(cleaned), axis=0)

    iqr[all_invalid] = _INVALID_SENTINEL
    iqr[np.isnan(iqr)] = _INVALID_SENTINEL
    iqr[np.isposinf(iqr)] = _INVALID_SENTINEL
    iqr[np.isneginf(iqr)] = _INVALID_SENTINEL

    return iqr.astype(np.float32, copy=False)


class CPCVObjectiveAggregator:
    """Compute CPCV distributional metrics for the current population."""

    def __init__(
        self,
        scheduler: CPCVScheduler,
        evaluation_config: dict,
        cpcv_objectives: dict,
        is_objectives: dict,
        oos_objectives: dict,
        device: str,
        min_trades_per_split: int = 32,
        parallel_chunks: Optional[int] = None,
        reuse_population: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.scheduler = scheduler
        self.evaluation_config = evaluation_config
        self.cpcv_objectives = cpcv_objectives
        self.is_objectives = is_objectives
        self.oos_objectives = oos_objectives
        self.device = device
        self.min_trades_per_split = max(1, int(min_trades_per_split))
        self.complexity_cfg = self.evaluation_config.get('complexity', {}) if isinstance(self.evaluation_config, dict) else {}
        self.objective_names = list(cpcv_objectives.get('objectives', cpcv_objectives.get('names', [])))
        self.minimize_flags = list(cpcv_objectives.get('minimize', []))
        default_chunks = getattr(self.scheduler, 'parallel_chunks', 1)
        chunks = parallel_chunks if parallel_chunks is not None else default_chunks
        self.parallel_chunks = max(1, int(chunks))
        self.reuse_population = bool(reuse_population)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._frame_cache: Dict[int, Dict[str, List[pd.DataFrame]]] = {}
        pruning_cfg = self.evaluation_config.get('pruning', {}) if isinstance(self.evaluation_config, dict) else {}
        collect_flag = bool(self.evaluation_config.get('collect_node_stats', False)) if isinstance(self.evaluation_config, dict) else False
        self.want_node_stats = bool(collect_flag and pruning_cfg.get('enabled', False))

    def evaluate(
        self,
        population: GATreePop,
        data: pd.DataFrame,
        all_feature_names: Sequence[str],
    ) -> CPCVAggregationResult:
        pop_size = population.pop_size
        simulator_factory = FitnessSimulator
        base_simulator = simulator_factory() if self.reuse_population else None

        train_metrics_nat: List[np.ndarray] = []
        oos_metrics_nat: List[np.ndarray] = []
        oos_metrics_min: List[np.ndarray] = []
        oos_returns: List[np.ndarray] = []
        combo_ids: List[int] = []
        node_stats_global: Optional[Dict[str, torch.Tensor]] = None

        chunk_iter = self.scheduler.iter_split_batches(
            batch_size=self.parallel_chunks,
            require_valid=True,
        )
        # tqdm으로 루프 감싸기 (아래 6줄이 추가/수정된 부분입니다)
        num_combinations = len(self.scheduler._combos)
        num_chunks = math.ceil(num_combinations / self.parallel_chunks)
        
        pbar_combos = tqdm(chunk_iter, total=num_chunks, desc="  CPCV Combinations", leave=False, position=1)

        for batch in pbar_combos:
            simulator = base_simulator if base_simulator is not None else simulator_factory()
            for split in tqdm(batch, leave=False, desc="    CPCV Splits"):
                if split.train_indices.size < self.min_trades_per_split or split.test_indices.size < self.min_trades_per_split:
                    continue

                segments = self._get_split_segments(split, data)
                if not segments['test']:
                    continue

                train_eval_cfg = _build_eval_cfg(self.evaluation_config, self.is_objectives)
                oos_eval_cfg = _build_eval_cfg(self.evaluation_config, self.oos_objectives)

                train_result = self._evaluate_segment_group(
                    simulator=simulator,
                    population=population,
                    segments=segments['train'],
                    eval_cfg=train_eval_cfg,
                    objective_cfg=self.is_objectives,
                    all_feature_names=all_feature_names,
                    min_trades=self.min_trades_per_split,
                    collect_returns=False,
                    collect_node_stats=self.want_node_stats,
                )
                if train_result is None:
                    continue
                train_nat, _, _, _, train_node_stats = train_result
                train_metrics_nat.append(train_nat)
                if self.want_node_stats:
                    node_stats_global = _merge_node_stats(node_stats_global, train_node_stats)

                test_result = self._evaluate_segment_group(
                    simulator=simulator,
                    population=population,
                    segments=segments['test'],
                    eval_cfg=oos_eval_cfg,
                    objective_cfg=self.oos_objectives,
                    all_feature_names=all_feature_names,
                    min_trades=self.min_trades_per_split,
                    collect_returns=True,
                    collect_node_stats=self.want_node_stats,
                )
                if test_result is None:
                    continue

                oos_nat, oos_min, combined_returns, _, test_node_stats = test_result
                oos_metrics_nat.append(oos_nat)
                oos_metrics_min.append(oos_min)
                oos_returns.append(combined_returns)
                if self.want_node_stats:
                    node_stats_global = _merge_node_stats(node_stats_global, test_node_stats)
                combo_ids.append(split.combo_id)

            if not self.reuse_population:
                del simulator
                base_simulator = None

        if not oos_metrics_nat:
            empty = np.full((pop_size, len(self.objective_names)), 1e9, dtype=np.float32)
            return CPCVAggregationResult(empty, summary={}, combo_ids=[])

        oos_stack = np.stack(oos_metrics_nat, axis=0)
        oos_stack_min = np.stack(oos_metrics_min, axis=0)
        train_stack = np.stack(train_metrics_nat, axis=0) if train_metrics_nat else np.zeros((0, pop_size, 0), dtype=np.float32)
        returns_stack = oos_returns

        summary = self._aggregate_metrics(oos_stack, train_stack, returns_stack, population)
        objective_matrix = self._build_objective_matrix(summary, pop_size)

        raw_train = train_stack if train_stack.size else None
        raw_oos = oos_stack_min if oos_stack_min.size else None
        raw_returns = returns_stack if returns_stack else None
        finalized_node_stats = _finalize_node_stats(node_stats_global) if self.want_node_stats else None

        return CPCVAggregationResult(
            objectives=objective_matrix,
            summary=summary,
            combo_ids=combo_ids,
            raw_oos_metrics=raw_oos,
            raw_train_metrics=raw_train,
            raw_returns=raw_returns,
            node_stats=finalized_node_stats,
        )

    def _get_split_segments(
        self,
        split: CPCVSplit,
        data: pd.DataFrame,
    ) -> Dict[str, List[pd.DataFrame]]:
        cache_hit = self._frame_cache.get(split.combo_id)
        if cache_hit is not None:
            return {
                'train': [segment.copy(deep=False) for segment in cache_hit['train']],
                'test': [segment.copy(deep=False) for segment in cache_hit['test']],
            }

        train_segments = [
            data.iloc[start:stop].copy()
            for start, stop in split.train_segments()
            if (stop - start) > 1
        ]
        test_segments = [
            data.iloc[start:stop].copy()
            for start, stop in split.test_segments()
            if (stop - start) > 1
        ]

        if self.cache_dir is not None:
            for kind, segments in (('train', train_segments), ('test', test_segments)):
                for idx, seg in enumerate(segments):
                    path = self.cache_dir / f'combo_{split.combo_id}_{kind}_{idx}.pkl'
                    if not path.exists():
                        seg.to_pickle(path)

        self._frame_cache[split.combo_id] = {'train': train_segments, 'test': test_segments}
        return {
            'train': [segment.copy(deep=False) for segment in train_segments],
            'test': [segment.copy(deep=False) for segment in test_segments],
        }

    def _evaluate_segment_group(
        self,
        simulator: FitnessSimulator,
        population: GATreePop,
        segments: List[pd.DataFrame],
        eval_cfg: dict,
        objective_cfg: dict,
        all_feature_names: Sequence[str],
        min_trades: int,
        collect_returns: bool,
        collect_node_stats: bool,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object], Optional[Dict[str, torch.Tensor]]]]:
        if not segments:
            return None

        pop_size = population.pop_size
        aggregated = _init_env_accumulator(pop_size)
        returns_segments: List[np.ndarray] = []
        bench_segments: List[np.ndarray] = []
        node_stats_accum: Optional[Dict[str, torch.Tensor]] = None

        
        for segment in segments:
            if len(segment) < 2:
                continue

            sim_result = simulator.simulate(
                population=population,
                data=segment,
                all_feature_names=list(all_feature_names),
                start_data_cnt=0,
                stop_data_cnt=len(segment),
                device=self.device,
                evaluation_config=eval_cfg,
                return_returns=True,
                return_node_stats=collect_node_stats,
                return_env_state=True,
            )
            if not isinstance(sim_result, tuple):
                continue
            if collect_node_stats:
                if len(sim_result) < 4:
                    continue
                _, returns, node_stats, env_state = sim_result
                node_stats_accum = _merge_node_stats(node_stats_accum, node_stats)
            else:
                if len(sim_result) < 3:
                    continue
                _, returns, env_state = sim_result
            if not isinstance(env_state, dict):
                continue

            _accumulate_env_state(aggregated, env_state)
            returns_segments.append(returns.astype(np.float32, copy=False))
            bench_arr = _stack_benchmark_returns(env_state.get('bench_returns', []), pop_size)
            if bench_arr.size:
                bench_segments.append(bench_arr)

        combined_returns = _combine_returns(returns_segments, pop_size)
        if combined_returns.size == 0:
            return None

        combined_bench = _combine_returns(bench_segments, pop_size)

        natural_metrics = _compute_objective_values(
            returns=combined_returns,
            bench_returns=combined_bench,
            state=aggregated,
            objective_cfg=objective_cfg,
            population=population,
        )

        natural_metrics = _apply_trade_filters(
            natural_metrics,
            aggregated['trade_total_count'],
            min_trades,
            objective_cfg,
        )

        minimization_metrics = _to_minimization_space(natural_metrics, objective_cfg)
        returns_payload = combined_returns if collect_returns else np.zeros((0, pop_size), dtype=np.float32)

        return (
            natural_metrics.astype(np.float32, copy=False),
            minimization_metrics.astype(np.float32, copy=False),
            returns_payload.astype(np.float32, copy=False),
            aggregated,
            node_stats_accum,
        )

    def _aggregate_metrics(
        self,
        oos_stack: np.ndarray,
        train_stack: np.ndarray,
        returns_stack: List[np.ndarray],
        population: GATreePop,
    ) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        names = self.oos_objectives.get('objectives', [])
        minimize = self.oos_objectives.get('minimize', [True] * len(names))
        name_to_idx = {name: idx for idx, name in enumerate(names)}

        if 'mean_return' in name_to_idx:
            idx = name_to_idx['mean_return']
            result['median_oos_return'] = _reduce_with_nan(
                oos_stack[:, :, idx],
                lambda arr, axis=0: np.nanmedian(arr, axis=axis),
                minimize=False,
            )
        if 'sharpe_ratio' in name_to_idx:
            idx = name_to_idx['sharpe_ratio']
            shard = oos_stack[:, :, idx]
            result['median_oos_sharpe'] = _reduce_with_nan(
                shard,
                lambda arr, axis=0: np.nanmedian(arr, axis=axis),
                minimize=False,
            )
            result['iqr_oos_sharpe'] = _compute_iqr_with_nan(shard)
        if 'max_drawdown' in name_to_idx:
            idx = name_to_idx['max_drawdown']
            result['iqr_oos_drawdown'] = _compute_iqr_with_nan(oos_stack[:, :, idx])
        if 'psr' in name_to_idx:
            idx = name_to_idx['psr']
            result['median_oos_psr'] = _reduce_with_nan(
                oos_stack[:, :, idx],
                lambda arr, axis=0: np.nanmedian(arr, axis=axis),
                minimize=False,
            )
        if 'profit_factor' in name_to_idx:
            idx = name_to_idx['profit_factor']
            result['median_oos_profit_factor'] = _reduce_with_nan(
                oos_stack[:, :, idx],
                lambda arr, axis=0: np.nanmedian(arr, axis=axis),
                minimize=False,
            )
        if 'turnover' in name_to_idx:
            idx = name_to_idx['turnover']
            result['oos_turnover'] = _reduce_with_nan(
                oos_stack[:, :, idx],
                lambda arr, axis=0: _nanmean_axis(arr, axis=axis),
                minimize=True,
            )

        result['cvar95_oos_return'] = _compute_cvar95(returns_stack)

        # Tree complexity
        result['tree_complexity'] = _compute_tree_complexity(population, self.complexity_cfg)

        # Aggregate IS metrics if available
        train_names = self.is_objectives.get('objectives', [])
        train_name_to_idx = {name: idx for idx, name in enumerate(train_names)}
        if train_stack.size > 0:
            if 'mean_return' in train_name_to_idx:
                idx = train_name_to_idx['mean_return']
                result['is_mean_return'] = _reduce_with_nan(
                    train_stack[:, :, idx],
                    lambda arr, axis=0: np.nanmedian(arr, axis=axis),
                    minimize=False,
                )
            if 'max_drawdown' in train_name_to_idx:
                idx = train_name_to_idx['max_drawdown']
                result['is_max_drawdown'] = _reduce_with_nan(
                    train_stack[:, :, idx],
                    lambda arr, axis=0: np.nanmedian(arr, axis=axis),
                    minimize=True,
                )
            if 'psr' in train_name_to_idx:
                idx = train_name_to_idx['psr']
                result['is_psr'] = _reduce_with_nan(
                    train_stack[:, :, idx],
                    lambda arr, axis=0: np.nanmedian(arr, axis=axis),
                    minimize=False,
                )
            if 'root_balance_penalty' in train_name_to_idx:
                idx = train_name_to_idx['root_balance_penalty']
                result['is_root_balance_penalty'] = _reduce_with_nan(
                    train_stack[:, :, idx],
                    lambda arr, axis=0: np.nanmedian(arr, axis=axis),
                    minimize=True,
                )

        return result

    def _build_objective_matrix(self, summary: Dict[str, np.ndarray], pop_size: int) -> np.ndarray:
        names = self.objective_names
        minimize_flags = list(self.minimize_flags) if self.minimize_flags else [True] * len(names)
        if len(minimize_flags) < len(names):
            minimize_flags.extend([True] * (len(names) - len(minimize_flags)))
        matrix = np.full((pop_size, len(names)), 1e9, dtype=np.float32)
        for idx, name in enumerate(names):
            values = summary.get(name)
            if values is None:
                continue
            vals = values.astype(np.float32, copy=False)
            matrix[:, idx] = vals if minimize_flags[idx] else -vals
        return matrix


def _build_eval_cfg(base_cfg: dict, objectives: dict) -> dict:
    cfg = dict(base_cfg)
    cfg['objectives'] = objectives
    cfg['multi_objectives'] = objectives
    return cfg


def _compute_cvar95(returns_stack: List[np.ndarray]) -> np.ndarray:
    if not returns_stack:
        return np.zeros((0,), dtype=np.float32)
    reshaped = [r.reshape(-1, r.shape[-1]) for r in returns_stack if r.size > 0]
    if not reshaped:
        pop_size = returns_stack[0].shape[-1]
        return np.zeros((pop_size,), dtype=np.float32)

    combined = np.concatenate(reshaped, axis=0)
    pop_size = combined.shape[1]
    cvar = np.zeros((pop_size,), dtype=np.float32)
    alpha = 0.95
    tail_prob = 1.0 - alpha
    for col in range(pop_size):
        series = combined[:, col]
        if series.size == 0:
            cvar[col] = 0.0
            continue
        k = max(1, int(math.ceil(tail_prob * series.size)))
        tail = np.partition(series, k - 1)[:k]
        losses = np.minimum(tail, 0.0)
        if losses.size == 0:
            cvar[col] = 0.0
            continue
        cvar[col] = -float(np.mean(losses))
    return cvar


def _compute_tree_complexity(population: GATreePop, complexity_cfg: dict) -> np.ndarray:
    pt = population.population_tensor
    node_types = pt[:, :, COL_NODE_TYPE]
    depths = pt[:, :, COL_DEPTH]
    used_mask = node_types != float(NODE_TYPE_UNUSED)
    size = used_mask.sum(dim=1).to(torch.float32)
    depth = torch.where(used_mask, depths, torch.zeros_like(depths)).max(dim=1).values.to(torch.float32)

    max_nodes = max(1, population.max_nodes)
    max_depth = max(1, population.max_depth)
    size_norm = (size / float(max_nodes)).clamp(0.0, 1.0)
    depth_norm = (depth / float(max_depth)).clamp(0.0, 1.0)

    mode = str(complexity_cfg.get('mode', 'weighted')).lower()
    size_weight = float(complexity_cfg.get('size_weight', 0.5))
    depth_weight = float(complexity_cfg.get('depth_weight', 0.5))

    if mode == 'max':
        complexity = torch.maximum(size_norm, depth_norm)
    elif mode in ('geomean', 'geo', 'gmean'):
        complexity = torch.sqrt(size_norm * depth_norm)
    else:
        denom = size_weight + depth_weight
        if denom == 0:
            denom = 1.0
        complexity = (size_weight * size_norm + depth_weight * depth_norm) / denom

    return complexity.detach().cpu().numpy().astype(np.float32, copy=False)


def _init_env_accumulator(pop_size: int) -> Dict[str, object]:
    zeros = np.zeros((pop_size,), dtype=np.float64)
    return {
        'sum_returns': zeros.copy(),
        'count_returns': zeros.copy(),
        'total_profit_agg': zeros.copy(),
        'total_loss_agg': zeros.copy(),
        'count_wins': zeros.copy(),
        'trade_total_count': zeros.copy(),
        'trade_win_count': zeros.copy(),
        'total_steps': 0,
        'risk_free_rate': None,
        'lookback_period': None,
        'mar': None,
    }


def _accumulate_env_state(accumulator: Dict[str, object], env_state: Dict[str, object]) -> None:
    for key in ('sum_returns', 'count_returns', 'total_profit_agg', 'total_loss_agg', 'count_wins', 'trade_total_count', 'trade_win_count'):
        if key in env_state and isinstance(env_state[key], np.ndarray):
            accumulator[key] = accumulator.get(key, np.zeros_like(env_state[key], dtype=np.float64)) + env_state[key].astype(np.float64, copy=False)

    accumulator['total_steps'] = int(accumulator.get('total_steps', 0)) + int(env_state.get('total_steps', 0))

    for meta_key in ('risk_free_rate', 'lookback_period', 'mar'):
        if accumulator.get(meta_key) is None and env_state.get(meta_key) is not None:
            accumulator[meta_key] = env_state[meta_key]


def _stack_benchmark_returns(bench_returns: Sequence[np.ndarray], pop_size: int) -> np.ndarray:
    if not bench_returns:
        return np.zeros((0, pop_size), dtype=np.float32)
    stacked: List[np.ndarray] = []
    for arr in bench_returns:
        if arr is None:
            continue
        arr_np = np.asarray(arr, dtype=np.float32)
        if arr_np.size == 0:
            continue

        flat = arr_np.reshape(-1)
        if flat.size == 1:
            row = np.full((1, pop_size), float(flat[0]), dtype=np.float32)
            stacked.append(row)
            continue

        if flat.size % pop_size != 0:
            # Skip inconsistent benchmark payloads to avoid invalid reshapes.
            continue

        reshaped = flat.reshape(-1, pop_size)
        stacked.append(reshaped)
    if not stacked:
        return np.zeros((0, pop_size), dtype=np.float32)
    return np.concatenate(stacked, axis=0).astype(np.float32, copy=False)


def _ensure_cpu_tensor(value) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        return None
    if tensor.is_floating_point():
        return tensor.to(torch.float32)
    return tensor.to(torch.int32)


def _merge_node_stats(
    accumulator: Optional[Dict[str, torch.Tensor]],
    new_stats: Optional[Dict[str, object]],
) -> Optional[Dict[str, torch.Tensor]]:
    if new_stats is None:
        return accumulator

    if accumulator is None:
        accumulator = {}

    for key, value in new_stats.items():
        if value is None:
            continue
        if key == 'total_steps':
            tensor = torch.tensor(int(value), dtype=torch.int32)
        else:
            tensor = _ensure_cpu_tensor(value)
        if tensor is None:
            continue

        if key in accumulator:
            accumulator[key] = accumulator[key] + tensor
        else:
            accumulator[key] = tensor.clone()

    return accumulator


def _finalize_node_stats(stats: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, Union[int, torch.Tensor]]]:
    if stats is None:
        return None
    finalized: Dict[str, Union[int, torch.Tensor]] = {}
    for key, value in stats.items():
        if value is None:
            continue
        if key == 'total_steps':
            if isinstance(value, torch.Tensor):
                finalized[key] = int(value.detach().cpu().item())
            else:
                finalized[key] = int(value)
        else:
            finalized[key] = value.detach().cpu()
    return finalized


def _combine_returns(returns_list: Sequence[np.ndarray], pop_size: int) -> np.ndarray:
    valid = [np.asarray(ret, dtype=np.float32) for ret in returns_list if isinstance(ret, np.ndarray) and ret.size]
    if not valid:
        return np.zeros((0, pop_size), dtype=np.float32)
    return np.concatenate(valid, axis=0)


def _to_minimization_space(values: np.ndarray, objectives: dict) -> np.ndarray:
    minimize_flags = list(objectives.get('minimize', [True] * values.shape[1]))
    out = values.astype(np.float32, copy=True)
    for idx, minimize in enumerate(minimize_flags):
        if idx >= out.shape[1]:
            break
        if not minimize:
            out[:, idx] = -out[:, idx]
    return out


def _compute_objective_values(
    returns: np.ndarray,
    bench_returns: np.ndarray,
    state: Dict[str, object],
    objective_cfg: dict,
    population: Optional[GATreePop],
) -> np.ndarray:
    pop_size = returns.shape[1]
    objective_names = list(objective_cfg.get('objectives', []))
    result = np.zeros((pop_size, len(objective_names)), dtype=np.float64)

    sum_returns = np.asarray(state.get('sum_returns', np.zeros(pop_size)), dtype=np.float64)
    count_returns = np.asarray(state.get('count_returns', np.zeros(pop_size)), dtype=np.float64)
    total_profit = np.asarray(state.get('total_profit_agg', np.zeros(pop_size)), dtype=np.float64)
    total_loss = np.asarray(state.get('total_loss_agg', np.zeros(pop_size)), dtype=np.float64)
    count_wins = np.asarray(state.get('count_wins', np.zeros(pop_size)), dtype=np.float64)
    trade_total = np.asarray(state.get('trade_total_count', np.zeros(pop_size)), dtype=np.float64)
    trade_wins = np.asarray(state.get('trade_win_count', np.zeros(pop_size)), dtype=np.float64)
    total_steps = int(state.get('total_steps', returns.shape[0]))

    risk_free = float(state.get('risk_free_rate', objective_cfg.get('risk_free_rate', 0.0)))
    lookback = int(state.get('lookback_period', objective_cfg.get('lookback_period', objective_cfg.get('volatility_lookback', 252))))
    lookback = max(lookback, 1)
    mar = float(state.get('mar', objective_cfg.get('mar', 0.0)))
    psr_benchmark = float(objective_cfg.get('psr_sr_benchmark', 0.0))

    eq_curve = np.cumprod(1.0 + returns, axis=0)
    running_max = np.maximum.accumulate(eq_curve, axis=0)
    drawdowns = 1.0 - eq_curve / np.maximum(running_max, 1e-12)
    max_drawdown = np.clip(drawdowns.max(axis=0), 0.0, 1.0)

    final_equity = eq_curve[-1] if returns.shape[0] > 0 else np.ones(pop_size, dtype=np.float64)
    compound_value = final_equity - 1.0

    mean_return = np.full(pop_size, -1e9, dtype=np.float64)
    valid_mask = count_returns > 0
    mean_return[valid_mask] = sum_returns[valid_mask] / count_returns[valid_mask]

    win_rate = np.full(pop_size, -1e9, dtype=np.float64)
    win_rate[valid_mask] = count_wins[valid_mask] / np.maximum(count_returns[valid_mask], 1.0)
    win_rate = np.clip(win_rate, 0.0, 1.0)

    trade_win_rate = np.full(pop_size, -1e9, dtype=np.float64)
    trade_mask = trade_total > 0
    trade_win_rate[trade_mask] = trade_wins[trade_mask] / np.maximum(trade_total[trade_mask], 1.0)
    trade_win_rate = np.clip(trade_win_rate, 0.0, 1.0)

    profit_factor = np.full(pop_size, 100.0, dtype=np.float64)
    loss_mask = total_loss > 0
    profit_factor[loss_mask] = np.clip(total_profit[loss_mask] / (total_loss[loss_mask] + 1e-9), 0.0, 100.0)

    turnover = np.full(pop_size, 1e9, dtype=np.float64)
    if total_steps > 0:
        turnover = trade_total / float(total_steps)

    annualization = np.sqrt(float(lookback))
    period_rf = risk_free / lookback

    if returns.shape[0] > 1:
        excess = returns - period_rf
        mean_excess = excess.mean(axis=0)
        std_excess = excess.std(axis=0, ddof=1)
        sharpe = np.zeros(pop_size, dtype=np.float64)
        sharpe_mask = std_excess > 1e-8
        sharpe[sharpe_mask] = mean_excess[sharpe_mask] / std_excess[sharpe_mask] * annualization

        negative = np.where(excess < 0, excess, 0.0)
        downside = np.sqrt(np.mean(negative ** 2, axis=0))
        sortino = np.zeros(pop_size, dtype=np.float64)
        sortino_mask = downside > 1e-8
        sortino[sortino_mask] = mean_excess[sortino_mask] / downside[sortino_mask] * annualization

        volatility = std_excess * annualization
        psr = _compute_psr_numpy(excess, psr_benchmark)

        std_returns = returns.std(axis=0, ddof=1)
        t_stat = np.zeros(pop_size, dtype=np.float64)
        t_mask = std_returns > 1e-8
        t_stat[t_mask] = returns.mean(axis=0)[t_mask] / (std_returns[t_mask] / np.sqrt(returns.shape[0]))
    else:
        sharpe = np.zeros(pop_size, dtype=np.float64)
        sortino = np.zeros(pop_size, dtype=np.float64)
        volatility = np.zeros(pop_size, dtype=np.float64)
        psr = np.zeros(pop_size, dtype=np.float64)
        t_stat = np.zeros(pop_size, dtype=np.float64)

    omega = np.zeros(pop_size, dtype=np.float64)
    if returns.shape[0] > 1:
        period_mar = mar / lookback
        excess_mar = returns - period_mar
        gains = np.clip(excess_mar, 0.0, None).sum(axis=0)
        losses = np.clip(-excess_mar, 0.0, None).sum(axis=0)
        omega_mask = losses > 1e-12
        omega[omega_mask] = gains[omega_mask] / (losses[omega_mask] + 1e-12)

    information_ratio = np.zeros(pop_size, dtype=np.float64)
    if bench_returns.size and returns.shape[0] > 1:
        t = min(len(returns), len(bench_returns))
        active = returns[-t:] - bench_returns[-t:]
        mean_active = active.mean(axis=0)
        tracking_err = active.std(axis=0, ddof=1)
        mask = tracking_err > 1e-8
        information_ratio[mask] = mean_active[mask] / tracking_err[mask] * annualization

    annualized_return = np.zeros(pop_size, dtype=np.float64)
    if returns.shape[0] > 0:
        total_return = compound_value
        periods = max(returns.shape[0], 1)
        base = 1.0 + total_return
        annualized_return = np.full(pop_size, -1.0, dtype=np.float64)
        valid_mask = base > 0.0
        if np.any(valid_mask):
            exponent = float(lookback) / float(periods)
            annualized_return[valid_mask] = np.power(base[valid_mask], exponent) - 1.0
    calmar = np.zeros(pop_size, dtype=np.float64)
    calmar_mask = max_drawdown > 1e-8
    calmar[calmar_mask] = annualized_return[calmar_mask] / max_drawdown[calmar_mask]

    metric_map = {
        'mean_return': mean_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'psr': psr,
        't_stat': t_stat,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'compound_value': compound_value,
        'trade_win_rate': trade_win_rate,
        'turnover': turnover,
        'volatility': volatility,
        'calmar_ratio': calmar,
        'sortino_ratio': sortino,
        'omega_ratio': omega,
        'information_ratio': information_ratio,
    }

    if 'root_balance_penalty' in objective_names:
        if population is None:
            raise ValueError("Objective 'root_balance_penalty' requires population context.")
        penalty_tensor = _compute_root_branch_balance_penalty(population)
        penalty = penalty_tensor.detach().to('cpu').numpy().astype(np.float64, copy=False)
        metric_map['root_balance_penalty'] = penalty

    for idx, name in enumerate(objective_names):
        metric = metric_map.get(name)
        if metric is None:
            raise ValueError(f"Objective '{name}' not supported in CPCV aggregation.")
        result[:, idx] = metric

    return result


def _compute_psr_numpy(excess_returns: np.ndarray, sr_benchmark: float) -> np.ndarray:
    pop_size = excess_returns.shape[1]
    if excess_returns.shape[0] < 3:
        return np.zeros(pop_size, dtype=np.float64)

    mean = excess_returns.mean(axis=0)
    std = excess_returns.std(axis=0, ddof=1)
    safe_mask = std > 1e-12
    sr_hat = np.zeros(pop_size, dtype=np.float64)
    sr_hat[safe_mask] = mean[safe_mask] / std[safe_mask]

    xc = excess_returns - mean
    m2 = np.mean(xc ** 2, axis=0)
    m3 = np.mean(xc ** 3, axis=0)
    m4 = np.mean(xc ** 4, axis=0)

    eps = 1e-12
    denom = np.sqrt(
        np.maximum(
            1.0 - (m3 / np.maximum(m2, eps) ** 1.5) * sr_hat + ((m4 / np.maximum(m2, eps) ** 2 - 1.0) / 4.0) * (sr_hat ** 2),
            eps,
        )
    )
    z = np.zeros(pop_size, dtype=np.float64)
    n = excess_returns.shape[0]
    mask = (denom > 0) & safe_mask
    z[mask] = (sr_hat[mask] - sr_benchmark) * np.sqrt(max(n - 1, 1)) / denom[mask]
    psr = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    psr[~safe_mask] = 0.0
    return np.clip(psr, 0.0, 1.0)


def _apply_trade_filters(
    metrics_nat: np.ndarray,
    trade_counts: np.ndarray,
    min_trades: int,
    objective_cfg: dict,
) -> np.ndarray:
    if min_trades <= 0:
        return metrics_nat
    filtered = metrics_nat.copy()
    below = trade_counts < float(min_trades)
    if not np.any(below):
        return filtered

    minimize_flags = list(objective_cfg.get('minimize', [True] * filtered.shape[1]))
    for idx, minimize in enumerate(minimize_flags):
        if idx >= filtered.shape[1]:
            break
        if minimize:
            filtered[below, idx] = 1e9
        else:
            filtered[below, idx] = -1e9
    return filtered
