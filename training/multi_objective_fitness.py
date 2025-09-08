# training/multi_objective_fitness.py

import numpy as np
import torch
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
# Note: Now using unified trading_env which includes all multi-objective functionality
# from .trading_env import TradingEnvironment, fitness_fn


# --- Module-level CUDA crowding distance helper ---
def _crowding_distance_cuda(front_obj: torch.Tensor) -> torch.Tensor:
    """
    CUDA crowding distance for a single front.
    front_obj: [F, M], minimization 기준.
    반환: [F] float32, 경계점=+inf
    """
    device = front_obj.device
    F, M = front_obj.shape
    if F <= 2:
        return torch.full((F,), float('inf'), dtype=torch.float32, device=device)

    dist = torch.zeros((F,), dtype=torch.float32, device=device)

    for m in range(M):
        vals = front_obj[:, m]
        sorted_vals, sorted_idx = torch.sort(vals)  # CUDA sort
        dist[sorted_idx[0]]  = float('inf')
        dist[sorted_idx[-1]] = float('inf')

        rng = sorted_vals[-1] - sorted_vals[0]
        if rng > 1e-8:
            prev = sorted_vals[:-2]
            nxt  = sorted_vals[2:]
            add = (nxt - prev) / rng  # [F-2]
            core_idx = sorted_idx[1:-1]
            # 경계가 이미 inf여도 합산은 무의미하므로 그대로 덮어쓰기 되지 않음.
            dist[core_idx] += add.to(dist.dtype)

    return dist


class MultiObjectiveFitnessEvaluator:
    """
    Multi-objective fitness evaluator that provides utilities for managing
    and evaluating multiple objectives in the genetic algorithm context.
    """
    
    def __init__(self, objectives_config: dict):
        """
        Initialize the multi-objective fitness evaluator.
        
        Args:
            objectives_config (dict): Configuration for objectives including:
                - objectives: List of objective names
                - minimize: List of booleans indicating minimization direction
                - weights: Optional weights for weighted sum evaluation
                - normalization: Normalization strategy ('minmax', 'zscore', 'none')
        """
        self.objectives_config = objectives_config
        self.objective_names = objectives_config['objectives']
        self.minimize_objectives = objectives_config['minimize']
        self.num_objectives = len(self.objective_names)
        
        # Validation
        if len(self.minimize_objectives) != self.num_objectives:
            raise ValueError("minimize list must match objectives list length")
        
        # Optional configuration
        self.weights = objectives_config.get('weights', None)
        self.normalization = objectives_config.get('normalization', 'minmax')
        
        # Statistics tracking for normalization
        self.objective_stats = {}
        self.history = []
    
    def evaluate_population(
        self, 
        population,
        data: pd.DataFrame,
        all_feature_names: list,
        adj_offsets: torch.Tensor,
        adj_indices: torch.Tensor,
        start_data_cnt: int,
        stop_data_cnt: int,
        device: str,
        evaluation_config: dict
    ) -> np.ndarray:
        """
        Evaluate population using multi-objective fitness function.
        
        Returns:
            objectives: [population_size, num_objectives] array of objective values
        """
        # Use the simulator's wrapper to avoid legacy dependency on trading_env.fitness_fn
        from fitness.simulator import fitness_fn
        return fitness_fn(
            population=population,
            data=data,
            all_feature_names=all_feature_names,
            # Simulator builds adjacency internally; passed adj_* are unused in wrapper
            start_data_cnt=start_data_cnt,
            stop_data_cnt=stop_data_cnt,
            device=device,
            evaluation_config=evaluation_config
        )
    
    def normalize_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """
        Normalize objectives according to the configured strategy.
        
        Args:
            objectives: [population_size, num_objectives] raw objective values
            
        Returns:
            normalized_objectives: [population_size, num_objectives] normalized values
        """
        if self.normalization == 'none':
            return objectives
        
        normalized = objectives.copy()
        
        for i in range(self.num_objectives):
            obj_values = objectives[:, i]
            valid_mask = ~np.isin(obj_values, [-1e9, 1e9]) & ~np.isnan(obj_values) & ~np.isinf(obj_values)
            
            if not np.any(valid_mask):
                # No valid values - keep as is
                continue
            
            valid_values = obj_values[valid_mask]
            
            if self.normalization == 'minmax':
                min_val, max_val = np.min(valid_values), np.max(valid_values)
                if min_val != max_val:
                    normalized[valid_mask, i] = (obj_values[valid_mask] - min_val) / (max_val - min_val)
                else:
                    normalized[valid_mask, i] = 0.5
            
            elif self.normalization == 'zscore':
                mean_val, std_val = np.mean(valid_values), np.std(valid_values)
                if std_val > 1e-8:
                    normalized[valid_mask, i] = (obj_values[valid_mask] - mean_val) / std_val
                else:
                    normalized[valid_mask, i] = 0.0
            
            # Update statistics for future use
            if valid_mask.any():
                self.objective_stats[self.objective_names[i]] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values)
                }
        
        return normalized
    
    def calculate_weighted_fitness(self, objectives: np.ndarray) -> np.ndarray:
        """
        Calculate weighted fitness scores from multi-objective values.
        
        Args:
            objectives: [population_size, num_objectives] objective values
            
        Returns:
            fitness: [population_size] weighted fitness scores
        """
        if self.weights is None:
            # Equal weights
            weights = np.ones(self.num_objectives) / self.num_objectives
        else:
            weights = np.array(self.weights)
            if len(weights) != self.num_objectives:
                raise ValueError("Weights length must match number of objectives")
        
        # Normalize objectives first
        normalized = self.normalize_objectives(objectives)
        
        # For minimization objectives, we want lower values to contribute positively to fitness
        # For maximization objectives (stored as negated), we want higher absolute values
        fitness_contrib = normalized.copy()
        
        for i in range(self.num_objectives):
            if self.minimize_objectives[i]:
                # Minimization: lower normalized value = higher fitness contribution
                fitness_contrib[:, i] = 1.0 - fitness_contrib[:, i]
            # For maximization objectives stored as negated values, 
            # higher normalized values already represent better fitness
        
        # Weighted sum
        fitness = np.sum(fitness_contrib * weights, axis=1)
        
        # Handle invalid individuals
        invalid_mask = np.any(np.isin(objectives, [-1e9, 1e9]), axis=1)
        fitness[invalid_mask] = -1.0
        
        return fitness
    
    def pareto_rank(self, objectives: np.ndarray, device: Union[str, torch.device, None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        CUDA 가속 Pareto 정렬 + Crowding Distance.
        - 입력 objectives는 'minimization 기준'이어야 합니다(원 코드와 동일).
        - 무효 개체(±1e9, NaN/Inf 포함)는 최하위 랭크로 보냅니다.
        반환:
            ranks: [N] (0이 최상위 전선)
            distances: [N] crowding distance (경계점=+inf)
        """
        # Honor explicitly provided device; otherwise fall back to availability
        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')
        else:
            device = torch.device(device)

        # np.ndarray 혹은 torch.Tensor 모두 허용
        if isinstance(objectives, np.ndarray):
            obj = torch.from_numpy(objectives)
        else:
            obj = objectives
        obj = obj.to(device=device, dtype=torch.float32)

        N, M = obj.shape if obj.ndim == 2 else (0, 0)
        if N == 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

        # 유효/무효 마스크 (행 기준): 어느 목표라도 sentinels/NaN/Inf면 무효
        row_invalid = ((obj == -1e9) | (obj == 1e9) | ~torch.isfinite(obj)).any(dim=1)
        valid_mask = ~row_invalid
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

        # 모두 무효한 경우
        if valid_idx.numel() == 0:
            ranks = torch.full((N,), 0, dtype=torch.int32, device=device)
            distances = torch.zeros((N,), dtype=torch.float32, device=device)
            # 전선이 없으니 무효는 최하위(=0) 대신 0으로 두고, 아래에서 len_fronts로 덮어씀
            ranks[~valid_mask] = 1  # fronts 수 = 1 로 취급
            return ranks.cpu().numpy(), distances.cpu().numpy()

        vobj = obj[valid_idx]                 # [Nv, M]
        Nv = vobj.shape[0]

        # DOM: [Nv, Nv] bool, DOM[i, j] = i가 j 지배?
        # 메모리/성능 균형을 위해 i축 chunking
        DOM = torch.zeros((Nv, Nv), dtype=torch.bool, device=device)
        chunk = 1024 if Nv >= 1024 else Nv  # 필요 시 조정

        for i0 in range(0, Nv, chunk):
            i1 = min(i0 + chunk, Nv)
            vi = vobj[i0:i1]  # [Ci, M]

            # [Ci, 1, M] vs [1, Nv, M]
            le = (vi[:, None, :] <= vobj[None, :, :]).all(dim=-1)  # [Ci, Nv]
            lt = (vi[:, None, :] <  vobj[None, :, :]).any(dim=-1)  # [Ci, Nv]
            block = le & lt

            # 대각(자기 자신 지배 금지)
            if i0 < i1:
                diag_j = torch.arange(i0, i1, device=device)
                block[torch.arange(i1 - i0, device=device), diag_j] = False

            DOM[i0:i1, :] = block

        # domination_counts[j] = 다른 개체가 j를 지배하는 수 = DOM[:, j].sum()
        counts = DOM.sum(dim=0).to(torch.int32)  # [Nv]

        # 전선 빌드
        assigned = torch.zeros(Nv, dtype=torch.bool, device=device)
        ranks_valid = torch.full((Nv,), -1, dtype=torch.int32, device=device)

        fronts: list[torch.Tensor] = []
        cur_front = torch.nonzero((counts == 0) & (~assigned), as_tuple=False).squeeze(1)
        while cur_front.numel() > 0:
            fronts.append(cur_front)
            ranks_valid[cur_front] = len(fronts) - 1
            assigned[cur_front] = True

            # 현재 전선이 지배하는 개수만큼 후속 개체들의 counts 감소
            decrements = DOM[cur_front].sum(dim=0).to(torch.int32)  # [Nv]
            counts -= decrements
            counts[cur_front] = -1  # 처리 완료 표식

            # 다음 전선: 아직 미할당 & counts==0
            cur_front = torch.nonzero((counts == 0) & (~assigned), as_tuple=False).squeeze(1)

        num_fronts = len(fronts)

        # Crowding Distance (전선별, CUDA 정렬 사용)
        distances_valid = torch.zeros((Nv,), dtype=torch.float32, device=device)
        for f in fronts:
            if f.numel() == 0:
                continue
            _d = _crowding_distance_cuda(vobj[f])  # [|f|]
            distances_valid[f] = _d

        # 전역 결과로 되돌리기
        ranks = torch.full((N,), num_fronts, dtype=torch.int32, device=device)  # 무효는 최하위(fronts 개수)
        distances = torch.zeros((N,), dtype=torch.float32, device=device)

        ranks[valid_idx] = ranks_valid
        distances[valid_idx] = distances_valid
        # 무효는 ranks= num_fronts (원 코드: len(fronts)), distances는 0

        return ranks.cpu().numpy(), distances.cpu().numpy()

    @staticmethod
    def _crowding_distance_cpu_fallback(front_objectives: np.ndarray) -> np.ndarray:
        """(비상용) CPU 폴백. CUDA 환경이 없을 때만 사용."""
        front_size = len(front_objectives)
        if front_size <= 2:
            return np.full(front_size, float('inf'), dtype=np.float32)
        distances = np.zeros(front_size, dtype=np.float32)
        F = front_objectives.astype(np.float32, copy=False)
        M = F.shape[1]
        for m in range(M):
            order = np.argsort(F[:, m], kind='mergesort')
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            rng = F[order[-1], m] - F[order[0], m]
            if rng > 1e-8:
                prev = F[order[:-2], m]
                nxt  = F[order[2:],  m]
                distances[order[1:-1]] += (nxt - prev) / rng
        return distances

    def _find_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask indicating Pareto-optimal (non-dominated) solutions.

        Assumes objectives are in minimization form (lower is better). Invalid
        rows marked with sentinel values (e.g., -1e9 or 1e9) are excluded.

        Args:
            objectives: [population_size, num_objectives] objective values

        Returns:
            mask: [population_size] boolean array, True for Pareto front members
        """
        if objectives.size == 0:
            return np.zeros((0,), dtype=bool)

        invalid_mask = np.any(np.isin(objectives, [-1e9, 1e9]), axis=1)
        ranks, _ = self.pareto_rank(objectives)
        return (ranks == 0) & (~invalid_mask)

    def _calculate_crowding_distances(self, front_objectives: np.ndarray) -> np.ndarray:
        """
        Calculate crowding distances for a given front.

        This wraps `_crowding_distance` while handling potential invalid rows.
        Assumes the input set is intended to represent a single Pareto front.

        Args:
            front_objectives: [front_size, num_objectives] objective values

        Returns:
            distances: [front_size] crowding distances (invalid rows get -inf)
        """
        if front_objectives.size == 0:
            return np.zeros((0,), dtype=float)

        valid_mask = ~np.any(np.isin(front_objectives, [-1e9, 1e9]), axis=1)
        distances = np.full(len(front_objectives), -float('inf'), dtype=float)

        if np.any(valid_mask):
            valid_distances = self._crowding_distance(front_objectives[valid_mask])
            distances[valid_mask] = valid_distances

        return distances
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """
        Check if obj1 Pareto dominates obj2.
        
        Args:
            obj1: First objective vector
            obj2: Second objective vector
            
        Returns:
            True if obj1 dominates obj2
        """
        # obj1 dominates obj2 if:
        # - obj1 is <= obj2 in all objectives (for minimization)
        # - obj1 is < obj2 in at least one objective
        
        all_better_or_equal = True
        at_least_one_better = False
        
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:  # Worse in this objective
                all_better_or_equal = False
                break
            elif obj1[i] < obj2[i]:  # Better in this objective
                at_least_one_better = True
        
        return all_better_or_equal and at_least_one_better
    
    def _crowding_distance(self, front_objectives: np.ndarray) -> np.ndarray:
        """
        Calculate crowding distance for a front.
        
        Args:
            front_objectives: [front_size, num_objectives] objectives for one front
            
        Returns:
            distances: [front_size] crowding distances
        """
        front_size = len(front_objectives)
        if front_size <= 2:
            return np.full(front_size, float('inf'))
        
        distances = np.zeros(front_size)
        
        for obj_idx in range(self.num_objectives):
            # Sort by this objective
            sorted_indices = np.argsort(front_objectives[:, obj_idx])
            
            # Boundary points get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate range for normalization
            obj_range = (front_objectives[sorted_indices[-1], obj_idx] - 
                        front_objectives[sorted_indices[0], obj_idx])
            
            if obj_range > 1e-8:
                for i in range(1, front_size - 1):
                    distance = (front_objectives[sorted_indices[i + 1], obj_idx] - 
                              front_objectives[sorted_indices[i - 1], obj_idx]) / obj_range
                    distances[sorted_indices[i]] += distance
        
        return distances
    
    def get_objective_summary(self, objectives: np.ndarray) -> Dict:
        """
        Generate summary statistics for objectives.
        
        Args:
            objectives: [population_size, num_objectives] objective values
            
        Returns:
            summary: Dictionary with summary statistics
        """
        summary = {
            'objective_names': self.objective_names,
            'minimize_objectives': self.minimize_objectives,
            'population_size': len(objectives),
            'statistics': {}
        }
        
        for i, obj_name in enumerate(self.objective_names):
            obj_values = objectives[:, i]
            valid_mask = ~np.isin(obj_values, [-1e9, 1e9]) & ~np.isnan(obj_values) & ~np.isinf(obj_values)
            
            if np.any(valid_mask):
                # Convert to natural display form: maximize metrics were negated upstream
                if self.minimize_objectives[i]:
                    valid_values = obj_values[valid_mask]
                else:
                    valid_values = -obj_values[valid_mask]
                summary['statistics'][obj_name] = {
                    'mean': float(np.mean(valid_values)),
                    'std': float(np.std(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'median': float(np.median(valid_values)),
                    'valid_count': int(np.sum(valid_mask)),
                    'minimize': self.minimize_objectives[i]
                }
            else:
                summary['statistics'][obj_name] = {
                    'mean': None, 'std': None, 'min': None, 'max': None,
                    'median': None, 'valid_count': 0,
                    'minimize': self.minimize_objectives[i]
                }
        
        return summary


def create_multi_objective_evaluator(objectives_config: dict) -> MultiObjectiveFitnessEvaluator:
    """
    Factory function to create a multi-objective fitness evaluator.
    
    Args:
        objectives_config (dict): Configuration for objectives
        
    Returns:
        evaluator: MultiObjectiveFitnessEvaluator instance
    """
    return MultiObjectiveFitnessEvaluator(objectives_config)


# Predefined objective configurations
PRESET_OBJECTIVES = {
    'balanced': {
        'objectives': ['mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate'],
        'minimize': [False, True, False, False],
        'weights': [0.3, 0.3, 0.2, 0.2],
        'normalization': 'minmax'
    },
    
    'return_focused': {
        'objectives': ['mean_return', 'compound_value', 'max_drawdown'],
        'minimize': [False, False, True],
        'weights': [0.4, 0.4, 0.2],
        'normalization': 'minmax'
    },
    
    'risk_focused': {
        'objectives': ['max_drawdown', 'volatility', 'sharpe_ratio', 'sortino_ratio'],
        'minimize': [True, True, False, False],
        'weights': [0.3, 0.2, 0.25, 0.25],
        'normalization': 'minmax'
    },
    
    'comprehensive': {
        'objectives': ['mean_return', 'max_drawdown', 'sharpe_ratio', 'win_rate', 'profit_factor', 'calmar_ratio'],
        'minimize': [False, True, False, False, False, False],
        'weights': [0.2, 0.2, 0.2, 0.15, 0.15, 0.1],
        'normalization': 'minmax'
    }
}


def get_preset_objectives(preset_name: str) -> dict:
    """
    Get a predefined objective configuration.
    
    Args:
        preset_name (str): Name of the preset ('balanced', 'return_focused', 'risk_focused', 'comprehensive')
        
    Returns:
        objectives_config: Configuration dictionary
    """
    if preset_name not in PRESET_OBJECTIVES:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_OBJECTIVES.keys())}")
    
    return PRESET_OBJECTIVES[preset_name].copy()


# ##############################################################################
#           Walk-Forward Optimization (WFO) Utilities
# ##############################################################################

class WalkForwardOptimizer:
    """
    Walk-Forward Optimization coordinator that manages rolling window training
    and out-of-sample evaluation with cross-window objective aggregation.
    """
    
    def __init__(self, wfo_config: dict, evaluation_config: dict):
        """
        Initialize the WFO optimizer.
        
        Args:
            wfo_config: WFO configuration from experiment_config.yaml
            evaluation_config: Evaluation configuration including IS/OOS objectives
        """
        self.wfo_config = wfo_config
        self.evaluation_config = evaluation_config
        
        # Validate configuration
        self._validate_config()
        
        # Initialize evaluators for different objective contexts
        self.is_evaluator = None
        self.oos_evaluator = None
        self.aggregate_config = evaluation_config.get('oos_aggregate', {})
        
        # WFO tracking
        self.window_results = []  # Per-window results storage
        self.candidate_archives = {}  # Per-candidate cross-window results (objectives by window)
        self.candidate_chromosomes = {}  # candidate_id -> torch.Tensor (chromosome)
        
    def _validate_config(self):
        """Validate WFO configuration parameters."""
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
        """Get the in-sample evaluator."""
        if self.is_evaluator is None:
            is_config = self.evaluation_config.get('is_objectives', 
                                                  self.evaluation_config.get('objectives'))
            self.is_evaluator = MultiObjectiveFitnessEvaluator(is_config)
        return self.is_evaluator
    
    def get_oos_evaluator(self) -> MultiObjectiveFitnessEvaluator:
        """Get the out-of-sample evaluator."""
        if self.oos_evaluator is None:
            oos_config = self.evaluation_config.get('oos_objectives',
                                                   self.evaluation_config.get('objectives'))
            self.oos_evaluator = MultiObjectiveFitnessEvaluator(oos_config)
        return self.oos_evaluator
        
    def generate_windows(self, total_length: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate WFO window indices.
        
        Args:
            total_length: Total data length
            
        Returns:
            windows: List of (train_start, train_end, oos_start, oos_end) tuples
        """
        windows = []
        train_bars = self.wfo_config['train_bars']
        oos_bars = self.wfo_config['oos_bars']
        step_bars = self.wfo_config['step_bars']
        warmup_bars = self.wfo_config.get('warmup_bars', 0)
        embargo_bars = self.wfo_config.get('embargo_bars', 0)
        scheme = self.wfo_config.get('scheme', 'sliding')
        
        start_pos = warmup_bars
        
        while True:
            # Calculate window boundaries
            if scheme == 'expanding':
                train_start = warmup_bars
            else:  # sliding
                train_start = start_pos
                
            train_end = start_pos + train_bars
            oos_start = train_end + embargo_bars
            oos_end = oos_start + oos_bars
            
            # Check if we have enough data
            if oos_end > total_length:
                break
                
            windows.append((train_start, train_end, oos_start, oos_end))
            
            # Move to next window
            start_pos += step_bars
            
        return windows
    
    def record_window_result(self, window_idx: int, candidate_id: int,
                             oos_objectives: np.ndarray,
                             chromosome: Optional[torch.Tensor] = None):
        """
        Record OOS results for a candidate in a specific window.
        
        Args:
            window_idx: Window index
            candidate_id: Candidate identifier 
            oos_objectives: OOS objective vector for this candidate
            chromosome: Optional chromosome tensor to store for reconstruction
        """
        if candidate_id not in self.candidate_archives:
            self.candidate_archives[candidate_id] = {}
            
        self.candidate_archives[candidate_id][window_idx] = oos_objectives.copy()

        # Store chromosome once (CPU copy) for later reconstruction
        if chromosome is not None and candidate_id not in self.candidate_chromosomes:
            try:
                # Store on CPU to avoid GPU memory growth
                self.candidate_chromosomes[candidate_id] = chromosome.detach().to('cpu').clone()
            except Exception:
                # Best-effort; ignore if tensor-like not provided
                pass

    def get_candidate_chromosome(self, candidate_id: int) -> Optional[torch.Tensor]:
        """Return stored chromosome tensor for a candidate if available."""
        return self.candidate_chromosomes.get(candidate_id, None)
    
    def aggregate_candidate_objectives(self, candidate_id: int) -> np.ndarray:
        """
        Aggregate OOS objectives across all windows for a specific candidate.
        
        Args:
            candidate_id: Candidate identifier
            
        Returns:
            aggregated_objectives: Aggregated objective vector
        """
        if candidate_id not in self.candidate_archives:
            raise ValueError(f"No results found for candidate {candidate_id}")
        
        window_results = self.candidate_archives[candidate_id]
        window_objectives = np.array(list(window_results.values()))
        
        return aggregate_oos_objectives(
            window_objectives, 
            self.aggregate_config
        )
    
    def get_final_rankings(self) -> Tuple[List[int], np.ndarray]:
        """
        Get final candidate rankings based on aggregated OOS objectives.
        
        Returns:
            candidate_ids: Ranked list of candidate IDs (best first)
            aggregated_objectives: Aggregated objective matrix
        """
        if not self.candidate_archives:
            return [], np.array([])
        
        candidate_ids = list(self.candidate_archives.keys())
        aggregated_matrix = []
        
        for candidate_id in candidate_ids:
            agg_obj = self.aggregate_candidate_objectives(candidate_id)
            aggregated_matrix.append(agg_obj)
        
        aggregated_matrix = np.array(aggregated_matrix)
        
        # Create temporary evaluator for ranking
        agg_evaluator = MultiObjectiveFitnessEvaluator(self.aggregate_config)
        ranks, distances = agg_evaluator.pareto_rank(aggregated_matrix)
        
        # Sort by rank (ascending) and distance (descending)
        sort_indices = np.lexsort((-distances, ranks))
        ranked_candidates = [candidate_ids[i] for i in sort_indices]
        
        return ranked_candidates, aggregated_matrix[sort_indices]


def aggregate_oos_objectives(objectives_per_window: np.ndarray, 
                           aggregate_config: dict) -> np.ndarray:
    """
    Aggregate per-window OOS objective matrices using specified methods.
    
    Args:
        objectives_per_window: [num_windows, num_base_objectives] matrix of per-window OOS objectives
        aggregate_config: Configuration specifying aggregation methods
        
    Returns:
        aggregated_objectives: [num_aggregate_objectives] aggregated objective vector
    """
    if objectives_per_window.size == 0:
        return np.array([])
    
    methods = aggregate_config.get('methods', ['mean'] * objectives_per_window.shape[1])
    thresholds = aggregate_config.get('thresholds', {})
    base_minimize = aggregate_config.get('base_minimize', [False] * objectives_per_window.shape[1])
    
    aggregated = []
    
    for i, method in enumerate(methods):
        if i < objectives_per_window.shape[1]:
            # Standard aggregation on base objectives
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
                # Percentile (e.g., 'p95', 'p05')
                percentile = float(method[1:])
                result = np.percentile(valid_values, percentile)
            else:
                result = np.mean(valid_values)  # Default fallback
                
            aggregated.append(result)
        else:
            # Special aggregation methods (beyond base objectives)
            if method == 'hit_rate':
                # Consistency objective: fraction of windows meeting criteria
                result = calculate_consistency_rate(objectives_per_window, thresholds)
                aggregated.append(result)
            else:
                # Unknown method - use base minimize flag if available
                if i < len(base_minimize):
                    aggregated.append(1e9 if base_minimize[i] else -1e9)
                else:
                    aggregated.append(0.0)
    
    return np.array(aggregated)


def calculate_consistency_rate(objectives_per_window: np.ndarray, 
                             thresholds: dict) -> float:
    """
    Calculate consistency rate (hit rate) across windows.
    
    Args:
        objectives_per_window: [num_windows, num_objectives] per-window objectives
        thresholds: Threshold configuration (e.g., min_return, max_dd)
        
    Returns:
        consistency_rate: Fraction of windows meeting all criteria [0, 1]
    """
    if objectives_per_window.size == 0:
        return 0.0
    
    num_windows = objectives_per_window.shape[0]
    meets_criteria = np.ones(num_windows, dtype=bool)
    
    # Apply return threshold (typically mean_return > min_return)
    min_return = thresholds.get('min_return', 0.0)
    if objectives_per_window.shape[1] > 0:  # Assume first column is mean_return
        return_values = objectives_per_window[:, 0]
        valid_mask = ~np.isin(return_values, [-1e9, 1e9]) & ~np.isnan(return_values)
        meets_criteria &= (return_values >= min_return) & valid_mask
    
    # Apply drawdown threshold (typically max_drawdown < max_dd)  
    max_dd = thresholds.get('max_dd', 0.3)
    if objectives_per_window.shape[1] > 1:  # Assume second column is max_drawdown
        dd_values = objectives_per_window[:, 1]
        valid_mask = ~np.isin(dd_values, [-1e9, 1e9]) & ~np.isnan(dd_values)
        meets_criteria &= (dd_values <= max_dd) & valid_mask
    
    return np.mean(meets_criteria)


def generate_wfo_windows(total_length: int, train_bars: int, oos_bars: int, 
                        step_bars: int, warmup_bars: int = 0, 
                        embargo_bars: int = 0, scheme: str = 'sliding') -> List[Tuple[int, int, int, int]]:
    """
    Generate Walk-Forward Optimization windows.
    
    Args:
        total_length: Total data length
        train_bars: In-sample window size
        oos_bars: Out-of-sample window size  
        step_bars: Step size between windows
        warmup_bars: Warmup period at start of data
        embargo_bars: Gap between IS and OOS to prevent leakage
        scheme: 'sliding' or 'expanding' window scheme
        
    Returns:
        windows: List of (train_start, train_end, oos_start, oos_end) tuples
    """
    windows = []
    start_pos = warmup_bars
    
    while True:
        # Calculate window boundaries
        if scheme == 'expanding':
            train_start = warmup_bars
        else:  # sliding
            train_start = start_pos
            
        train_end = start_pos + train_bars
        oos_start = train_end + embargo_bars
        oos_end = oos_start + oos_bars
        
        # Check if we have enough data
        if oos_end > total_length:
            break
            
        windows.append((train_start, train_end, oos_start, oos_end))
        
        # Move to next window
        start_pos += step_bars
        
    return windows
