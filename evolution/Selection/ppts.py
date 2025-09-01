# evolution/Selection/ppts.py
import torch
from .base import BaseSelection

# Import CUDA acceleration
try:
    import gatree_cuda_compat as gatree_cuda
    CUDA_AVAILABLE = gatree_cuda.ppts_select_elites is not None
except ImportError:
    CUDA_AVAILABLE = False
    gatree_cuda = None

class PPTSSelection(BaseSelection):
    """
    Parameterless Batched Pareto Tournament Selection (PPTS) - CUDA accelerated.
    
    High-performance multi-objective selection using CUDA kernels for massive populations.
    Achieves 1400x+ speedup over Python implementations.
    """
    
    def __init__(self, chunk_size: int = 8192, generator: torch.Generator = None):
        """
        Initialize PPTS selection with CUDA acceleration.
        
        Args:
            chunk_size (int): Tournament batch size for memory management (-1 = auto)
            generator (torch.Generator): Random number generator for reproducibility
        """
        self.chunk_size = chunk_size if chunk_size > 0 else -1
        self.generator = generator
        
        if not CUDA_AVAILABLE:
            raise RuntimeError("PPTS requires CUDA acceleration. Please build gatree_selection extension.")

    def _get_rng_seed(self) -> int:
        """Get RNG seed from generator if available."""
        if self.generator is not None:
            try:
                return self.generator.initial_seed()
            except:
                pass
        return 0

    def select_elites(self, fitness_or_objectives: torch.Tensor, num_selects: int) -> torch.Tensor:
        """
        Select elite individuals using CUDA-accelerated PPTS.
        
        Args:
            fitness_or_objectives: [N] for single-objective or [N, M] for multi-objective
            num_selects: Number to select
            
        Returns:
            selected_indices: [num_selects] tensor of indices
        """
        # Ensure objectives are on GPU
        if not fitness_or_objectives.is_cuda:
            raise RuntimeError("PPTS selection requires CUDA tensors. Move data to GPU first.")
        
        # Convert to multi-objective format if needed
        if fitness_or_objectives.dim() == 1:
            # Single-objective: convert to multi-objective (negate for minimization)
            objectives = (-fitness_or_objectives).unsqueeze(-1)
        else:
            # Multi-objective: assume minimization format
            objectives = fitness_or_objectives
            
        N = objectives.shape[0]
        if num_selects >= N:
            return torch.arange(N, device=objectives.device)
        
        # Call CUDA implementation
        return gatree_cuda.ppts_select_elites(
            objectives=objectives,
            num_elites=num_selects,
            chunk_size=self.chunk_size,
            rng_seed=self._get_rng_seed()
        )
    
    def pick_parents(self, fitness_or_objectives: torch.Tensor, num_parents: int) -> torch.Tensor:
        """
        Select parent individuals using CUDA-accelerated PPTS.
        
        Args:
            fitness_or_objectives: [N] for single-objective or [N, M] for multi-objective
            num_parents: Number of parents to select
            
        Returns:
            parent_indices: [num_parents] tensor of indices
        """
        # Ensure objectives are on GPU
        if not fitness_or_objectives.is_cuda:
            raise RuntimeError("PPTS selection requires CUDA tensors. Move data to GPU first.")
        
        # Convert to multi-objective format if needed
        if fitness_or_objectives.dim() == 1:
            # Single-objective: convert to multi-objective (negate for minimization)
            objectives = (-fitness_or_objectives).unsqueeze(-1)
        else:
            # Multi-objective: assume minimization format
            objectives = fitness_or_objectives
            
        # Call CUDA implementation (duplicates allowed for parents)
        return gatree_cuda.ppts_pick_parents(
            objectives=objectives,
            num_parents=num_parents,
            chunk_size=self.chunk_size,
            rng_seed=self._get_rng_seed()
        )
