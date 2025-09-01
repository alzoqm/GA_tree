# evolution/Selection/ppts.py
import torch
import math
from .base import BaseSelection

# Import CUDA acceleration if available
try:
    import gatree_cuda_compat as gatree_cuda
    CUDA_AVAILABLE = gatree_cuda.ppts_select_elites is not None
except ImportError:
    CUDA_AVAILABLE = False

class PPTSSelection(BaseSelection):
    """
    Parameterless Batched Pareto Tournament Selection (PPTS) with ε-lexicase tiebreaking.
    
    This implementation provides multi-objective selection without fixed weights, avoiding O(N²) 
    non-dominated sorting, and scaling to 100k+ individuals through GPU vectorization.
    
    Core Algorithm:
    1. Replace global sorting with many small parallel tournaments
    2. Select non-dominated candidates by Pareto dominance (scale-invariant, no weights)
    3. If multiple remain, tiebreak via ε-lexicase using robust, data-driven ε per objective
    4. Entire pipeline runs as large batched tensor ops on GPU
    """
    
    def __init__(self, chunk_size: int = 8192, generator: torch.Generator = None, use_cuda: bool = True):
        """
        Initialize PPTS selection.
        
        Args:
            chunk_size (int): Maximum number of tournaments to process in parallel (memory control)
            generator (torch.Generator): Random number generator for reproducibility
            use_cuda (bool): Whether to use CUDA acceleration if available (default: True)
        """
        self.chunk_size = chunk_size
        self.generator = generator
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        
        if use_cuda and not CUDA_AVAILABLE:
            print("Warning: CUDA acceleration requested but not available, falling back to Python implementation")

    def _get_device_generator(self, device: torch.device | str | None):
        """Return a torch.Generator compatible with the given device.

        If a generator was provided at construction time but lives on a different
        device than requested here, create a new generator on the target device
        using the same initial seed for reproducibility. If no generator was
        provided, return None to use PyTorch's default RNG.
        """
        if device is None:
            return self.generator
        if self.generator is None:
            return None
        target = torch.device(device)
        # Some torch.Generator implementations expose `.device`
        gen_device = getattr(self.generator, 'device', torch.device('cpu'))
        if torch.device(gen_device).type == target.type:
            return self.generator
        # Create a new generator on the target device with the same seed
        try:
            seed = self.generator.initial_seed()
        except Exception:
            # Fallback: let PyTorch choose a seed if not available
            return None
        new_gen = torch.Generator(device=target)
        new_gen.manual_seed(seed)
        return new_gen
    
    def _auto_tournament_size(self, population_size: int) -> int:
        """Auto-size tournament using k ≈ ceil(log2(N)) rule."""
        return max(2, math.ceil(math.log2(population_size)))
    
    def _compute_pareto_dominance(self, objectives: torch.Tensor) -> torch.Tensor:
        """
        Compute Pareto dominance relationships within tournaments.
        
        Args:
            objectives: [B, k, M] tensor where B=batch_size, k=tournament_size, M=num_objectives
            
        Returns:
            non_dominated_mask: [B, k] boolean tensor indicating non-dominated individuals
        """
        B, k, M = objectives.shape
        
        # Pairwise comparisons: [B, k, k, M]
        obj_i = objectives.unsqueeze(2)  # [B, k, 1, M]
        obj_j = objectives.unsqueeze(1)  # [B, 1, k, M]
        
        # i dominates j if: i <= j in all objectives AND i < j in at least one
        le = (obj_i <= obj_j)  # [B, k, k, M] - less than or equal in all objectives
        lt = (obj_i < obj_j)   # [B, k, k, M] - strictly less in some objective
        
        # i dominates j: all(i <= j) AND any(i < j)
        dominates = le.all(dim=-1) & lt.any(dim=-1)  # [B, k, k]
        
        # Individual i is dominated if any j dominates i
        dominated_by_any = dominates.transpose(-2, -1).any(dim=-1)  # [B, k]
        
        # Non-dominated mask
        non_dominated_mask = ~dominated_by_any  # [B, k]
        
        return non_dominated_mask
    
    def _epsilon_lexicase_tiebreak(self, objectives: torch.Tensor, candidates_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform ε-lexicase tiebreaking on non-dominated candidates.
        
        Args:
            objectives: [B, k, M] tournament objectives
            candidates_mask: [B, k] boolean mask of candidates to consider
            
        Returns:
            winners_k: [B] tensor of winner indices in original k-space for each tournament
        """
        B, k, M = objectives.shape
        device = objectives.device
        
        # Extract candidates for each tournament
        # For tournaments with no candidates, we'll handle separately
        num_candidates = candidates_mask.sum(dim=-1)  # [B]
        max_candidates = num_candidates.max().item()

        if max_candidates == 0:
            # No candidates anywhere - return first index as fallback
            return torch.zeros(B, dtype=torch.long, device=device)

        # Create dense candidate matrix [B, max_candidates, M] and index map back to original k
        candidates = torch.zeros(B, max_candidates, M, device=device)
        valid_mask = torch.zeros(B, max_candidates, dtype=torch.bool, device=device)
        index_map = torch.zeros(B, max_candidates, dtype=torch.long, device=device)

        for b in range(B):
            candidate_indices = torch.where(candidates_mask[b])[0]
            n_cand = len(candidate_indices)
            if n_cand > 0:
                candidates[b, :n_cand] = objectives[b, candidate_indices]
                valid_mask[b, :n_cand] = True
                index_map[b, :n_cand] = candidate_indices  # map compacted -> original k
        
        # Robust ε calculation using MAD (Median Absolute Deviation)
        # For each tournament, compute per-objective epsilon
        eps = torch.zeros(B, 1, M, device=device)
        
        for b in range(B):
            if num_candidates[b] > 0:
                cand_obj = candidates[b, valid_mask[b]]  # [n_cand, M]
                if len(cand_obj) > 1:
                    # Median and MAD per objective
                    med = torch.median(cand_obj, dim=0)[0]  # [M]
                    mad = torch.median(torch.abs(cand_obj - med.unsqueeze(0)), dim=0)[0]  # [M]
                    eps[b, 0] = mad
                # If only one candidate or MAD is 0, eps remains 0
        
        # Random objective permutation per tournament
        gen = self._get_device_generator(device)
        noise = torch.randn(B, M, device=device, generator=gen)
        perm = torch.argsort(noise, dim=-1)  # [B, M]
        
        # ε-lexicase filtering
        survivors = valid_mask.clone()  # [B, max_candidates]
        
        for obj_idx in range(M):
            # Current objective for each tournament (permuted)
            current_obj_indices = perm[:, obj_idx]  # [B]
            
            for b in range(B):
                if survivors[b].sum() <= 1:
                    continue  # Already down to one or zero survivors
                
                current_obj = current_obj_indices[b].item()
                survivor_mask = survivors[b]
                survivor_values = candidates[b, survivor_mask, current_obj]
                
                if len(survivor_values) > 1:
                    min_val = survivor_values.min()
                    threshold = min_val + eps[b, 0, current_obj]
                    keep = survivor_values <= threshold
                    
                    # Update survivors for this tournament
                    survivor_indices = torch.where(survivor_mask)[0]
                    new_survivors = torch.zeros_like(survivor_mask)
                    new_survivors[survivor_indices[keep]] = True
                    survivors[b] = new_survivors
        
        # Select final winners (random choice among remaining survivors)
        winners_k = torch.zeros(B, dtype=torch.long, device=device)

        for b in range(B):
            survivor_indices = torch.where(survivors[b])[0]
            if len(survivor_indices) > 0:
                # Random choice among survivors
                choice_idx = torch.randint(0, len(survivor_indices), (1,), 
                                        generator=gen, device=device)[0]
                compact_idx = survivor_indices[choice_idx]
                # Map compacted candidate index back to original k index
                winners_k[b] = index_map[b, compact_idx]
            # else: winners_k[b] remains 0 (fallback -> first participant)
        
        return winners_k
    
    def _run_tournaments(self, objectives: torch.Tensor, num_tournaments: int, k: int) -> torch.Tensor:
        """
        Run batched tournaments with chunking for memory management.
        
        Args:
            objectives: [N, M] population objectives (minimization)
            num_tournaments: Number of tournaments to run
            k: Tournament size
            
        Returns:
            winner_indices: [num_tournaments] absolute indices of winners
        """
        N, M = objectives.shape
        device = objectives.device
        
        all_winners = []
        
        # Process tournaments in chunks
        for start_idx in range(0, num_tournaments, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_tournaments)
            B = end_idx - start_idx
            
            # Sample tournament participants [B, k]
            gen = self._get_device_generator(device)
            tournament_indices = torch.randint(0, N, (B, k), 
                                             generator=gen, device=device)
            
            # Gather objectives for tournaments [B, k, M]
            tournament_objectives = objectives[tournament_indices]  # [B, k, M]
            
            # Compute Pareto dominance
            non_dominated_mask = self._compute_pareto_dominance(tournament_objectives)
            
            # Count non-dominated candidates per tournament
            non_dom_counts = non_dominated_mask.sum(dim=-1)
            
            # Tournaments with exactly one non-dominated candidate
            single_winner_mask = (non_dom_counts == 1)
            
            # Initialize winners
            chunk_winners = torch.zeros(B, dtype=torch.long, device=device)
            
            # Handle single winners
            if single_winner_mask.any():
                single_winner_tournaments = torch.where(single_winner_mask)[0]
                for b_idx in single_winner_tournaments:
                    winner_pos = torch.where(non_dominated_mask[b_idx])[0][0]
                    chunk_winners[b_idx] = tournament_indices[b_idx, winner_pos]
            
            # Handle ties with ε-lexicase
            tie_mask = (non_dom_counts > 1)
            if tie_mask.any():
                tie_tournaments = torch.where(tie_mask)[0]
                
                # Extract tied tournaments
                tied_objectives = tournament_objectives[tie_tournaments]  # [n_ties, k, M]
                tied_candidates_mask = non_dominated_mask[tie_tournaments]  # [n_ties, k]
                tied_indices = tournament_indices[tie_tournaments]  # [n_ties, k]
                
                # Run ε-lexicase on tied tournaments
                tie_winners_pos = self._epsilon_lexicase_tiebreak(tied_objectives, tied_candidates_mask)
                
                # Convert to absolute indices
                for i, b_idx in enumerate(tie_tournaments):
                    winner_pos = tie_winners_pos[i]
                    chunk_winners[b_idx] = tied_indices[i, winner_pos]
            
            # Handle no winners (fallback to random)
            no_winner_mask = (non_dom_counts == 0)
            if no_winner_mask.any():
                no_winner_tournaments = torch.where(no_winner_mask)[0]
                for b_idx in no_winner_tournaments:
                    random_pos = torch.randint(0, k, (1,), generator=gen, device=device)[0]
                    chunk_winners[b_idx] = tournament_indices[b_idx, random_pos]
            
            all_winners.append(chunk_winners)
        
        return torch.cat(all_winners, dim=0)
    
    
    # Override BaseSelection interface for multi-objective compatibility
    def select_elites(self, fitness_or_objectives: torch.Tensor, num_selects: int) -> torch.Tensor:
        """
        Unified interface supporting both single and multi-objective optimization.
        
        Args:
            fitness_or_objectives: [N] for single-objective or [N, M] for multi-objective
            num_selects: Number to select
            
        Returns:
            selected_indices: [num_selects] tensor of indices
        """
        if fitness_or_objectives.dim() == 1:
            # Single-objective: convert to multi-objective format
            # Higher fitness is better, so negate for minimization
            objectives = (-fitness_or_objectives).unsqueeze(-1)
        else:
            # Multi-objective: assume minimization format
            objectives = fitness_or_objectives
            
        return self._select_elites_multi(objectives, num_selects)
    
    def pick_parents(self, fitness_or_objectives: torch.Tensor, num_parents: int) -> torch.Tensor:
        """
        Unified interface supporting both single and multi-objective optimization.
        
        Args:
            fitness_or_objectives: [N] for single-objective or [N, M] for multi-objective
            num_parents: Number of parents to select
            
        Returns:
            parent_indices: [num_parents] tensor of indices
        """
        if fitness_or_objectives.dim() == 1:
            # Single-objective: convert to multi-objective format
            objectives = (-fitness_or_objectives).unsqueeze(-1)
        else:
            # Multi-objective: assume minimization format
            objectives = fitness_or_objectives
            
        return self._pick_parents_multi(objectives, num_parents)
    
    def _select_elites_multi(self, objectives: torch.Tensor, num_elites: int) -> torch.Tensor:
        """Internal multi-objective elite selection."""
        N, M = objectives.shape
        
        if num_elites >= N:
            return torch.arange(N, device=objectives.device)
        
        # Use CUDA acceleration if available and objectives are on GPU
        if self.use_cuda and objectives.is_cuda:
            try:
                # Get seed from generator if available
                rng_seed = 0
                if self.generator is not None:
                    try:
                        rng_seed = self.generator.initial_seed()
                    except:
                        rng_seed = 0
                
                # Call CUDA implementation directly
                return gatree_cuda.ppts_select_elites(
                    objectives=objectives,
                    num_elites=num_elites,
                    chunk_size=self.chunk_size if self.chunk_size > 0 else -1,
                    rng_seed=rng_seed
                )
            except Exception as e:
                print(f"CUDA elite selection failed, falling back to Python: {e}")
                # Fall through to Python implementation
        
        # Python fallback implementation
        # Use larger tournament size for elites to increase selection pressure
        k = min(self._auto_tournament_size(N) * 2, N)
        
        # Run tournaments for elites with potential duplicates
        num_tournaments = num_elites * 3  # Oversample to ensure uniqueness
        winners = self._run_tournaments(objectives, num_tournaments, k)

        # Remove duplicates and take top num_elites
        unique_winners = torch.unique(winners)

        if len(unique_winners) >= num_elites:
            return unique_winners[:num_elites]
        else:
            # If we don't have enough unique winners, supplement with additional tournaments
            needed = num_elites - len(unique_winners)
            additional_tournaments = needed * 2
            additional_winners = self._run_tournaments(objectives, additional_tournaments, k)
            
            # Combine and remove duplicates
            all_winners = torch.cat([unique_winners, additional_winners])
            final_unique = torch.unique(all_winners)
            if len(final_unique) >= num_elites:
                return final_unique[:num_elites]
            # Still short: fill deterministically from remaining indices
            missing = num_elites - len(final_unique)
            device = objectives.device
            all_indices = torch.arange(N, device=device)
            # Build mask of already selected
            selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
            selected_mask[final_unique] = True
            remaining = all_indices[~selected_mask]
            if len(remaining) > 0:
                # If fewer remaining than needed, just take all remaining and then repeat from start
                take = min(missing, len(remaining))
                filler = remaining[:take]
                combined = torch.cat([final_unique, filler])
                if len(combined) < num_elites:
                    # As an absolute fallback (rare), repeat from beginning to reach exact size
                    repeat_needed = num_elites - len(combined)
                    combined = torch.cat([combined, final_unique[:repeat_needed]])
                return combined[:num_elites]
            else:
                # No remaining (shouldn't happen unless N==0); fallback to return what we have
                return final_unique
    
    def _pick_parents_multi(self, objectives: torch.Tensor, num_parents: int) -> torch.Tensor:
        """Internal multi-objective parent selection."""
        N, M = objectives.shape
        
        # Use CUDA acceleration if available and objectives are on GPU
        if self.use_cuda and objectives.is_cuda:
            try:
                # Get seed from generator if available
                rng_seed = 0
                if self.generator is not None:
                    try:
                        rng_seed = self.generator.initial_seed()
                    except:
                        rng_seed = 0
                
                # Call CUDA implementation directly
                return gatree_cuda.ppts_pick_parents(
                    objectives=objectives,
                    num_parents=num_parents,
                    chunk_size=self.chunk_size if self.chunk_size > 0 else -1,
                    rng_seed=rng_seed
                )
            except Exception as e:
                print(f"CUDA parent selection failed, falling back to Python: {e}")
                # Fall through to Python implementation
        
        # Python fallback implementation
        # Standard tournament size for parent selection
        k = self._auto_tournament_size(N)
        
        # Run tournaments (duplicates allowed for parents)
        return self._run_tournaments(objectives, num_parents, k)
