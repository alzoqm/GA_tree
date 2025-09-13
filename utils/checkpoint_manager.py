# /utils/checkpoint_manager.py
"""
Checkpoint Manager for Training Resumption

This module provides utilities for managing training checkpoints, enabling
robust resumption of interrupted GA tree training sessions.
"""

import os
import json
import torch
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path


class CheckpointManager:
    """Manages training checkpoints for resumption capability."""

    def __init__(self, output_dir: str, config: Dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.config = config
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.main_checkpoint_file = self.checkpoint_dir / "main_checkpoint.json"
        self.config_hash = self._compute_config_hash(config)
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute a hash of the configuration for compatibility checking."""
        # Create a deterministic string representation of the config
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_data_fingerprint(self, data_path: str) -> Optional[str]:
        """Generate a fingerprint of the data file for validation."""
        try:
            if os.path.exists(data_path):
                stat = os.stat(data_path)
                return f"{stat.st_size}_{stat.st_mtime}"
            return None
        except Exception:
            return None
    
    def save_main_checkpoint(self, 
                           current_window: int,
                           total_windows: int,
                           completed_generations: int,
                           wandb_run_id: Optional[str] = None,
                           wfo_state: Optional[Dict] = None,
                           **extra_state) -> None:
        """Save the main training checkpoint."""
        checkpoint_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "config_hash": self.config_hash,
            "current_window": current_window,
            "total_windows": total_windows,
            "completed_generations": completed_generations,
            "wandb_run_id": wandb_run_id,
            "wfo_state": wfo_state or {},
            "data_fingerprint": self._get_data_fingerprint(
                os.path.join(self.config.get('environment', {}).get('data_dir', ''),
                           self.config.get('data', {}).get('final_features_path', ''))
            ),
            **extra_state
        }
        
        try:
            with open(self.main_checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logging.info(f"Main checkpoint saved: window {current_window}/{total_windows}, "
                        f"generations {completed_generations}")
        except Exception as e:
            logging.error(f"Failed to save main checkpoint: {e}")
    
    def load_main_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load and validate the main checkpoint."""
        if not self.main_checkpoint_file.exists():
            return None
            
        try:
            with open(self.main_checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Validate config compatibility
            if checkpoint_data.get("config_hash") != self.config_hash:
                logging.warning("Configuration has changed since checkpoint was created. "
                              "Proceeding with caution.")
            
            # Validate data fingerprint
            current_fingerprint = self._get_data_fingerprint(
                os.path.join(self.config.get('environment', {}).get('data_dir', ''),
                           self.config.get('data', {}).get('final_features_path', ''))
            )
            if (checkpoint_data.get("data_fingerprint") != current_fingerprint and 
                current_fingerprint is not None):
                logging.warning("Data files may have changed since checkpoint was created.")
            
            logging.info(f"Loaded main checkpoint from {checkpoint_data.get('timestamp')}")
            return checkpoint_data
            
        except Exception as e:
            logging.error(f"Failed to load main checkpoint: {e}")
            return None
    
    def get_window_checkpoint_dir(self, window_idx: int) -> Path:
        """Get the checkpoint directory for a specific window."""
        return self.checkpoint_dir / f"wf_window_{window_idx}"
    
    def save_window_checkpoint(self, 
                             window_idx: int,
                             generation: int,
                             population_state: Dict,
                             best_chromosomes: Optional[torch.Tensor] = None,
                             best_objectives: Optional[torch.Tensor] = None,
                             best_node_stats: Optional[Dict[str, torch.Tensor]] = None,
                             evolution_state: Optional[Dict] = None,
                             **extra_state) -> None:
        """Save checkpoint for a specific window and generation."""
        window_dir = self.get_window_checkpoint_dir(window_idx)
        window_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare population state for JSON serialization (separate tensors)
        json_safe_population_state = {}
        tensor_files = {}
        
        for key, value in population_state.items():
            if isinstance(value, torch.Tensor):
                # Save tensor separately and store filename reference
                tensor_file = window_dir / f"population_{key}_gen_{generation}.pth"
                torch.save(value, tensor_file)
                tensor_files[key] = tensor_file.name
                json_safe_population_state[key] = f"__TENSOR_FILE__{tensor_file.name}"
            else:
                json_safe_population_state[key] = value
        
        # Save the generation state
        gen_checkpoint = {
            "window_idx": window_idx,
            "generation": generation,
            "timestamp": datetime.utcnow().isoformat(),
            "population_state": json_safe_population_state,
            "evolution_state": evolution_state or {},
            "tensor_files": tensor_files,
            **extra_state
        }
        
        gen_file = window_dir / f"generation_{generation}.json"
        try:
            with open(gen_file, 'w') as f:
                json.dump(gen_checkpoint, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save generation checkpoint: {e}")
        
        # Save best chromosomes and objectives if provided
        if best_chromosomes is not None:
            torch.save(best_chromosomes, window_dir / f"best_chromosomes_gen_{generation}.pth")
        
        if best_objectives is not None:
            torch.save(best_objectives, window_dir / f"best_objectives_gen_{generation}.pth")
        
        # Save best node stats if provided
        if best_node_stats is not None:
            try:
                # Move to CPU for portability
                cpu_stats = {}
                for k, v in best_node_stats.items():
                    if isinstance(v, torch.Tensor):
                        cpu_stats[k] = v.detach().cpu()
                    else:
                        cpu_stats[k] = v
                torch.save(cpu_stats, window_dir / f"best_node_stats_gen_{generation}.pth")
            except Exception as e:
                logging.warning(f"Failed to save best_node_stats: {e}")
    
    def load_window_checkpoint(self, window_idx: int, generation: int) -> Optional[Dict[str, Any]]:
        """Load checkpoint for a specific window and generation."""
        window_dir = self.get_window_checkpoint_dir(window_idx)
        gen_file = window_dir / f"generation_{generation}.json"
        
        if not gen_file.exists():
            return None
            
        try:
            with open(gen_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Restore tensors from separate files
            population_state = checkpoint_data.get("population_state", {})
            tensor_files = checkpoint_data.get("tensor_files", {})
            
            for key, value in population_state.items():
                if isinstance(value, str) and value.startswith("__TENSOR_FILE__"):
                    tensor_filename = value.replace("__TENSOR_FILE__", "")
                    tensor_path = window_dir / tensor_filename
                    if tensor_path.exists():
                        population_state[key] = torch.load(tensor_path, weights_only=False)
                    else:
                        logging.warning(f"Tensor file not found: {tensor_path}")
            
            # Load best chromosomes and objectives if they exist
            best_chromosomes_file = window_dir / f"best_chromosomes_gen_{generation}.pth"
            if best_chromosomes_file.exists():
                checkpoint_data["best_chromosomes"] = torch.load(best_chromosomes_file, weights_only=False)
            
            best_objectives_file = window_dir / f"best_objectives_gen_{generation}.pth"
            if best_objectives_file.exists():
                checkpoint_data["best_objectives"] = torch.load(best_objectives_file, weights_only=False)
            
            # Load best node stats if it exists
            best_node_stats_file = window_dir / f"best_node_stats_gen_{generation}.pth"
            if best_node_stats_file.exists():
                checkpoint_data["best_node_stats"] = torch.load(best_node_stats_file, weights_only=False)
            
            return checkpoint_data
            
        except Exception as e:
            logging.error(f"Failed to load window checkpoint: {e}")
            return None
    
    def find_latest_checkpoint(self) -> Optional[Tuple[int, int]]:
        """Find the latest completed checkpoint (window_idx, generation)."""
        main_checkpoint = self.load_main_checkpoint()
        if not main_checkpoint:
            return None
        
        current_window = main_checkpoint.get("current_window", 0)
        
        # Look for the latest generation in the current window
        window_dir = self.get_window_checkpoint_dir(current_window)
        if not window_dir.exists():
            return None
        
        # Find the highest generation number
        max_generation = -1
        for gen_file in window_dir.glob("generation_*.json"):
            try:
                gen_num = int(gen_file.stem.split('_')[1])
                max_generation = max(max_generation, gen_num)
            except (ValueError, IndexError):
                continue
        
        if max_generation >= 0:
            return current_window, max_generation
        
        return None
    
    def get_completed_windows(self) -> List[int]:
        """Get list of completed window indices."""
        completed = []
        for window_dir in self.checkpoint_dir.glob("wf_window_*"):
            try:
                window_idx = int(window_dir.name.split('_')[2])
                # Check if window has any generation checkpoints
                if any(window_dir.glob("generation_*.json")):
                    completed.append(window_idx)
            except (ValueError, IndexError):
                continue
        
        return sorted(completed)
    
    def cleanup_old_checkpoints(self, keep_last_n_generations: int = 5):
        """Clean up old generation checkpoints to save disk space."""
        for window_dir in self.checkpoint_dir.glob("wf_window_*"):
            gen_files = list(window_dir.glob("generation_*.json"))
            
            if len(gen_files) <= keep_last_n_generations:
                continue
            
            # Sort by generation number
            gen_files.sort(key=lambda x: int(x.stem.split('_')[1]))
            
            # Remove older checkpoints
            for gen_file in gen_files[:-keep_last_n_generations]:
                try:
                    gen_num = gen_file.stem.split('_')[1]
                    gen_file.unlink()
                    
                    # Also remove associated tensor files
                    for tensor_file in window_dir.glob(f"*_gen_{gen_num}.pth"):
                        tensor_file.unlink()
                        
                except Exception as e:
                    logging.warning(f"Failed to cleanup checkpoint {gen_file}: {e}")
    
    def is_resumable(self) -> bool:
        """Check if there's a valid checkpoint to resume from."""
        return self.find_latest_checkpoint() is not None
    
    def clear_checkpoints(self):
        """Clear all checkpoints - use with caution!"""
        try:
            import shutil
            if self.checkpoint_dir.exists():
                shutil.rmtree(self.checkpoint_dir)
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info("All checkpoints cleared")
        except Exception as e:
            logging.error(f"Failed to clear checkpoints: {e}")


def detect_resumable_training(output_dir: str, config: Dict[str, Any]) -> Optional[CheckpointManager]:
    """
    Detect if there's a resumable training session in the output directory.
    
    Returns CheckpointManager if resumable, None otherwise.
    """
    try:
        checkpoint_manager = CheckpointManager(output_dir, config)
        if checkpoint_manager.is_resumable():
            return checkpoint_manager
        return None
    except Exception as e:
        logging.error(f"Error detecting resumable training: {e}")
        return None
