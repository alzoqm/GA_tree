#!/usr/bin/env python3
"""
Harsh validation test for CUDA delete node mutation.
Tests edge cases, stress conditions, and invariant preservation under extreme conditions.
"""

import torch
import random
import numpy as np
from typing import Dict, Any, List, Tuple

from models.constants import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, COL_PARAM_1, COL_PARAM_2, COL_PARAM_3, COL_PARAM_4,
    NODE_TYPE_UNUSED, NODE_TYPE_DECISION, NODE_TYPE_ROOT_BRANCH, NODE_TYPE_ACTION,
    ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT,
    COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_FEAT, COMP_TYPE_FEAT_BOOL,
    OP_GTE, OP_LTE, ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_CLOSE_ALL
)

try:
    import gatree_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: gatree_cuda not available. Skipping CUDA tests.")

from evolution.Mutation.delete_node import DeleteNodeMutation, _validate_tree_constraints


def create_complex_tree(device='cuda', max_nodes=128, depth=6) -> torch.Tensor:
    """Create a complex tree with multiple branches and types."""
    trees = torch.zeros((1, max_nodes, 7), dtype=torch.float32, device=device)
    node_idx = 0
    
    # Root branches
    for root_type in [ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT]:
        trees[0, node_idx, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
        trees[0, node_idx, COL_PARENT_IDX] = -1
        trees[0, node_idx, COL_DEPTH] = 0
        trees[0, node_idx, COL_PARAM_1] = root_type
        node_idx += 1
    
    # Build tree recursively
    def add_subtree(parent_idx: int, current_depth: int, max_depth: int, branch_factor: int = 2):
        nonlocal node_idx
        if current_depth >= max_depth or node_idx >= max_nodes - 1:
            return
        
        parent_depth = int(trees[0, parent_idx, COL_DEPTH].item())
        
        # Add decision nodes
        children_added = 0
        for i in range(branch_factor):
            if node_idx >= max_nodes - 1:
                break
            
            if current_depth == max_depth - 1:
                # Add action node at leaf
                trees[0, node_idx, COL_NODE_TYPE] = NODE_TYPE_ACTION
                trees[0, node_idx, COL_PARENT_IDX] = parent_idx
                trees[0, node_idx, COL_DEPTH] = parent_depth + 1
                trees[0, node_idx, COL_PARAM_1] = random.choice([ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_CLOSE_ALL])
                trees[0, node_idx, COL_PARAM_2] = random.random()
                trees[0, node_idx, COL_PARAM_3] = random.uniform(1.0, 3.0)
                node_idx += 1
                children_added += 1
            else:
                # Add decision node
                trees[0, node_idx, COL_NODE_TYPE] = NODE_TYPE_DECISION
                trees[0, node_idx, COL_PARENT_IDX] = parent_idx
                trees[0, node_idx, COL_DEPTH] = parent_depth + 1
                trees[0, node_idx, COL_PARAM_1] = random.choice([COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_FEAT, COMP_TYPE_FEAT_BOOL])
                trees[0, node_idx, COL_PARAM_2] = random.choice([OP_GTE, OP_LTE])
                trees[0, node_idx, COL_PARAM_3] = random.randint(0, 50)
                trees[0, node_idx, COL_PARAM_4] = random.random()
                
                child_idx = node_idx
                node_idx += 1
                children_added += 1
                
                # Recursively add subtree
                add_subtree(child_idx, current_depth + 1, max_depth, max(1, branch_factor - 1))
    
    # Add subtrees to each root
    for root_idx in range(3):
        if node_idx < max_nodes - 1:
            add_subtree(root_idx, 1, depth)
    
    return trees


def create_pathological_tree(device='cuda', max_nodes=64) -> torch.Tensor:
    """Create a tree designed to test edge cases."""
    trees = torch.zeros((1, max_nodes, 7), dtype=torch.float32, device=device)
    
    # Root
    trees[0, 0, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
    trees[0, 0, COL_PARENT_IDX] = -1
    trees[0, 0, COL_DEPTH] = 0
    trees[0, 0, COL_PARAM_1] = ROOT_BRANCH_LONG
    
    # Create a chain of decision nodes (linear tree)
    for i in range(1, min(20, max_nodes - 1)):
        trees[0, i, COL_NODE_TYPE] = NODE_TYPE_DECISION
        trees[0, i, COL_PARENT_IDX] = i - 1
        trees[0, i, COL_DEPTH] = i
        trees[0, i, COL_PARAM_1] = COMP_TYPE_FEAT_NUM
        trees[0, i, COL_PARAM_2] = OP_GTE
        trees[0, i, COL_PARAM_3] = i
        trees[0, i, COL_PARAM_4] = 0.5
    
    # Add action at the end
    final_idx = min(19, max_nodes - 1)
    trees[0, final_idx, COL_NODE_TYPE] = NODE_TYPE_ACTION
    trees[0, final_idx, COL_PARENT_IDX] = final_idx - 1
    trees[0, final_idx, COL_DEPTH] = final_idx
    trees[0, final_idx, COL_PARAM_1] = ACTION_NEW_LONG
    
    return trees


def test_stress_large_batch():
    """Test with large batches and high deletion rates."""
    print("Testing stress conditions with large batches...")
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA not available")
        return True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    max_nodes = 64
    
    # Create batch of complex trees
    trees_list = []
    for b in range(batch_size):
        tree = create_complex_tree(device=device, max_nodes=max_nodes, depth=5)
        trees_list.append(tree[0])
    
    trees = torch.stack(trees_list, dim=0)
    
    # Validate initial batch
    valid_initial = _validate_tree_constraints(trees)
    print(f"Initial valid trees: {valid_initial.sum().item()}/{batch_size}")
    
    # Keep only valid trees for testing
    valid_mask = valid_initial
    if valid_mask.sum() == 0:
        print("No valid initial trees generated, skipping")
        return True
    
    trees = trees[valid_mask]
    
    # Apply aggressive mutation
    config = {'max_children': 3, 'max_depth': 8}
    mutation = DeleteNodeMutation(prob=1.0, config=config, max_delete_nodes=10, max_nodes=max_nodes)
    
    mutated_trees = mutation(trees)
    
    # Validate all mutated trees
    valid_mutated = _validate_tree_constraints(mutated_trees)
    assert valid_mutated.all(), f"All mutated trees should be valid, but {(~valid_mutated).sum().item()} failed"
    
    print(f"Successfully processed {trees.size(0)} complex trees")
    print("PASSED: Stress test with large batches")
    return True


def test_extreme_constraints():
    """Test with extremely restrictive constraints."""
    print("Testing extreme constraint conditions...")
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA not available")
        return True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test with max_children = 1 (very restrictive)
    trees = create_complex_tree(device=device, max_nodes=32, depth=4)
    valid_initial = _validate_tree_constraints(trees)
    
    if not valid_initial.all():
        print("Initial tree invalid, creating simpler tree")
        trees = torch.zeros((1, 32, 7), dtype=torch.float32, device=device)
        # Simple valid tree
        trees[0, 0, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
        trees[0, 0, COL_PARENT_IDX] = -1
        trees[0, 0, COL_DEPTH] = 0
        trees[0, 1, COL_NODE_TYPE] = NODE_TYPE_DECISION
        trees[0, 1, COL_PARENT_IDX] = 0
        trees[0, 1, COL_DEPTH] = 1
        trees[0, 2, COL_NODE_TYPE] = NODE_TYPE_ACTION
        trees[0, 2, COL_PARENT_IDX] = 1
        trees[0, 2, COL_DEPTH] = 2
        trees[0, 2, COL_PARAM_1] = ACTION_NEW_LONG
    
    config = {'max_children': 1, 'max_depth': 3}  # Extremely restrictive
    mutation = DeleteNodeMutation(prob=1.0, config=config, max_delete_nodes=5, max_nodes=32)
    
    mutated_trees = mutation(trees)
    valid_mutated = _validate_tree_constraints(mutated_trees)
    assert valid_mutated.all(), "Even with extreme constraints, trees should remain valid"
    
    print("PASSED: Extreme constraints test")
    return True


def test_pathological_cases():
    """Test pathological tree structures."""
    print("Testing pathological tree structures...")
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA not available")
        return True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test 1: Linear chain tree
    trees = create_pathological_tree(device=device, max_nodes=32)
    valid_initial = _validate_tree_constraints(trees)
    assert valid_initial.all(), "Pathological tree should be valid initially"
    
    config = {'max_children': 5, 'max_depth': 15}
    mutation = DeleteNodeMutation(prob=1.0, config=config, max_delete_nodes=8, max_nodes=32)
    
    mutated_trees = mutation(trees)
    valid_mutated = _validate_tree_constraints(mutated_trees)
    assert valid_mutated.all(), "Linear tree should remain valid after deletion"
    
    # Test 2: Tree with maximum children per node
    trees = torch.zeros((1, 32, 7), dtype=torch.float32, device=device)
    trees[0, 0, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
    trees[0, 0, COL_PARENT_IDX] = -1
    trees[0, 0, COL_DEPTH] = 0
    
    # Add decision node with maximum children
    trees[0, 1, COL_NODE_TYPE] = NODE_TYPE_DECISION
    trees[0, 1, COL_PARENT_IDX] = 0
    trees[0, 1, COL_DEPTH] = 1
    
    # Add many children to test max_children constraint
    for i in range(2, 8):  # 6 children
        trees[0, i, COL_NODE_TYPE] = NODE_TYPE_DECISION
        trees[0, i, COL_PARENT_IDX] = 1
        trees[0, i, COL_DEPTH] = 2
        
        # Add action child to each
        trees[0, i + 6, COL_NODE_TYPE] = NODE_TYPE_ACTION
        trees[0, i + 6, COL_PARENT_IDX] = i
        trees[0, i + 6, COL_DEPTH] = 3
        trees[0, i + 6, COL_PARAM_1] = ACTION_NEW_LONG
    
    valid_initial = _validate_tree_constraints(trees)
    if valid_initial.all():
        config = {'max_children': 3, 'max_depth': 10}  # Should restrict some deletions
        mutation = DeleteNodeMutation(prob=1.0, config=config, max_delete_nodes=3, max_nodes=32)
        
        mutated_trees = mutation(trees)
        valid_mutated = _validate_tree_constraints(mutated_trees)
        assert valid_mutated.all(), "Tree with many children should remain valid"
    
    print("PASSED: Pathological cases test")
    return True


def test_memory_safety():
    """Test for memory safety and bounds checking."""
    print("Testing memory safety...")
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA not available")
        return True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test with various buffer sizes and tree configurations
    test_configs = [
        (8, 16),   # Small
        (16, 32),  # Medium  
        (32, 64),  # Large
        (1, 256),  # Very wide
        (64, 8),   # Many small trees
    ]
    
    for batch_size, max_nodes in test_configs:
        print(f"  Testing batch_size={batch_size}, max_nodes={max_nodes}")
        
        # Create batch
        trees = torch.zeros((batch_size, max_nodes, 7), dtype=torch.float32, device=device)
        
        # Fill with simple valid trees
        for b in range(batch_size):
            trees[b, 0, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
            trees[b, 0, COL_PARENT_IDX] = -1
            trees[b, 0, COL_DEPTH] = 0
            
            if max_nodes > 1:
                trees[b, 1, COL_NODE_TYPE] = NODE_TYPE_DECISION
                trees[b, 1, COL_PARENT_IDX] = 0
                trees[b, 1, COL_DEPTH] = 1
                
            if max_nodes > 2:
                trees[b, 2, COL_NODE_TYPE] = NODE_TYPE_ACTION
                trees[b, 2, COL_PARENT_IDX] = 1
                trees[b, 2, COL_DEPTH] = 2
                trees[b, 2, COL_PARAM_1] = ACTION_NEW_LONG
        
        # Test mutation
        config = {'max_children': 5, 'max_depth': 10}
        mutation = DeleteNodeMutation(prob=0.5, config=config, max_delete_nodes=3, max_nodes=max_nodes)
        
        try:
            mutated_trees = mutation(trees)
            valid_mutated = _validate_tree_constraints(mutated_trees)
            assert valid_mutated.all(), f"Memory safety test failed for {batch_size}x{max_nodes}"
        except Exception as e:
            print(f"    FAILED: {str(e)}")
            raise
    
    print("PASSED: Memory safety test")
    return True


def test_invariant_stress():
    """Stress test all invariants extensively."""
    print("Testing invariant stress conditions...")
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA not available")
        return True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate many random tree configurations
    num_tests = 50
    passed_tests = 0
    
    for test_idx in range(num_tests):
        try:
            # Random tree parameters
            batch_size = random.randint(1, 16)
            max_nodes = random.randint(8, 64)
            max_delete = random.randint(1, min(8, max_nodes // 4))
            max_children = random.randint(2, 6)
            max_depth = random.randint(4, 12)
            prob = random.uniform(0.5, 1.0)
            
            # Create random trees
            trees = torch.zeros((batch_size, max_nodes, 7), dtype=torch.float32, device=device)
            
            for b in range(batch_size):
                # Create a random valid tree
                tree = create_complex_tree(device=device, max_nodes=max_nodes, 
                                         depth=min(max_depth - 2, 6))
                trees[b] = tree[0]
            
            # Validate initial trees
            valid_initial = _validate_tree_constraints(trees)
            if valid_initial.sum() == 0:
                continue  # Skip if no valid trees
            
            # Keep only valid trees
            trees = trees[valid_initial]
            
            # Apply mutation
            config = {'max_children': max_children, 'max_depth': max_depth}
            mutation = DeleteNodeMutation(prob=prob, config=config, 
                                        max_delete_nodes=max_delete, max_nodes=max_nodes)
            
            mutated_trees = mutation(trees)
            
            # Validate results
            valid_mutated = _validate_tree_constraints(mutated_trees)
            
            if not valid_mutated.all():
                print(f"FAILED test {test_idx}: batch_size={batch_size}, max_nodes={max_nodes}")
                print(f"  max_delete={max_delete}, max_children={max_children}, max_depth={max_depth}")
                print(f"  Invalid trees: {(~valid_mutated).sum().item()}/{mutated_trees.size(0)}")
                raise AssertionError("Invariant violation detected")
            
            passed_tests += 1
            
        except Exception as e:
            print(f"Exception in test {test_idx}: {str(e)}")
            raise
    
    print(f"PASSED: Invariant stress test ({passed_tests}/{num_tests} configurations)")
    return True


def test_cuda_memory_limits():
    """Test CUDA memory limits and large allocations."""
    print("Testing CUDA memory limits...")
    
    if not CUDA_AVAILABLE or not torch.cuda.is_available():
        print("SKIPPED: CUDA not available")
        return True
    
    device = 'cuda'
    
    # Test progressively larger allocations
    max_batch_sizes = [32, 64, 128, 256]
    
    for batch_size in max_batch_sizes:
        try:
            print(f"  Testing batch_size={batch_size}")
            
            max_nodes = 32
            trees = torch.zeros((batch_size, max_nodes, 7), dtype=torch.float32, device=device)
            
            # Fill with simple trees
            for b in range(batch_size):
                trees[b, 0, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
                trees[b, 0, COL_PARENT_IDX] = -1
                trees[b, 0, COL_DEPTH] = 0
                trees[b, 1, COL_NODE_TYPE] = NODE_TYPE_ACTION
                trees[b, 1, COL_PARENT_IDX] = 0
                trees[b, 1, COL_DEPTH] = 1
                trees[b, 1, COL_PARAM_1] = ACTION_NEW_LONG
            
            config = {'max_children': 4, 'max_depth': 8}
            mutation = DeleteNodeMutation(prob=0.1, config=config, max_delete_nodes=1, max_nodes=max_nodes)
            
            mutated_trees = mutation(trees)
            valid_mutated = _validate_tree_constraints(mutated_trees)
            assert valid_mutated.all(), f"Large batch test failed for batch_size={batch_size}"
            
            # Clear memory
            del trees, mutated_trees
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    Hit memory limit at batch_size={batch_size}")
                break
            else:
                raise
        except Exception as e:
            print(f"    Error at batch_size={batch_size}: {str(e)}")
            raise
    
    print("PASSED: CUDA memory limits test")
    return True


def main():
    """Run all harsh validation tests."""
    print("="*60)
    print("HARSH VALIDATION TESTS FOR DELETE NODE MUTATION")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(12345)
    random.seed(12345)
    np.random.seed(12345)
    
    tests = [
        test_stress_large_batch,
        test_extreme_constraints,
        test_pathological_cases,
        test_memory_safety,
        test_invariant_stress,
        test_cuda_memory_limits,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"FAILED: {test.__name__} - {str(e)}")
            import traceback
            traceback.print_exc()
            print()
    
    print("="*60)
    print(f"HARSH VALIDATION RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("✅ All harsh validation tests PASSED!")
        return True
    else:
        print("❌ Some harsh validation tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)