#!/usr/bin/env python3
"""
Basic validation test for CUDA delete node mutation.
Tests basic functionality and invariant preservation under normal conditions.
"""

import torch
import random
import numpy as np
from typing import Dict, Any

from models.constants import (
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, COL_PARAM_1, COL_PARAM_2, COL_PARAM_3, COL_PARAM_4,
    NODE_TYPE_UNUSED, NODE_TYPE_DECISION, NODE_TYPE_ROOT_BRANCH, NODE_TYPE_ACTION,
    ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT,
    COMP_TYPE_FEAT_NUM, OP_GTE, ACTION_NEW_LONG
)

try:
    import gatree_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: gatree_cuda not available. Skipping CUDA tests.")

from evolution.Mutation.delete_node import DeleteNodeMutation, _validate_tree_constraints


def create_simple_test_tree(device='cuda', max_nodes=64) -> torch.Tensor:
    """Create a simple valid tree for testing."""
    trees = torch.zeros((1, max_nodes, 7), dtype=torch.float32, device=device)
    
    # Root branch (LONG)
    trees[0, 0, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
    trees[0, 0, COL_PARENT_IDX] = -1
    trees[0, 0, COL_DEPTH] = 0
    trees[0, 0, COL_PARAM_1] = ROOT_BRANCH_LONG
    
    # Decision node 1 (child of root)
    trees[0, 1, COL_NODE_TYPE] = NODE_TYPE_DECISION
    trees[0, 1, COL_PARENT_IDX] = 0
    trees[0, 1, COL_DEPTH] = 1
    trees[0, 1, COL_PARAM_1] = COMP_TYPE_FEAT_NUM
    trees[0, 1, COL_PARAM_2] = OP_GTE
    trees[0, 1, COL_PARAM_3] = 5.0  # feature index
    trees[0, 1, COL_PARAM_4] = 0.5  # threshold
    
    # Decision node 2 (child of decision 1)
    trees[0, 2, COL_NODE_TYPE] = NODE_TYPE_DECISION
    trees[0, 2, COL_PARENT_IDX] = 1
    trees[0, 2, COL_DEPTH] = 2
    trees[0, 2, COL_PARAM_1] = COMP_TYPE_FEAT_NUM
    trees[0, 2, COL_PARAM_2] = OP_GTE
    trees[0, 2, COL_PARAM_3] = 10.0
    trees[0, 2, COL_PARAM_4] = 0.3
    
    # Action node 1 (child of decision 2)
    trees[0, 3, COL_NODE_TYPE] = NODE_TYPE_ACTION
    trees[0, 3, COL_PARENT_IDX] = 2
    trees[0, 3, COL_DEPTH] = 3
    trees[0, 3, COL_PARAM_1] = ACTION_NEW_LONG
    trees[0, 3, COL_PARAM_2] = 0.5  # size
    trees[0, 3, COL_PARAM_3] = 2.0  # leverage
    
    # Decision node 3 (another child of decision 1)
    trees[0, 4, COL_NODE_TYPE] = NODE_TYPE_DECISION
    trees[0, 4, COL_PARENT_IDX] = 1
    trees[0, 4, COL_DEPTH] = 2
    trees[0, 4, COL_PARAM_1] = COMP_TYPE_FEAT_NUM
    trees[0, 4, COL_PARAM_2] = OP_GTE
    trees[0, 4, COL_PARAM_3] = 15.0
    trees[0, 4, COL_PARAM_4] = 0.7
    
    # Action node 2 (child of decision 3)
    trees[0, 5, COL_NODE_TYPE] = NODE_TYPE_ACTION
    trees[0, 5, COL_PARENT_IDX] = 4
    trees[0, 5, COL_DEPTH] = 3
    trees[0, 5, COL_PARAM_1] = ACTION_NEW_LONG
    trees[0, 5, COL_PARAM_2] = 0.3
    trees[0, 5, COL_PARAM_3] = 1.5
    
    return trees


def test_basic_delete_functionality():
    """Test basic delete functionality."""
    print("Testing basic delete functionality...")
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA not available")
        return True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trees = create_simple_test_tree(device=device)
    
    # Validate initial tree
    valid_initial = _validate_tree_constraints(trees)
    assert valid_initial.all(), "Initial tree should be valid"
    
    # Count initial nodes
    initial_decision_count = (trees[0, :, COL_NODE_TYPE] == NODE_TYPE_DECISION).sum().item()
    print(f"Initial decision nodes: {initial_decision_count}")
    
    # Apply delete mutation
    config = {'max_children': 5, 'max_depth': 10}
    mutation = DeleteNodeMutation(prob=1.0, config=config, max_delete_nodes=1, max_nodes=64)
    
    mutated_trees = mutation(trees)
    
    # Validate mutated tree
    valid_mutated = _validate_tree_constraints(mutated_trees)
    assert valid_mutated.all(), "Mutated tree should be valid"
    
    # Count final nodes
    final_decision_count = (mutated_trees[0, :, COL_NODE_TYPE] == NODE_TYPE_DECISION).sum().item()
    print(f"Final decision nodes: {final_decision_count}")
    
    # Should have deleted at most 1 node
    assert final_decision_count <= initial_decision_count, "Should not have gained nodes"
    assert final_decision_count >= initial_decision_count - 1, "Should not have deleted more than 1 node"
    
    print("PASSED: Basic delete functionality")
    return True


def test_invariant_preservation():
    """Test that tree invariants are preserved."""
    print("Testing invariant preservation...")
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA not available")
        return True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test multiple random trees
    config = {'max_children': 4, 'max_depth': 8}
    mutation = DeleteNodeMutation(prob=1.0, config=config, max_delete_nodes=3, max_nodes=32)
    
    for i in range(10):
        trees = create_simple_test_tree(device=device, max_nodes=32)
        
        # Add some randomness to the tree structure
        if i > 0:
            # Add a few more decision nodes randomly
            for j in range(3):
                # Find unused slots
                unused_mask = trees[0, :, COL_NODE_TYPE] == NODE_TYPE_UNUSED
                unused_indices = unused_mask.nonzero(as_tuple=True)[0]
                if len(unused_indices) == 0:
                    break
                
                # Find used decision nodes to attach to
                decision_mask = trees[0, :, COL_NODE_TYPE] == NODE_TYPE_DECISION
                decision_indices = decision_mask.nonzero(as_tuple=True)[0]
                if len(decision_indices) == 0:
                    break
                
                new_idx = unused_indices[0].item()
                parent_idx = decision_indices[torch.randint(0, len(decision_indices), (1,))].item()
                parent_depth = int(trees[0, parent_idx, COL_DEPTH].item())
                
                # Add new decision node
                trees[0, new_idx, COL_NODE_TYPE] = NODE_TYPE_DECISION
                trees[0, new_idx, COL_PARENT_IDX] = parent_idx
                trees[0, new_idx, COL_DEPTH] = parent_depth + 1
                trees[0, new_idx, COL_PARAM_1] = COMP_TYPE_FEAT_NUM
                trees[0, new_idx, COL_PARAM_2] = OP_GTE
                trees[0, new_idx, COL_PARAM_3] = random.randint(0, 20)
                trees[0, new_idx, COL_PARAM_4] = random.random()
        
        # Validate initial tree
        valid_initial = _validate_tree_constraints(trees)
        if not valid_initial.all():
            print(f"Skipping test {i}: Initial tree invalid")
            continue
        
        # Apply mutation
        mutated_trees = mutation(trees)
        
        # Validate mutated tree
        valid_mutated = _validate_tree_constraints(mutated_trees)
        assert valid_mutated.all(), f"Test {i}: Mutated tree should be valid"
    
    print("PASSED: Invariant preservation")
    return True


def test_batch_processing():
    """Test batch processing with multiple trees."""
    print("Testing batch processing...")
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA not available")
        return True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    max_nodes = 32
    
    # Create batch of trees
    trees = torch.zeros((batch_size, max_nodes, 7), dtype=torch.float32, device=device)
    
    for b in range(batch_size):
        single_tree = create_simple_test_tree(device=device, max_nodes=max_nodes)
        trees[b] = single_tree[0]
    
    # Validate initial batch
    valid_initial = _validate_tree_constraints(trees)
    assert valid_initial.all(), "All initial trees should be valid"
    
    # Apply mutation
    config = {'max_children': 5, 'max_depth': 10}
    mutation = DeleteNodeMutation(prob=0.8, config=config, max_delete_nodes=2, max_nodes=max_nodes)
    
    mutated_trees = mutation(trees)
    
    # Validate mutated batch
    valid_mutated = _validate_tree_constraints(mutated_trees)
    assert valid_mutated.all(), "All mutated trees should be valid"
    
    print("PASSED: Batch processing")
    return True


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases...")
    
    if not CUDA_AVAILABLE:
        print("SKIPPED: CUDA not available")
        return True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test 1: Tree with only root and one action child (should not delete)
    trees = torch.zeros((1, 32, 7), dtype=torch.float32, device=device)
    trees[0, 0, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
    trees[0, 0, COL_PARENT_IDX] = -1
    trees[0, 0, COL_DEPTH] = 0
    trees[0, 0, COL_PARAM_1] = ROOT_BRANCH_LONG
    
    trees[0, 1, COL_NODE_TYPE] = NODE_TYPE_ACTION
    trees[0, 1, COL_PARENT_IDX] = 0
    trees[0, 1, COL_DEPTH] = 1
    trees[0, 1, COL_PARAM_1] = ACTION_NEW_LONG
    
    config = {'max_children': 5, 'max_depth': 10}
    mutation = DeleteNodeMutation(prob=1.0, config=config, max_delete_nodes=1, max_nodes=32)
    
    mutated_trees = mutation(trees)
    valid_mutated = _validate_tree_constraints(mutated_trees)
    assert valid_mutated.all(), "Simple tree should remain valid"
    
    # Should not have deleted anything (no decision nodes to delete)
    action_count_before = (trees[0, :, COL_NODE_TYPE] == NODE_TYPE_ACTION).sum().item()
    action_count_after = (mutated_trees[0, :, COL_NODE_TYPE] == NODE_TYPE_ACTION).sum().item()
    assert action_count_before == action_count_after, "Action nodes should not be deleted"
    
    # Test 2: Max children constraint
    trees = create_simple_test_tree(device=device, max_nodes=32)
    config = {'max_children': 1, 'max_depth': 10}  # Very restrictive
    mutation = DeleteNodeMutation(prob=1.0, config=config, max_delete_nodes=1, max_nodes=32)
    
    mutated_trees = mutation(trees)
    valid_mutated = _validate_tree_constraints(mutated_trees)
    assert valid_mutated.all(), "Restrictive max_children should still produce valid tree"
    
    print("PASSED: Edge cases")
    return True


def main():
    """Run all basic validation tests."""
    print("="*60)
    print("BASIC VALIDATION TESTS FOR DELETE NODE MUTATION")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    tests = [
        test_basic_delete_functionality,
        test_invariant_preservation,
        test_batch_processing,
        test_edge_cases,
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
    print(f"BASIC VALIDATION RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("✅ All basic validation tests PASSED!")
        return True
    else:
        print("❌ Some basic validation tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)