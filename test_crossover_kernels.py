#!/usr/bin/env python3
"""
Test script to verify the divided crossover kernel functionality.
Tests all three crossover methodologies: node, subtree, and root branch.
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import gatree_cuda_compat as gatree_cuda
    print("✓ gatree_cuda module loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import gatree_cuda: {e}")
    print("Please build the CUDA extension first:")
    print("python setup.py build_ext --inplace")
    sys.exit(1)

from models.constants import NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH, NODE_TYPE_DECISION, NODE_TYPE_ACTION
from models.constants import COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, COL_PARAM_1

def create_simple_test_trees():
    """Create simple test trees for crossover testing."""
    batch_size = 2
    max_nodes = 10
    node_info_dim = 7
    
    # Initialize trees with UNUSED nodes
    trees = torch.full((batch_size, max_nodes, node_info_dim), 0.0, dtype=torch.float32)
    trees[:, :, COL_NODE_TYPE] = NODE_TYPE_UNUSED
    trees[:, :, COL_PARENT_IDX] = -1
    
    # Tree 1: Simple structure
    # Root branches (0=LONG, 1=HOLD, 2=SHORT)
    trees[0, 0, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
    trees[0, 0, COL_PARENT_IDX] = -1
    trees[0, 0, COL_DEPTH] = 0
    trees[0, 0, COL_PARAM_1] = 0  # LONG branch
    
    trees[0, 1, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
    trees[0, 1, COL_PARENT_IDX] = -1
    trees[0, 1, COL_DEPTH] = 0
    trees[0, 1, COL_PARAM_1] = 1  # HOLD branch
    
    trees[0, 2, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
    trees[0, 2, COL_PARENT_IDX] = -1
    trees[0, 2, COL_DEPTH] = 0
    trees[0, 2, COL_PARAM_1] = 2  # SHORT branch
    
    # Add some children
    trees[0, 3, COL_NODE_TYPE] = NODE_TYPE_ACTION
    trees[0, 3, COL_PARENT_IDX] = 0  # Child of LONG branch
    trees[0, 3, COL_DEPTH] = 1
    trees[0, 3, COL_PARAM_1] = 1  # NEW_LONG action
    
    trees[0, 4, COL_NODE_TYPE] = NODE_TYPE_DECISION
    trees[0, 4, COL_PARENT_IDX] = 1  # Child of HOLD branch
    trees[0, 4, COL_DEPTH] = 1
    trees[0, 4, COL_PARAM_1] = 0  # Comparison type
    
    # Tree 2: Different structure
    trees[1, 0, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
    trees[1, 0, COL_PARENT_IDX] = -1
    trees[1, 0, COL_DEPTH] = 0
    trees[1, 0, COL_PARAM_1] = 0  # LONG branch
    
    trees[1, 1, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
    trees[1, 1, COL_PARENT_IDX] = -1
    trees[1, 1, COL_DEPTH] = 0
    trees[1, 1, COL_PARAM_1] = 1  # HOLD branch
    
    trees[1, 2, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
    trees[1, 2, COL_PARENT_IDX] = -1
    trees[1, 2, COL_DEPTH] = 0
    trees[1, 2, COL_PARAM_1] = 2  # SHORT branch
    
    trees[1, 5, COL_NODE_TYPE] = NODE_TYPE_ACTION
    trees[1, 5, COL_PARENT_IDX] = 2  # Child of SHORT branch
    trees[1, 5, COL_DEPTH] = 1
    trees[1, 5, COL_PARAM_1] = 2  # NEW_SHORT action
    
    return trees

def test_node_crossover():
    """Test node parameter crossover functionality."""
    print("\n=== Testing Node Crossover ===")
    
    batch_size = 2
    max_nodes = 10
    node_info_dim = 7
    
    # Create test trees and masks
    trees = create_simple_test_trees()
    if torch.cuda.is_available():
        trees = trees.cuda()
    
    # Create masks for nodes to swap (only non-root nodes)
    mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    mask[0, 3] = True  # Action node in tree 1
    mask[0, 4] = True  # Decision node in tree 1
    mask[1, 5] = True  # Action node in tree 2
    
    if torch.cuda.is_available():
        mask = mask.cuda()
    
    # Create output masks for contextual selection
    output_mask = torch.zeros_like(mask)
    
    try:
        # Test contextual mask generation
        gatree_cuda.get_contextual_mask_cuda(
            trees, output_mask, NODE_TYPE_ACTION, 0  # Look for ACTION nodes in LONG branch
        )
        print("✓ get_contextual_mask_cuda executed successfully")
        
        # Test node parameter swapping
        c1 = trees.clone()
        c2 = trees.clone()
        
        gatree_cuda.swap_node_params_cuda(c1, c2, mask, mask)
        print("✓ swap_node_params_cuda executed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Node crossover test failed: {e}")
        return False

def test_subtree_crossover():
    """Test subtree crossover functionality."""
    print("\n=== Testing Subtree Crossover ===")
    
    batch_size = 2
    max_nodes = 20
    max_depth = 5
    max_retries = 3
    
    # Create test trees
    p1_batch = create_simple_test_trees()
    p2_batch = create_simple_test_trees()
    
    if torch.cuda.is_available():
        p1_batch = p1_batch.cuda()
        p2_batch = p2_batch.cuda()
    
    # Create output tensors
    child1_out = torch.zeros_like(p1_batch)
    child2_out = torch.zeros_like(p2_batch)
    
    # Create branch permutation (for context mode)
    branch_perm = torch.randint(0, 3, (batch_size, 3), dtype=torch.int32)
    if torch.cuda.is_available():
        branch_perm = branch_perm.cuda()
    
    # Create work buffers
    bfs_queue_buffer = torch.zeros((batch_size, 2 * max_nodes), dtype=torch.int32)
    result_indices_buffer = torch.zeros((batch_size, 2 * max_nodes), dtype=torch.int32)
    old_to_new_map_buffer = torch.zeros((batch_size, max_nodes), dtype=torch.int32)
    p1_candidates_buffer = torch.zeros((batch_size, max_nodes), dtype=torch.int32)
    p2_candidates_buffer = torch.zeros((batch_size, max_nodes), dtype=torch.int32)
    
    if torch.cuda.is_available():
        bfs_queue_buffer = bfs_queue_buffer.cuda()
        result_indices_buffer = result_indices_buffer.cuda()
        old_to_new_map_buffer = old_to_new_map_buffer.cuda()
        p1_candidates_buffer = p1_candidates_buffer.cuda()
        p2_candidates_buffer = p2_candidates_buffer.cuda()
    
    try:
        gatree_cuda.subtree_crossover_batch_cuda(
            child1_out, child2_out,
            p1_batch, p2_batch,
            0,  # mode: 0=free, 1=context
            max_depth, max_nodes, max_retries,
            branch_perm,
            bfs_queue_buffer, result_indices_buffer, old_to_new_map_buffer,
            p1_candidates_buffer, p2_candidates_buffer
        )
        print("✓ subtree_crossover_batch_cuda executed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Subtree crossover test failed: {e}")
        return False

def test_root_crossover():
    """Test root branch crossover functionality."""
    print("\n=== Testing Root Branch Crossover ===")
    
    batch_size = 2
    max_nodes = 20
    
    # Create test trees
    p1_batch = create_simple_test_trees()
    p2_batch = create_simple_test_trees()
    child_batch = torch.zeros_like(p1_batch)
    
    if torch.cuda.is_available():
        p1_batch = p1_batch.cuda()
        p2_batch = p2_batch.cuda()
        child_batch = child_batch.cuda()
    
    # Create root branches in child (normally done by Python)
    child_batch[:, 0:3] = p1_batch[:, 0:3].clone()
    
    # Create donor map (which parent to choose from for each branch)
    donor_map = torch.randint(0, 2, (batch_size, 3), dtype=torch.int32)  # 0=p1, 1=p2
    if torch.cuda.is_available():
        donor_map = donor_map.cuda()
    
    # Create work buffers
    bfs_queue_buffer = torch.zeros((batch_size, max_nodes), dtype=torch.int32)
    result_indices_buffer = torch.zeros((batch_size, max_nodes), dtype=torch.int32)
    old_to_new_map_buffer = torch.zeros((batch_size, max_nodes), dtype=torch.int32)
    
    if torch.cuda.is_available():
        bfs_queue_buffer = bfs_queue_buffer.cuda()
        result_indices_buffer = result_indices_buffer.cuda()
        old_to_new_map_buffer = old_to_new_map_buffer.cuda()
    
    try:
        # Test branch copying
        gatree_cuda.copy_branches_batch_cuda(
            child_batch, p1_batch, p2_batch, donor_map,
            bfs_queue_buffer, result_indices_buffer, old_to_new_map_buffer
        )
        print("✓ copy_branches_batch_cuda executed successfully")
        
        # Test tree repair
        child_count_buffer = torch.zeros((batch_size, max_nodes), dtype=torch.int32)
        act_cnt_buffer = torch.zeros((batch_size, max_nodes), dtype=torch.int32)
        dec_cnt_buffer = torch.zeros((batch_size, max_nodes), dtype=torch.int32)
        repair_bfs_queue = torch.zeros((batch_size, 2 * max_nodes), dtype=torch.int32)
        repair_result_buffer = torch.zeros((batch_size, 2 * max_nodes), dtype=torch.int32)
        
        if torch.cuda.is_available():
            child_count_buffer = child_count_buffer.cuda()
            act_cnt_buffer = act_cnt_buffer.cuda()
            dec_cnt_buffer = dec_cnt_buffer.cuda()
            repair_bfs_queue = repair_bfs_queue.cuda()
            repair_result_buffer = repair_result_buffer.cuda()
        
        gatree_cuda.repair_after_root_branch_cuda(
            child_batch, child_count_buffer, act_cnt_buffer, dec_cnt_buffer,
            repair_bfs_queue, repair_result_buffer
        )
        print("✓ repair_after_root_branch_cuda executed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Root branch crossover test failed: {e}")
        return False

def run_all_tests():
    """Run all crossover kernel tests."""
    print("Starting Crossover Kernel Tests...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
    
    tests = [
        ("Node Crossover", test_node_crossover),
        ("Subtree Crossover", test_subtree_crossover),
        ("Root Branch Crossover", test_root_crossover)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} crashed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("CROSSOVER KERNEL TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:<25}: {status}")
        if not passed:
            all_passed = False
    
    print("="*50)
    overall_status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"Overall Result: {overall_status}")
    print("="*50)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)