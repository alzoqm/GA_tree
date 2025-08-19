#!/usr/bin/env python3
"""
Simple test for the CUDA-accelerated AddNodeMutation
"""
import torch
from evolution.Mutation.add_node import AddNodeMutation
from models.constants import *

def create_simple_config():
    """Create a minimal config for testing"""
    return {
        'max_depth': 10,
        'all_features': ['close', 'RSI', 'MACD', 'volume', 'InvertedHammers'],
        'feature_num': {
            'close': [1000, 2000],
            'RSI': [0, 100],
            'MACD': [-50, 50]
        },
        'feature_comparison': ['close', 'RSI', 'MACD'],
        'feature_bool': ['InvertedHammers']
    }

def create_test_population(batch_size=4, max_nodes=50):
    """Create a simple test population with basic tree structure"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trees = torch.zeros((batch_size, max_nodes, NODE_INFO_DIM), dtype=torch.float32, device=device)
    
    for b in range(batch_size):
        # Root branches
        trees[b, 0, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
        trees[b, 0, COL_PARENT_IDX] = -1
        trees[b, 0, COL_DEPTH] = 0
        trees[b, 0, COL_PARAM_1] = ROOT_BRANCH_LONG
        
        trees[b, 1, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
        trees[b, 1, COL_PARENT_IDX] = -1
        trees[b, 1, COL_DEPTH] = 0
        trees[b, 1, COL_PARAM_1] = ROOT_BRANCH_HOLD
        
        trees[b, 2, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
        trees[b, 2, COL_PARENT_IDX] = -1
        trees[b, 2, COL_DEPTH] = 0
        trees[b, 2, COL_PARAM_1] = ROOT_BRANCH_SHORT
        
        # Simple decision nodes
        trees[b, 3, COL_NODE_TYPE] = NODE_TYPE_DECISION
        trees[b, 3, COL_PARENT_IDX] = 0  # child of first root branch
        trees[b, 3, COL_DEPTH] = 1
        
        trees[b, 4, COL_NODE_TYPE] = NODE_TYPE_DECISION
        trees[b, 4, COL_PARENT_IDX] = 1  # child of second root branch
        trees[b, 4, COL_DEPTH] = 1
        
        # Action leaves
        trees[b, 5, COL_NODE_TYPE] = NODE_TYPE_ACTION
        trees[b, 5, COL_PARENT_IDX] = 3
        trees[b, 5, COL_DEPTH] = 2
        trees[b, 5, COL_PARAM_1] = ACTION_NEW_LONG
        
        trees[b, 6, COL_NODE_TYPE] = NODE_TYPE_ACTION
        trees[b, 6, COL_PARENT_IDX] = 4
        trees[b, 6, COL_DEPTH] = 2
        trees[b, 6, COL_PARAM_1] = ACTION_CLOSE_ALL
    
    return trees

def validate_tree_invariants(trees):
    """Validate that tree invariants are preserved"""
    B = trees.shape[0]
    
    print("Validating tree invariants...")
    
    for b in range(B):
        tree = trees[b]
        active_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        
        print(f"Tree {b}: {len(active_indices)} active nodes")
        
        # Check that all leaf nodes are ACTION nodes
        leaf_count = 0
        action_leaf_count = 0
        
        for idx in active_indices:
            node_type = int(tree[idx, COL_NODE_TYPE].item())
            
            # Find children
            children = []
            for j in active_indices:
                if int(tree[j, COL_PARENT_IDX].item()) == idx:
                    children.append(j)
            
            # If no children, this is a leaf
            if len(children) == 0 and node_type != NODE_TYPE_ROOT_BRANCH:
                leaf_count += 1
                if node_type == NODE_TYPE_ACTION:
                    action_leaf_count += 1
                else:
                    print(f"  ERROR: Leaf node {idx} is not ACTION (type={node_type})")
            
            # Check for mixed child types
            if len(children) > 0:
                child_types = set()
                for child_idx in children:
                    child_type = int(tree[child_idx, COL_NODE_TYPE].item())
                    if child_type != NODE_TYPE_ROOT_BRANCH:
                        child_types.add(child_type)
                
                if NODE_TYPE_ACTION in child_types and NODE_TYPE_DECISION in child_types:
                    print(f"  ERROR: Node {idx} has mixed ACTION and DECISION children")
                
                if NODE_TYPE_ACTION in child_types and len(children) > 1:
                    print(f"  ERROR: Node {idx} has ACTION child but {len(children)} total children")
        
        print(f"  Leaves: {leaf_count}, ACTION leaves: {action_leaf_count}")
        if leaf_count == action_leaf_count:
            print(f"  ✓ All leaves are ACTION nodes")
        else:
            print(f"  ✗ Some leaves are not ACTION nodes")

def test_add_node_mutation():
    """Test the AddNodeMutation implementation"""
    print("Testing CUDA-accelerated AddNodeMutation...")
    
    config = create_simple_config()
    mutation = AddNodeMutation(prob=1.0, config=config, max_add_nodes=3, max_nodes=50)
    
    # Create test population
    population = create_test_population(batch_size=2, max_nodes=50)
    print(f"Initial population shape: {population.shape}")
    print(f"Device: {population.device}")
    
    # Show initial active nodes
    for b in range(population.shape[0]):
        active_count = (population[b, :, COL_NODE_TYPE] != NODE_TYPE_UNUSED).sum().item()
        print(f"Tree {b}: {active_count} active nodes initially")
    
    # Validate initial invariants
    validate_tree_invariants(population)
    
    print("\nApplying AddNodeMutation...")
    try:
        mutated_population = mutation(population)
        print("✓ Mutation completed successfully")
        
        # Show results
        for b in range(mutated_population.shape[0]):
            active_count = (mutated_population[b, :, COL_NODE_TYPE] != NODE_TYPE_UNUSED).sum().item()
            original_count = (population[b, :, COL_NODE_TYPE] != NODE_TYPE_UNUSED).sum().item()
            added = active_count - original_count
            print(f"Tree {b}: {original_count} -> {active_count} nodes (+{added})")
        
        # Validate post-mutation invariants
        validate_tree_invariants(mutated_population)
        
        print("\n✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during mutation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    success = test_add_node_mutation()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")