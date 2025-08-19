#!/usr/bin/env python3
"""
Stress test for the CUDA-accelerated AddNodeMutation
"""
import torch
from evolution.Mutation.add_node import AddNodeMutation
from models.constants import *

def create_config():
    """Create config based on experiment_config.yaml"""
    return {
        'max_depth': 20,
        'all_features': [
            'close', 'open', 'high', 'low', 'volume',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Mid', 'BB_Lower', 'ATR',
            'InvertedHammers', 'Hammers', 'Dojis'
        ],
        'feature_num': {
            'close': [1000, 50000],
            'RSI': [0, 100],
            'MACD': [-100, 100],
            'ATR': [0, 500],
            'volume': [1000, 1000000]
        },
        'feature_comparison': [
            'close', 'open', 'high', 'low', 'RSI', 'MACD', 
            'BB_Upper', 'BB_Mid', 'BB_Lower'
        ],
        'feature_bool': ['InvertedHammers', 'Hammers', 'Dojis']
    }

def create_complex_population(batch_size=32, max_nodes=100):
    """Create a more complex test population"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trees = torch.zeros((batch_size, max_nodes, NODE_INFO_DIM), dtype=torch.float32, device=device)
    
    for b in range(batch_size):
        node_idx = 0
        
        # Root branches
        for branch_type in [ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT]:
            trees[b, node_idx, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
            trees[b, node_idx, COL_PARENT_IDX] = -1
            trees[b, node_idx, COL_DEPTH] = 0
            trees[b, node_idx, COL_PARAM_1] = branch_type
            node_idx += 1
        
        # Add some decision nodes with varying depths
        for i in range(min(15, max_nodes - node_idx - 10)):  # Leave space for actions
            parent = torch.randint(0, min(node_idx, 20), (1,)).item() if node_idx > 3 else i % 3
            parent_depth = int(trees[b, parent, COL_DEPTH].item())
            
            if parent_depth + 1 < 15:  # Depth limit
                trees[b, node_idx, COL_NODE_TYPE] = NODE_TYPE_DECISION
                trees[b, node_idx, COL_PARENT_IDX] = parent
                trees[b, node_idx, COL_DEPTH] = parent_depth + 1
                
                # Random decision params
                trees[b, node_idx, COL_PARAM_1] = torch.randint(0, 5, (1,)).float()  # feature idx
                trees[b, node_idx, COL_PARAM_2] = torch.randint(0, 2, (1,)).float()  # op
                trees[b, node_idx, COL_PARAM_3] = torch.randint(0, 3, (1,)).float()  # comp_type
                trees[b, node_idx, COL_PARAM_4] = torch.rand(1) * 100  # value
                
                node_idx += 1
        
        # Ensure ALL leaf nodes are ACTION nodes
        # First, find all nodes that currently have no children
        leaf_candidates = []
        for potential_parent in range(node_idx):
            node_type = int(trees[b, potential_parent, COL_NODE_TYPE].item())
            if node_type in [NODE_TYPE_DECISION, NODE_TYPE_ROOT_BRANCH]:
                # Check if it has children
                has_children = False
                for j in range(node_idx):
                    if int(trees[b, j, COL_PARENT_IDX].item()) == potential_parent:
                        has_children = True
                        break
                
                if not has_children:
                    leaf_candidates.append(potential_parent)
        
        # Add ACTION children to ALL leaf candidates
        for potential_parent in leaf_candidates:
            if node_idx >= max_nodes:
                break
            parent_depth = int(trees[b, potential_parent, COL_DEPTH].item())
            trees[b, node_idx, COL_NODE_TYPE] = NODE_TYPE_ACTION
            trees[b, node_idx, COL_PARENT_IDX] = potential_parent
            trees[b, node_idx, COL_DEPTH] = parent_depth + 1
            trees[b, node_idx, COL_PARAM_1] = torch.randint(1, 7, (1,)).float()  # action type
            node_idx += 1
    
    return trees

def detailed_invariant_check(trees):
    """Comprehensive invariant validation"""
    B = trees.shape[0]
    total_errors = 0
    
    print("Performing detailed invariant check...")
    
    for b in range(B):
        tree = trees[b]
        active_mask = tree[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        
        errors = 0
        
        # Build parent-child mapping
        children_map = {}
        for idx in active_indices:
            parent = int(tree[idx, COL_PARENT_IDX].item())
            if parent >= 0:
                if parent not in children_map:
                    children_map[parent] = []
                children_map[parent].append(idx.item())
        
        # Check each node
        for idx in active_indices:
            idx_val = idx.item()
            node_type = int(tree[idx, COL_NODE_TYPE].item())
            children = children_map.get(idx_val, [])
            
            # Check leaf constraint: all leaves must be ACTION
            if len(children) == 0 and node_type != NODE_TYPE_ROOT_BRANCH:
                if node_type != NODE_TYPE_ACTION:
                    print(f"  Tree {b}: ERROR - Leaf node {idx_val} is not ACTION (type={node_type})")
                    errors += 1
            
            # Check child type mixing
            if len(children) > 1:
                child_types = set()
                for child_idx in children:
                    child_type = int(tree[child_idx, COL_NODE_TYPE].item())
                    if child_type != NODE_TYPE_ROOT_BRANCH:
                        child_types.add(child_type)
                
                if NODE_TYPE_ACTION in child_types and NODE_TYPE_DECISION in child_types:
                    print(f"  Tree {b}: ERROR - Node {idx_val} has mixed ACTION/DECISION children")
                    errors += 1
            
            # Check ACTION parent constraint
            action_children = [c for c in children if int(tree[c, COL_NODE_TYPE].item()) == NODE_TYPE_ACTION]
            if action_children and len(children) > 1:
                print(f"  Tree {b}: ERROR - Node {idx_val} has ACTION child but {len(children)} total children")
                errors += 1
            
            # Check depth consistency
            if node_type != NODE_TYPE_ROOT_BRANCH:
                parent_idx = int(tree[idx, COL_PARENT_IDX].item())
                if parent_idx >= 0:
                    parent_depth = int(tree[parent_idx, COL_DEPTH].item())
                    node_depth = int(tree[idx, COL_DEPTH].item())
                    if node_depth != parent_depth + 1:
                        print(f"  Tree {b}: ERROR - Node {idx_val} depth {node_depth} != parent depth {parent_depth} + 1")
                        errors += 1
        
        if errors == 0:
            print(f"  Tree {b}: ✓ All invariants satisfied ({len(active_indices)} nodes)")
        else:
            print(f"  Tree {b}: ✗ {errors} invariant violations")
            total_errors += errors
    
    return total_errors == 0

def stress_test():
    """Run comprehensive stress test"""
    print("=== CUDA AddNodeMutation Stress Test ===")
    
    config = create_config()
    
    # Test with different parameters
    test_cases = [
        (16, 64, 0.5, 2),   # Medium batch, medium trees, 50% mutation
        (32, 100, 1.0, 5),  # Large batch, large trees, 100% mutation
        (8, 200, 0.3, 3),   # Small batch, very large trees, 30% mutation
    ]
    
    for batch_size, max_nodes, prob, max_add in test_cases:
        print(f"\nTesting: batch_size={batch_size}, max_nodes={max_nodes}, prob={prob}, max_add={max_add}")
        
        mutation = AddNodeMutation(prob=prob, config=config, max_add_nodes=max_add, max_nodes=max_nodes)
        population = create_complex_population(batch_size=batch_size, max_nodes=max_nodes)
        
        # Count initial nodes
        initial_counts = []
        for b in range(batch_size):
            count = (population[b, :, COL_NODE_TYPE] != NODE_TYPE_UNUSED).sum().item()
            initial_counts.append(count)
        
        print(f"Initial node counts: avg={sum(initial_counts)/len(initial_counts):.1f}, "
              f"min={min(initial_counts)}, max={max(initial_counts)}")
        
        # Validate initial state
        if not detailed_invariant_check(population):
            print("✗ Initial population has invariant violations!")
            return False
        
        # Apply mutation
        try:
            mutated = mutation(population)
            
            # Count final nodes
            final_counts = []
            total_added = 0
            for b in range(batch_size):
                final_count = (mutated[b, :, COL_NODE_TYPE] != NODE_TYPE_UNUSED).sum().item()
                final_counts.append(final_count)
                total_added += final_count - initial_counts[b]
            
            print(f"Final node counts: avg={sum(final_counts)/len(final_counts):.1f}, "
                  f"min={min(final_counts)}, max={max(final_counts)}")
            print(f"Total nodes added: {total_added}")
            
            # Validate final state
            if not detailed_invariant_check(mutated):
                print("✗ Mutated population has invariant violations!")
                return False
            
            print("✓ Test case passed")
            
        except Exception as e:
            print(f"✗ Exception during mutation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n=== All stress tests PASSED! ===")
    return True

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    success = stress_test()
    print(f"\nStress test {'PASSED' if success else 'FAILED'}")