# gatree_cuda_compat/__init__.py - Compatibility layer for split CUDA extensions
"""
This module provides backward compatibility by importing all functions
from the new split CUDA extensions and exposing them under the original
gatree_cuda namespace.
"""

try:
    # Import all the new extensions
    import gatree_predict
    import gatree_mutation
    import gatree_crossover  
    import gatree_utils
    
    # Prediction functions (from gatree_predict)
    predict = gatree_predict.predict
    count_and_create_offsets = gatree_predict.count_and_create_offsets
    fill_child_indices = gatree_predict.fill_child_indices
    validate_adjacency = gatree_predict.validate_adjacency
    
    # Mutation functions (from gatree_mutation)
    node_param_mutate = gatree_mutation.node_param_mutate
    add_subtrees_batch = gatree_mutation.add_subtrees_batch
    delete_subtrees_batch = gatree_mutation.delete_subtrees_batch
    add_decision_nodes_batch = gatree_mutation.add_decision_nodes_batch
    delete_nodes_batch = gatree_mutation.delete_nodes_batch
    reinitialize_node_mutate = gatree_mutation.reinitialize_node_mutate
    
    # Crossover functions (from gatree_crossover)
    swap_node_params = gatree_crossover.swap_node_params
    swap_node_params_cuda = gatree_crossover.swap_node_params_cuda
    get_contextual_mask = gatree_crossover.get_contextual_mask
    get_contextual_mask_cuda = gatree_crossover.get_contextual_mask_cuda
    subtree_crossover_batch = gatree_crossover.subtree_crossover_batch
    subtree_crossover_batch_cuda = gatree_crossover.subtree_crossover_batch_cuda
    copy_branches_batch = gatree_crossover.copy_branches_batch
    copy_branches_batch_cuda = gatree_crossover.copy_branches_batch_cuda
    repair_after_root_branch = gatree_crossover.repair_after_root_branch
    repair_after_root_branch_cuda = gatree_crossover.repair_after_root_branch_cuda
    
    # Utility functions (from gatree_utils)
    validate_trees = gatree_utils.validate_trees
    init_population_cuda = gatree_utils.init_population_cuda
    reorganize_population_with_arrays = gatree_utils.reorganize_population_with_arrays
    
except ImportError as e:
    print("="*60)
    print(">>> 경고: CUDA 확장 모듈을 찾을 수 없습니다.")
    print(">>> C++/CUDA 코드를 먼저 컴파일해야 합니다.")
    print(">>> 프로젝트 루트에서 다음 명령을 실행하세요:")
    print(">>> python setup.py build_ext --inplace")
    print(f">>> 오류 세부사항: {e}")
    print("="*60)
    
    # Set all functions to None for graceful degradation
    predict = None
    count_and_create_offsets = None
    fill_child_indices = None
    validate_adjacency = None
    node_param_mutate = None
    add_subtrees_batch = None
    delete_subtrees_batch = None
    add_decision_nodes_batch = None
    delete_nodes_batch = None
    reinitialize_node_mutate = None
    swap_node_params = None
    swap_node_params_cuda = None
    get_contextual_mask = None
    get_contextual_mask_cuda = None
    subtree_crossover_batch = None
    subtree_crossover_batch_cuda = None
    copy_branches_batch = None
    copy_branches_batch_cuda = None
    repair_after_root_branch = None
    repair_after_root_branch_cuda = None
    validate_trees = None
    init_population_cuda = None
    reorganize_population_with_arrays = None