# /gatree_cuda_compat/__init__.py
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
    import gatree_selection
    import gatree_env
    
    # Prediction functions (from gatree_predict)
    predict = gatree_predict.predict
    count_and_create_offsets = gatree_predict.count_and_create_offsets
    fill_child_indices = gatree_predict.fill_child_indices
    validate_adjacency = gatree_predict.validate_adjacency
    # [추가] 인접리스트 빌더 노출
    if hasattr(gatree_predict, 'build_adjacency_dicts'):
        build_adjacency_dicts = gatree_predict.build_adjacency_dicts
    else:
        build_adjacency_dicts = None
    if hasattr(gatree_predict, 'build_adjacency_dicts_optimized'):
        build_adjacency_dicts_optimized = gatree_predict.build_adjacency_dicts_optimized
    else:
        build_adjacency_dicts_optimized = None
    
    # Mutation functions (from gatree_mutation)
    node_param_mutate = gatree_mutation.node_param_mutate
    add_subtrees_batch = gatree_mutation.add_subtrees_batch
    delete_subtrees_batch = gatree_mutation.delete_subtrees_batch
    add_decision_nodes_batch = gatree_mutation.add_decision_nodes_batch
    delete_nodes_batch = gatree_mutation.delete_nodes_batch
    reinitialize_node_mutate = gatree_mutation.reinitialize_node_mutate
    critical_repair_batch = gatree_mutation.critical_repair_batch
    
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
    root_branch_sampling_crossover = gatree_crossover.root_branch_sampling_crossover
    root_branch_sampling_crossover_cuda = gatree_crossover.root_branch_sampling_crossover_cuda
    
    # Utility functions (from gatree_utils)
    validate_trees = gatree_utils.validate_trees
    init_population_cuda = gatree_utils.init_population_cuda
    # Canonical API name: reorganize_population
    # Prefer array-backed version when available; otherwise, support legacy signature transparently.
    if hasattr(gatree_utils, 'reorganize_population_with_arrays'):
        def reorganize_population(population_tensor, active_counts_per_tree, old_gid_to_new_gid_map):
            return gatree_utils.reorganize_population_with_arrays(
                population_tensor, active_counts_per_tree, old_gid_to_new_gid_map
            )
    elif hasattr(gatree_utils, 'reorganize_population'):
        def reorganize_population(population_tensor, active_counts_per_tree=None, old_gid_to_new_gid_map=None):
            # Support both new (3-arg) and legacy (1-arg) bindings transparently
            if active_counts_per_tree is not None and old_gid_to_new_gid_map is not None:
                return gatree_utils.reorganize_population(
                    population_tensor, active_counts_per_tree, old_gid_to_new_gid_map
                )
            else:
                return gatree_utils.reorganize_population(population_tensor)
    else:
        reorganize_population = None

    # Pruning (from gatree_utils)
    prune_mark_and_cleanup = getattr(gatree_utils, 'prune_mark_and_cleanup', None)
    
    # Selection functions (from gatree_selection)
    ppts_select_elites = gatree_selection.ppts_select_elites
    ppts_pick_parents = gatree_selection.ppts_pick_parents

    # Trading environment functions (from gatree_env)
    execute_actions_batch = gatree_env.execute_actions_batch
    execute_actions_with_attrib = getattr(gatree_env, 'execute_actions_with_attrib', None)
    accumulate_realized_stats = getattr(gatree_env, 'accumulate_realized_stats', None)
    check_liquidation_batch = getattr(gatree_env, 'check_liquidation_batch', None)
    check_liquidation_with_attrib = getattr(gatree_env, 'check_liquidation_with_attrib', None)
    compute_equity_batch = getattr(gatree_env, 'compute_equity_batch', None)
    compute_unrealized_pnl = getattr(gatree_env, 'compute_unrealized_pnl', None)
    update_holding_periods = getattr(gatree_env, 'update_holding_periods', None)
    prepare_and_deduplicate_actions = getattr(gatree_env, 'prepare_and_deduplicate_actions', None)
    update_dedup_buffers_after_exec = getattr(gatree_env, 'update_dedup_buffers_after_exec', None)

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
    build_adjacency_dicts = None
    build_adjacency_dicts_optimized = None
    node_param_mutate = None
    add_subtrees_batch = None
    delete_subtrees_batch = None
    add_decision_nodes_batch = None
    delete_nodes_batch = None
    reinitialize_node_mutate = None
    critical_repair_batch = None
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
    root_branch_sampling_crossover = None
    root_branch_sampling_crossover_cuda = None
    validate_trees = None
    init_population_cuda = None
    reorganize_population = None
    ppts_select_elites = None
    ppts_pick_parents = None
    execute_actions_batch = None
    execute_actions_with_attrib = None
    accumulate_realized_stats = None
    check_liquidation_batch = None
    check_liquidation_with_attrib = None
    compute_equity_batch = None
    compute_unrealized_pnl = None
    update_holding_periods = None
    prepare_and_deduplicate_actions = None
    update_dedup_buffers_after_exec = None
