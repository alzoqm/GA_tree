import torch
import os
from model import (
    GATree,
    # --- 테스트에 필요한 상수들을 직접 임포트하여 가독성 확보 ---
    NODE_INFO_DIM,
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH,
    COL_PARAM_1, COL_PARAM_2, COL_PARAM_3, COL_PARAM_4,
    NODE_TYPE_ROOT_BRANCH, NODE_TYPE_DECISION, NODE_TYPE_ACTION,
    ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT,
    OP_GTE, OP_LTE,
    COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_FEAT,
    POS_TYPE_LONG, POS_TYPE_SHORT,
    # --- 테스트용 피쳐 정보 (model.py와 동일하게 유지) ---
    FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL
)

def create_manual_test_tree():
    """
    테스트 시나리오를 검증하기 위한 고정된 GATree를 수동으로 생성합니다.
    이 트리는 BFS의 최단 경로 탐색 능력을 검증하도록 설계되었습니다.
    """
    print("--- 1. Creating a manual test tree ---")
    max_nodes = 30  # 테스트에 충분한 크기
    tree = GATree(max_nodes, 10, 5, FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL)

    # 피쳐 이름과 인덱스를 매핑 (가독성 향상)
    feat_map = {name: i for i, name in enumerate(tree.all_features)}

    # [Node ID] Description (Condition -> Expected Result for test features)
    # --------------------------------------------------------------------
    # [ 0] ROOT_BRANCH (LONG)
    # │
    # ├──[ 3] DECISION (RSI >= 80) -> FALSE
    # │   └──[ 6] ACTION (DECOY) -> 도달해서는 안 됨
    # │
    # └──[ 4] DECISION (ATR <= 0.5) -> TRUE
    #     │
    #     ├──[ 7] ACTION (LONG, 0.5, 20x) -> *** 정답! 최단 경로(깊이 2)의 Action ***
    #     │
    #     └──[ 8] DECISION (SMA_5 >= SMA_20) -> TRUE
    #         └──[ 9] ACTION (SHORT, 0.9, 50x) -> 깊이 3의 유효한 Action. BFS는 여기 오기 전에 7번에서 멈춰야 함.
    #
    # [ 1] ROOT_BRANCH (HOLD)
    # │
    # └──[ 5] DECISION (STOCH_K >= 50) -> TRUE
    #     └──[10] ACTION (SHORT, 0.25, 5x) -> 유일한 경로
    #
    # [ 2] ROOT_BRANCH (SHORT)
    # │
    # └──[11] DECISION (WR <= -90) -> FALSE
    #     └──[12] ACTION (DECOY) -> 도달해서는 안 됨 (Dead End)

    # 데이터 텐서를 직접 채워넣기
    tree.data[0] = torch.tensor([NODE_TYPE_ROOT_BRANCH, -1, 0, ROOT_BRANCH_LONG, 0, 0, 0])
    tree.data[1] = torch.tensor([NODE_TYPE_ROOT_BRANCH, -1, 0, ROOT_BRANCH_HOLD, 0, 0, 0])
    tree.data[2] = torch.tensor([NODE_TYPE_ROOT_BRANCH, -1, 0, ROOT_BRANCH_SHORT, 0, 0, 0])

    # --- LONG Branch ---
    tree.data[3] = torch.tensor([NODE_TYPE_DECISION, 0, 1, feat_map['RSI'], OP_GTE, COMP_TYPE_FEAT_NUM, 80])
    tree.data[4] = torch.tensor([NODE_TYPE_DECISION, 0, 1, feat_map['ATR'], OP_LTE, COMP_TYPE_FEAT_NUM, 0.5])
    tree.data[6] = torch.tensor([NODE_TYPE_ACTION, 3, 2, POS_TYPE_SHORT, 0.1, 1, 0]) # Decoy Action
    tree.data[7] = torch.tensor([NODE_TYPE_ACTION, 4, 2, POS_TYPE_LONG, 0.5, 20, 0]) # *** SHALLOWEST VALID ACTION ***
    tree.data[8] = torch.tensor([NODE_TYPE_DECISION, 4, 2, feat_map['SMA_5'], OP_GTE, COMP_TYPE_FEAT_FEAT, feat_map['SMA_20']])
    tree.data[9] = torch.tensor([NODE_TYPE_ACTION, 8, 3, POS_TYPE_SHORT, 0.9, 50, 0]) # Deeper valid action

    # --- HOLD Branch ---
    tree.data[5] = torch.tensor([NODE_TYPE_DECISION, 1, 1, feat_map['STOCH_K'], OP_GTE, COMP_TYPE_FEAT_NUM, 50])
    tree.data[10] = torch.tensor([NODE_TYPE_ACTION, 5, 2, POS_TYPE_SHORT, 0.25, 5, 0])

    # --- SHORT Branch ---
    tree.data[11] = torch.tensor([NODE_TYPE_DECISION, 2, 1, feat_map['WR'], OP_LTE, COMP_TYPE_FEAT_NUM, -90])
    tree.data[12] = torch.tensor([NODE_TYPE_ACTION, 11, 2, POS_TYPE_LONG, 0.3, 3, 0]) # Decoy Action

    # 트리 상태 마무리
    tree.next_idx = 13
    tree._build_adjacency_list()  # predict를 위해 필수
    tree.initialized = True
    print("Manual tree created successfully.")
    return tree

def define_test_feature_values():
    """
    수동으로 만든 트리의 특정 경로를 활성화/비활성화하기 위한 피쳐 값들을 정의합니다.
    """
    print("--- 2. Defining test feature values ---")
    feature_values = {
        'RSI': 75,        # -> RSI >= 80 (FALSE)
        'ATR': 0.4,       # -> ATR <= 0.5 (TRUE)
        'SMA_5': 101,     # -> SMA_5 >= SMA_20 (TRUE)
        'SMA_20': 100,
        'STOCH_K': 60,    # -> STOCH_K >= 50 (TRUE)
        'WR': -85,        # -> WR <= -90 (FALSE)
        # 테스트에 사용되지 않는 나머지 피쳐들
        'IsBullishMarket': False,
        'IsHighVolatility': True,
        'EMA_10': 0, 'EMA_30': 0, 'BB_upper': 0, 'BB_lower': 0,
    }
    print(f"Features: {feature_values}")
    return feature_values

def run_tests():
    """메인 테스트 함수"""
    test_tree = create_manual_test_tree()
    feature_values = define_test_feature_values()

    # 시각화를 통해 트리 구조 확인 (디버깅에 매우 유용)
    vis_file = "test_tree_structure.html"
    test_tree.visualize_graph(file=vis_file, open_browser=False)
    print(f"\nTest tree structure saved to '{vis_file}'. Open this file in a browser to see the visual layout.")

    print("\n--- 3. Running Prediction Tests ---")

    # === Test Case 1: LONG Position (최단 경로 검증) ===
    print("\n[Test 1] Position: LONG")
    expected_long_action = ('LONG', 0.5, 20)
    print(f"Expected Action: {expected_long_action} (from shallowest valid node ID 7)")
    actual_long_action = test_tree.predict(feature_values, 'LONG')
    print(f"Actual Action:   {actual_long_action}")

    assert actual_long_action == expected_long_action, "FAIL: Did not select the shortest path action!"
    print("✅ PASS: Correctly selected the shallowest valid action.")

    # === Test Case 2: HOLD Position (단일 경로 검증) ===
    print("\n[Test 2] Position: HOLD")
    expected_hold_action = ('SHORT', 0.25, 5)
    print(f"Expected Action: {expected_hold_action} (from the only valid node ID 10)")
    actual_hold_action = test_tree.predict(feature_values, 'HOLD')
    print(f"Actual Action:   {actual_hold_action}")

    assert actual_hold_action == expected_hold_action, "FAIL: Did not find the correct action for HOLD branch!"
    print("✅ PASS: Correctly found the action in a single-path branch.")

    # === Test Case 3: SHORT Position (유효 경로 없음 검증) ===
    print("\n[Test 3] Position: SHORT")
    expected_short_action = ('HOLD', 0.0, 0)
    print(f"Expected Action: {expected_short_action} (default action as no path is valid)")
    actual_short_action = test_tree.predict(feature_values, 'SHORT')
    print(f"Actual Action:   {actual_short_action}")

    assert actual_short_action == expected_short_action, "FAIL: Did not return the default HOLD action for a dead-end branch!"
    print("✅ PASS: Correctly returned default action when no path was valid.")

    print("\n\n🎉 All tests passed successfully! 🎉")


if __name__ == '__main__':
    run_tests()