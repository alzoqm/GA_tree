# test_prediction_advanced.py
import torch
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GA_tree.models.model import (
    GATree,
    # 상수 임포트
    COL_NODE_TYPE, COL_PARENT_IDX, COL_DEPTH, COL_PARAM_1, COL_PARAM_2, COL_PARAM_3, COL_PARAM_4, NODE_INFO_DIM,
    NODE_TYPE_UNUSED, NODE_TYPE_ROOT_BRANCH, NODE_TYPE_DECISION, NODE_TYPE_ACTION,
    ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT,
    COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_FEAT,
    OP_GT, OP_LT, OP_EQ,
    POS_TYPE_LONG, POS_TYPE_SHORT,
    # 설정 임포트
    FEATURE_NUM, FEATURE_PAIR, ALL_FEATURES
)

def create_advanced_test_tree():
    """
    더 복잡하고 다양한 엣지 케이스를 포함하는 GATree 객체를 생성합니다.
    """
    print("--- Creating an advanced, manually defined GATree for testing ---")
    
    config = {
        'max_nodes': 50, 'max_depth': 10, 'max_children': 5,
        'feature_num': FEATURE_NUM, 'feature_pair': FEATURE_PAIR
    }
    tree = GATree(**config)
    feat_map = {name: i for i, name in enumerate(ALL_FEATURES)}

    # [TYPE, PARENT, DEPTH, P1, P2, P3, P4]
    test_tree_data_list = [
        # 0: ROOT_BRANCH_LONG
        [NODE_TYPE_ROOT_BRANCH, -1, 0, ROOT_BRANCH_LONG, 0, 0, 0],
        # 1: ROOT_BRANCH_HOLD
        [NODE_TYPE_ROOT_BRANCH, -1, 0, ROOT_BRANCH_HOLD, 0, 0, 0],
        # 2: ROOT_BRANCH_SHORT
        [NODE_TYPE_ROOT_BRANCH, -1, 0, ROOT_BRANCH_SHORT, 0, 0, 0],
        
        # --- 'LONG' 경로 (복잡) ---
        # 3: (P:0) IF RSI > 70
        [NODE_TYPE_DECISION, 0, 1, feat_map['RSI'], OP_GT, COMP_TYPE_FEAT_NUM, 70],
        # 4: (P:3) IF SMA_5 > SMA_20
        [NODE_TYPE_DECISION, 3, 2, feat_map['SMA_5'], OP_GT, COMP_TYPE_FEAT_FEAT, feat_map['SMA_20']],
        # 5: (P:4) IF ATR > 0.5
        [NODE_TYPE_DECISION, 4, 3, feat_map['ATR'], OP_GT, COMP_TYPE_FEAT_NUM, 0.5],
        # 6: (P:5) ACTION: SHORT, 0.5, 10x
        [NODE_TYPE_ACTION, 5, 4, POS_TYPE_SHORT, 0.5, 10, 0],
        # 7: (P:3) IF WR < -80
        [NODE_TYPE_DECISION, 3, 2, feat_map['WR'], OP_LT, COMP_TYPE_FEAT_NUM, -80],
        # 8: (P:7) ACTION: LONG, 1.0, 20x
        [NODE_TYPE_ACTION, 7, 3, POS_TYPE_LONG, 1.0, 20, 0],
        # 9: (P:3) IF STOCH_K == 50.0
        [NODE_TYPE_DECISION, 3, 2, feat_map['STOCH_K'], OP_EQ, COMP_TYPE_FEAT_NUM, 50.0],
        # 10: (P:9) ACTION: SHORT, 0.8, 50x
        [NODE_TYPE_ACTION, 9, 3, POS_TYPE_SHORT, 0.8, 50, 0],

        # --- 'HOLD' 경로 ---
        # 11: (P:1) IF ATR > 0.5
        [NODE_TYPE_DECISION, 1, 1, feat_map['ATR'], OP_GT, COMP_TYPE_FEAT_NUM, 0.5],
        # 12: (P:11) IF RSI < 30
        [NODE_TYPE_DECISION, 11, 2, feat_map['RSI'], OP_LT, COMP_TYPE_FEAT_NUM, 30],
        # 13: (P:12) ACTION: LONG, 0.3, 3x
        [NODE_TYPE_ACTION, 12, 3, POS_TYPE_LONG, 0.3, 3, 0],
        # 14: (P:1) IF BB_upper > SMA_20 (고아 Decision 노드)
        [NODE_TYPE_DECISION, 1, 1, feat_map['BB_upper'], OP_GT, COMP_TYPE_FEAT_FEAT, feat_map['SMA_20']],

        # --- 'SHORT' 경로 ---
        # 15: (P:2) ACTION: LONG, 0.2, 5x (우선순위 1)
        [NODE_TYPE_ACTION, 2, 1, POS_TYPE_LONG, 0.2, 5, 0],
        # 16: (P:2) IF RSI < 30 (우선순위 2)
        [NODE_TYPE_DECISION, 2, 1, feat_map['RSI'], OP_LT, COMP_TYPE_FEAT_NUM, 30],
        # 17: (P:16) ACTION: SHORT, 0.9, 15x
        [NODE_TYPE_ACTION, 16, 2, POS_TYPE_SHORT, 0.9, 15, 0],
    ]
    
    num_nodes = len(test_tree_data_list)
    test_tensor = torch.zeros((config['max_nodes'], NODE_INFO_DIM))
    test_tensor[:num_nodes] = torch.tensor(test_tree_data_list, dtype=torch.float32)
    
    tree.data.copy_(test_tensor)
    tree.next_idx = num_nodes
    tree.initialized = True
    
    print(f"Advanced manual tree created successfully with {num_nodes} nodes.")
    return tree

def compare_results(actual, expected):
    """결과 튜플을 비교 (부동소수점 오차 고려)"""
    if not isinstance(actual, tuple) or len(actual) != 3: return False
    if actual[0] != expected[0]: return False
    if abs(actual[1] - expected[1]) > 1e-6: return False
    if actual[2] != expected[2]: return False
    return True

if __name__ == '__main__':
    test_tree = create_advanced_test_tree()
    print("\nVisualizing the advanced test tree... (saved to test_tree_advanced_visualization.html)")
    test_tree.visualize_graph(file="test_tree_advanced_visualization.html", open_browser=False)

    test_cases = [
        # --- 기본 경로 테스트 ---
        {
            "id": 1, "description": "가장 깊은 경로 성공 (0->3->4->5->6)",
            "position": "LONG",
            "features": {'RSI': 80, 'SMA_5': 110, 'SMA_20': 100, 'ATR': 0.6, 'WR': -50, 'STOCH_K': 20},
            "expected": ('SHORT', 0.5, 10)
        },
        {
            "id": 2, "description": "형제 분기 경로 성공 (0->3->7->8)",
            "position": "LONG",
            "features": {'RSI': 80, 'SMA_5': 90, 'SMA_20': 100, 'ATR': 0.6, 'WR': -90, 'STOCH_K': 20},
            "expected": ('LONG', 1.0, 20)
        },
        {
            "id": 3, "description": "부동소수점 비교 경로 성공 (0->3->9->10)",
            "position": "LONG",
            "features": {'RSI': 80, 'SMA_5': 90, 'SMA_20': 100, 'ATR': 0.6, 'WR': -50, 'STOCH_K': 50.0},
            "expected": ('SHORT', 0.8, 50)
        },
        # --- 엣지 케이스 및 실패 경로 테스트 ---
        {
            "id": 4, "description": "중간 경로 실패 후 다른 분기 성공 (0->3->(4실패)->7->8)",
            "position": "LONG",
            "features": {'RSI': 80, 'SMA_5': 100, 'SMA_20': 110, 'ATR': 0.4, 'WR': -90, 'STOCH_K': 20},
            "expected": ('LONG', 1.0, 20)
        },
        {
            "id": 5, "description": "DFS 우선순위 확인 (모든 경로 참일 때 첫 경로 선택)",
            "position": "LONG",
            "features": {'RSI': 80, 'SMA_5': 110, 'SMA_20': 100, 'ATR': 0.6, 'WR': -90, 'STOCH_K': 50.0},
            "expected": ('SHORT', 0.5, 10) # 0->3->4... 경로가 가장 먼저 탐색됨
        },
        {
            "id": 6, "description": "HOLD 경로 성공 (1->11->12->13)",
            "position": "HOLD",
            "features": {'ATR': 0.6, 'RSI': 20},
            "expected": ('LONG', 0.3, 3)
        },
        {
            "id": 7, "description": "고아 Decision 노드 경로 (HOLD 반환)",
            "position": "HOLD",
            "features": {'ATR': 0.4, 'BB_upper': 120, 'SMA_20': 100}, # 11번 경로는 실패, 14번 경로 참
            "expected": ('HOLD', 0.0, 0)
        },
        {
            "id": 8, "description": "필수 피처 부재 (HOLD 반환)",
            "position": "LONG",
            "features": {'SMA_5': 110, 'SMA_20': 100}, # RSI가 없어 3번 노드에서 실패
            "expected": ('HOLD', 0.0, 0)
        },
        {
            "id": 9, "description": "SHORT 경로 탐색 시 Action 우선순위 확인",
            "position": "SHORT",
            "features": {'RSI': 20}, # 16번 경로도 참이지만 15번 Action이 먼저 발견되어야 함
            "expected": ('LONG', 0.2, 5)
        },
        {
            "id": 10, "description": "모든 경로 실패 (HOLD 반환)",
            "position": "LONG",
            "features": {'RSI': 60, 'SMA_5': 100, 'SMA_20': 110, 'ATR': 0.4, 'WR': -50, 'STOCH_K': 20},
            "expected": ('HOLD', 0.0, 0)
        }
    ]

    print("\n--- Running Advanced Prediction Tests ---")
    passed_count = 0
    for case in test_cases:
        print(f"\n[Test Case #{case['id']:02d}: {case['description']}]")
        print(f"  - Input Position: {case['position']}")
        print(f"  - Input Features: {case['features']}")
        
        try:
            actual_result = test_tree.predict(case['features'], case['position'])
            print(f"  - Expected Result: {case['expected']}")
            print(f"  - Actual Result:   {actual_result}")
            
            if compare_results(actual_result, case['expected']):
                print("  - Verdict: PASSED ✔️")
                passed_count += 1
            else:
                print("  - Verdict: FAILED ❌")
        except Exception as e:
            print(f"  - AN ERROR OCCURRED: {e}")
            print(f"  - Expected Result: {case['expected']}")
            print("  - Verdict: ERROR ❌")

    print("\n--- Test Summary ---")
    print(f"Total tests: {len(test_cases)}, Passed: {passed_count}, Failed/Error: {len(test_cases) - passed_count}")
    
    if passed_count == len(test_cases):
        print("\nAll advanced predict() tests passed successfully!")
    else:
        print("\nSome advanced predict() tests failed. Please review the output.")