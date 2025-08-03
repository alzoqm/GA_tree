# --- START OF FILE models/model.py ---

import torch
import random
import os
import webbrowser
import networkx as nx

# pyvis 라이브러리 가용성 확인
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

# -------------------------------------
# --- 상수 정의 (C/C++ 연동을 위해 중요)
# -------------------------------------

# Tensor Column Indices
COL_NODE_TYPE = 0
COL_PARENT_IDX = 1
COL_DEPTH = 2
COL_PARAM_1 = 3
COL_PARAM_2 = 4
COL_PARAM_3 = 5
COL_PARAM_4 = 6
NODE_INFO_DIM = 7  # 총 컬럼 수

# Node Types
NODE_TYPE_UNUSED = 0
NODE_TYPE_ROOT_BRANCH = 1
NODE_TYPE_DECISION = 2
NODE_TYPE_ACTION = 3

# Root Branch Types
ROOT_BRANCH_LONG = 0
ROOT_BRANCH_HOLD = 1
ROOT_BRANCH_SHORT = 2

# Decision Node: Comparison Types
COMP_TYPE_FEAT_NUM = 0  # Feature vs Number
COMP_TYPE_FEAT_FEAT = 1 # Feature vs Feature
COMP_TYPE_FEAT_BOOL = 2 # Feature vs Boolean

# Decision Node: Operators
OP_GTE = 0  # >= (Greater Than or Equal)
OP_LTE = 1  # <= (Less Than or Equal)

# --- [신규] Action Node: Action Types (COL_PARAM_1에 저장) ---
ACTION_NEW_LONG        = 1  # 신규 롱 포지션 진입
ACTION_NEW_SHORT       = 2  # 신규 숏 포지션 진입
ACTION_CLOSE_ALL       = 3  # 현재 포지션 전체 청산
ACTION_CLOSE_PARTIAL   = 4  # 현재 포지션 부분 청산 (비율 지정)
ACTION_ADD_POSITION    = 5  # 현재 포지션 추가 진입 (피라미딩)
ACTION_FLIP_POSITION   = 6  # 현재 포지션 청산 후 즉시 반대 포지션 진입

# --- 예시 Feature 및 설정 (실제 사용 시 변경) ---
FEATURE_NUM = {'RSI': (0, 100), 'ATR': (0, 1), 'WR': (-100, 0), 'STOCH_K':(0, 100)}
FEATURE_BOOL = ['IsBullishMarket', 'IsHighVolatility']

FEATURE_COMPARISON_MAP = {
    'SMA_5': ['SMA_20', 'EMA_10', 'EMA_30'],
    'SMA_20': ['SMA_5', 'EMA_10', 'EMA_30'],
    'EMA_10': ['SMA_5', 'SMA_20', 'BB_upper', 'BB_lower'],
    'EMA_30': ['SMA_5', 'SMA_20'],
    'BB_upper': ['EMA_10', 'BB_lower'],
    'BB_lower': ['EMA_10', 'BB_upper']
}

def get_all_features(feature_num, feature_map, feature_bool):
    comp_features = set(feature_map.keys())
    for v_list in feature_map.values():
        comp_features.update(v_list)
    return list(feature_num.keys()) + sorted(list(comp_features)) + feature_bool

ALL_FEATURES = get_all_features(FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL)


# --- Helper 딕셔너리 (시각화 및 디버깅용) ---
NODE_TYPE_MAP = {
    NODE_TYPE_UNUSED: "UNUSED",
    NODE_TYPE_ROOT_BRANCH: "ROOT_BRANCH",
    NODE_TYPE_DECISION: "DECISION",
    NODE_TYPE_ACTION: "ACTION",
}
ROOT_BRANCH_MAP = {ROOT_BRANCH_LONG: "IF_POS_IS_LONG", ROOT_BRANCH_HOLD: "IF_POS_IS_HOLD", ROOT_BRANCH_SHORT: "IF_POS_IS_SHORT"}
OPERATOR_MAP = {OP_GTE: ">=", OP_LTE: "<="}
# --- [신규] 기존 POS_TYPE_MAP을 ACTION_TYPE_MAP으로 대체 ---
ACTION_TYPE_MAP = {
    ACTION_NEW_LONG: "NEW_LONG",
    ACTION_NEW_SHORT: "NEW_SHORT",
    ACTION_CLOSE_ALL: "CLOSE_ALL",
    ACTION_CLOSE_PARTIAL: "CLOSE_PARTIAL",
    ACTION_ADD_POSITION: "ADD_POSITION",
    ACTION_FLIP_POSITION: "FLIP_POSITION",
}

class GATree:
    """
    하나의 유전 알고리즘 트리를 나타내는 클래스.
    이 클래스의 데이터는 C++/CUDA에서 직접 처리할 수 있는 torch.Tensor 형식으로 저장됩니다.
    """
    def __init__(self, max_nodes, max_depth, max_children, feature_num, feature_comparison_map, feature_bool, data_tensor=None):
        """
        GATree 초기화.

        Args:
            max_nodes (int): 트리가 가질 수 있는 최대 노드 수.
            max_depth (int): 트리의 최대 깊이.
            max_children (int): Decision 노드가 가질 수 있는 최대 자식 노드 수.
            feature_num (dict): 숫자와 비교할 피쳐와 (min, max) 범위.
            feature_comparison_map (dict): 피처 간 비교 규칙을 정의한 맵.
            feature_bool (list): Boolean과 비교할 피쳐 리스트.
            data_tensor (torch.Tensor, optional): 외부에서 생성된 텐서의 view.
                                                  None이면 자체적으로 텐서를 생성합니다.
        """
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_children = max_children
        self.feature_num = feature_num
        self.feature_comparison_map = feature_comparison_map
        self.feature_bool = feature_bool

        comp_features = set(self.feature_comparison_map.keys())
        for v_list in self.feature_comparison_map.values():
            comp_features.update(v_list)
        self.all_features = list(feature_num.keys()) + sorted(list(comp_features)) + feature_bool

        self.initialized = False
        self.next_idx = 0

        if data_tensor is None:
            self.data = torch.zeros((self.max_nodes, NODE_INFO_DIM), dtype=torch.float32)
        else:
            if data_tensor.shape != (self.max_nodes, NODE_INFO_DIM):
                raise ValueError("Provided data_tensor has incorrect shape")
            self.data = data_tensor

        self._bfs_queue = torch.zeros(max_nodes, dtype=torch.int32)
        self._queue_head = 0
        self._queue_tail = 0
        self._adjacency_list = {}

    def _get_root_branch_type_from_child(self, start_node_idx: int) -> int:
        """
        [신규 헬퍼 함수]
        주어진 노드에서부터 부모를 거슬러 올라가 최상위 루트 분기의 타입을 찾습니다.
        """
        current_idx = start_node_idx
        # 부모 인덱스가 -1 (루트)가 아닐 동안 계속 올라감
        while self.data[current_idx, COL_PARENT_IDX].item() != -1:
            current_idx = int(self.data[current_idx, COL_PARENT_IDX].item())
        
        # 최상위 노드 (루트 분기 노드)의 타입을 반환
        return int(self.data[current_idx, COL_PARAM_1].item())

    def _build_adjacency_list(self):
        """
        self.data 텐서가 완성된 후, 이를 기반으로 부모-자식 관계를 맵(딕셔너리)으로 만들어
        _adjacency_list 속성에 저장합니다.
        """
        self._adjacency_list.clear()
        for i in range(self.next_idx):
            if self.data[i, COL_NODE_TYPE] == NODE_TYPE_UNUSED:
                continue

            parent_id = int(self.data[i, COL_PARENT_IDX].item())
            if parent_id != -1:
                if parent_id not in self._adjacency_list:
                    self._adjacency_list[parent_id] = []
                self._adjacency_list[parent_id].append(i)


    def make_tree(self):
        """
        [수정됨] 요구사항에 맞는 랜덤 트리를 생성하고 현재 객체를 초기화합니다.
        브랜치 간 노드 불균형 문제를 해결하기 위해 예산을 사전에 균등하게 분배합니다.
        """
        self.data.zero_()
        self.initialized = False
        self.next_idx = 0

        root_branches = [ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT]
        root_branch_ids = []
        for branch_type in root_branches:
            idx = self._get_next_idx()
            self.data[idx, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
            self.data[idx, COL_PARENT_IDX] = -1
            self.data[idx, COL_DEPTH] = 0
            self.data[idx, COL_PARAM_1] = branch_type
            root_branch_ids.append(idx)
        
        # [수정된 로직 1] 전체 생성할 노드 수를 합리적인 범위 내에서 결정
        min_nodes_to_create = (self.max_nodes - 3) // 2
        max_nodes_to_create = self.max_nodes - 3
        # 생성할 노드가 0보다 작아지는 경우 방지
        if min_nodes_to_create > max_nodes_to_create:
            min_nodes_to_create = max_nodes_to_create
        
        total_nodes = random.randint(min_nodes_to_create, max_nodes_to_create)

        # [수정된 로직 2] 브랜치별 예산을 최대한 균등하게 분배
        nodes_per_branch = total_nodes // 3
        remainder = total_nodes % 3
        budgets = [nodes_per_branch] * 3
        for i in range(remainder):
            budgets[i] += 1
        random.shuffle(budgets)  # 어떤 브랜치가 더 많은 예산을 받을지 무작위로 결정

        for i, branch_id in enumerate(root_branch_ids):
            self._grow_branch(branch_id, budgets[i])

        self._build_adjacency_list()
        self.initialized = True
        print(f"Tree created with {self.next_idx} nodes.")

    def _grow_branch(self, branch_root_id, budget):
        """
        [전면 수정됨] 한 분기(LONG/HOLD/SHORT) 아래의 트리를 성장시키는 내부 함수.
        'Two-List 관리 시스템'과 '미래 예측 안전 점검' 로직을 사용하여 안정적인 트리를 생성합니다.
        """
        nodes_to_create = budget
        if nodes_to_create <= 0:
            if self.next_idx < self.max_nodes:
                self._create_action_node(branch_root_id)
            return

        # "Two-List 관리 시스템" 초기화
        # open_list: 자식을 더 가질 수 있는 Decision 노드 목록 (성장 후보)
        open_list = [branch_root_id]
        # leaf_list: 현재 자식이 없는 순수 말단 노드 목록 (종료 대상)
        leaf_list = [branch_root_id]
        # child_counts: 각 부모 노드의 현재 자식 수를 추적
        child_counts = {branch_root_id: 0}

        while nodes_to_create > 0 and open_list:
            # [핵심 로직 1] 미래 예측 안전 점검 (Lookahead Safety Check)
            # 남은 예산이 모든 리프를 Action 노드로 덮기에도 빠듯하거나 부족한지 확인
            if nodes_to_create <= len(leaf_list)+1:
                # "강제 종료 모드"로 전환하여 모든 리프에 Action 노드를 붙이고 종료
                for parent_id in leaf_list:
                    if self.next_idx < self.max_nodes:
                        self._create_action_node(parent_id)
                break  # 브랜치 생성 종료

            # [핵심 로직 2] 일반 성장 모드
            parent_id = random.choice(open_list)
            parent_depth = self.data[parent_id, COL_DEPTH].item()

            # CASE 1: 최대 깊이에 도달하여 Action 노드를 생성해야 하는 경우
            if parent_depth >= self.max_depth - 1:
                new_node_id = self._create_action_node(parent_id)
                if new_node_id is not None:
                    nodes_to_create -= 1
                
                # 리스트 관리: 해당 부모는 더 이상 성장/리프가 아님
                open_list.remove(parent_id)
                if parent_id in leaf_list:
                    leaf_list.remove(parent_id)
                continue

            # CASE 2: 일반적인 Decision 노드 생성
            child_id = self._create_decision_node(parent_id)
            
            if child_id is not None:
                # 노드 생성 성공 시
                nodes_to_create -= 1
                
                # 자식 수 카운트 업데이트
                child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
                child_counts[child_id] = 0

                # 리스트 관리
                if parent_id in leaf_list:
                    leaf_list.remove(parent_id) # 부모는 더 이상 리프가 아님
                leaf_list.append(child_id)      # 자식은 새로운 리프임
                open_list.append(child_id)      # 자식은 새로운 성장 후보임

                # 부모의 자식 수가 최대치에 도달하면 성장 후보에서 제외
                if child_counts[parent_id] >= self.max_children:
                    open_list.remove(parent_id)
            else:
                # 노드 생성 실패 시 (e.g. 전역 노드 풀 고갈)
                # 해당 부모는 더 이상 성장할 수 없으므로 성장 후보에서 제외
                open_list.remove(parent_id)

    def _create_action_node(self, parent_id):
        """[전면 수정] Action 노드 하나를 문맥에 맞게 생성하고 Tensor에 기록"""
        idx = self._get_next_idx()
        if idx is None: return None

        self.data[idx, COL_NODE_TYPE] = NODE_TYPE_ACTION
        self.data[idx, COL_PARENT_IDX] = parent_id
        self.data[idx, COL_DEPTH] = self.data[parent_id, COL_DEPTH] + 1

        # 1. 부모로부터 루트 분기 타입을 결정
        root_branch_type = self._get_root_branch_type_from_child(parent_id)

        # 2. 루트 분기 타입(문맥)에 따라 가능한 Action 리스트를 정의
        if root_branch_type == ROOT_BRANCH_HOLD:
            possible_actions = [ACTION_NEW_LONG, ACTION_NEW_SHORT]
        elif root_branch_type == ROOT_BRANCH_LONG:
            possible_actions = [ACTION_CLOSE_ALL, ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION, ACTION_FLIP_POSITION]
        elif root_branch_type == ROOT_BRANCH_SHORT:
            possible_actions = [ACTION_CLOSE_ALL, ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION, ACTION_FLIP_POSITION]
        else: # 혹시 모를 예외 처리
            possible_actions = []

        if not possible_actions:
            # 생성할 유효한 액션이 없으면 노드를 UNUSED로 처리하고 반납
            self.data[idx, COL_NODE_TYPE] = NODE_TYPE_UNUSED
            self.next_idx -= 1
            return None

        # 3. 선택된 Action 타입에 따라 파라미터를 랜덤하게 생성
        chosen_action = random.choice(possible_actions)
        self.data[idx, COL_PARAM_1] = chosen_action
        
        # 파라미터 2, 3은 Action 종류에 따라 의미가 달라짐
        if chosen_action in [ACTION_NEW_LONG, ACTION_NEW_SHORT]:
            # Param2: 진입 비중 (0~1), Param3: 레버리지 (1~100)
            self.data[idx, COL_PARAM_2] = random.random()
            self.data[idx, COL_PARAM_3] = random.randint(1, 100)
        elif chosen_action == ACTION_FLIP_POSITION:
            # Param2: 새로운 포지션의 진입 비중, Param3: 새로운 포지션의 레버리지
            self.data[idx, COL_PARAM_2] = random.random()
            self.data[idx, COL_PARAM_3] = random.randint(1, 100)
        elif chosen_action == ACTION_CLOSE_PARTIAL:
            # Param2: 청산 비율 (0~1), Param3: 사용 안함
            self.data[idx, COL_PARAM_2] = random.random()
        elif chosen_action == ACTION_ADD_POSITION:
            # Param2: 추가 진입 비중 (0~1), Param3: 사용 안함
            self.data[idx, COL_PARAM_2] = random.random()
        # ACTION_CLOSE_ALL은 추가 파라미터가 필요 없음

        return idx

    def _create_decision_node(self, parent_id):
        """Decision 노드 하나를 생성하고 Tensor에 기록"""
        idx = self._get_next_idx()
        if idx is None: return None

        self.data[idx, COL_NODE_TYPE] = NODE_TYPE_DECISION
        self.data[idx, COL_PARENT_IDX] = parent_id
        self.data[idx, COL_DEPTH] = self.data[parent_id, COL_DEPTH] + 1

        comp_type_choices = [COMP_TYPE_FEAT_NUM]
        if self.feature_comparison_map:
            comp_type_choices.append(COMP_TYPE_FEAT_FEAT)
        if self.feature_bool:
            comp_type_choices.append(COMP_TYPE_FEAT_BOOL)

        comp_type = random.choice(comp_type_choices)
        self.data[idx, COL_PARAM_3] = comp_type

        if comp_type == COMP_TYPE_FEAT_NUM or comp_type == COMP_TYPE_FEAT_FEAT:
            self.data[idx, COL_PARAM_2] = random.choice([OP_GTE, OP_LTE])

        if comp_type == COMP_TYPE_FEAT_NUM:
            feat_name = random.choice(list(self.feature_num.keys()))
            feat_idx = self.all_features.index(feat_name)
            min_val, max_val = self.feature_num[feat_name]
            comp_val = random.uniform(min_val, max_val)

            self.data[idx, COL_PARAM_1] = feat_idx
            self.data[idx, COL_PARAM_4] = comp_val

        elif comp_type == COMP_TYPE_FEAT_FEAT:
            possible_feat1 = [k for k, v in self.feature_comparison_map.items() if v]
            if not possible_feat1:
                self.data[idx, COL_NODE_TYPE] = NODE_TYPE_UNUSED
                self.next_idx -= 1
                return None

            feat1_name = random.choice(possible_feat1)
            feat2_name = random.choice(self.feature_comparison_map[feat1_name])

            feat1_idx = self.all_features.index(feat1_name)
            feat2_idx = self.all_features.index(feat2_name)

            self.data[idx, COL_PARAM_1] = feat1_idx
            self.data[idx, COL_PARAM_4] = feat2_idx

        elif comp_type == COMP_TYPE_FEAT_BOOL:
            feat_name = random.choice(self.feature_bool)
            feat_idx = self.all_features.index(feat_name)
            comp_val = random.choice([0.0, 1.0])

            self.data[idx, COL_PARAM_1] = feat_idx
            self.data[idx, COL_PARAM_4] = comp_val

        return idx

    def _get_next_idx(self):
        """사용 가능한 다음 노드 인덱스를 반환"""
        if self.next_idx < self.max_nodes:
            idx = self.next_idx
            self.next_idx += 1
            return idx
        return None

    def reorganize_nodes(self):
        """
        텐서 내의 노드들을 재조직하여 빈 공간(단편화)을 제거합니다.
        """
        if not self.initialized:
            print("Warning: Reorganizing an uninitialized tree.")
            return

        active_mask = self.data[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED
        active_indices = active_mask.nonzero(as_tuple=True)[0]

        if len(active_indices) == 0:
            self.data.zero_()
            self.next_idx = 0
            self._adjacency_list.clear()
            return

        active_node_data = self.data[active_indices]
        num_active_nodes = len(active_indices)

        old_to_new_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(active_indices)}

        new_data = torch.zeros_like(self.data)
        new_data[:num_active_nodes] = active_node_data

        for i in range(num_active_nodes):
            old_parent_idx = int(new_data[i, COL_PARENT_IDX].item())

            if old_parent_idx != -1:
                if old_parent_idx in old_to_new_map:
                    new_parent_idx = old_to_new_map[old_parent_idx]
                    new_data[i, COL_PARENT_IDX] = new_parent_idx
                else:
                    new_data[i].zero_()
                    new_data[i, COL_NODE_TYPE] = NODE_TYPE_UNUSED

        self.data.copy_(new_data)
        self.next_idx = num_active_nodes
        
        self._build_adjacency_list()


    def set_next_idx(self):
        """
        현재 텐서 상태를 기반으로 next_idx 값을 재설정합니다.
        """
        if not self.initialized:
            self.next_idx = 0
            return

        num_active_nodes = (self.data[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED).sum().item()
        self.next_idx = num_active_nodes

    def return_next_idx(self) -> int:
        """
        현재 트리의 next_idx 값을 반환합니다.
        """
        return self.next_idx

    def _evaluate_node(self, node_idx, feature_values):
        """주어진 Decision 노드의 조건을 평가하여 참/거짓을 반환합니다."""
        node_info = self.data[node_idx]

        comp_type = int(node_info[COL_PARAM_3].item())
        feat1_idx = int(node_info[COL_PARAM_1].item())
        feat1_name = self.all_features[feat1_idx]

        if feat1_name not in feature_values:
            return False
        val1 = feature_values[feat1_name]

        if comp_type == COMP_TYPE_FEAT_NUM:
            op = int(node_info[COL_PARAM_2].item())
            val2 = node_info[COL_PARAM_4].item()
            if op == OP_GTE: return val1 >= val2
            if op == OP_LTE: return val1 <= val2

        elif comp_type == COMP_TYPE_FEAT_FEAT:
            op = int(node_info[COL_PARAM_2].item())
            feat2_idx = int(node_info[COL_PARAM_4].item())
            feat2_name = self.all_features[feat2_idx]
            if feat2_name not in feature_values:
                return False
            val2 = feature_values[feat2_name]
            if op == OP_GTE: return val1 >= val2
            if op == OP_LTE: return val1 <= val2

        elif comp_type == COMP_TYPE_FEAT_BOOL:
            val2 = node_info[COL_PARAM_4].item()
            return val1 == val2

        return False

    def predict(self, feature_values, current_position):
        """
        [수정] BFS를 사용하여 최단 경로의 Action을 찾고, 새로운 Action 체계의 파라미터를 반환합니다.
        """
        if not self.initialized:
            raise RuntimeError("Tree is not initialized. Call make_tree() or load() first.")

        pos_map = {'LONG': ROOT_BRANCH_LONG, 'HOLD': ROOT_BRANCH_HOLD, 'SHORT': ROOT_BRANCH_SHORT}
        if current_position not in pos_map:
            raise ValueError(f"Invalid current_position: {current_position}")
        target_branch_type = pos_map[current_position]

        start_node_idx = -1
        for i in range(3):
            if self.data[i, COL_NODE_TYPE] == NODE_TYPE_ROOT_BRANCH and \
               self.data[i, COL_PARAM_1] == target_branch_type:
                start_node_idx = i
                break

        if start_node_idx == -1:
            return (None, 0.0, 0.0, 0.0) # 기본 HOLD 동작 유도

        self._queue_head = 0
        self._queue_tail = 0
        self._bfs_queue[self._queue_tail] = start_node_idx
        self._queue_tail += 1

        while self._queue_head < self._queue_tail:
            current_node_idx = self._bfs_queue[self._queue_head]
            self._queue_head += 1

            child_indices = self._adjacency_list.get(int(current_node_idx.item()), [])

            for child_idx in child_indices:
                child_node_type = int(self.data[child_idx, COL_NODE_TYPE].item())

                if child_node_type == NODE_TYPE_ACTION:
                    # [수정] Action 노드의 파라미터를 그대로 반환
                    action_type = int(self.data[child_idx, COL_PARAM_1].item())
                    param_2 = self.data[child_idx, COL_PARAM_2].item()
                    param_3 = self.data[child_idx, COL_PARAM_3].item()
                    param_4 = self.data[child_idx, COL_PARAM_4].item()
                    return (action_type, param_2, param_3, param_4)

                elif child_node_type == NODE_TYPE_DECISION:
                    if self._evaluate_node(child_idx, feature_values):
                        if self._queue_tail < self.max_nodes:
                            self._bfs_queue[self._queue_tail] = child_idx
                            self._queue_tail += 1
                        else:
                            print("Warning: BFS queue is full. Prediction might be incomplete.")
                            return (None, 0.0, 0.0, 0.0)

        # Action을 찾지 못하면 None을 반환하여 HOLD 동작 유도
        return (None, 0.0, 0.0, 0.0)


    def save(self, filepath):
        """트리의 상태를 파일로 저장합니다."""
        if not self.initialized:
            print("Warning: Saving uninitialized tree.")

        state = {
            'data': self.data,
            'next_idx': self.next_idx,
            'max_nodes': self.max_nodes,
            'max_depth': self.max_depth,
            'max_children': self.max_children,
            'feature_num': self.feature_num,
            'feature_comparison_map': self.feature_comparison_map,
            'feature_bool': self.feature_bool,
        }
        torch.save(state, filepath)
        print(f"Tree saved to {filepath}")

    def load(self, source):
        """파일 또는 state_dict로부터 트리 상태를 로드합니다."""
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"File not found: {source}")
            state = torch.load(source)
        elif isinstance(source, dict):
            state = source
        else:
            raise TypeError("source must be a filepath string or a state_dict")

        feature_comparison_map = state.get('feature_comparison_map', {})
        if not feature_comparison_map and 'feature_pair' in state:
            print("Warning: Loading legacy 'feature_pair'. Converting to an empty map.")

        feature_bool = state.get('feature_bool', [])

        self.__init__(
            state['max_nodes'], state['max_depth'], state['max_children'],
            state['feature_num'], feature_comparison_map, feature_bool
        )
        self.data.copy_(state['data'])
        self.next_idx = state['next_idx']
        
        self._build_adjacency_list()
        self.initialized = True
        print(f"Tree loaded successfully.")

    def _node_label_color(self, idx):
        """[수정] 시각화를 위한 노드의 레이블과 색상을 생성합니다."""
        node = self.data[idx]
        node_type = int(node[COL_NODE_TYPE].item())

        label = f"ID: {idx}\n"
        color = "grey"

        if node_type == NODE_TYPE_ROOT_BRANCH:
            branch_type = int(node[COL_PARAM_1].item())
            label += f"START\n{ROOT_BRANCH_MAP.get(branch_type, 'UNKNOWN')}"
            color = "#FFD700"
        elif node_type == NODE_TYPE_DECISION:
            comp_type = int(node[COL_PARAM_3].item())
            feat1_idx = int(node[COL_PARAM_1].item())
            feat1_name = self.all_features[feat1_idx]

            if comp_type == COMP_TYPE_FEAT_NUM:
                op = OPERATOR_MAP.get(int(node[COL_PARAM_2].item()))
                val = node[COL_PARAM_4].item()
                label += f"IF {feat1_name} {op} {val:.2f}"
            elif comp_type == COMP_TYPE_FEAT_FEAT:
                op = OPERATOR_MAP.get(int(node[COL_PARAM_2].item()))
                feat2_idx = int(node[COL_PARAM_4].item())
                feat2_name = self.all_features[feat2_idx]
                label += f"IF {feat1_name} {op} {feat2_name}"
            elif comp_type == COMP_TYPE_FEAT_BOOL:
                bool_val = "True" if node[COL_PARAM_4].item() == 1.0 else "False"
                label += f"IF {feat1_name} == {bool_val}"
            color = "#1E90FF"
        elif node_type == NODE_TYPE_ACTION:
            # [수정] 새로운 Action 체계에 맞게 레이블 생성
            action_type = int(node[COL_PARAM_1].item())
            action_name = ACTION_TYPE_MAP.get(action_type, 'UNKNOWN_ACTION')
            label += f"ACTION: {action_name}"

            if action_type in [ACTION_NEW_LONG, ACTION_NEW_SHORT, ACTION_FLIP_POSITION]:
                size = node[COL_PARAM_2].item()
                lev = int(node[COL_PARAM_3].item())
                label += f"\nSize: {size:.2f}, Lev: {lev}x"
            elif action_type == ACTION_CLOSE_PARTIAL:
                ratio = node[COL_PARAM_2].item()
                label += f"\nRatio: {ratio:.2f}"
            elif action_type == ACTION_ADD_POSITION:
                add_size = node[COL_PARAM_2].item()
                label += f"\nAdd Size: {add_size:.2f}"
            
            color = "#32CD32"
        else:
            label += "UNUSED"

        return label, color

    def visualize_graph(self, file="ga_tree.html", open_browser=True):
        """시각화 로직은 인접 리스트를 활용하여 효율적으로 구성합니다."""
        if not PYVIS_AVAILABLE:
            print("Install `networkx pyvis` for graph view")
            return
        if not self.initialized:
            print("Tree not initialized. Call make_tree() first.")
            return

        g = nx.DiGraph()

        for idx in range(int(self.next_idx)):
            node_type = int(self.data[idx, COL_NODE_TYPE].item())
            if node_type == NODE_TYPE_UNUSED:
                continue
            label, color = self._node_label_color(idx)
            g.add_node(idx, label=label, color=color, title=label.replace("\n", "<br>"), shape='box')
        
        for parent_id, children in self._adjacency_list.items():
            for child_id in children:
                if parent_id in g and child_id in g:
                    g.add_edge(parent_id, child_id)

        if not g.nodes:
            print("No nodes to visualize.")
            return

        net = Network(height="800px", width="100%", directed=True, notebook=False, cdn_resources='remote')
        net.from_nx(g)
        try:
            net.set_options("""
            var options = {
              "layout": { "hierarchical": { "enabled": true, "levelSeparation": 200, "nodeSpacing": 150, "treeSpacing": 250, "direction": "UD", "sortMethod": "directed"}},
              "physics": { "enabled": false }
            }
            """)
        except Exception as e:
            print(f"Pyvis options error: {e}. Using default layout.")

        try:
            net.save_graph(file)
            print(f"Graph saved -> {file}")
            if open_browser:
                webbrowser.open("file://" + os.path.realpath(file))
        except Exception as e:
            print(f"Error saving or opening graph: {e}")

class GATreePop:
    """
    GATree의 집단(Population)을 관리하는 클래스.
    """
    def __init__(self, pop_size, max_nodes, max_depth, max_children, feature_num, feature_comparison_map, feature_bool):
        """GATreePop 초기화"""
        self.pop_size = pop_size
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_children = max_children
        self.feature_num = feature_num
        self.feature_comparison_map = feature_comparison_map
        self.feature_bool = feature_bool

        self.initialized = False
        self.population_tensor = torch.zeros((pop_size, max_nodes, NODE_INFO_DIM), dtype=torch.float32)
        self.population = []

    def make_population(self):
        """설정된 pop_size만큼 GATree 개체를 생성하여 집단을 초기화합니다."""
        # 추후 multiprocessing을 통한 생성
        self.population = []
        for i in range(self.pop_size):
            print(f"--- Creating Tree {i+1}/{self.pop_size} ---")
            tree_data_view = self.population_tensor[i]
            tree = GATree(
                self.max_nodes, self.max_depth, self.max_children,
                self.feature_num, self.feature_comparison_map, self.feature_bool,
                data_tensor=tree_data_view
            )
            tree.make_tree()
            self.population.append(tree)
        self.initialized = True
        print("\nPopulation created successfully.")

    def reorganize_nodes(self):
        """
        집단 내의 모든 GATree 개체에 대해 노드 재조직을 수행합니다.
        """
        if not self.initialized:
            print("Warning: Reorganizing an uninitialized population.")
            return

        print(f"Reorganizing all {self.pop_size} trees in the population...")
        for i, tree in enumerate(self.population):
            tree.reorganize_nodes()
        print("Population reorganization complete.")

    def set_next_idx(self):
        """
        집단 내의 모든 GATree 개체에 대해 next_idx를 재설정합니다.
        """
        if not self.initialized:
            print("Warning: Setting next_idx for an uninitialized population.")
            return

        for tree in self.population:
            tree.set_next_idx()

    def return_next_idx(self) -> list[int]:
        """
        집단 내 모든 GATree 개체의 next_idx 값을 리스트로 반환합니다.
        """
        if not self.initialized:
            return [0] * self.pop_size

        return [tree.return_next_idx() for tree in self.population]

    def save(self, filepath):
        """집단 전체의 상태를 파일로 저장합니다."""
        if not self.initialized:
            print("Warning: Saving uninitialized population.")

        state = {
            'population_tensor': self.population_tensor,
            'pop_size': self.pop_size,
            'max_nodes': self.max_nodes,
            'max_depth': self.max_depth,
            'max_children': self.max_children,
            'feature_num': self.feature_num,
            'feature_comparison_map': self.feature_comparison_map,
            'feature_bool': self.feature_bool,
        }
        torch.save(state, filepath)
        print(f"Population saved to {filepath}")

    def load(self, filepath):
        """파일로부터 집단 전체의 상태를 로드합니다."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        state = torch.load(filepath)

        feature_comparison_map = state.get('feature_comparison_map', {})
        if not feature_comparison_map and 'feature_pair' in state:
             print("Warning: Loading legacy 'feature_pair'. Converting to an empty map.")

        feature_bool = state.get('feature_bool', [])

        self.__init__(
            state['pop_size'], state['max_nodes'], state['max_depth'],
            state['max_children'], state['feature_num'], feature_comparison_map,
            feature_bool
        )
        self.population_tensor.copy_(state['population_tensor'])

        self.population = []
        for i in range(self.pop_size):
            tree_data_view = self.population_tensor[i]
            tree = GATree(
                self.max_nodes, self.max_depth, self.max_children,
                self.feature_num, self.feature_comparison_map, self.feature_bool,
                data_tensor=tree_data_view
            )
            tree.load({
                'data': tree_data_view,
                'next_idx': (tree_data_view[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED).sum().item(),
                'max_nodes': self.max_nodes,
                'max_depth': self.max_depth,
                'max_children': self.max_children,
                'feature_num': self.feature_num,
                'feature_comparison_map': self.feature_comparison_map,
                'feature_bool': self.feature_bool
            })
            self.population.append(tree)

        self.initialized = True
        print(f"Population loaded successfully from {filepath}")


if __name__ == '__main__':
    # --- 시뮬레이션 파라미터 ---
    MAX_NODES = 2048
    MAX_DEPTH = 200
    MAX_CHILDREN = 200
    POP_SIZE = 5

    # =======================================================
    # === 1. 단일 GATree 생성, 시각화, 저장 및 로드 테스트 ===
    # =======================================================
    print("===== [Phase 1] Single GATree Demo =====")

    print("\n1. Creating a standalone GATree...")
    tree1 = GATree(MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL)
    tree1.make_tree()
    tree1.visualize_graph(file="single_tree_generated.html")
    tree1.save("single_tree.pth")

    print("\n2. Loading the tree into a new GATree object...")
    tree2 = GATree(MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL)
    tree2.load("single_tree.pth")
    tree2.visualize_graph(file="single_tree_loaded.html")

    assert torch.equal(tree1.data, tree2.data), "Saved and Loaded trees are not identical!"
    print("\nStandalone GATree save/load test PASSED.")

    print("\n3. Testing the new BFS predict method...")
    test_features = {
        'RSI': 50, 'ATR': 0.5, 'WR': -50, 'STOCH_K': 50,
        'SMA_5': 100, 'SMA_20': 98, 'EMA_10': 101, 'EMA_30': 97,
        'BB_upper': 105, 'BB_lower': 95,
        'IsBullishMarket': True, 'IsHighVolatility': False
    }
    action_long = tree1.predict(test_features, 'LONG')
    print(f"Prediction for 'LONG' position: {action_long} -> Action: {ACTION_TYPE_MAP.get(action_long[0])}")
    action_hold = tree1.predict(test_features, 'HOLD')
    print(f"Prediction for 'HOLD' position: {action_hold} -> Action: {ACTION_TYPE_MAP.get(action_hold[0])}")
    action_short = tree1.predict(test_features, 'SHORT')
    print(f"Prediction for 'SHORT' position: {action_short} -> Action: {ACTION_TYPE_MAP.get(action_short[0])}")


    # =====================================================
    # === 2. GATreePop 생성, 시각화, 저장 및 로드 테스트 ===
    # =====================================================
    print("\n\n===== [Phase 2] GATreePop Demo =====")

    print("\n1. Creating a population of GATrees...")
    population1 = GATreePop(POP_SIZE, MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL)
    population1.make_population()

    print("\nVisualizing the first tree from the population...")
    first_tree_from_pop = population1.population[0]
    first_tree_from_pop.visualize_graph(file="population_tree_generated.html")
    population1.save("population.pth")

    print("\n2. Loading the population into a new GATreePop object...")
    population2 = GATreePop(POP_SIZE, MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL)
    population2.load("population.pth")

    print("\nVisualizing the first tree from the loaded population...")
    first_tree_from_loaded_pop = population2.population[0]
    first_tree_from_loaded_pop.visualize_graph(file="population_tree_loaded.html")

    assert torch.equal(population1.population_tensor, population2.population_tensor), "Saved and Loaded populations are not identical!"
    print("\nGATreePop save/load test PASSED.")

    assert torch.equal(population1.population[0].data, population2.population[0].data)
    print("Memory view reference in loaded population confirmed.")