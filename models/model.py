# --- START OF FILE models/model.py ---

import torch
import random
import os
import webbrowser
import networkx as nx
import torch.multiprocessing as mp # [신규] 멀티프로세싱 라이브러리 임포트
from itertools import repeat # [신규] 멀티프로세싱 인자 전달을 위한 임포트


from .constants import *

# pyvis 라이브러리 가용성 확인
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

try:
    import gatree_cuda
except ImportError:
    gatree_cuda = None

# -------------------------------------
# --- 상수 정의 (C/C++ 연동을 위해 중요)
# -------------------------------------



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
    """[수정] 클래스 외부에서도 사용할 수 있도록 전역 함수로 변경"""
    comp_features = set(feature_map.keys())
    for v_list in feature_map.values():
        comp_features.update(v_list)
    # [수정] 피처 리스트의 순서를 항상 동일하게 보장하기 위해 정렬 로직 추가
    return sorted(list(feature_num.keys())) + sorted(list(comp_features)) + sorted(feature_bool)

ALL_FEATURES = get_all_features(FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL)


# ==============================================================================
# [신규] 멀티프로세싱을 위한 최상위 레벨 워커 함수
# ==============================================================================
def _create_tree_worker(args):
    """멀티프로세싱 워커: 단일 트리를 생성하고 공유 텐서에 기록합니다."""
    i, population_tensor, config = args
    tree_data_view = population_tensor[i]
    
    # GATree 생성 시 config에서 모든 필요 인자를 추출하여 전달
    tree = GATree(
        max_nodes=config['max_nodes'],
        max_depth=config['max_depth'],
        max_children=config['max_children'],
        feature_num=config['feature_num'],
        feature_comparison_map=config['feature_comparison_map'],
        feature_bool=config['feature_bool'],
        all_features=config['all_features'], # all_features 전달
        data_tensor=tree_data_view
    )
    tree.make_tree()
    # 반환값은 필요 없음 (공유 메모리에 직접 쓰기 때문)

def _reorganize_worker(args):
    """
    [수정] 멀티프로세싱 워커: 단일 트리의 노드를 재구성하고,
    새로운 인접 리스트와 next_idx를 계산하여 반환합니다.
    """
    i, population_tensor, config = args
    tree_data_view = population_tensor[i]

    tree = GATree(
        max_nodes=config['max_nodes'],
        max_depth=config['max_depth'],
        max_children=config['max_children'],
        feature_num=config['feature_num'],
        feature_comparison_map=config['feature_comparison_map'],
        feature_bool=config['feature_bool'],
        all_features=config['all_features'],
        data_tensor=tree_data_view
    )
    # 텐서 뷰만으로 객체 상태를 동기화
    tree.set_next_idx()
    tree.initialized = True
    
    # 1. 노드 재구성 수행
    tree.reorganize_nodes()
    
    # 2. 재구성된 텐서 기반으로 새로운 인접 리스트와 next_idx 계산
    new_adj_list = tree._adjacency_list
    new_next_idx = tree.next_idx
    
    # 3. 객체 ID와 함께 결과 반환
    return (i, new_next_idx, new_adj_list)


class GATree:
    """
    하나의 유전 알고리즘 트리를 나타내는 클래스.
    이 클래스의 데이터는 C++/CUDA에서 직접 처리할 수 있는 torch.Tensor 형식으로 저장됩니다.
    """
    def __init__(self, max_nodes, max_depth, max_children, feature_num, feature_comparison_map, feature_bool, all_features, data_tensor=None):
        """
        [수정] GATree 초기화. all_features 리스트를 외부에서 주입받도록 변경.

        Args:
            max_nodes (int): 트리가 가질 수 있는 최대 노드 수.
            max_depth (int): 트리의 최대 깊이.
            max_children (int): Decision 노드가 가질 수 있는 최대 자식 노드 수.
            feature_num (dict): 숫자와 비교할 피쳐와 (min, max) 범위.
            feature_comparison_map (dict): 피처 간 비교 규칙을 정의한 맵.
            feature_bool (list): Boolean과 비교할 피쳐 리스트.
            all_features (list): 전체 피처의 순서가 정의된 리스트. (중요)
            data_tensor (torch.Tensor, optional): 외부에서 생성된 텐서의 view.
                                                  None이면 자체적으로 텐서를 생성합니다.
        """
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_children = max_children
        self.feature_num = feature_num
        self.feature_num = {
    k: (float(v[0]), float(v[1])) for k, v in feature_num.items()
}
        self.feature_comparison_map = feature_comparison_map
        self.feature_bool = feature_bool

        # [수정] all_features를 내부에서 계산하지 않고, 주입받은 것을 그대로 사용
        self.all_features = all_features
        if not self.all_features:
            raise ValueError("all_features list cannot be empty.")

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
        
        # [수정] total_nodes가 음수가 되지 않도록 방어 코드 추가
        if max_nodes_to_create < 1:
            total_nodes = 0
        else:
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
        # 멀티프로세싱 환경에서는 print 출력이 섞일 수 있으므로 주석 처리하거나 logging 사용을 권장
        # print(f"Tree created with {self.next_idx} nodes.")

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
            if nodes_to_create <= len(leaf_list): # [수정] 논리 오류 수정 (+1 제거)
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
            'all_features': self.all_features, # [수정] all_features 저장
        }
        torch.save(state, filepath)
        print(f"Tree saved to {filepath}")

    def load(self, source):
        """[수정] 파일 또는 state_dict로부터 트리 상태를 로드합니다."""
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"File not found: {source}")
            state = torch.load(source)
        elif isinstance(source, dict):
            state = source
        else:
            raise TypeError("source must be a filepath string or a state_dict")

        # [수정] __init__ 호출 시 `all_features`를 전달해야 하므로,
        # state 딕셔너리에 `all_features`가 있는지 확인하고, 없으면 생성.
        # 이는 하위 호환성을 위함.
        if 'all_features' in state:
            all_features = state['all_features']
        else:
            print("Warning: 'all_features' not found in state. Re-creating from components.")
            feature_comparison_map = state.get('feature_comparison_map', {})
            feature_bool = state.get('feature_bool', [])
            all_features = get_all_features(state['feature_num'], feature_comparison_map, feature_bool)

        # GATree의 init을 다시 호출하여 객체를 재설정
        self.__init__(
            max_nodes=state['max_nodes'],
            max_depth=state['max_depth'],
            max_children=state['max_children'],
            feature_num=state['feature_num'],
            feature_comparison_map=state.get('feature_comparison_map', {}),
            feature_bool=state.get('feature_bool', []),
            all_features=all_features # [수정] all_features 전달
        )
        self.data.copy_(state['data'])
        self.next_idx = state['next_idx']
        
        self._build_adjacency_list()
        self.initialized = True
        # print(f"Tree loaded successfully.") # 개별 로드는 조용히 처리

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
    def __init__(self, pop_size, max_nodes, max_depth, max_children, feature_num, feature_comparison_map, feature_bool, all_features):
        """[수정] GATreePop 초기화. all_features를 명시적으로 받음."""
        self.pop_size = pop_size
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_children = max_children
        self.feature_num = feature_num
        self.feature_comparison_map = feature_comparison_map
        self.feature_bool = feature_bool
        self.all_features = all_features

        self.initialized = False
        self.population_tensor = torch.zeros((pop_size, max_nodes, NODE_INFO_DIM), dtype=torch.float32)
        self.population = []

    def make_population(self, num_processes: int = 1, device: str = 'cuda',
                        init_mode: str = 'cuda', node_budget: int | None = None):
        """
        [수정] 설정된 pop_size만큼 GATree 개체를 생성하여 집단을 초기화합니다.
        CUDA 초기화와 멀티프로세싱을 지원합니다.
        
        Args:
            num_processes: CPU 초기화 시 사용할 프로세스 수
            device: 텐서를 저장할 디바이스 ('cuda' 또는 'cpu')
            init_mode: 초기화 방식 ('cuda' 또는 'cpu')
            node_budget: 트리당 노드 예산 (None이면 자동 설정)
        """
        self.population = []
        B, N, D = self.pop_size, self.max_nodes, NODE_INFO_DIM

        if init_mode == 'cuda' and device.startswith('cuda') and gatree_cuda is not None:
            # --- Host-side feasibility checks ---
            assert self.max_nodes >= 6, "max_nodes must be >= 6 (need 3 roots + 3 actions for minimal tree)"
            assert self.max_depth >= 2, "max_depth must be >= 2 (root-action minimal tree)"
            assert self.max_children >= 1, "max_children must be >= 1"
            if node_budget is not None:
                assert node_budget >= 0

            print(f"[CUDA] Initializing {B} trees on GPU (N={N}, D={D}) with invariant guards.")

            # 1) population tensor on CUDA
            self.population_tensor = torch.zeros((B, N, D), dtype=torch.float32, device=device)

            # 2) budgets
            if node_budget is None:
                node_budget = max(1, (N - 3) // 2)
            total_budget = torch.full((B,), int(node_budget), dtype=torch.int32, device=device)

            # 3) work buffers (all on CUDA)
            bfs_q     = torch.empty((B, 2 * N), dtype=torch.int32,   device=device)
            scratch   = torch.empty((B, 2 * N), dtype=torch.int32,   device=device)
            child_cnt = torch.empty((B, N),     dtype=torch.int32,   device=device)
            act_cnt   = torch.empty((B, N),     dtype=torch.int32,   device=device)
            dec_cnt   = torch.empty((B, N),     dtype=torch.int32,   device=device)
            cand_idx  = torch.empty((B, N),     dtype=torch.int32,   device=device)
            cand_w    = torch.empty((B, N),     dtype=torch.float32, device=device)

            # 4) feature tables → ALL_FEATURES index space
            all_feats = self.all_features

            # numeric family — zero-length when empty
            num_names = list(self.feature_num.keys())
            if len(num_names) == 0:
                num_feat_indices = torch.empty((0,), dtype=torch.int32, device=device)
                num_feat_minmax  = torch.empty((0, 2), dtype=torch.float32, device=device)
            else:
                num_feat_indices = torch.tensor([all_feats.index(k) for k in num_names],
                                                dtype=torch.int32, device=device)
                num_feat_minmax  = torch.tensor([[float(self.feature_num[k][0]), float(self.feature_num[k][1])]
                                                 for k in num_names],
                                                dtype=torch.float32, device=device)

            # boolean family — zero-length when empty
            if len(self.feature_bool) == 0:
                bool_feat_indices = torch.empty((0,), dtype=torch.int32, device=device)
            else:
                bool_feat_indices = torch.tensor([all_feats.index(k) for k in self.feature_bool],
                                                 dtype=torch.int32, device=device)

            # feat-feat pairs — zero-length when empty
            pairs = []
            for f1, lst in self.feature_comparison_map.items():
                if not lst: continue
                f1i = all_feats.index(f1)
                for f2 in lst:
                    pairs.append([f1i, all_feats.index(f2)])
            if len(pairs) == 0:
                ff_pairs = torch.empty((0, 2), dtype=torch.int32, device=device)
            else:
                ff_pairs = torch.tensor(pairs, dtype=torch.int32, device=device)

            # 5) action allow-lists per root-context
            long_actions  = torch.tensor(
                [ACTION_CLOSE_ALL, ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION, ACTION_FLIP_POSITION],
                dtype=torch.int32, device=device)
            hold_actions  = torch.tensor(
                [ACTION_NEW_LONG, ACTION_NEW_SHORT],
                dtype=torch.int32, device=device)
            short_actions = torch.tensor(
                [ACTION_CLOSE_ALL, ACTION_CLOSE_PARTIAL, ACTION_ADD_POSITION, ACTION_FLIP_POSITION],
                dtype=torch.int32, device=device)

            # 6) launch CUDA initializer
            gatree_cuda.init_population_cuda(
                self.population_tensor, total_budget,
                self.max_children, self.max_depth, self.max_nodes,
                bfs_q, scratch, child_cnt, act_cnt, dec_cnt, cand_idx, cand_w,
                num_feat_indices, num_feat_minmax,
                bool_feat_indices, ff_pairs,
                long_actions, hold_actions, short_actions
            )

            # Validate population structure after CUDA init
            try:
                gatree_cuda.validate_trees(self.population_tensor.contiguous())
            except Exception as e:
                print(f"Warning: validate_trees failed after init_population_cuda: {e}")

            # 7) Stitch Python objects to GPU views (no CPU traversal here)
            self.population = []
            for i in range(B):
                tree_data_view = self.population_tensor[i]
                tree = GATree(
                    self.max_nodes, self.max_depth, self.max_children,
                    self.feature_num, self.feature_comparison_map, self.feature_bool,
                    self.all_features, data_tensor=tree_data_view
                )
                tree.initialized = True
                tree.set_next_idx()  # uses device tensor shape logically; do not CPU-iterate
                self.population.append(tree)

            self.initialized = True
            print("[CUDA] Population initialized and synchronized.")
            return

        # --- CPU fallback path ---
        if num_processes > 1 and self.pop_size > 1:
            print(f"--- Creating {self.pop_size} trees using {num_processes} processes (CPU) ---")
            
            # 공유 메모리에 텐서 올리기
            self.population_tensor = torch.zeros((self.pop_size, self.max_nodes, NODE_INFO_DIM), dtype=torch.float32, device=device)
            self.population_tensor.share_memory_()

            # 워커에 전달할 설정 딕셔너리
            config = self._get_config_dict()
            
            # 워커에 전달할 인자 리스트 생성
            args = zip(range(self.pop_size), repeat(self.population_tensor), repeat(config))
            
            # 멀티프로세싱 풀 생성 및 실행
            with mp.Pool(processes=num_processes) as pool:
                list(pool.map(_create_tree_worker, args))
        else:
            print(f"--- Creating {self.pop_size} trees sequentially (CPU) ---")
            self.population_tensor = torch.zeros((self.pop_size, self.max_nodes, NODE_INFO_DIM), dtype=torch.float32, device=device)
            for i in range(self.pop_size):
                tree_data_view = self.population_tensor[i]
                tree = GATree(
                    self.max_nodes, self.max_depth, self.max_children,
                    self.feature_num, self.feature_comparison_map, self.feature_bool,
                    self.all_features,
                    data_tensor=tree_data_view
                )
                tree.make_tree()
        
        # 멀티프로세싱 실행 후, self.population 리스트를 다시 채움
        for i in range(self.pop_size):
            tree_data_view = self.population_tensor[i]
            tree = GATree(
                self.max_nodes, self.max_depth, self.max_children,
                self.feature_num, self.feature_comparison_map, self.feature_bool,
                self.all_features,
                data_tensor=tree_data_view
            )
            # 텐서는 이미 채워져 있으므로, 내부 상태만 동기화
            tree.load({
                'data': tree_data_view,
                'next_idx': (tree_data_view[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED).sum().item(),
                'max_nodes': self.max_nodes, 'max_depth': self.max_depth, 'max_children': self.max_children,
                'feature_num': self.feature_num, 'feature_comparison_map': self.feature_comparison_map,
                'feature_bool': self.feature_bool, 'all_features': self.all_features
            })
            self.population.append(tree)

        self.initialized = True
        print("Population created successfully (CPU).")

    def reorganize_nodes(self):
        """
        [수정] 집단 내의 모든 GATree 개체에 대해 노드 재조직을 GPU에서 병렬로 수행합니다.
        Python 멀티프로세싱 로직을 CUDA 커널 직접 호출로 대체합니다.
        """
        if not self.initialized:
            print("Warning: Reorganizing an uninitialized population.")
            return

        # [수정] CUDA가 아닌 장치에 대한 예외 처리 또는 대체 로직
        if not self.population_tensor.device.type.startswith('cuda'):
            print("Warning: CUDA-based reorganize_nodes is only available for GPU tensors. Falling back to sequential CPU method.")
            # CPU에서는 기존의 순차적 방식 실행
            for tree in self.population:
                tree.reorganize_nodes()
            return
        
        # [수정] C++/CUDA 확장 모듈 직접 호출
        gatree_cuda.reorganize_population(self.population_tensor)
        # Validate population after reorganization
        try:
            gatree_cuda.validate_trees(self.population_tensor.contiguous())
        except Exception as e:
            print(f"Warning: validate_trees failed after reorganize_population: {e}")
        
        # [중요] CUDA 연산 후, Python GATree 객체들의 내부 상태를 텐서와 동기화합니다.
        # 이 과정이 없으면, 다음 연산(예: 시각화, CPU 기반 예측)에서 오류가 발생합니다.
        self.set_next_idx()  # 모든 트리의 next_idx 값을 텐서 기준으로 다시 계산
        for tree in self.population:
            tree._build_adjacency_list() # 모든 트리의 인접 리스트를 텐서 기준으로 재생성

    def set_next_idx(self):
        """
        [수정] 집단 내 모든 GATree 개체의 next_idx를 텐서 기반으로 재설정합니다.
        이 함수는 이제 CUDA 연산 후 동기화를 위해 더욱 중요해졌습니다.
        """
        if not self.initialized:
            return

        for tree in self.population:
            # GATree 객체의 텐서 뷰(tree.data)를 직접 참조하여 상태를 업데이트합니다.
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
            'all_features': self.all_features,
        }
        torch.save(state, filepath)
        print(f"Population saved to {filepath}")

    def load(self, filepath):
        """[수정] 파일로부터 집단 전체의 상태를 로드합니다."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        state = torch.load(filepath)

        # [수정] 하위 호환성을 위해 all_features가 없으면 생성
        if 'all_features' not in state:
             print("Warning: 'all_features' not found in state file. Re-creating.")
             state['all_features'] = get_all_features(
                 state['feature_num'], 
                 state.get('feature_comparison_map', {}), 
                 state.get('feature_bool', [])
             )
        
        # __init__ 호출 시 all_features 전달
        self.__init__(
            state['pop_size'], state['max_nodes'], state['max_depth'],
            state['max_children'], state['feature_num'], 
            state.get('feature_comparison_map', {}),
            state.get('feature_bool', []), 
            state['all_features']
        )
        self.population_tensor.copy_(state['population_tensor'])

        self.population = []
        for i in range(self.pop_size):
            tree_data_view = self.population_tensor[i]
            tree = GATree(
                self.max_nodes, self.max_depth, self.max_children,
                self.feature_num, self.feature_comparison_map, self.feature_bool,
                self.all_features, # [수정] all_features 전달
                data_tensor=tree_data_view
            )
            # GATree.load는 state dict를 받아 내부 상태를 채움
            tree.load({
                'data': tree_data_view,
                'next_idx': (tree_data_view[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED).sum().item(),
                'max_nodes': self.max_nodes,
                'max_depth': self.max_depth,
                'max_children': self.max_children,
                'feature_num': self.feature_num,
                'feature_comparison_map': self.feature_comparison_map,
                'feature_bool': self.feature_bool,
                'all_features': self.all_features # GATree.load에 전달
            })
            self.population.append(tree)

        self.initialized = True
        print(f"Population loaded successfully from {filepath}")

    def _get_config_dict(self):
        """멀티프로세싱 워커에 전달할 설정 딕셔너리를 반환합니다."""
        return {
            'max_nodes': self.max_nodes,
            'max_depth': self.max_depth,
            'max_children': self.max_children,
            'feature_num': self.feature_num,
            'feature_comparison_map': self.feature_comparison_map,
            'feature_bool': self.feature_bool,
            'all_features': self.all_features,
        }


if __name__ == '__main__':
    # 멀티프로세싱 사용 시 main 진입점 보호
    # Windows/macOS에서 "fork"가 아닌 "spawn" 시작 방식을 사용하므로 필수
    mp.set_start_method("spawn", force=True)

    # --- 시뮬레이션 파라미터 ---
    MAX_NODES = 2048
    MAX_DEPTH = 200
    MAX_CHILDREN = 200
    POP_SIZE = 10 # 멀티프로세싱 테스트를 위해 크기 증가
    NUM_PROCESSES = 4 # 사용할 프로세스 수

    # =======================================================
    # === 1. 단일 GATree 생성, 시각화, 저장 및 로드 테스트 ===
    # =======================================================
    print("===== [Phase 1] Single GATree Demo =====")

    print("\n1. Creating a standalone GATree...")
    # [수정] GATree 생성자에 all_features 전달
    tree1 = GATree(MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL, ALL_FEATURES)
    tree1.make_tree()
    tree1.visualize_graph(file="single_tree_generated.html", open_browser=False)
    tree1.save("single_tree.pth")

    print("\n2. Loading the tree into a new GATree object...")
    # [수정] 로드 전 객체 생성 시에도 all_features 전달
    tree2 = GATree(MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL, ALL_FEATURES)
    tree2.load("single_tree.pth")
    # tree2.visualize_graph(file="single_tree_loaded.html", open_browser=False)

    assert torch.equal(tree1.data, tree2.data), "Saved and Loaded trees are not identical!"
    assert tree1.all_features == tree2.all_features, "Saved and Loaded trees have different all_features list!"
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

    print(f"\n1. Creating a population of GATrees using {NUM_PROCESSES} processes...")
    # [수정] GATreePop 생성자에 all_features 전달
    population1 = GATreePop(POP_SIZE, MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL, ALL_FEATURES)
    population1.make_population(num_processes=NUM_PROCESSES)

    print("\nVisualizing the first tree from the population...")
    first_tree_from_pop = population1.population[0]
    first_tree_from_pop.visualize_graph(file="population_tree_generated.html", open_browser=False)
    population1.save("population.pth")

    print("\n2. Loading the population into a new GATreePop object...")
    # [수정] GATreePop 생성자에 all_features 전달
    population2 = GATreePop(POP_SIZE, MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_COMPARISON_MAP, FEATURE_BOOL, ALL_FEATURES)
    population2.load("population.pth")

    # print("\nVisualizing the first tree from the loaded population...")
    # first_tree_from_loaded_pop = population2.population[0]
    # first_tree_from_loaded_pop.visualize_graph(file="population_tree_loaded.html", open_browser=False)

    assert torch.equal(population1.population_tensor, population2.population_tensor), "Saved and Loaded populations are not identical!"
    assert population1.all_features == population2.all_features, "Saved and loaded populations have different all_features lists!"
    print("\nGATreePop save/load test PASSED.")

    assert torch.equal(population1.population[0].data, population2.population[0].data)
    print("Memory view reference in loaded population confirmed.")

    # [신규] 멀티프로세싱 재구성 테스트
    print(f"\n3. Testing reorganize_nodes with {NUM_PROCESSES} processes...")
    population1.reorganize_nodes(num_processes=NUM_PROCESSES)
    print("Reorganization test finished.")
