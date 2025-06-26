import numpy as np
import random
from enum import IntEnum

# --- 상수 정의 (C/CUDA의 #define과 유사) ---

class NodeType(IntEnum):
    """노드 타입을 나타내는 정수 Enum. C에서 enum으로 변환 가능."""
    EMPTY = 0
    DECISION_NUM = 1  # Feature vs Number
    DECISION_FEAT = 2 # Feature1 vs Feature2
    ACTION = 3        # 최종 행동 노드

class Operator(IntEnum):
    """비교 연산자 Enum."""
    GT = 0  # >
    LT = 1  # <
    EQ = 2  # ==

# 노드 배열의 각 열(column) 인덱스 정의
# C에서 struct 멤버에 접근하는 것과 유사한 효과
NODE_TYPE = 0
# For Decision Nodes
PARAM_1 = 1  # feat1_idx
PARAM_2 = 2  # feat2_idx or comparison_value
PARAM_3 = 3  # operator_idx
TRUE_CHILD_IDX = 4
FALSE_CHILD_IDX = 5
# For Action Nodes
ACTION_ENTRY = 1      # 0 (False) or 1 (True)
ACTION_PROPORTION = 2 # 0.0 ~ 1.0
ACTION_LEVERAGE = 3   # e.g., 1, 5, 10

class GeneticTree:
    """
    CUDA/C 호환성을 위해 평탄화된 Numpy 배열로 구현된 유전 알고리즘 트리.
    하나의 인스턴스가 하나의 염색체(개체)를 나타냅니다.
    """
    def __init__(self, config):
        """
        트리를 초기화합니다. config 딕셔너리로부터 설정을 불러옵니다.
        config 예시:
        {
            'feature_num': ['RSI', 'ATR', 'WR'],
            'feature_pair': ['SMA', 'EMA', 'BB_upper', 'BB_lower'],
            'feature_ranges': {'RSI': (0, 100), 'ATR': (0.1, 5.0), 'WR': (-100, 0)},
            'max_nodes': 256,
            'max_depth': 8
        }
        """
        self.config = config
        self.max_nodes = config['max_nodes']
        self.max_depth = config['max_depth']

        # 전체 피처 리스트와 룩업 테이블 생성
        self.features = config['feature_num'] + config['feature_pair']
        self.feature_to_idx = {name: i for i, name in enumerate(self.features)}

        # --- 핵심 데이터 구조: 평탄화된 노드 배열 ---
        # [node_idx, params(6)] 형태의 2D 배열
        # 각 행: [NodeType, Param1, Param2, Param3, Param4, Param5]
        # GPU에 직접 복사될 데이터입니다.
        self.nodes = np.zeros((self.max_nodes, 6), dtype=np.float32)

        # 현재 사용된 노드의 개수 추적
        self.node_count = 1  # 인덱스 0은 비워둠 (루트 역할)

        # 3개의 고정된 분기에 대한 시작 노드 인덱스
        self.root_indices = {
            'LONG': -1,
            'HOLD': -1,
            'SHORT': -1
        }
        
        # 초기 랜덤 트리 생성
        self._initialize_random_tree()

    def _get_new_node_idx(self):
        """새로운 노드를 할당하고 인덱스를 반환합니다."""
        if self.node_count >= self.max_nodes:
            # 이 경우, 더 이상 노드를 추가하지 않고 액션 노드로 강제 종료해야 함
            return -1
        idx = self.node_count
        self.node_count += 1
        return idx

    def _create_random_branch(self, current_depth):
        """
        재귀적으로 랜덤한 서브트리를 생성하고 시작 노드의 인덱스를 반환합니다.
        """
        # --- 종료 조건 (잎 노드 생성) ---
        # 1. 최대 깊이에 도달한 경우
        # 2. 할당 가능한 노드가 없는 경우
        # 3. 일정 확률로 조기 종료하여 다양한 형태의 트리를 생성
        if (current_depth >= self.max_depth or 
            self.node_count >= self.max_nodes - 2 or # 자식 노드 2개 공간 확보
            random.random() < 0.3 and current_depth > 1): # 조기 종료 확률
            
            node_idx = self._get_new_node_idx()
            if node_idx == -1: return -1 # 사실상 이 코드는 실행되지 않아야 함
            
            # 액션 노드 생성
            self.nodes[node_idx, NODE_TYPE] = NodeType.ACTION
            self.nodes[node_idx, ACTION_ENTRY] = random.choice([0.0, 1.0]) # 진입 여부
            self.nodes[node_idx, ACTION_PROPORTION] = random.uniform(0.1, 1.0) # 진입 비중
            self.nodes[node_idx, ACTION_LEVERAGE] = float(random.choice([1, 2, 3, 5, 10])) # 레버리지
            return node_idx

        # --- 재귀 단계 (결정 노드 생성) ---
        node_idx = self._get_new_node_idx()
        if node_idx == -1: return -1

        # 결정 노드 타입 랜덤 선택 (Feature vs Num 또는 Feature vs Feature)
        if random.random() < 0.5 and len(self.config['feature_num']) > 0:
            # 타입 1: Feature vs Number
            self.nodes[node_idx, NODE_TYPE] = NodeType.DECISION_NUM
            
            feat_name = random.choice(self.config['feature_num'])
            feat_idx = self.feature_to_idx[feat_name]
            min_val, max_val = self.config['feature_ranges'][feat_name]
            
            self.nodes[node_idx, PARAM_1] = float(feat_idx)
            self.nodes[node_idx, PARAM_2] = random.uniform(min_val, max_val)
            self.nodes[node_idx, PARAM_3] = random.choice([Operator.GT, Operator.LT, Operator.EQ])

        else:
            # 타입 2: Feature vs Feature
            self.nodes[node_idx, NODE_TYPE] = NodeType.DECISION_FEAT
            
            f1_name, f2_name = random.sample(self.config['feature_pair'], 2)
            f1_idx = self.feature_to_idx[f1_name]
            f2_idx = self.feature_to_idx[f2_name]

            self.nodes[node_idx, PARAM_1] = float(f1_idx)
            self.nodes[node_idx, PARAM_2] = float(f2_idx)
            self.nodes[node_idx, PARAM_3] = random.choice([Operator.GT, Operator.LT, Operator.EQ])

        # 자식 노드 재귀 생성
        true_child = self._create_random_branch(current_depth + 1)
        false_child = self._create_random_branch(current_depth + 1)

        self.nodes[node_idx, TRUE_CHILD_IDX] = true_child
        self.nodes[node_idx, FALSE_CHILD_IDX] = false_child

        return node_idx

    def _initialize_random_tree(self):
        """3개의 고정된 포지션에 대해 각각 랜덤 서브트리를 생성합니다."""
        for position in ['LONG', 'HOLD', 'SHORT']:
            start_node_idx = self._create_random_branch(current_depth=0)
            self.root_indices[position] = start_node_idx

    def save(self, filepath):
        """트리의 상태(노드 배열, 루트 인덱스)를 파일에 저장합니다."""
        # 루트 인덱스는 별도 배열로 변환하여 저장
        root_indices_arr = np.array([
            self.root_indices['LONG'],
            self.root_indices['HOLD'],
            self.root_indices['SHORT']
        ])
        np.savez(filepath, nodes=self.nodes, root_indices=root_indices_arr)
        print(f"Tree saved to {filepath}")

    def load(self, filepath):
        """파일로부터 트리 상태를 불러옵니다."""
        data = np.load(filepath)
        self.nodes = data['nodes']
        root_indices_arr = data['root_indices']
        self.root_indices['LONG'] = int(root_indices_arr[0])
        self.root_indices['HOLD'] = int(root_indices_arr[1])
        self.root_indices['SHORT'] = int(root_indices_arr[2])
        # node_count는 실제 사용된 노드 수로 업데이트 (로딩 후 변이를 위함)
        self.node_count = np.where(self.nodes[:, NODE_TYPE] != NodeType.EMPTY)[0].max() + 1
        print(f"Tree loaded from {filepath}")

    def predict(self, position, feature_values):
        """
        주어진 포지션과 시장 데이터로 트리를 순회하여 최종 액션을 반환합니다.
        이 로직은 CUDA 커널에서 스레드별로 실행될 로직과 거의 동일합니다.
        
        Args:
            position (str): 'LONG', 'HOLD', 'SHORT'
            feature_values (np.array): 모든 feature의 값을 담은 1D Numpy 배열
                                      (self.features 순서와 일치해야 함)
        
        Returns:
            list: [진입여부, 진입비중, 레버리지]
        """
        if position not in self.root_indices:
            raise ValueError(f"Invalid position: {position}")

        current_idx = self.root_indices[position]
        
        # C/CUDA에서 무한 루프는 위험하므로, max_depth를 이용해 탈출 조건 추가
        for _ in range(self.max_depth + 1): 
            if current_idx == -1: # 유효하지 않은 자식 노드
                return [0.0, 0.0, 1.0] # 기본값 (아무것도 안 함)

            node = self.nodes[int(current_idx)]
            node_type = NodeType(node[NODE_TYPE])

            if node_type == NodeType.ACTION:
                return [node[ACTION_ENTRY], node[ACTION_PROPORTION], node[ACTION_LEVERAGE]]
            
            elif node_type == NodeType.DECISION_NUM:
                feat_idx = int(node[PARAM_1])
                comp_val = node[PARAM_2]
                op = Operator(node[PARAM_3])
                
                feat_val = feature_values[feat_idx]
                
                result = False
                if op == Operator.GT: result = feat_val > comp_val
                elif op == Operator.LT: result = feat_val < comp_val
                elif op == Operator.EQ: result = feat_val == comp_val
                
                current_idx = node[TRUE_CHILD_IDX] if result else node[FALSE_CHILD_IDX]

            elif node_type == NodeType.DECISION_FEAT:
                f1_idx = int(node[PARAM_1])
                f2_idx = int(node[PARAM_2])
                op = Operator(node[PARAM_3])

                f1_val = feature_values[f1_idx]
                f2_val = feature_values[f2_idx]
                
                result = False
                if op == Operator.GT: result = f1_val > f2_val
                elif op == Operator.LT: result = f1_val < f2_val
                elif op == Operator.EQ: result = f1_val == f2_val

                current_idx = node[TRUE_CHILD_IDX] if result else node[FALSE_CHILD_IDX]

            else: # NodeType.EMPTY or other
                return [0.0, 0.0, 1.0] # 에러 상황, 기본값 반환

        # 최대 깊이를 초과했는데도 액션 노드에 도달하지 못한 경우
        print("Warning: Max depth exceeded during prediction.")
        return [0.0, 0.0, 1.0]


# --- 사용 예시 ---
if __name__ == '__main__':
    # 1. 설정 정의
    ga_config = {
        'feature_num': ['RSI', 'ATR', 'WR'],
        'feature_pair': ['SMA', 'EMA', 'BB_upper', 'BB_lower'],
        'feature_ranges': {
            'RSI': (0, 100), 
            'ATR': (0.1, 5.0), 
            'WR': (-100, 0)
        },
        'max_nodes': 256,
        'max_depth': 8
    }

    # 2. 새로운 랜덤 트리 생성
    print("--- Creating a new random tree ---")
    my_tree = GeneticTree(ga_config)
    print("Root indices:", my_tree.root_indices)
    print("Total nodes used:", my_tree.node_count)

    # 3. 예측 실행
    print("\n--- Running prediction ---")
    # 예시 시장 데이터 (my_tree.features 순서에 맞춰 생성)
    # ['RSI', 'ATR', 'WR', 'SMA', 'EMA', 'BB_upper', 'BB_lower']
    market_data = np.array([75.0, 1.2, -15.0, 100.5, 101.2, 105.0, 95.0], dtype=np.float32)
    
    # 현재 포지션이 'HOLD'일 때의 결정
    action = my_tree.predict('HOLD', market_data)
    print(f"Position: HOLD, Market Data: {market_data}")
    print(f"Predicted Action: [Entry: {bool(action[0])}, Proportion: {action[1]:.2f}, Leverage: {action[2]:.0f}]")
    
    # 4. 트리 저장 및 불러오기
    print("\n--- Saving and Loading Tree ---")
    save_path = "genetic_tree_01.npz"
    my_tree.save(save_path)
    
    # 새로운 트리 객체를 만들고 저장된 데이터 로드
    loaded_tree = GeneticTree(ga_config)
    loaded_tree.load(save_path)
    
    print("Loaded tree root indices:", loaded_tree.root_indices)
    
    # 로드된 트리로 동일한 예측 수행 -> 결과가 같아야 함
    loaded_action = loaded_tree.predict('HOLD', market_data)
    print(f"Loaded Tree Predicted Action: [Entry: {bool(loaded_action[0])}, Proportion: {loaded_action[1]:.2f}, Leverage: {loaded_action[2]:.0f}]")
    
    # 결과가 같은지 확인
    assert np.allclose(action, loaded_action), "Prediction mismatch after loading!"
    print("\nPrediction consistency confirmed.")