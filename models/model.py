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

# Decision Node: Operators
OP_GT = 0  # >
OP_LT = 1  # <
OP_EQ = 2  # =

# Action Node: Position Types
POS_TYPE_LONG = 0
POS_TYPE_SHORT = 1

# --- 예시 Feature 및 설정 (실제 사용 시 변경) ---
FEATURE_NUM = {'RSI': (0, 100), 'ATR': (0, 1), 'WR': (-100, 0), 'STOCH_K':(0, 100)}
FEATURE_PAIR = ['SMA_5', 'SMA_20', 'EMA_10', 'EMA_30', 'BB_upper', 'BB_lower']
ALL_FEATURES = list(FEATURE_NUM.keys()) + FEATURE_PAIR

# --- Helper 딕셔너리 (시각화 및 디버깅용) ---
NODE_TYPE_MAP = {
    NODE_TYPE_UNUSED: "UNUSED",
    NODE_TYPE_ROOT_BRANCH: "ROOT_BRANCH",
    NODE_TYPE_DECISION: "DECISION",
    NODE_TYPE_ACTION: "ACTION",
}
ROOT_BRANCH_MAP = {ROOT_BRANCH_LONG: "IF_POS_IS_LONG", ROOT_BRANCH_HOLD: "IF_POS_IS_HOLD", ROOT_BRANCH_SHORT: "IF_POS_IS_SHORT"}
OPERATOR_MAP = {OP_GT: ">", OP_LT: "<", OP_EQ: "="}
POS_TYPE_MAP = {POS_TYPE_LONG: "LONG", POS_TYPE_SHORT: "SHORT"}

class GATree:
    """
    하나의 유전 알고리즘 트리를 나타내는 클래스.
    이 클래스의 데이터는 C++/CUDA에서 직접 처리할 수 있는 torch.Tensor 형식으로 저장됩니다.
    """
    def __init__(self, max_nodes, max_depth, max_children, feature_num, feature_pair, data_tensor=None):
        """
        GATree 초기화.

        Args:
            max_nodes (int): 트리가 가질 수 있는 최대 노드 수.
            max_depth (int): 트리의 최대 깊이.
            max_children (int): Decision 노드가 가질 수 있는 최대 자식 노드 수.
            feature_num (dict): 숫자와 비교할 피쳐와 (min, max) 범위.
            feature_pair (list): 피쳐끼리 비교할 피쳐 리스트.
            data_tensor (torch.Tensor, optional): 외부에서 생성된 텐서의 view. 
                                                  None이면 자체적으로 텐서를 생성합니다.
        """
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_children = max_children
        self.feature_num = feature_num
        self.feature_pair = feature_pair
        self.all_features = list(feature_num.keys()) + feature_pair
        
        self.initialized = False
        self.next_idx = 0

        # GATree가 텐서를 직접 소유할지, 외부 텐서의 view를 참조할지 결정
        if data_tensor is None:
            # 독립적인 텐서 생성 (단일 트리 테스트용)
            self.data = torch.zeros((self.max_nodes, NODE_INFO_DIM), dtype=torch.float32)
        else:
            # 외부 텐서(GATreePop의 텐서)의 view를 참조 (메모리 효율성)
            if data_tensor.shape != (self.max_nodes, NODE_INFO_DIM):
                raise ValueError("Provided data_tensor has incorrect shape")
            self.data = data_tensor

    def make_tree(self):
        """
        요구사항에 맞는 랜덤 트리를 생성하고 현재 객체를 초기화합니다.
        """
        # 1. 초기화
        self.data.zero_()
        self.initialized = False
        self.next_idx = 0
        
        # 2. 고정된 루트 분기 3개 생성
        root_branches = [ROOT_BRANCH_LONG, ROOT_BRANCH_HOLD, ROOT_BRANCH_SHORT]
        root_branch_ids = []
        for branch_type in root_branches:
            idx = self._get_next_idx()
            self.data[idx, COL_NODE_TYPE] = NODE_TYPE_ROOT_BRANCH
            self.data[idx, COL_PARENT_IDX] = -1 # Root
            self.data[idx, COL_DEPTH] = 0
            self.data[idx, COL_PARAM_1] = branch_type
            root_branch_ids.append(idx)
        
        # 3. 노드 예산 분배
        # 최소 노드 수를 보장하고, max_nodes 내에서 랜덤하게 결정
        min_total_nodes = 3 + 3 * self.max_children # 루트 3개 + 각 분기별 최소 자식
        total_nodes = random.randint(min_total_nodes, self.max_nodes)
        
        node_budget = total_nodes - 3 # 루트 분기 3개 제외
        nodes_per_branch = node_budget // 3
        budgets = [nodes_per_branch] * 3
        # 남은 예산을 랜덤하게 분배
        for i in range(node_budget % 3):
            budgets[random.randint(0, 2)] += 1

        # 4. 각 분기별로 트리 생성
        for i, branch_id in enumerate(root_branch_ids):
            self._grow_branch(branch_id, budgets[i])
            
        self.initialized = True
        print(f"Tree created with {self.next_idx} nodes.")

    def _grow_branch(self, branch_root_id, budget):
        """한 분기(LONG/HOLD/SHORT) 아래의 트리를 성장시키는 내부 함수"""
        if budget <= 0:
            # 예산이 없으면 최소한의 Action 노드 하나만 생성
            self._create_action_node(branch_root_id)
            return

        # 열린 노드 리스트: 자식을 추가해야 할 Decision 노드들의 ID를 관리
        open_list = [branch_root_id]
        
        nodes_to_create = budget
        
        while nodes_to_create > 0 and open_list:
            # 1. 부모 노드 선택 (리스트에서 랜덤하게 선택하여 다양한 트리 구조 유도)
            parent_id = random.choice(open_list)
            parent_depth = int(self.data[parent_id, COL_DEPTH].item())
            
            # 2. 자식 타입 결정 (Action or Decision)
            # 최대 깊이에 도달하면 무조건 Action 생성
            create_action = (parent_depth + 1 >= self.max_depth)
            # 또는, 확률적으로 Action 생성 (트리가 너무 깊어지는 것을 방지)
            if not create_action:
                # 남은 노드가 적을수록 Action 생성 확률 증가
                prob_action = 0.2 + 0.5 * (1 - nodes_to_create / budget)
                if random.random() < prob_action:
                    create_action = True
            
            # 3. 자식 노드 생성
            if create_action:
                if nodes_to_create >= 1:
                    self._create_action_node(parent_id)
                    nodes_to_create -= 1
                    # Action 노드를 자식으로 가졌으므로 이 부모는 닫힘
                    open_list.remove(parent_id)
            else:
                # Decision 노드 생성
                num_children = random.randint(1, self.max_children)
                # 예산과 남은 노드 슬롯을 초과하지 않도록 조정
                num_children = min(num_children, nodes_to_create, self.max_nodes - self.next_idx)
                
                if num_children > 0:
                    for _ in range(num_children):
                        child_id = self._create_decision_node(parent_id)
                        open_list.append(child_id) # 새로 생긴 Decision 노드는 열린 노드
                    nodes_to_create -= num_children
                    # 자식을 할당했으므로 이 부모는 닫힘
                    open_list.remove(parent_id)
                else: # 생성할 자식이 없으면 닫음
                    open_list.remove(parent_id)

        # 4. 후처리: 루프가 끝났는데도 열려있는 노드가 있다면, 강제로 Action 노드 할당
        for parent_id in open_list:
            if self.next_idx < self.max_nodes:
                 self._create_action_node(parent_id)

    def _create_action_node(self, parent_id):
        """Action 노드 하나를 생성하고 Tensor에 기록"""
        idx = self._get_next_idx()
        if idx is None: return None
        
        self.data[idx, COL_NODE_TYPE] = NODE_TYPE_ACTION
        self.data[idx, COL_PARENT_IDX] = parent_id
        self.data[idx, COL_DEPTH] = self.data[parent_id, COL_DEPTH] + 1
        
        # 랜덤 파라미터 생성
        self.data[idx, COL_PARAM_1] = random.choice([POS_TYPE_LONG, POS_TYPE_SHORT])
        self.data[idx, COL_PARAM_2] = random.random() # 진입 비중 (0~1)
        self.data[idx, COL_PARAM_3] = random.randint(1, 100) # 레버리지 (1~100)
        return idx

    def _create_decision_node(self, parent_id):
        """Decision 노드 하나를 생성하고 Tensor에 기록"""
        idx = self._get_next_idx()
        if idx is None: return None

        self.data[idx, COL_NODE_TYPE] = NODE_TYPE_DECISION
        self.data[idx, COL_PARENT_IDX] = parent_id
        self.data[idx, COL_DEPTH] = self.data[parent_id, COL_DEPTH] + 1
        
        # 랜덤 비교 구문 생성
        comp_type = random.choice([COMP_TYPE_FEAT_NUM, COMP_TYPE_FEAT_FEAT])
        self.data[idx, COL_PARAM_3] = comp_type
        self.data[idx, COL_PARAM_2] = random.choice([OP_GT, OP_LT, OP_EQ]) # Operator

        if comp_type == COMP_TYPE_FEAT_NUM:
            feat_name = random.choice(list(self.feature_num.keys()))
            feat_idx = self.all_features.index(feat_name)
            min_val, max_val = self.feature_num[feat_name]
            comp_val = random.uniform(min_val, max_val)
            
            self.data[idx, COL_PARAM_1] = feat_idx
            self.data[idx, COL_PARAM_4] = comp_val
        else: # COMP_TYPE_FEAT_FEAT
            feat1_name, feat2_name = random.sample(self.feature_pair, 2)
            feat1_idx = self.all_features.index(feat1_name)
            feat2_idx = self.all_features.index(feat2_name)
            
            self.data[idx, COL_PARAM_1] = feat1_idx
            self.data[idx, COL_PARAM_4] = feat2_idx

        return idx

    def _get_next_idx(self):
        """사용 가능한 다음 노드 인덱스를 반환"""
        if self.next_idx < self.max_nodes:
            idx = self.next_idx
            self.next_idx += 1
            return idx
        return None

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
            'feature_pair': self.feature_pair
        }
        torch.save(state, filepath)
        print(f"Tree saved to {filepath}")

    def load(self, source):
        """파일 또는 state_dict로부터 트리 상태를 로드합니다."""
        if isinstance(source, str): # 파일 경로인 경우
            if not os.path.exists(source):
                raise FileNotFoundError(f"File not found: {source}")
            state = torch.load(source)
        elif isinstance(source, dict): # state_dict인 경우
            state = source
        else:
            raise TypeError("source must be a filepath string or a state_dict")

        self.__init__(
            state['max_nodes'], state['max_depth'], state['max_children'],
            state['feature_num'], state['feature_pair']
        )
        self.data.copy_(state['data'])
        self.next_idx = state['next_idx']
        self.initialized = True
        print(f"Tree loaded successfully.")
    
    def _node_label_color(self, idx):
        """시각화를 위한 노드의 레이블과 색상을 생성합니다."""
        node = self.data[idx]
        node_type = int(node[COL_NODE_TYPE].item())
        
        label = f"ID: {idx}\n"
        color = "grey"

        if node_type == NODE_TYPE_ROOT_BRANCH:
            branch_type = int(node[COL_PARAM_1].item())
            label += f"START\n{ROOT_BRANCH_MAP.get(branch_type, 'UNKNOWN')}"
            color = "#FFD700" # Gold
        elif node_type == NODE_TYPE_DECISION:
            feat1_idx = int(node[COL_PARAM_1].item())
            op = OPERATOR_MAP.get(int(node[COL_PARAM_2].item()))
            comp_type = int(node[COL_PARAM_3].item())
            
            feat1_name = self.all_features[feat1_idx]
            
            if comp_type == COMP_TYPE_FEAT_NUM:
                val = node[COL_PARAM_4].item()
                label += f"IF {feat1_name} {op} {val:.2f}"
            else:
                feat2_idx = int(node[COL_PARAM_4].item())
                feat2_name = self.all_features[feat2_idx]
                label += f"IF {feat1_name} {op} {feat2_name}"
            color = "#1E90FF" # DodgerBlue
        elif node_type == NODE_TYPE_ACTION:
            pos = POS_TYPE_MAP.get(int(node[COL_PARAM_1].item()))
            size = node[COL_PARAM_2].item()
            lev = int(node[COL_PARAM_3].item())
            label += f"ACTION: {pos}\nSize: {size:.2f}, Lev: {lev}x"
            color = "#32CD32" # LimeGreen
        else:
            label += "UNUSED"

        return label, color

    # ───── graph visualization ─────
    def visualize_graph(self, file="ga_tree.html", open_browser=True):
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

            parent_id = int(self.data[idx, COL_PARENT_IDX].item())
            if parent_id != -1:
                if int(self.data[parent_id, COL_NODE_TYPE].item()) != NODE_TYPE_UNUSED:
                    g.add_edge(parent_id, idx)

        if not g.nodes:
            print("No nodes to visualize.")
            return

        net = Network(height="800px", width="100%", directed=True, notebook=False, cdn_resources='remote')
        
        net.from_nx(g)
        
        try:
            # 계층적 레이아웃 옵션 적용
            net.set_options("""
            var options = {
              "layout": {
                "hierarchical": {
                  "enabled": true,
                  "levelSeparation": 200,
                  "nodeSpacing": 150,
                  "treeSpacing": 250,
                  "direction": "UD",
                  "sortMethod": "directed"
                }
              },
              "physics": { "enabled": false }
            }
            """)
        except Exception as e:
            print(f"Pyvis options error: {e}. Using default layout.")

        try:
            net.save_graph(file)
            print(f"Graph saved -> {file}")
            if open_browser:
                # 절대 경로로 파일 열기
                webbrowser.open("file://" + os.path.realpath(file))
        except Exception as e:
            print(f"Error saving or opening graph: {e}")

class GATreePop:
    """
    GATree의 집단(Population)을 관리하는 클래스.
    모든 트리의 데이터를 하나의 거대한 텐서로 관리하여 CUDA 연산에 최적화합니다.
    """
    def __init__(self, pop_size, max_nodes, max_depth, max_children, feature_num, feature_pair):
        self.pop_size = pop_size
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_children = max_children
        self.feature_num = feature_num
        self.feature_pair = feature_pair
        
        self.initialized = False
        # 모든 개체의 데이터를 담을 거대한 3D 텐서
        self.population_tensor = torch.zeros((pop_size, max_nodes, NODE_INFO_DIM), dtype=torch.float32)
        # 각 GATree 객체를 담을 리스트
        self.population = []

    def make_population(self):
        """
        설정된 pop_size만큼 GATree 개체를 생성하여 집단을 초기화합니다.
        """
        self.population = []
        for i in range(self.pop_size):
            print(f"--- Creating Tree {i+1}/{self.pop_size} ---")
            # population_tensor의 일부(view)를 GATree에 전달
            tree_data_view = self.population_tensor[i]
            tree = GATree(
                self.max_nodes, self.max_depth, self.max_children,
                self.feature_num, self.feature_pair,
                data_tensor=tree_data_view
            )
            tree.make_tree()
            self.population.append(tree)
        self.initialized = True
        print("\nPopulation created successfully.")
    
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
            'feature_pair': self.feature_pair
        }
        torch.save(state, filepath)
        print(f"Population saved to {filepath}")

    def load(self, filepath):
        """파일로부터 집단 전체의 상태를 로드합니다."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        state = torch.load(filepath)

        self.__init__(
            state['pop_size'], state['max_nodes'], state['max_depth'],
            state['max_children'], state['feature_num'], state['feature_pair']
        )
        # 텐서 데이터 복사
        self.population_tensor.copy_(state['population_tensor'])
        
        # 로드된 텐서로부터 GATree 객체 리스트를 재생성
        self.population = []
        for i in range(self.pop_size):
            tree_data_view = self.population_tensor[i]
            tree = GATree(
                self.max_nodes, self.max_depth, self.max_children,
                self.feature_num, self.feature_pair,
                data_tensor=tree_data_view
            )
            # 로드된 데이터로부터 next_idx를 복원해야 시각화 등이 올바르게 동작
            # UNUSED가 아닌 노드의 수를 세어 next_idx 추정
            tree.next_idx = (tree_data_view[:, COL_NODE_TYPE] != NODE_TYPE_UNUSED).sum().item()
            tree.initialized = True
            self.population.append(tree)

        self.initialized = True
        print(f"Population loaded successfully from {filepath}")


if __name__ == '__main__':
    # --- 시뮬레이션 파라미터 ---
    MAX_NODES = 1024
    MAX_DEPTH = 10
    MAX_CHILDREN = 5
    POP_SIZE = 5

    # =======================================================
    # === 1. 단일 GATree 생성, 시각화, 저장 및 로드 테스트 ===
    # =======================================================
    print("===== [Phase 1] Single GATree Demo =====")
    
    # GATree가 자체 텐서를 소유하는 경우
    print("\n1. Creating a standalone GATree...")
    tree1 = GATree(MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_PAIR)
    tree1.make_tree()
    
    # 생성된 트리 시각화
    tree1.visualize_graph(file="single_tree_generated.html")
    
    # 트리 저장
    tree1.save("single_tree.pth")
    
    # 새로운 객체에 트리 로드
    print("\n2. Loading the tree into a new GATree object...")
    tree2 = GATree(MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_PAIR)
    tree2.load("single_tree.pth")
    
    # 로드된 트리 시각화 (결과가 동일한지 확인)
    tree2.visualize_graph(file="single_tree_loaded.html")

    # 데이터가 동일한지 확인
    assert torch.equal(tree1.data, tree2.data), "Saved and Loaded trees are not identical!"
    print("\nStandalone GATree save/load test PASSED.")


    # =====================================================
    # === 2. GATreePop 생성, 시각화, 저장 및 로드 테스트 ===
    # =====================================================
    print("\n\n===== [Phase 2] GATreePop Demo =====")
    
    # GATree가 GATreePop의 텐서 view를 참조하는 경우
    print("\n1. Creating a population of GATrees...")
    population1 = GATreePop(POP_SIZE, MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_PAIR)
    population1.make_population()

    # 집단의 첫 번째 트리 시각화
    print("\nVisualizing the first tree from the population...")
    first_tree_from_pop = population1.population[0]
    first_tree_from_pop.visualize_graph(file="population_tree_generated.html")

    # 집단 저장
    population1.save("population.pth")
    
    # 새로운 객체에 집단 로드
    print("\n2. Loading the population into a new GATreePop object...")
    population2 = GATreePop(POP_SIZE, MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_PAIR)
    population2.load("population.pth")

    # 로드된 집단의 첫 번째 트리 시각화 (결과가 동일한지 확인)
    print("\nVisualizing the first tree from the loaded population...")
    first_tree_from_loaded_pop = population2.population[0]
    first_tree_from_loaded_pop.visualize_graph(file="population_tree_loaded.html")
    
    # 데이터가 동일한지 확인
    assert torch.equal(population1.population_tensor, population2.population_tensor), "Saved and Loaded populations are not identical!"
    print("\nGATreePop save/load test PASSED.")
    
    # population1의 첫번째 트리 데이터와 population2의 첫번째 트리 데이터가 동일한지 확인
    # 이는 GATree 객체들이 텐서의 view를 올바르게 참조하고 있음을 증명
    assert torch.equal(population1.population[0].data, population2.population[0].data)
    print("Memory view reference in loaded population confirmed.")