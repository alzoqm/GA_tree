import torch
import os
import shutil

# --- 제공해주신 코드의 클래스들을 임포트합니다 ---

# 모델 관련
from models.model import (GATree, GATreePop, FEATURE_NUM, FEATURE_PAIR, ALL_FEATURES,
                   COL_NODE_TYPE, NODE_TYPE_UNUSED)

# 변이 연산자 관련
from evolution.Mutation.base import BaseMutation
from evolution.Mutation.chain import ChainMutation
from evolution.Mutation.node_param import NodeParamMutation
from evolution.Mutation.reinitialize_node import ReinitializeNodeMutation
from evolution.Mutation.add_node import AddNodeMutation
from evolution.Mutation.delete_node import DeleteNodeMutation
from evolution.Mutation.add_subtree import AddSubtreeMutation
from evolution.Mutation.delete_subtree import DeleteSubtreeMutation


# --- 테스트 환경 설정 ---
POP_SIZE = 1  # 하나의 트리에 집중하여 변화를 명확히 보기 위함
MAX_NODES = 32
MAX_DEPTH = 5
MAX_CHILDREN = 3
TEST_RESULTS_DIR = "test_results"

# --- 테스트에 필요한 설정(config) 딕셔너리 ---
# 변이 연산자들이 트리의 제약조건을 알아야 하므로 config가 필요합니다.
config = {
    'max_nodes': MAX_NODES,
    'max_depth': MAX_DEPTH,
    'max_children': MAX_CHILDREN,
    'feature_num': FEATURE_NUM,
    'feature_pair': FEATURE_PAIR,
    'all_features': ALL_FEATURES,
}

def run_mutation_test(name: str, mutation_op: BaseMutation, population: GATreePop, if_save_model=False):
    """
    특정 변이 연산자를 테스트하고 결과를 시각화하는 헬퍼 함수.
    
    Args:
        name (str): 테스트 이름 (파일 이름에 사용됨).
        mutation_op (BaseMutation): 테스트할 변이 연산자 객체.
        population (GATreePop): 원본 GATree 집단.
    """
    print(f"\n===== [Testing: {name}] =====")
    
    # 1. 테스트 전 상태 저장 (깊은 복사)
    original_tensor = population.population_tensor.clone()
    
    # 원본 GATree 객체 생성 (시각화용)
    tree_before = GATree(MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_PAIR,
                         data_tensor=original_tensor[0])

    # 로드된 텐서로부터 next_idx 복원
    tree_before.initialized = True
    tree_before.set_next_idx()
    
    if if_save_model:
        tree_before.save(f'{TEST_RESULTS_DIR}/{name}_before.pth')

    print(f"Nodes before mutation: {tree_before.next_idx}")
    tree_before.visualize_graph(file=os.path.join(TEST_RESULTS_DIR, f"{name}_01_before.html"), open_browser=False)

    # 2. 변이 연산 수행
    mutated_tensor = mutation_op(original_tensor)

    # 3. 변이 후 결과 확인
    # 변이 후 GATree 객체 생성 (시각화용)
    tree_after = GATree(MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_PAIR,
                        data_tensor=mutated_tensor[0])
    tree_after.initialized = True
    tree_after.reorganize_nodes()
    tree_after.set_next_idx()
    

    print(f"Nodes after mutation:  {tree_after.next_idx}")
    tree_after.visualize_graph(file=os.path.join(TEST_RESULTS_DIR, f"{name}_02_after.html"), open_browser=False)

    # 4. 변이가 실제로 일어났는지 확인 (텐서가 달라졌는지)
    if not torch.equal(population.population_tensor, mutated_tensor):
        print(f"✅ SUCCESS: Tensor has changed after {name}.")
    else:
        # 변이는 확률적이므로, 일어나지 않을 수도 있습니다.
        print(f"⚠️ INFO: Tensor did not change. This can happen if the mutation was not triggered by chance or no valid target was found.")

    print(f"-> Check HTML files in '{TEST_RESULTS_DIR}/' directory to see the changes.")
    if if_save_model:
        tree_after.save(f'{TEST_RESULTS_DIR}/{name}_after.pth')


if __name__ == '__main__':
    # 테스트 결과 디렉토리 정리 및 생성
    if os.path.exists(TEST_RESULTS_DIR):
        shutil.rmtree(TEST_RESULTS_DIR)
    os.makedirs(TEST_RESULTS_DIR)

    # 1. 테스트를 위한 초기 집단 생성
    print("===== [Setup] Creating initial GATree population... =====")
    initial_population = GATreePop(POP_SIZE, MAX_NODES, MAX_DEPTH, MAX_CHILDREN, FEATURE_NUM, FEATURE_PAIR)
    initial_population.make_population()
    print("Population created successfully.")
    
    # 원본 트리 저장
    initial_tree = initial_population.population[0]
    initial_tree.visualize_graph(file=os.path.join(TEST_RESULTS_DIR, "original_tree.html"), open_browser=False)
    print(f"Original tree saved to '{TEST_RESULTS_DIR}/original_tree.html'")

    # --- 각 변이 연산자 테스트 실행 ---
    # 각 변이 연산자의 prob를 1.0으로 설정하여, 테스트 시 반드시 실행되도록 강제합니다.
    
    # 값 기반 변이 (구조 변경 없음)
    run_mutation_test("NodeParamMutation", NodeParamMutation(prob=1.0, config=config), initial_population)
    run_mutation_test("ReinitializeNodeMutation", ReinitializeNodeMutation(prob=1.0, config=config), initial_population)

    # 구조 변경 변이
    run_mutation_test("DeleteNodeMutation", DeleteNodeMutation(prob=1.0, config=config), initial_population, if_save_model=False)
    run_mutation_test("AddNodeMutation", AddNodeMutation(prob=1.0, config=config), initial_population, if_save_model=True)
    run_mutation_test("AddSubtreeMutation", AddSubtreeMutation(prob=1.0, config=config, node_count_range=(3, 6)), initial_population, if_save_model=False)
    run_mutation_test("DeleteSubtreeMutation", DeleteSubtreeMutation(prob=1.0, config=config), initial_population)
    
    # 체인 변이 테스트
    chained_mutations = ChainMutation(mutations=[
        DeleteSubtreeMutation(prob=1.0, config=config),
        DeleteNodeMutation(prob=1.0, config=config),
        AddNodeMutation(prob=1.0, config=config),
        AddSubtreeMutation(prob=1.0, config=config, node_count_range=(3, 6)),
        # NodeParamMutation(prob=0.5, config=config),
        
    ])
    run_mutation_test("ChainMutation", chained_mutations, initial_population)
    
    print("\n\n===== All mutation tests finished. =====")
    print("Please open the HTML files in the 'test_results' folder to visually inspect the results.")