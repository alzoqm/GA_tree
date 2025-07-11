# test_crossover.py
import torch
import os

# --- 테스트에 필요한 클래스들을 임포트합니다. ---
# 프로젝트 루트에서 실행한다고 가정합니다.
from models.model import GATree, GATreePop, NODE_TYPE_UNUSED, FEATURE_NUM, FEATURE_PAIR
from evolution.Crossover.subtree import SubtreeCrossover

# ----------------------------------------------------
# --- 테스트 환경 설정 ---
# ----------------------------------------------------
POP_SIZE = 2  # 테스트를 위해 부모 2개만 생성
MAX_NODES = 100
MAX_DEPTH = 7
MAX_CHILDREN = 3
CROSSOVER_RATE = 1.0 # 테스트를 위해 교차가 반드시 일어나도록 설정
MAX_RETRIES = 10 # 교차점 찾기 재시도 횟수
OUTPUT_DIR = "crossover_test_results"

# --- 헬퍼 함수: 시각화를 위해 raw 텐서를 GATree 객체로 변환 ---
def tensor_to_gatree(tensor: torch.Tensor, config: dict) -> GATree:
    """
    단일 트리 텐서(2D)를 시각화 가능한 GATree 객체로 래핑합니다.
    """
    tree = GATree(
        max_nodes=config['max_nodes'],
        max_depth=config['max_depth'],
        max_children=config['max_children'],
        feature_num=config['feature_num'],
        feature_pair=config['feature_pair'],
        data_tensor=tensor
    )
    # 로드된 데이터로부터 next_idx를 복원해야 시각화가 올바르게 동작합니다.
    # UNUSED가 아닌 노드의 수를 세어 next_idx를 추정합니다.
    tree.next_idx = (tensor[:, NODE_TYPE_UNUSED] != NODE_TYPE_UNUSED).sum().item()
    tree.initialized = True
    return tree

# --- 메인 테스트 로직 ---
if __name__ == '__main__':
    print("===== SubtreeCrossover 테스트를 시작합니다. =====")

    # 1. 결과물을 저장할 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"결과 HTML 파일은 '{OUTPUT_DIR}/' 폴더에 저장됩니다.")

    # 2. GATreePop을 사용해 2개의 부모 트리 생성
    print("\n[1] 2개의 랜덤 부모 트리를 생성합니다...")
    config = {
        'max_nodes': MAX_NODES,
        'max_depth': MAX_DEPTH,
        'max_children': MAX_CHILDREN,
        'feature_num': FEATURE_NUM,
        'feature_pair': FEATURE_PAIR,
        'all_features': list(FEATURE_NUM.keys()) + FEATURE_PAIR
    }
    population = GATreePop(
        pop_size=POP_SIZE,
        max_nodes=config['max_nodes'],
        max_depth=config['max_depth'],
        max_children=config['max_children'],
        feature_num=config['feature_num'],
        feature_pair=config['feature_pair']
    )
    population.make_population()
    
    # 3. 교차 전 부모 트리 시각화
    print("\n[2] 교차 전(Before) 부모 트리들을 시각화합니다...")
    parents_tensor = population.population_tensor
    parent1_tensor = parents_tensor[0]
    parent2_tensor = parents_tensor[1]

    parent1_tree = tensor_to_gatree(parent1_tensor, config)
    parent2_tree = tensor_to_gatree(parent2_tensor, config)

    p1_nodes = parent1_tree.next_idx
    p2_nodes = parent2_tree.next_idx
    
    parent1_tree.visualize_graph(file=os.path.join(OUTPUT_DIR, "parent_1_before.html"), open_browser=False)
    parent2_tree.visualize_graph(file=os.path.join(OUTPUT_DIR, "parent_2_before.html"), open_browser=False)
    
    print(f"  - Parent 1 (노드 수: {p1_nodes}): parent_1_before.html")
    print(f"  - Parent 2 (노드 수: {p2_nodes}): parent_2_before.html")

    # 4. SubtreeCrossover 실행
    print("\n[3] SubtreeCrossover 연산을 수행합니다...")
    crossover = SubtreeCrossover(
        rate=CROSSOVER_RATE,
        max_nodes=MAX_NODES,
        max_depth=MAX_DEPTH,
        max_retries=MAX_RETRIES,
        mode='context'
    )
    
    # crossover.__call__은 (num_parents, ...) -> (num_offspring, ...) 형태입니다.
    # 부모 2명을 넣으면 자식 1명이 나옵니다.
    children_tensor = crossover(parents_tensor)
    child1_tensor = children_tensor[0]

    print("  - Crossover 연산 완료.")

    # 5. 교차 후 자식 트리 시각화
    print("\n[4] 교차 후(After) 생성된 자식 트리를 시각화합니다...")
    child1_tree = tensor_to_gatree(child1_tensor, config)
    c1_nodes = child1_tree.next_idx
    
    child1_tree.visualize_graph(file=os.path.join(OUTPUT_DIR, "child_1_after.html"), open_browser=False)
    print(f"  - Child 1 (노드 수: {c1_nodes}): child_1_after.html")

    # 6. 최종 결과 안내
    print("\n===== 테스트 완료 =====")
    print("분석 방법:")
    print(f"1. '{OUTPUT_DIR}' 폴더를 엽니다.")
    print("2. 'parent_1_before.html' 과 'parent_2_before.html' 파일을 웹 브라우저로 엽니다.")
    print("3. 'child_1_after.html' 파일을 웹 브라우저로 엽니다.")
    print("\n-> Child 1 트리는 Parent 1의 몸체에 Parent 2의 특정 서브트리(가지)가 이식된 형태일 것입니다.")
    print("-> 이 과정을 통해 SubtreeCrossover의 서브트리 교환, 노드/깊이 제약조건 검사, 지능적 재시도 기능이 올바르게 동작함을 확인할 수 있습니다.")
    print(f"-> 브라우저에서 '{os.path.realpath(OUTPUT_DIR)}' 경로의 파일들을 확인해주세요.")