# models/README.md

GA 트리 표현, 상수 정의, 집단(population) 관리가 구현되어 있으며, CPU/ CUDA 초기화 경로와 선택적 시각화를 지원합니다.

## 파일 구성

- `constants.py`: 공용 열거형과 텐서 레이아웃
  - 텐서 스키마(`NODE_INFO_DIM = 7`):
    - `COL_NODE_TYPE`, `COL_PARENT_IDX`, `COL_DEPTH`, `COL_PARAM_1..4`
  - 노드 타입: `NODE_TYPE_ROOT_BRANCH`, `NODE_TYPE_DECISION`, `NODE_TYPE_ACTION`
  - 루트 브랜치: `ROOT_BRANCH_LONG`, `ROOT_BRANCH_HOLD`, `ROOT_BRANCH_SHORT`
  - 비교 타입: `COMP_TYPE_FEAT_NUM`, `COMP_TYPE_FEAT_FEAT`, `COMP_TYPE_FEAT_BOOL`; 연산자 `OP_GTE`, `OP_LTE`
  - 액션 타입: `ACTION_NEW_LONG`, `ACTION_NEW_SHORT`, `ACTION_CLOSE_ALL`, `ACTION_CLOSE_PARTIAL`, `ACTION_ADD_POSITION`, `ACTION_FLIP_POSITION` 및 디버깅용 문자 맵

- `model.py`: 핵심 GA 트리 클래스 및 헬퍼; `gatree_cuda_compat`가 있으면 CUDA 커널과 연동
  - 피처 헬퍼:
    - `get_all_features(feature_num, feature_map, feature_bool)`: 숫자/비교/불리언 피처명을 안정적(정렬)으로 결합한 전역 인덱스 공간을 생성
  - `GATree` 클래스:
    - 단일 트리를 `constants.py` 스키마를 따르는 고정 크기 `torch.Tensor`로 저장
    - 트리 생성: `make_tree()`에서 3개 루트 브랜치(LONG/HOLD/SHORT) 생성 후, 브랜치별 예산으로 `_grow_branch`를 통해 Decision/Action 노드를 확장
    - 노드 생성:
      - `_create_decision_node(parent_id)`: 비교 타입과 파라미터를 무작위로 선택하고 텐서에 인덱스/값 기록
      - `_create_action_node(parent_id)`: 루트 컨텍스트(HOLD는 진입, LONG/SHORT는 청산/추가/플립 우선)에 따라 액션과 파라미터 기록
    - 구조 헬퍼: `_build_adjacency_list()`(부모→자식 맵), `_get_root_branch_type_from_child()`(루트 컨텍스트 추정), `set_next_idx()`(다음 인덱스 동기화)
    - 시각화: `visualize(file='tree.html', open_browser=False)`로 `networkx`/`pyvis` 기반 HTML 그래프 저장
  - `GATreePop` 클래스:
    - 집단 텐서 `(B, N, D)`와 트리 목록을 관리
    - `make_population(num_processes=1, device='cuda', init_mode='cuda', node_budget=None)`:
      - CUDA 경로: GPU 버퍼를 할당하고 `init_population_cuda` 실행 후 `validate_trees`로 검증, 각 트리를 GPU 텐서 뷰로 연결
      - CPU 경로: 멀티프로세싱/공유메모리 기반 초기화 (CUDA 미사용 시)
    - 커널용 피처 테이블(숫자 min/max, 불리언, 피처쌍)을 `ALL_FEATURES` 인덱스 공간으로 매핑 보관

## 참고 사항

- 각 트리는 고정 2D 텐서로 인코딩되어 CUDA 커널 효율을 높입니다.
- 데이터 생성과 예측 시 피처 이름/순서는 반드시 일치해야 하며, 피처 엔지니어링과 동일한 설정으로 `all_features`를 구성해야 합니다.
- 시각화는 선택 사항이며 `pyvis`가 필요합니다.

## 예시

### 1) GA 트리 집단 생성 (CUDA/CPU)

```python
from models.model import GATreePop, get_all_features

# data 단계에서 생성된 model_config 사용 예시
feature_num = model_config['feature_num']
feature_map = model_config['feature_comparison_map']
feature_bool = model_config['feature_bool']
all_features = get_all_features(feature_num, feature_map, feature_bool)

pop = GATreePop(
    pop_size=128,
    max_nodes=64,
    max_depth=6,
    max_children=3,
    feature_num=feature_num,
    feature_comparison_map=feature_map,
    feature_bool=feature_bool,
    all_features=all_features,
)

# CUDA 경로 (확장 모듈이 빌드되어 있고 GPU 사용 가능 시)
try:
    pop.make_population(init_mode='cuda', device='cuda:0')
except Exception:
    # CPU 대체 경로
    pop.make_population(init_mode='cpu', device='cpu', num_processes=1)

assert pop.initialized
```

### 2) 트리 시각화 (선택)

```python
tree0 = pop.population[0]
tree0.visualize('tree0.html', open_browser=False)
```
