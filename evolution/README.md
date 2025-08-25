# evolution/README.md

GA 트리 집단을 세대별로 진화시키는 엔진과 오퍼레이터 패키지(Selection/Crossover/Mutation)를 묶는 상위 폴더입니다.

## 파일 구성

- `evolution.py`: 진화 컨트롤러 `Evolution` 클래스
  - 초기화: `(population, selection, crossover, mutation, parent_size, num_elites)`을 받습니다.
  - `evolve(fitness: torch.Tensor)`: 한 세대를 수행합니다.
    - 엘리트 보존: `select_elites`로 상위 `num_elites` 개체를 보존(GPU 텐서로 이동)
    - 교배 풀 구성: 상위 `parent_size` 인덱스를 선발 후, `pick_parents`로 실제 부모 인덱스 샘플링
    - 교차: 부모 텐서(배치)를 `crossover(parents_gpu)`로 전달하여 자손 생성(GPU)
    - 변이: 자손 텐서를 `mutation(children_gpu)`로 변이(GPU)
    - 다음 세대: `elite + offspring`을 합쳐 집단 텐서를 업데이트(최종 CPU 반영)
- `__init__.py`: `Evolution`와 하위 패키지 네임스페이스를 내보냅니다.

## 참고 사항

- CUDA 확장(`gatree_cuda_compat`)이 컴파일되어 있어야 대부분의 연산을 GPU에서 수행할 수 있습니다.
- `fitness`는 GPU 텐서여야 하며, 내부에서 부모/자손 텐서도 GPU에 생성/유지됩니다.
- 집단 텐서 스키마(노드 정보 차원, 노드 타입 등)는 `models/constants.py`와 동일해야 합니다.
- `parent_size`는 최소 2 이상이어야 하며, `num_elites ≤ parent_size ≤ pop_size` 제약을 가집니다.

## 예시

### 1) Selection/Crossover/Mutation 구성 후 한 세대 수행

```python
import torch
from evolution.evolution import Evolution
from evolution.Selection.tournament import TournamentSelection
from evolution.Selection.roulette import RouletteSelection
from evolution.Crossover.chain import ChainCrossover
from evolution.Crossover.root_branch import RootBranchCrossover
from evolution.Crossover.subtree import SubtreeCrossover
from evolution.Crossover.node import NodeCrossover
from evolution.Mutation.chain import ChainMutation
from evolution.Mutation.add_subtree import AddSubtreeMutation
from evolution.Mutation.add_node import AddNodeMutation
from evolution.Mutation.delete_subtree import DeleteSubtreeMutation
from evolution.Mutation.delete_node import DeleteNodeMutation
from evolution.Mutation.node_param import NodeParamMutation
from evolution.Mutation.reinitialize_node import ReinitializeNodeMutation
from models.model import GATreePop, get_all_features

# data 단계에서 생성된 model_config 사용
feature_num = model_config['feature_num']
feature_map = model_config['feature_comparison_map']
feature_bool = model_config['feature_bool']
all_features = get_all_features(feature_num, feature_map, feature_bool)

pop = GATreePop(
    pop_size=128, max_nodes=256, max_depth=8, max_children=3,
    feature_num=feature_num,
    feature_comparison_map=feature_map,
    feature_bool=feature_bool,
    all_features=all_features,
)
pop.make_population(init_mode='cuda', device='cuda:0')

# 공통 변이/교차 설정에 필요한 config
op_config = {
    'all_features': all_features,
    'feature_num': feature_num,
    'feature_comparison_map': feature_map,
    'feature_bool': feature_bool,
    'max_depth': pop.max_depth,
    'max_children': pop.max_children,
}

selection = TournamentSelection(k=5)  # 또는 RouletteSelection()

crossover = ChainCrossover(
    crossovers=[
        RootBranchCrossover(rate=0.5, max_nodes=pop.max_nodes),
        SubtreeCrossover(rate=0.35, max_nodes=pop.max_nodes, max_depth=pop.max_depth, mode='context'),
        NodeCrossover(rate=0.15, mode='free'),
    ],
    weights=[0.4, 0.4, 0.2],
)

mutation = ChainMutation([
    AddSubtreeMutation(prob=0.20, config=op_config, node_count_range=(2, 6), max_nodes=pop.max_nodes),
    AddNodeMutation(prob=0.15, config=op_config, max_add_nodes=3, max_nodes=pop.max_nodes),
    DeleteSubtreeMutation(prob=0.10, config=op_config, max_nodes=pop.max_nodes),
    DeleteNodeMutation(prob=0.10, config=op_config, max_delete_nodes=3, max_nodes=pop.max_nodes),
    NodeParamMutation(prob=0.20, config=op_config, noise_ratio=0.1, leverage_change=5),
    ReinitializeNodeMutation(prob=0.05, config=op_config),
])

evo = Evolution(population=pop, selection=selection, crossover=crossover,
                mutation=mutation, parent_size=64, num_elites=2)

# 가상의 GPU fitness
fitness = torch.rand(pop.pop_size, device=torch.device('cuda'))
evo.evolve(fitness)
```

