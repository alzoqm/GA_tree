# evolution/Crossover/README.md

유전 알고리즘의 GA 트리 표현에 특화된 교차(Crossover) 오퍼레이터들이 포함되어 있습니다. 모든 연산은 배치 텐서 입력을 받아 GPU에서 동작하도록 설계되었습니다.

## 파일 구성

- `base.py`: `BaseCrossover`
  - 인터페이스: `__call__(parents: torch.Tensor) -> torch.Tensor`
  - 입력/출력 규약:
    - 입력 `parents`: `(2 * num_offspring, max_nodes, node_dim)` (GPU / float32)
    - 출력 `children`: `(num_offspring, max_nodes, node_dim)` (GPU / float32)
  - `rate`: 교차 시도 확률(나머지는 부모 복제)

- `root_branch.py`: `RootBranchCrossover`
  - 루트 3개 분기(LONG/HOLD/SHORT)를 부모 간에 재조합하여 자손을 생성
  - Python에서 작업 버퍼를 할당하고 CUDA 커널(`copy_branches_batch`, `repair_after_root_branch`)로 구조 복구 및 검증 수행
  - 파라미터: `rate`, `max_nodes`

- `subtree.py`: `SubtreeCrossover`
  - 임의의 서브트리를 서로 교환하여 자손 생성
  - 모드: `free`(노드 타입만 일치), `context`(루트 분기까지 문맥 일치 요구)
  - Python에서 BFS 큐/결과/매핑 버퍼를 충분히 크게 할당하여 커널 오버플로 방지
  - 파라미터: `rate`, `max_nodes`, `max_depth`, `max_retries`, `mode`

- `node.py`: `NodeCrossover`
  - 동일 타입(DECISION, ACTION) 노드 파라미터를 문맥 선택에 따라 교환
  - 모드: `free`/`context`(루트 분기 문맥 고려)
  - CUDA 헬퍼(`get_contextual_mask`, `swap_node_params`) 기반 벡터화 교환

- `chain.py`: `ChainCrossover`
  - 여러 교차 연산자를 비율대로 분할 적용 후 결과를 합쳐 섞음
  - `weights` 합은 1.0이어야 하며, 각 슬라이스는 `(count*2)` 부모를 사용

## 참고 사항

- CUDA 확장(`gatree_cuda_compat`)이 필요합니다. `python setup.py build_ext --inplace`로 빌드하세요.
- 모든 텐서는 `torch.float32` 스키마로 인코딩(정수 필드도 float)되어야 합니다.
- 입력 `parents`는 반드시 GPU에 있어야 하며, 반환 `children` 또한 GPU입니다.
- 내부에서 구조 제약(깊이/부모-자식 일관성/리프 규칙)을 커널 레벨에서 검증합니다(`validate_trees`).

## 예시

### 1) 교차 연산자 단독 실행

```python
import torch
from evolution.Crossover.root_branch import RootBranchCrossover

op = RootBranchCrossover(rate=0.8, max_nodes=max_nodes)
# parents: (2 * num_offspring, max_nodes, node_dim) on CUDA
children = op(parents)
assert children.shape[0] == parents.shape[0] // 2
```

### 2) 체인 조합 사용

```python
from evolution.Crossover.chain import ChainCrossover
from evolution.Crossover.subtree import SubtreeCrossover
from evolution.Crossover.node import NodeCrossover

crossover = ChainCrossover(
    crossovers=[
        RootBranchCrossover(rate=0.5, max_nodes=max_nodes),
        SubtreeCrossover(rate=0.35, max_nodes=max_nodes, max_depth=max_depth, mode='context'),
        NodeCrossover(rate=0.15, mode='free'),
    ],
    weights=[0.4, 0.4, 0.2]
)
children = crossover(parents)
```

