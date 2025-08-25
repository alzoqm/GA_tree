# evolution/Mutation/README.md

GA 트리의 구조/파라미터를 확률적으로 변화시키는 변이(Mutation) 오퍼레이터 모음입니다. 모든 연산은 배치 텐서를 입력으로 받아 CUDA 커널에서 한 번에 처리(one-shot)하도록 설계되었습니다.

## 파일 구성

- `base.py`: `BaseMutation`
  - 인터페이스: `__call__(chromosomes: torch.Tensor) -> torch.Tensor`
  - 입력/출력: `(B, max_nodes, node_dim)` (GPU / float32)
  - `prob`: 개체 단위 변이 확률(연산자별로 보유)

- `add_node.py`: `AddNodeMutation`
  - 임의의 엣지를 분할하여 DECISION 노드를 삽입하고 깊이/부모를 일관되게 업데이트
  - 새로 삽입된 DECISION 노드 파라미터는 `DecisionParamSampler`가 GPU 벡터화로 채움
  - 파라미터: `prob`, `config`, `max_add_nodes`, `max_nodes`

- `add_subtree.py`: `AddSubtreeMutation`
  - 임의 위치에 DECISION/ACTION으로 구성된 서브트리를 다수(범위) 추가
  - DECISION/ACTION 파라미터를 각각 벡터 샘플러로 일괄 채움
  - 안전 계약: `max_new_nodes >= 2 * max(node_count_range)` 보장 필요
  - 파라미터: `prob`, `config`, `node_count_range`, `max_nodes`, `max_new_nodes`

- `delete_node.py`: `DeleteNodeMutation`
  - 규칙을 지키며 다수 노드를 제거(후속 수리 포함), 서브트리/자식 수 제약을 커널에서 보증
  - 파라미터: `prob`, `config`, `max_delete_nodes`, `max_nodes`

- `delete_subtree.py`: `DeleteSubtreeMutation`
  - 서브트리 단위로 삭제하고, 리페어 마스크를 결합해 일괄 삭제/수리
  - 옵션: UNUSED 노드의 `parent_idx=-1` 설정, 루트 분기 고아 방지 크리티컬 리페어 수행
  - 파라미터: `prob`, `config`, `max_nodes`, `alpha`, `ensure_action_left`, `set_unused_parent_idx`

- `node_param.py`: `NodeParamMutation`
  - DECISION 비교 임계/피처, ACTION 레버리지/비율 등 노드 파라미터를 미세 조정(노이즈/증감)
  - CUDA 커널 `node_param_mutate`로 배치 전체를 한 번에 수정
  - 파라미터: `prob`, `config`, `noise_ratio`, `leverage_change`

- `reinitialize_node.py`: `ReinitializeNodeMutation`
  - 노드 파라미터를 새로 샘플링하여 완전 재초기화(DECISION/BOOL/PAIR 및 ACTION)
  - CUDA 커널 `reinitialize_node_mutate` 사용
  - 파라미터: `prob`, `config`

- `chain.py`: `ChainMutation`
  - 여러 변이를 순차 적용(각 연산자 고유 `prob`로 발생)

- `utils.py`
  - Python 사이드 유틸: 서브트리 탐색/깊이 갱신/랜덤 파라미터 생성 등

## 참고 사항

- CUDA 확장(`gatree_cuda_compat`)이 필요합니다. `python setup.py build_ext --inplace`로 빌드하세요.
- 모든 텐서는 `torch.float32`여야 하며, 디바이스는 CUDA여야 합니다.
- 공통 `config` 예시(필수 키):
  - `all_features`: 전역 피처 인덱스 공간(정렬/고정)
  - `feature_num`: `{feat_name: (min, max)}` 범위 사전
  - `feature_comparison_map`: `{feat_name: [feat_name, ...]}` 비교 허용 맵
  - `feature_bool`: 불리언 피처 목록
  - `max_depth`, `max_children`: 모델 제약과 일치해야 함

## 예시

### 1) 변이 체인 구성

```python
from evolution.Mutation.chain import ChainMutation
from evolution.Mutation.add_node import AddNodeMutation
from evolution.Mutation.add_subtree import AddSubtreeMutation
from evolution.Mutation.delete_node import DeleteNodeMutation
from evolution.Mutation.delete_subtree import DeleteSubtreeMutation
from evolution.Mutation.node_param import NodeParamMutation
from evolution.Mutation.reinitialize_node import ReinitializeNodeMutation

mutation = ChainMutation([
    AddSubtreeMutation(prob=0.20, config=op_config, node_count_range=(2, 6), max_nodes=max_nodes),
    AddNodeMutation(prob=0.15, config=op_config, max_add_nodes=3, max_nodes=max_nodes),
    DeleteSubtreeMutation(prob=0.10, config=op_config, max_nodes=max_nodes),
    DeleteNodeMutation(prob=0.10, config=op_config, max_delete_nodes=3, max_nodes=max_nodes),
    NodeParamMutation(prob=0.20, config=op_config, noise_ratio=0.1, leverage_change=5),
    ReinitializeNodeMutation(prob=0.05, config=op_config),
])

# trees: (B, max_nodes, node_dim) on CUDA
mutated = mutation(trees)
```

