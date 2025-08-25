# evolution/Selection/README.md

유전 알고리즘의 선택(Selection) 전략을 제공합니다. 엘리트 보존과 부모 선택(교차용)을 분리된 인터페이스로 지원합니다.

## 파일 구성

- `base.py`: `BaseSelection`
  - `select_elites(fitness, num_selects) -> Tensor`: 상위 개체 인덱스(중복 없음)
  - `pick_parents(fitness, num_parents) -> Tensor`: 부모 샘플(중복 허용 가능)

- `roulette.py`: `RouletteSelection`
  - 룰렛 휠 방식: 적합도에 비례하여 확률적으로 부모 선택
  - 음수 적합도 보정: 전체를 평행 이동해 0 이상으로 만든 뒤 정규화
  - `select_elites`: 단순 상위 정렬로 엘리트 반환(결정적)

- `tournament.py`: `TournamentSelection`
  - 토너먼트 방식: 무작위로 `k`명 추출 → 최고 적합도 승자 선발
  - `k >= 1` 제약, `select_elites`는 상위 정렬 기반

## 참고 사항

- 입력 `fitness`는 1D 텐서이며 GPU 상에서 작동하도록 설계되었습니다.
- `RouletteSelection`은 `torch.multinomial`로 확률 샘플링을 수행합니다.
- 엘리트 개수/부모 수는 `Evolution` 단계에서 일관되게 관리합니다.

## 예시

### 1) 엘리트와 부모 선택

```python
import torch
from evolution.Selection.roulette import RouletteSelection
from evolution.Selection.tournament import TournamentSelection

fitness = torch.tensor([0.1, 0.8, 0.3, 0.5], device='cuda')

sel = RouletteSelection()
elites = sel.select_elites(fitness, num_selects=2)   # 상위 2명
parents_rel = sel.pick_parents(fitness, num_parents=6)

sel2 = TournamentSelection(k=3)
elites2 = sel2.select_elites(fitness, num_elites=2)
parents_rel2 = sel2.pick_parents(fitness, num_parents=6)
```

