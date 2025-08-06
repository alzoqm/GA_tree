# evolution/Selection/roulette.py
import torch
from .base import BaseSelection

class RouletteSelection(BaseSelection):
    """
    룰렛 휠 선택(Roulette Wheel Selection) 방식 구현 클래스.
    적합도에 비례하여 부모를 확률적으로 선택합니다.
    """

    def select_elites(self, fitness: torch.Tensor, num_selects: int) -> torch.Tensor:
        """
        가장 높은 적합도를 가진 개체를 엘리트로 선택합니다.
        엘리트 선택은 확률적이지 않으므로, 가장 적합도가 높은 순서대로 반환합니다.

        Args:
            fitness (torch.Tensor): 집단 내 각 개체의 적합도를 담은 1D 텐서.
            num_selects (int): 선택할 엘리트 개체의 수.

        Returns:
            torch.Tensor: 선택된 엘리트 개체들의 인덱스를 담은 1D 텐서.
        """
        # 적합도를 기준으로 내림차순 정렬하여 상위 인덱스를 반환
        elite_indices = torch.argsort(fitness, descending=True)[:num_selects]
        return elite_indices

    def pick_parents(self, fitness: torch.Tensor, num_parents: int) -> torch.Tensor:
        """
        룰렛 휠 방식을 사용하여 교배에 참여할 부모를 선택합니다.
        적합도가 음수일 경우, 모든 값이 0 이상이 되도록 조정하여 처리합니다.

        Args:
            fitness (torch.Tensor): 집단 내 각 개체의 적합도를 담은 1D 텐서.
            num_parents (int): 선택할 부모 개체의 수 (중복 허용).

        Returns:
            torch.Tensor: 선택된 부모 개체들의 인덱스를 담은 1D 텐서.
        """
        # 1. 적합도 값 조정 (음수 처리)
        # 룰렛 휠 선택은 음수 적합도에서 동작하지 않으므로, 모든 값을 0 이상으로 만듭니다.
        min_fitness = torch.min(fitness)
        if min_fitness < 0:
            # 가장 작은 값이 0이 되도록 모든 값을 평행 이동
            shifted_fitness = fitness - min_fitness
        else:
            shifted_fitness = fitness

        # 2. 선택 확률 계산
        fitness_sum = torch.sum(shifted_fitness)

        if fitness_sum == 0:
            # 모든 개체의 적합도가 동일하게 최하위인 경우 (예: 전부 0), 균등 확률로 선택
            probs = torch.ones_like(fitness) / len(fitness)
        else:
            probs = shifted_fitness / fitness_sum

        # 3. 확률에 기반한 부모 선택
        # torch.multinomial을 사용하여 주어진 확률 분포에 따라 인덱스를 샘플링합니다.
        # replacement=True는 한 개체가 여러 번 부모로 선택될 수 있음을 의미합니다.
        parent_indices = torch.multinomial(probs, num_parents, replacement=True)

        return parent_indices