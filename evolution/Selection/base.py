# evolution/Selection/base.py
import torch
from abc import ABC, abstractmethod

class BaseSelection(ABC):
    """
    유전 알고리즘의 '선택' 연산을 위한 추상 기반 클래스.
    모든 선택 연산자 클래스는 이 클래스를 상속받아야 합니다.
    """

    @abstractmethod
    def select_elites(self, fitness: torch.Tensor, num_elites: int) -> torch.Tensor:
        """
        주어진 적합도(fitness)를 기반으로 엘리트 개체들의 인덱스를 선택합니다.

        Args:
            fitness (torch.Tensor): 집단 내 각 개체의 적합도를 담은 1D 텐서.
            num_elites (int): 선택할 엘리트 개체의 수.

        Returns:
            torch.Tensor: 선택된 엘리트 개체들의 인덱스를 담은 1D 텐서.
        """
        pass

    @abstractmethod
    def pick_parents(self, fitness: torch.Tensor, num_parents: int) -> torch.Tensor:
        """
        주어진 적합도(fitness)를 기반으로 교차에 사용할 부모 개체들의 인덱스를 선택합니다.

        Args:
            fitness (torch.Tensor): 집단 내 각 개체의 적합도를 담은 1D 텐서.
            num_parents (int): 선택할 부모 개체의 수.

        Returns:
            torch.Tensor: 선택된 부모 개체들의 인덱스를 담은 1D 텐서.
        """
        pass