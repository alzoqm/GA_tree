# /evolution/Selection/base.py
import torch
from abc import ABC, abstractmethod

class BaseSelection(ABC):
    """
    다중 목표 유전 알고리즘의 '선택' 연산을 위한 추상 기반 클래스.
    모든 선택 연산자 클래스는 이 클래스를 상속받아야 합니다.
    다중 목표 최적화를 지원합니다.
    """

    @abstractmethod
    def select_elites(self, objectives: torch.Tensor, num_selects: int) -> torch.Tensor:
        """
        주어진 목표값들을 기반으로 상위 개체들의 인덱스를 선택합니다.
        이 메소드는 중복되지 않는 고유한 인덱스 집합을 반환해야 합니다.
        (엘리트 보존 및 교배 풀 구성에 사용됩니다.)

        Args:
            objectives (torch.Tensor): 집단 내 각 개체의 목표값들을 담은 텐서.
                                     [N] 또는 [N, M] 형태 (N=개체수, M=목표수)
            num_selects (int): 선택할 상위 개체의 수.

        Returns:
            torch.Tensor: 선택된 상위 개체들의 인덱스를 담은 1D 텐서.
        """
        pass

    @abstractmethod
    def pick_parents(self, objectives: torch.Tensor, num_parents: int) -> torch.Tensor:
        """
        주어진 목표값들을 기반으로 교차에 사용할 부모 개체들의 인덱스를 선택합니다.
        이 메소드는 중복 선택(selection with replacement)을 허용할 수 있습니다.

        Args:
            objectives (torch.Tensor): 집단 내 각 개체의 목표값들을 담은 텐서.
                                      [N] 또는 [N, M] 형태 (N=개체수, M=목표수)
            num_parents (int): 선택할 부모 개체의 수.

        Returns:
            torch.Tensor: 선택된 부모 개체들의 인덱스를 담은 1D 텐서.
        """
        pass