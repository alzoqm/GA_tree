# evolution/Crossover/chain.py
import torch
from typing import List
from .base import BaseCrossover

class ChainCrossover(BaseCrossover):
    """
    여러 교차 연산자들을 가중치에 따라 조합하여 자손을 생성하는 분배자 클래스.
    """
    def __init__(self, crossovers: List[BaseCrossover], weights: List[float]):
        """
        Args:
            crossovers (List[BaseCrossover]): 사용할 BaseCrossover 객체의 리스트.
            weights (List[float]): 각 교차 연산자에 할당될 가중치. 합은 1.0이어야 함.
        """
        super().__init__(rate=1.0) # 체인 자체는 항상 실행됨
        if len(crossovers) != len(weights):
            raise ValueError("Length of crossovers and weights must be the same.")
        if not torch.isclose(torch.tensor(weights).sum(), torch.tensor(1.0)):
            raise ValueError("Sum of weights must be 1.0.")

        self.crossovers = crossovers
        self.weights = weights

    def __call__(self, parents: torch.Tensor) -> torch.Tensor:
        """
        가중치에 따라 부모 풀을 분할하고, 각 교차 연산자를 적용한 후 결과를 합칩니다.
        """
        num_parents = parents.shape[0]
        if num_parents % 2 != 0:
            raise ValueError("Number of parents must be even.")
        
        num_offspring = num_parents // 2
        
        # 1. 가중치에 따라 각 연산자가 생성할 자손 수 계산
        counts = [int(w * num_offspring) for w in self.weights]
        # 반올림 오차 보정
        remainder = num_offspring - sum(counts)
        counts[-1] += remainder

        # 2. 부모 풀을 섞어서 무작위성 확보
        shuffled_indices = torch.randperm(num_parents)
        shuffled_parents = parents[shuffled_indices]
        
        all_children = []
        current_parent_idx = 0
        
        # 3. 각 연산자에 부모 풀을 할당하고 자손 생성
        for op, count in zip(self.crossovers, counts):
            if count == 0:
                continue
            
            num_op_parents = count * 2
            parent_slice = shuffled_parents[current_parent_idx : current_parent_idx + num_op_parents]
            
            if parent_slice.shape[0] > 0:
                offspring_slice = op(parent_slice)
                all_children.append(offspring_slice)
            
            current_parent_idx += num_op_parents
            
        # 4. 생성된 모든 자손을 하나로 합침
        if not all_children:
            # 모든 카운트가 0인 극단적 경우, 부모 복사
            return parents[:num_offspring].clone()
        
        final_children = torch.cat(all_children, dim=0)

        # 5. 최종 자손 리스트를 섞어서 특정 연산자의 결과가 몰리는 것을 방지
        final_shuffled_indices = torch.randperm(final_children.shape[0])
        return final_children[final_shuffled_indices]