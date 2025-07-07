# evolution/Selection/tournament.py
import torch
from .base import BaseSelection

class TournamentSelection(BaseSelection):
    """
    토너먼트 선택 방식 구현 클래스.
    """
    def __init__(self, k: int = 5):
        """
        토너먼트 선택 초기화.

        Args:
            k (int): 각 토너먼트의 크기.
        """
        if k < 1:
            raise ValueError("Tournament size k must be at least 1.")
        self.k = k

    def select_elites(self, fitness: torch.Tensor, num_elites: int) -> torch.Tensor:
        """
        가장 높은 적합도를 가진 개체를 엘리트로 선택합니다.
        """
        # 적합도를 기준으로 내림차순 정렬하여 상위 인덱스를 반환
        elite_indices = torch.argsort(fitness, descending=True)[:num_elites]
        return elite_indices

    def pick_parents(self, fitness: torch.Tensor, num_parents: int) -> torch.Tensor:
        """
        토너먼트 방식을 사용하여 부모를 선택합니다.
        """
        population_size = len(fitness)
        parent_indices = torch.zeros(num_parents, dtype=torch.long)
        
        for i in range(num_parents):
            # 무작위로 k개의 참가자 인덱스를 뽑음
            tournament_contender_indices = torch.randint(0, population_size, (self.k,))
            # 참가자들의 적합도를 가져옴
            tournament_fitness = fitness[tournament_contender_indices]
            # 토너먼트에서 승자(가장 높은 적합도를 가진 개체)의 상대적 인덱스를 찾음
            winner_relative_idx = torch.argmax(tournament_fitness)
            # 실제 인덱스를 저장
            parent_indices[i] = tournament_contender_indices[winner_relative_idx]
            
        return parent_indices