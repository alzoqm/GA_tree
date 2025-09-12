from __future__ import annotations
import math
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch

from training.trading_env import TradingEnvironment
from models.constants import ACTION_NOT_FOUND


class ShapleyReplayer:
    """
    Deterministic trade-window replayer to compute Causal Shapley attributions
    for executed action events.
    """

    def __init__(self, env_config: dict, device: str = 'cuda'):
        self.env_config = env_config
        self.device = device

    def _run_window(self, data_slice: pd.DataFrame, step_indices: List[int], event_steps: List[int], event_actions: List[Tuple[int,float,float,float]], include_mask: np.ndarray) -> float:
        """
        Run a single replay for one subset include_mask over the window and return realized_cum reward.
        """
        B = 1
        env = TradingEnvironment(chromosomes_size=B, env_config=self.env_config, device=self.device)
        # Iterate over steps, injecting actions when event present and included in mask
        for s in step_indices:
            row = data_slice.loc[s]
            # next open price: use next index of full data, but data_slice index is original integer index
            # So we fetch Open at s+1 from the parent DataFrame if available in slice
            try:
                next_open_price = torch.as_tensor(float(data_slice.loc[s+1]['Open']), dtype=torch.float32, device=self.device)
            except Exception:
                # Fallback: reuse close as next open proxy (should not happen in proper slices)
                next_open_price = torch.as_tensor(float(row['Close']), dtype=torch.float32, device=self.device)

            # Build one-row actions tensor
            act = torch.zeros((B, 5), dtype=torch.float32, device=self.device)
            act[:, 0] = float(ACTION_NOT_FOUND)
            # Check if there is an event at this step
            if s in event_steps:
                idx = event_steps.index(s)
                if include_mask[idx]:
                    at, p2, p3, p4 = event_actions[idx]
                    act[0, 0] = float(at)
                    act[0, 1] = float(p2)
                    act[0, 2] = float(p3)
                    act[0, 3] = float(p4)
                    # node id is optional, not needed for replay
            env.step(row, next_open_price, act)

        # Use realized cumulative profit as reward
        return float(env.realized_cum.detach().to('cpu').numpy()[0])

    def shapley_exact(self, data: pd.DataFrame, start_step: int, end_step: int,
                      events: List[Dict]) -> List[float]:
        """
        Exact Shapley for small N via 2^N subsets.
        events: list of dicts with keys: step_index, action_type, param2, param3, param4
        """
        N = len(events)
        if N == 0:
            return []
        step_indices = list(range(start_step, end_step + 1))
        event_steps = [int(e['step_index']) for e in events]
        event_actions = [(int(e['action_type']), float(e['param2']), float(e['param3']), float(e['param4'])) for e in events]

        # Precompute rewards for all subsets by bitmask
        num_subsets = 1 << N
        rewards = np.zeros((num_subsets,), dtype=np.float64)
        for mask in range(num_subsets):
            inc = np.array([(mask >> i) & 1 for i in range(N)], dtype=bool)
            r = self._run_window(data, step_indices, event_steps, event_actions, inc)
            rewards[mask] = r

        # Shapley aggregation
        fact = math.factorial
        denom = fact(N)
        phi = np.zeros((N,), dtype=np.float64)
        for i in range(N):
            contrib = 0.0
            for mask in range(num_subsets):
                if (mask >> i) & 1:
                    without = mask & (~(1 << i))
                    k = bin(without).count('1')
                    w = fact(k) * fact(N - k - 1) / denom
                    contrib += w * (rewards[mask] - rewards[without])
            phi[i] = contrib
        return phi.tolist()

    def shapley_permutation(self, data: pd.DataFrame, start_step: int, end_step: int,
                             events: List[Dict], P: int = 256, seed: int = 1234) -> List[float]:
        """
        Approximate Shapley via permutation sampling.
        """
        N = len(events)
        if N == 0:
            return []
        rng = np.random.RandomState(seed)
        step_indices = list(range(start_step, end_step + 1))
        event_steps = [int(e['step_index']) for e in events]
        event_actions = [(int(e['action_type']), float(e['param2']), float(e['param3']), float(e['param4'])) for e in events]
        phi = np.zeros((N,), dtype=np.float64)

        for _ in range(P):
            perm = rng.permutation(N)
            inc = np.zeros((N,), dtype=bool)
            prev_reward = self._run_window(data, step_indices, event_steps, event_actions, inc)
            for idx in perm:
                inc[idx] = True
                new_reward = self._run_window(data, step_indices, event_steps, event_actions, inc)
                phi[idx] += (new_reward - prev_reward)
                prev_reward = new_reward
        phi /= float(P)
        return phi.tolist()


def compute_shapley_for_all(data: pd.DataFrame, env_config: dict, events_df: pd.DataFrame, windows_df: pd.DataFrame,
                            device: str = 'cuda') -> pd.DataFrame:
    """
    Compute Shapley per event and aggregate to a flat table with event_id/grouping.
    Returns DataFrame with columns: [generation, chromosome, trade_id, step_index, action_node_id, shapley]
    """
    replayer = ShapleyReplayer(env_config=env_config, device=device)
    records: List[Dict] = []
    # Group by window (generation, chromosome, trade_id)
    grouped = windows_df.groupby(['generation', 'chromosome', 'trade_id'])
    for (gen, chrom, tid), w in grouped:
        start_step = int(w['start_step'].iloc[0])
        end_step = int(w['end_step'].iloc[0])
        evts = events_df[(events_df['generation'] == gen) & (events_df['chromosome'] == chrom) & (events_df['trade_id'] == tid)]
        evts = evts.sort_values('step_index')
        events = evts.to_dict('records')
        N = len(events)
        if N == 0:
            continue
        if N <= 12:
            phi = replayer.shapley_exact(data, start_step, end_step, events)
        else:
            phi = replayer.shapley_permutation(data, start_step, end_step, events, P=256)
        for e, sh in zip(events, phi):
            records.append({
                'generation': gen,
                'chromosome': chrom,
                'trade_id': tid,
                'step_index': int(e['step_index']),
                'action_node_id': int(e.get('action_node_id', -1)),
                'action_type': int(e.get('action_type', 0)),
                'shapley': float(sh),
            })
    return pd.DataFrame(records)


def aggregate_node_contributions(shapley_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Shapley contributions per action_node_id.
    Returns DataFrame with columns: node_id, count, contrib_sum, contrib_mean, contrib_std, positive_rate
    """
    if shapley_df.empty:
        return pd.DataFrame(columns=['action_node_id', 'count', 'contrib_sum', 'contrib_mean', 'contrib_std', 'positive_rate'])
    grp = shapley_df.groupby('action_node_id')
    agg = grp['shapley'].agg(['count', 'sum', 'mean', 'std'])
    agg = agg.rename(columns={'sum': 'contrib_sum', 'mean': 'contrib_mean', 'std': 'contrib_std'})
    pos = grp['shapley'].apply(lambda s: float((s > 0).mean()))
    agg['positive_rate'] = pos
    agg = agg.reset_index()
    return agg

