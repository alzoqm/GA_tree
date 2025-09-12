import os
from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd


@dataclass
class EventRecord:
    generation: int
    chromosome: int
    step_index: int
    timestamp: str
    action_node_id: int
    action_type: int
    param2: float
    param3: float
    param4: float
    applied: bool
    pos_before: int
    pos_after: int
    lev_before: int
    er_before: float
    add_count_before: int
    trade_id: int
    trade_seq: int


@dataclass
class WindowRecord:
    generation: int
    chromosome: int
    trade_id: int
    start_step: int
    start_time: str
    end_step: int
    end_time: str


class EventLogger:
    """
    Buffers action events and trade windows, and writes them to disk (CSV) at the end.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._events: List[EventRecord] = []
        self._windows: List[WindowRecord] = []
        # Batch DataFrame buffers for vectorized logging
        self._events_df: List[pd.DataFrame] = []
        self._windows_df: List[pd.DataFrame] = []

    def log_event(self, rec: EventRecord):
        self._events.append(rec)

    def log_window(self, rec: WindowRecord):
        self._windows.append(rec)

    def flush(self):
        # Events
        dfs_events = []
        if self._events:
            dfs_events.append(pd.DataFrame([asdict(e) for e in self._events]))
        if self._events_df:
            dfs_events.extend(self._events_df)
        if dfs_events:
            df = pd.concat(dfs_events, ignore_index=True)
            path = os.path.join(self.output_dir, "events.csv")
            if os.path.exists(path):
                df0 = pd.read_csv(path)
                df = pd.concat([df0, df], ignore_index=True)
            df.to_csv(path, index=False)
        # Windows
        dfs_windows = []
        if self._windows:
            dfs_windows.append(pd.DataFrame([asdict(w) for w in self._windows]))
        if self._windows_df:
            dfs_windows.extend(self._windows_df)
        if dfs_windows:
            dfw = pd.concat(dfs_windows, ignore_index=True)
            pathw = os.path.join(self.output_dir, "windows.csv")
            if os.path.exists(pathw):
                df0 = pd.read_csv(pathw)
                dfw = pd.concat([df0, dfw], ignore_index=True)
            dfw.to_csv(pathw, index=False)
        # Clear buffers after flush
        self._events.clear()
        self._windows.clear()
        self._events_df.clear()
        self._windows_df.clear()

    # Vectorized batch logging methods
    def log_events_df(self, df: pd.DataFrame):
        if df is not None and not df.empty:
            self._events_df.append(df)

    def log_windows_df(self, df: pd.DataFrame):
        if df is not None and not df.empty:
            self._windows_df.append(df)
