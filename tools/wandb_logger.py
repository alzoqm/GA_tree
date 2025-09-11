from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


class _NoopWandb:
    def __init__(self):
        self.enabled = False
        self.run = None
        self._wandb = None

    def init(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        pass

    def Table(self, *args, **kwargs):
        return None

    def Image(self, *args, **kwargs):
        return None

    def Artifact(self, *args, **kwargs):
        class _NoopArtifact:
            def add_file(self, *a, **k):
                pass
        return _NoopArtifact()

    def save(self, *args, **kwargs):
        pass


_state = {
    "enabled": False,
    "wandb": _NoopWandb(),
}


def _import_wandb():
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return _NoopWandb()


def init_run(cfg: Dict[str, Any], run_name: str, out_dir: str, full_config: Optional[Dict[str, Any]] = None, resume_run_id: Optional[str] = None) -> None:
    """Initialize wandb if enabled, loading API key from .env.

    cfg keys (optional): enabled, project, entity, group, job_type, tags, dir, mode
    resume_run_id: If provided, resume an existing run with this ID
    """
    enabled = bool(cfg.get("enabled", False))
    _state["enabled"] = enabled
    if not enabled:
        _state["wandb"] = _NoopWandb()
        return

    # Load .env and set API key if present
    try:
        load_dotenv()
        api_key = os.getenv("WANDB_API_KEY", None)
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
    except Exception:
        pass

    # Respect mode
    mode = cfg.get("mode", "offline")
    os.environ.setdefault("WANDB_MODE", mode)

    wandb = _import_wandb()
    _state["wandb"] = wandb

    # If wandb import failed, wandb is _NoopWandb and will no-op
    try:
        dir_path = cfg.get("dir", out_dir)
        os.makedirs(dir_path, exist_ok=True)
    except Exception:
        dir_path = out_dir

    run = None
    try:
        # Prepare init arguments
        init_kwargs = {
            "project": cfg.get("project", "GA_tree"),
            "entity": cfg.get("entity", None),
            "group": cfg.get("group", None),
            "job_type": cfg.get("job_type", "train"),
            "tags": cfg.get("tags", []),
            "name": run_name,
            "dir": dir_path,
            "config": full_config or {},
            "reinit": True,
        }
        
        # Handle resumption
        if resume_run_id:
            try:
                init_kwargs["id"] = resume_run_id
                init_kwargs["resume"] = "must"
                
                logging.info(f"Resuming wandb run: {resume_run_id}")
            except Exception as e:
                logging.warning(f"Failed to resume wandb run {resume_run_id}, starting new run: {e}")
                # Fall back to new run without resume parameters
                init_kwargs.pop("id", None)
                init_kwargs.pop("resume", None)
        
        run = wandb.init(**init_kwargs)
        # Define custom step metrics so plots use logical x-axes
        try:
            # Generation-level metrics keyed by gen/index
            wandb.define_metric("gen/*", step_metric="gen/index")
            # Window-level metrics (IS/OOS/tables) keyed by window/index
            wandb.define_metric("is/*", step_metric="window/index")
            wandb.define_metric("oos/*", step_metric="window/index")
            wandb.define_metric("tables/*", step_metric="window/index")
            # Final aggregates keyed by final/windows when present
            wandb.define_metric("final/*", step_metric="final/windows")
        except Exception:
            pass
    except Exception:
        # Fallback to no-op if init fails
        _state["wandb"] = _NoopWandb()
        _state["enabled"] = False
        return

    _state["run"] = run


def log(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if not _state.get("enabled", False):
        return
    try:
        # Avoid passing explicit step to keep global step monotonic and
        # rely on define_metric(..., step_metric=...) for x-axis control.
        _state["wandb"].log(metrics)
    except Exception:
        pass


def log_table(name: str, columns: List[str], rows: List[List[Any]], step: Optional[int] = None) -> None:
    if not _state.get("enabled", False):
        return
    try:
        Table = _state["wandb"].Table
        if not Table:
            return
        table = Table(columns=columns)
        for r in rows:
            table.add_data(*r)
        # Do not pass explicit step; charts can bind to custom step metrics
        _state["wandb"].log({name: table})
    except Exception:
        pass


def log_image(name: str, path: str, step: Optional[int] = None) -> None:
    if not _state.get("enabled", False):
        return
    try:
        Image = _state["wandb"].Image
        if not Image:
            return
        img = Image(path)
        _state["wandb"].log({name: img})
    except Exception:
        pass


def log_artifacts(files: List[str], name: str, type_: str = "dataset") -> None:
    if not _state.get("enabled", False):
        return
    try:
        wandb = _state["wandb"]
        art = wandb.Artifact(name=name, type=type_)
        for f in files:
            if f and os.path.exists(f):
                art.add_file(f)
        if _state.get("run") is not None:
            _state["run"].log_artifact(art)
    except Exception:
        pass


def get_run_id() -> Optional[str]:
    """Get the current wandb run ID for checkpoint saving."""
    if not _state.get("enabled", False):
        return None
    try:
        run = _state.get("run")
        if run and hasattr(run, "id"):
            return run.id
        return None
    except Exception:
        return None


def is_run_active() -> bool:
    """Check if there's an active wandb run."""
    if not _state.get("enabled", False):
        return False
    try:
        run = _state.get("run")
        return run is not None
    except Exception:
        return False
