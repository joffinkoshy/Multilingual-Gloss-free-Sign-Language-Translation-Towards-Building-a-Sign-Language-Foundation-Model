"""Checkpoint save/resume (Section 2.6): model + optimizer + scheduler state
every epoch, resume-from-checkpoint built in from day one."""
import os
from typing import Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    best_score: float,
    rng_state: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "step": step,
            "best_score": best_score,
            "torch_rng_state": torch.get_rng_state(),
        },
        path,
    )


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None) -> dict:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    if optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    return state
