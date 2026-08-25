"""Optimizer/gradient-clipper/scheduler builders, config-driven per Section 2.3."""
from typing import Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


class ConstantLR(_LRScheduler):
    """No-op scheduler for scheduler.name=constant — the paper-literal
    reading (fixed LR=1e-3, no schedule, Section 2.3)."""

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def build_optimizer(config: dict, parameters) -> Optimizer:
    opt_cfg = config["optimizer"]
    name = opt_cfg["name"].lower()
    if name != "adam":
        raise ValueError(f"Only Adam is specified by the paper; got '{name}'")
    return torch.optim.Adam(
        parameters,
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    )


def build_gradient_clipper(config: dict):
    max_norm = config["training"]["grad_clip_norm"]
    if not max_norm:
        return None
    return lambda params: torch.nn.utils.clip_grad_norm_(params, max_norm)


def build_scheduler(config: dict, optimizer: Optimizer) -> Tuple[Optional[_LRScheduler], str]:
    """Returns (scheduler, step_at) where step_at is 'validation' or 'epoch'."""
    sched_cfg = config["scheduler"]
    name = sched_cfg["name"].lower()
    if name == "constant":
        return ConstantLR(optimizer), "epoch"
    if name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",  # tracks dev BLEU, higher is better
            factor=sched_cfg.get("decrease_factor", 0.1),
            patience=sched_cfg.get("patience", 10),
            threshold_mode="abs",
        )
        return scheduler, "validation"
    raise ValueError(f"Unknown scheduler '{name}'")
