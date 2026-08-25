"""Deterministic seeding across python/numpy/torch."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CTC/attention joint training is numerically sensitive; trade a little
    # speed for determinism so overfit/resume tests are meaningful.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
