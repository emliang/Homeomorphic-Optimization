"""Random seed helpers."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
