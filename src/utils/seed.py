"""Reproducible random-seed setup for Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a required project dependency
    torch = None


def set_seed(seed: int, deterministic: bool = True, num_threads: int | None = None) -> int:
    """Seed supported runtimes and optionally request deterministic execution."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if num_threads is not None:
        if not isinstance(num_threads, int) or isinstance(num_threads, bool) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if num_threads is not None:
            torch.set_num_threads(num_threads)

        if deterministic:
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        elif hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)

    return seed


def seed_everything(seed: int, deterministic: bool = True, num_threads: int | None = None) -> int:
    """Backward-friendly descriptive alias for set_seed."""
    return set_seed(seed, deterministic=deterministic, num_threads=num_threads)


def seed_worker(worker_id: int) -> None:
    """Initialize a NumPy seed for one PyTorch DataLoader worker."""
    del worker_id
    base_seed = torch.initial_seed() if torch is not None else None
    if base_seed is None:
        return
    np.random.seed(base_seed % (2**32))
