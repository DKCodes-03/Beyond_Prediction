"""Atomic SELR model and optimizer checkpoint persistence."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn


class CheckpointableModel(Protocol):
    """Model interface needed by checkpoint serialization."""

    def model_metadata(self) -> dict[str, Any]: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = ...) -> Any: ...


def save_checkpoint(
    path: str | Path,
    *,
    model: CheckpointableModel,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    seed: int,
    epoch: int,
    global_step: int,
    best_validation_exact_match: float,
    stale_epochs: int,
) -> Path:
    """Atomically persist model, optimizer, RNG, config, seed, and progress."""
    destination = Path(path)
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    _validate_seed(seed)
    for name, value in (
        ("epoch", epoch),
        ("global_step", global_step),
        ("stale_epochs", stale_epochs),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not math.isfinite(best_validation_exact_match) or not (
        0 <= best_validation_exact_match <= 1
    ):
        raise ValueError("best_validation_exact_match must be between 0 and 1")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = {
        "format_version": 1,
        "model_metadata": model.model_metadata(),
        "model_state_dict": model.state_dict(),
        "optimizer_metadata": {"class": type(optimizer).__name__},
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dict(config),
        "seed": seed,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_exact_match": best_validation_exact_match,
        "stale_epochs": stale_epochs,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: CheckpointableModel,
    optimizer: torch.optim.Optimizer,
    expected_config: Mapping[str, Any],
    expected_seed: int,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Restore a checkpoint, rejecting incompatible model/config/seed metadata."""
    _validate_seed(expected_seed)
    checkpoint = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format_version")
    if checkpoint.get("model_metadata") != model.model_metadata():
        raise ValueError("checkpoint model metadata does not match the model")
    if checkpoint.get("optimizer_metadata") != {
        "class": type(optimizer).__name__
    }:
        raise ValueError("checkpoint optimizer metadata does not match the optimizer")
    if checkpoint.get("config") != dict(expected_config):
        raise ValueError("checkpoint config does not match the requested config")
    if checkpoint.get("seed") != expected_seed:
        raise ValueError("checkpoint seed does not match the requested seed")
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "global_step",
        "best_validation_exact_match",
        "stale_epochs",
        "rng_state",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)!r}")
    rng_state = checkpoint["rng_state"]
    if not isinstance(rng_state, Mapping) or not {
        "python",
        "numpy",
        "torch",
        "cuda",
    }.issubset(rng_state):
        raise ValueError("checkpoint rng_state is incomplete")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    if rng_state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in rng_state["cuda"]]
        )
    return {
        key: checkpoint[key]
        for key in ("epoch", "global_step", "best_validation_exact_match", "stale_epochs")
    }


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
