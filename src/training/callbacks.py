"""Training callbacks for structured progress reporting."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from src.utils.logging_utils import log_event

if TYPE_CHECKING:
    from src.training.trainer import TrainingResult


class TrainingCallback(Protocol):
    """Observer interface for completed training epochs and runs."""

    def on_epoch_end(self, metrics: Mapping[str, Any]) -> None: ...

    def on_train_end(self, result: TrainingResult) -> None: ...


class LoggingCallback:
    """Emit epoch and run summaries through the project's structured logger."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def on_epoch_end(self, metrics: Mapping[str, Any]) -> None:
        log_event(self.logger, "training_epoch", **dict(metrics))

    def on_train_end(self, result: TrainingResult) -> None:
        log_event(
            self.logger,
            "training_complete",
            global_step=result.global_step,
            epochs_completed=result.epochs_completed,
            best_validation_exact_match=result.best_validation_exact_match,
            stop_reason=result.stop_reason,
            best_checkpoint=result.best_checkpoint,
            latest_checkpoint=result.latest_checkpoint,
        )
