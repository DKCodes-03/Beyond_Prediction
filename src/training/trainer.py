"""SELR answer/explanation training, validation, and resumable checkpoints."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.llm.answer_decoder import CandidateAnswerDecoder
from src.selr.bottleneck import LatentBottleneck
from src.selr.explanation_decoder import ExplanationDecoder
from src.selr.latent_reasoner import LatentReasoner
from src.selr.losses import answer_loss, explanation_loss
from src.selr.reasoning_head import ReasoningHead
from src.training.callbacks import LoggingCallback, TrainingCallback
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


_BATCH_FIELDS = (
    "encoder_memory",
    "encoder_memory_mask",
    "candidate_embeddings",
    "candidate_mask",
    "answer_targets",
)


class SELRAnswerModel(nn.Module):
    """Compose SELR answer modules and an optional latent-conditioned decoder."""

    def __init__(
        self,
        encoder_dim: int,
        latent_dim: int,
        max_steps: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder_dim = _positive_int(encoder_dim, "encoder_dim")
        self.latent_dim = _positive_int(latent_dim, "latent_dim")
        self.max_steps = _positive_int(max_steps, "max_steps")
        if hidden_dim is not None:
            _positive_int(hidden_dim, "hidden_dim")
        self.bottleneck = LatentBottleneck(
            self.encoder_dim, self.latent_dim, self.max_steps
        )
        self.reasoner = LatentReasoner(self.latent_dim, self.max_steps)
        self.answer_head = ReasoningHead(
            self.latent_dim, self.encoder_dim, hidden_dim
        )
        self.explanation_decoder: ExplanationDecoder | None = None
        self._explanation_metadata: dict[str, int] | None = None

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Return answer logits aligned with the candidate axis in ``batch``."""
        inputs = _batch_tensors(batch, device=None)
        reasoned_states, reasoned_mask = self._reason(inputs)
        return self.answer_head(
            reasoned_states,
            reasoned_mask,
            inputs["candidate_embeddings"],
            inputs["candidate_mask"],
        )

    def enable_explanation_decoder(
        self,
        *,
        vocab_size: int,
        max_decode_steps: int,
        pad_token_id: int,
    ) -> None:
        """Attach the bounded decoder for the post-answer explanation stage."""
        if self.explanation_decoder is not None:
            raise ValueError("the explanation decoder is already enabled")
        self.explanation_decoder = ExplanationDecoder(
            latent_dim=self.latent_dim,
            vocab_size=vocab_size,
            max_decode_steps=max_decode_steps,
            pad_token_id=pad_token_id,
        )
        self._explanation_metadata = {
            "vocab_size": vocab_size,
            "max_decode_steps": max_decode_steps,
            "pad_token_id": pad_token_id,
        }

    def forward_with_explanations(
        self, batch: Mapping[str, Tensor], decoder_input_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return answer logits and teacher-forced explanation token logits."""
        if self.explanation_decoder is None:
            raise ValueError("the explanation decoder is not enabled")
        inputs = _batch_tensors(batch, device=None)
        reasoned_states, reasoned_mask = self._reason(inputs)
        answer_logits = self.answer_head(
            reasoned_states,
            reasoned_mask,
            inputs["candidate_embeddings"],
            inputs["candidate_mask"],
        )
        token_logits = self.explanation_decoder(
            reasoned_states, reasoned_mask, decoder_input_ids
        )
        return answer_logits, token_logits

    def _reason(self, inputs: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        latent_states, latent_mask = self.bottleneck(
            inputs["encoder_memory"], inputs["encoder_memory_mask"]
        )
        return self.reasoner(latent_states, latent_mask)

    def model_metadata(self) -> dict[str, Any]:
        """Return the architecture metadata required to validate checkpoints."""
        metadata = {
            "class": type(self).__name__,
            "encoder_dim": self.encoder_dim,
            "latent_dim": self.latent_dim,
            "max_steps": self.max_steps,
            "hidden_dim": self.answer_head.hidden_dim,
        }
        if self._explanation_metadata is not None:
            metadata["explanation_decoder"] = dict(self._explanation_metadata)
        return metadata


class BaselineAnswerModel(nn.Module):
    """Batched candidate decoder using the frozen encoder context directly."""

    def __init__(self, encoder_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.encoder_dim = _positive_int(encoder_dim, "encoder_dim")
        self.hidden_dim = encoder_dim if hidden_dim is None else hidden_dim
        self.decoder = CandidateAnswerDecoder(encoder_dim, hidden_dim)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        inputs = _batch_tensors(batch, device=None)
        context = inputs.get("context_embeddings")
        if context is None:
            raise ValueError("baseline batches must contain context_embeddings")
        return self.decoder.score_embeddings(context, inputs["candidate_embeddings"])

    def model_metadata(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "encoder_dim": self.encoder_dim,
            "hidden_dim": self.hidden_dim,
        }


@dataclass(frozen=True)
class TrainingResult:
    """Metrics and checkpoint locations from one training invocation."""

    global_step: int
    epochs_completed: int
    best_validation_exact_match: float
    stop_reason: str
    best_checkpoint: str
    latest_checkpoint: str
    history: tuple[dict[str, Any], ...]


def validate(
    model: SELRAnswerModel | BaselineAnswerModel,
    batches: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute candidate-weighted BCE and set exact match without gradients."""
    resolved_device = torch.device(device)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number between 0 and 1")
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    candidate_count = 0
    example_count = 0
    exact_matches = 0
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(batches):
                inputs = _batch_tensors(batch, device=resolved_device)
                logits = model(inputs)
                loss = answer_loss(
                    logits, inputs["answer_targets"], inputs["candidate_mask"]
                )
                valid_count = int(inputs["candidate_mask"].sum().item())
                batch_size = logits.shape[0]
                loss_sum += float(loss.item()) * valid_count
                candidate_count += valid_count
                example_count += batch_size
                predictions = torch.sigmoid(logits) >= threshold
                predictions &= inputs["candidate_mask"]
                empty_rows = ~predictions.any(dim=1)
                if empty_rows.any():
                    top_candidates = logits.masked_fill(
                        ~inputs["candidate_mask"], torch.finfo(logits.dtype).min
                    ).argmax(dim=1)
                    predictions[
                        torch.arange(batch_size, device=logits.device)[empty_rows],
                        top_candidates[empty_rows],
                    ] = True
                gold = (inputs["answer_targets"] == 1) & inputs["candidate_mask"]
                exact_matches += int((predictions == gold).all(dim=1).sum().item())
    finally:
        model.train(was_training)

    if example_count == 0 or candidate_count == 0:
        raise ValueError("validation batches must contain examples and candidates")
    return {
        "loss": loss_sum / candidate_count,
        "exact_match_accuracy": exact_matches / example_count,
        "example_count": example_count,
        "candidate_count": candidate_count,
    }


def train_selr(
    model: SELRAnswerModel,
    train_batches: Iterable[Mapping[str, Tensor]],
    validation_batches: Iterable[Mapping[str, Tensor]],
    *,
    config: Mapping[str, Any],
    seed: int,
    resume_from: str | Path | None = None,
    logger: logging.Logger | None = None,
    callbacks: Sequence[TrainingCallback] | None = None,
    runtime_started_at: float | None = None,
) -> TrainingResult:
    """Train the answer-only SELR stage with the shared training loop."""
    if config.get("objective") != "answer_only":
        raise ValueError(
            "train_selr supports only 'answer_only'; use "
            "train_selr_explanations for stage two"
        )
    return _train_answer_model(
        model,
        train_batches,
        validation_batches,
        config=config,
        seed=seed,
        resume_from=resume_from,
        logger=logger,
        callbacks=callbacks,
        model_variant="selr",
        runtime_started_at=runtime_started_at,
    )


def train_selr_explanations(
    model: SELRAnswerModel,
    train_batches: Iterable[Mapping[str, Tensor]],
    validation_batches: Iterable[Mapping[str, Tensor]],
    *,
    answer_checkpoint: str | Path,
    answer_config: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
    resume_from: str | Path | None = None,
    logger: logging.Logger | None = None,
    callbacks: Sequence[TrainingCallback] | None = None,
    runtime_started_at: float | None = None,
) -> TrainingResult:
    """Restore an answer-only checkpoint, then train explanations jointly.

    The explanation stage retains answer loss and selects checkpoints using
    validation answer exact match. Explanation targets are tokenized using a
    vocabulary built from permitted training data and supplied in each batch.
    """
    _validate_training_config(config, seed, "selr")
    if config["objective"] != "answer_plus_explanation":
        raise ValueError(
            "explanation training config objective must be "
            "'answer_plus_explanation'"
        )
    _validate_training_config(answer_config, seed, "selr")
    if answer_config["objective"] != "answer_only":
        raise ValueError("answer_config objective must be 'answer_only'")
    if model.explanation_decoder is not None:
        raise ValueError("model must not have an explanation decoder before stage two")

    answer_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=answer_config["optimizer"]["learning_rate"],
        weight_decay=answer_config["optimizer"]["weight_decay"],
    )
    answer_device = _resolve_device(answer_config["device"])
    model.to(answer_device)
    load_checkpoint(
        answer_checkpoint,
        model=model,
        optimizer=answer_optimizer,
        expected_config=answer_config,
        expected_seed=seed,
        map_location=answer_device,
    )

    explanation = config["explanation"]
    model.enable_explanation_decoder(
        vocab_size=explanation["vocab_size"],
        max_decode_steps=explanation["max_decode_steps"],
        pad_token_id=explanation["pad_token_id"],
    )
    return _train_answer_model(
        model,
        train_batches,
        validation_batches,
        config=config,
        seed=seed,
        resume_from=resume_from,
        logger=logger,
        callbacks=callbacks,
        model_variant="selr",
        runtime_started_at=runtime_started_at,
    )


def train_baseline(
    model: BaselineAnswerModel,
    train_batches: Iterable[Mapping[str, Tensor]],
    validation_batches: Iterable[Mapping[str, Tensor]],
    *,
    config: Mapping[str, Any],
    seed: int,
    resume_from: str | Path | None = None,
    logger: logging.Logger | None = None,
    callbacks: Sequence[TrainingCallback] | None = None,
    runtime_started_at: float | None = None,
) -> TrainingResult:
    """Train the non-SELR candidate decoder under the same budgets."""
    return _train_answer_model(
        model,
        train_batches,
        validation_batches,
        config=config,
        seed=seed,
        resume_from=resume_from,
        logger=logger,
        callbacks=callbacks,
        model_variant="baseline",
        runtime_started_at=runtime_started_at,
    )


def _train_answer_model(
    model: SELRAnswerModel | BaselineAnswerModel,
    train_batches: Iterable[Mapping[str, Tensor]],
    validation_batches: Iterable[Mapping[str, Tensor]],
    *,
    config: Mapping[str, Any],
    seed: int,
    resume_from: str | Path | None,
    logger: logging.Logger | None,
    callbacks: Sequence[TrainingCallback] | None,
    model_variant: str,
    runtime_started_at: float | None,
) -> TrainingResult:
    """Fit an answer model with validation-only selection and explicit limits.

    Batches must provide ``encoder_memory``, ``encoder_memory_mask``,
    ``candidate_embeddings``, ``candidate_mask``, and ``answer_targets``.
    Iterables must be reusable for every epoch.
    """
    _validate_training_config(config, seed, model_variant)
    _validate_reusable(train_batches, "train_batches")
    _validate_reusable(validation_batches, "validation_batches")
    set_seed(
        seed,
        deterministic=bool(config["deterministic"]),
        num_threads=config["num_threads"],
    )
    device = _resolve_device(config["device"])
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    checkpoint_dir = Path(config["checkpoint_dir"])
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    start_epoch = 1
    global_step = 0
    best_metric = -1.0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if resume_from is not None:
        restored = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            expected_config=config,
            expected_seed=seed,
            map_location=device,
        )
        start_epoch = restored["epoch"] + 1
        global_step = restored["global_step"]
        best_metric = restored["best_validation_exact_match"]
        stale_epochs = restored["stale_epochs"]

    run_logger = logger or get_logger("beyond_prediction.training")
    run_callbacks = (
        (LoggingCallback(run_logger),) if callbacks is None else tuple(callbacks)
    )
    for callback_index, callback in enumerate(run_callbacks):
        if not callable(getattr(callback, "on_epoch_end", None)) or not callable(
            getattr(callback, "on_train_end", None)
        ):
            raise ValueError(
                f"callback {callback_index} must implement on_epoch_end and on_train_end"
            )
    if runtime_started_at is not None and (
        isinstance(runtime_started_at, bool)
        or not isinstance(runtime_started_at, (int, float))
        or not math.isfinite(runtime_started_at)
        or runtime_started_at > time.monotonic()
    ):
        raise ValueError("runtime_started_at must be a finite monotonic timestamp")
    started_at = (
        time.monotonic() if runtime_started_at is None else runtime_started_at
    )
    stop_reason = "max_epochs"
    epochs_completed = 0

    explanation_loss_sum = 0.0
    explanation_token_count = 0
    for epoch in range(start_epoch, config["max_epochs"] + 1):
        model.train()
        train_loss_sum = 0.0
        explanation_loss_sum = 0.0
        explanation_token_count = 0
        train_candidate_count = 0
        epoch_batch_count = 0
        interrupted = False
        for batch_index, batch in enumerate(train_batches):
            elapsed = time.monotonic() - started_at
            if elapsed >= config["max_runtime_seconds"]:
                stop_reason = "max_runtime"
                interrupted = True
                break
            if global_step >= config["max_steps"]:
                stop_reason = "max_steps"
                interrupted = True
                break
            inputs = _batch_tensors(batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            explanation_component = None
            if config["objective"] == "answer_plus_explanation":
                if not isinstance(model, SELRAnswerModel):
                    raise ValueError("explanation training requires SELRAnswerModel")
                explanation_inputs = _explanation_batch_inputs(
                    inputs, model, config["explanation"]
                )
                answer_logits, token_logits = model.forward_with_explanations(
                    inputs, explanation_inputs["decoder_input_ids"]
                )
                answer_component = answer_loss(
                    answer_logits,
                    inputs["answer_targets"],
                    inputs["candidate_mask"],
                )
                explanation_component = explanation_loss(
                    token_logits,
                    explanation_inputs["target_ids"],
                    explanation_inputs["target_mask"],
                )
                loss = (
                    answer_component
                    + config["explanation"]["loss_weight"] * explanation_component
                )
            else:
                answer_component = answer_loss(
                    model(inputs),
                    inputs["answer_targets"],
                    inputs["candidate_mask"],
                )
                loss = answer_component
            if not torch.isfinite(answer_component):
                raise FloatingPointError(
                    f"non-finite answer loss at epoch {epoch}, batch {batch_index}"
                )
            if explanation_component is not None and not torch.isfinite(
                explanation_component
            ):
                raise FloatingPointError(
                    f"non-finite explanation loss at epoch {epoch}, "
                    f"batch {batch_index}"
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch}, batch {batch_index}"
                )
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(
                        f"non-finite gradient for {name} at epoch {epoch}, "
                        f"batch {batch_index}"
                    )
            optimizer.step()
            batch_candidates = int(inputs["candidate_mask"].sum().item())
            train_loss_sum += float(loss.detach().item()) * batch_candidates
            train_candidate_count += batch_candidates
            if explanation_component is not None:
                token_count = int(
                    explanation_inputs["target_mask"].sum().item()
                )
                explanation_loss_sum += (
                    float(explanation_component.detach().item()) * token_count
                )
                explanation_token_count += token_count
            epoch_batch_count += 1
            global_step += 1

        if epoch_batch_count == 0:
            if global_step == 0:
                raise ValueError("training batches must contain at least one batch")
            break
        if not interrupted:
            epochs_completed += 1
        if (
            not interrupted
            and time.monotonic() - started_at >= config["max_runtime_seconds"]
        ):
            stop_reason = "max_runtime"
        should_validate = (
            epoch % config["validation"]["every_epochs"] == 0
            or epoch == config["max_epochs"]
            or interrupted
            or global_step >= config["max_steps"]
            or stop_reason == "max_runtime"
        )
        if not should_validate:
            epoch_result = {
                "epoch": epoch,
                "global_step": global_step,
                "training_loss": train_loss_sum / train_candidate_count,
                "validation": None,
                "improved": False,
                "stop_reason": None,
            }
            if config["objective"] == "answer_plus_explanation":
                epoch_result["training_explanation_loss"] = (
                    explanation_loss_sum / explanation_token_count
                )
            history.append(epoch_result)
            for callback in run_callbacks:
                callback.on_epoch_end(epoch_result)
            continue
        validation_metrics = validate(model, validation_batches, device=device)
        train_loss = train_loss_sum / train_candidate_count
        validation_score = float(validation_metrics["exact_match_accuracy"])
        improved = validation_score > best_metric
        if improved:
            best_metric = validation_score
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                config=config,
                seed=seed,
                epoch=epoch,
                global_step=global_step,
                best_validation_exact_match=best_metric,
                stale_epochs=stale_epochs,
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            config=config,
            seed=seed,
            epoch=epoch,
            global_step=global_step,
            best_validation_exact_match=best_metric,
            stale_epochs=stale_epochs,
        )
        epoch_result = {
            "epoch": epoch,
            "global_step": global_step,
            "training_loss": train_loss,
            "validation": validation_metrics,
            "improved": improved,
            "stop_reason": stop_reason if interrupted else None,
        }
        if config["objective"] == "answer_plus_explanation":
            epoch_result["training_explanation_loss"] = (
                explanation_loss_sum / explanation_token_count
            )
        history.append(epoch_result)
        for callback in run_callbacks:
            callback.on_epoch_end(epoch_result)
        if interrupted:
            break
        if stale_epochs >= config["validation"]["early_stopping_patience"]:
            stop_reason = "early_stopping"
            break
        if global_step >= config["max_steps"]:
            stop_reason = "max_steps"
            break
        if time.monotonic() - started_at >= config["max_runtime_seconds"]:
            stop_reason = "max_runtime"
            break

    if best_metric < 0:
        raise RuntimeError(
            "training stopped before a validation checkpoint could be selected"
        )
    if not best_path.is_file() or not latest_path.is_file():
        raise RuntimeError("training completed without writing required checkpoints")
    result = TrainingResult(
        global_step=global_step,
        epochs_completed=epochs_completed,
        best_validation_exact_match=best_metric,
        stop_reason=stop_reason,
        best_checkpoint=str(best_path),
        latest_checkpoint=str(latest_path),
        history=tuple(history),
    )
    for callback in run_callbacks:
        callback.on_train_end(result)
    return result


def _batch_tensors(
    batch: Mapping[str, Tensor], *, device: torch.device | None
) -> dict[str, Tensor]:
    if not isinstance(batch, Mapping):
        raise ValueError("each batch must be a mapping of tensors")
    missing = set(_BATCH_FIELDS).difference(batch)
    if missing:
        raise ValueError(f"batch is missing required fields: {sorted(missing)!r}")
    inputs: dict[str, Tensor] = {}
    for field in _BATCH_FIELDS:
        value = batch[field]
        if not isinstance(value, Tensor):
            raise ValueError(f"batch field {field} must be a torch.Tensor")
        inputs[field] = value.to(device) if device is not None else value
    if "context_embeddings" in batch:
        context = batch["context_embeddings"]
        candidates = inputs["candidate_embeddings"]
        if (
            not isinstance(context, Tensor)
            or candidates.ndim != 3
            or context.ndim != 2
            or context.shape[0] != candidates.shape[0]
            or context.shape[1] != candidates.shape[2]
            or context.dtype != candidates.dtype
            or context.device != candidates.device
        ):
            raise ValueError(
                "context_embeddings must have shape [B, D] and match candidate "
                "dtype and device"
            )
        inputs["context_embeddings"] = (
            context.to(device) if device is not None else context
        )
    if "encoder_memory_type_ids" in batch:
        type_ids = batch["encoder_memory_type_ids"]
        if (
            not isinstance(type_ids, Tensor)
            or type_ids.shape != inputs["encoder_memory_mask"].shape
            or type_ids.dtype != torch.int64
        ):
            raise ValueError(
                "encoder_memory_type_ids must be an int64 tensor matching "
                "encoder_memory_mask"
            )
        type_ids = type_ids.to(device) if device is not None else type_ids
        if type_ids.device != inputs["encoder_memory"].device:
            raise ValueError(
                "encoder_memory_type_ids and encoder_memory must share a device"
            )
        valid_type_ids = type_ids[inputs["encoder_memory_mask"]]
        if valid_type_ids.numel() and not torch.all(
            (valid_type_ids >= 0) & (valid_type_ids <= 2)
        ):
            raise ValueError("encoder_memory_type_ids values must be 0, 1, or 2")
        if torch.any(type_ids[~inputs["encoder_memory_mask"]] != 0):
            raise ValueError("padded encoder_memory_type_ids must be zero")
        inputs["encoder_memory_type_ids"] = type_ids
    has_target_ids = "explanation_target_ids" in batch
    has_target_mask = "explanation_target_mask" in batch
    if has_target_ids != has_target_mask:
        raise ValueError(
            "explanation_target_ids and explanation_target_mask must be provided together"
        )
    if has_target_ids:
        target_ids = batch["explanation_target_ids"]
        target_mask = batch["explanation_target_mask"]
        if (
            not isinstance(target_ids, Tensor)
            or target_ids.ndim != 2
            or target_ids.dtype != torch.long
            or target_ids.shape[0] != inputs["encoder_memory"].shape[0]
        ):
            raise ValueError(
                "explanation_target_ids must be an int64 tensor with shape [B, T]"
            )
        if (
            not isinstance(target_mask, Tensor)
            or target_mask.shape != target_ids.shape
            or target_mask.dtype != torch.bool
        ):
            raise ValueError(
                "explanation_target_mask must be a bool tensor matching target IDs"
            )
        target_ids = target_ids.to(device) if device is not None else target_ids
        target_mask = target_mask.to(device) if device is not None else target_mask
        if target_ids.device != inputs["encoder_memory"].device:
            raise ValueError("explanation targets and encoder memory must share a device")
        inputs["explanation_target_ids"] = target_ids
        inputs["explanation_target_mask"] = target_mask
    return inputs


def _explanation_batch_inputs(
    inputs: Mapping[str, Tensor],
    model: SELRAnswerModel,
    config: Mapping[str, Any],
) -> dict[str, Tensor]:
    if model.explanation_decoder is None:
        raise ValueError("explanation training requires an enabled decoder")
    try:
        target_ids = inputs["explanation_target_ids"]
        target_mask = inputs["explanation_target_mask"]
    except KeyError as exc:
        raise ValueError(
            "explanation training batches must include explanation_target_ids "
            "and explanation_target_mask"
        ) from exc
    if (
        target_ids.shape[1] < 1
        or target_ids.shape[1] > config["max_decode_steps"]
    ):
        raise ValueError(
            "explanation targets must have between 1 and max_decode_steps tokens"
        )
    if not torch.all(target_mask.any(dim=1)):
        raise ValueError("every explanation target must contain at least one token")
    if target_mask.shape[1] > 1 and torch.any(
        ~target_mask[:, :-1] & target_mask[:, 1:]
    ):
        raise ValueError("valid explanation targets must form a contiguous prefix")
    if torch.any(target_ids[~target_mask] != config["pad_token_id"]):
        raise ValueError("padded explanation targets must use pad_token_id")
    if torch.any(target_ids[target_mask] == config["pad_token_id"]):
        raise ValueError("valid explanation targets must not use pad_token_id")
    target_lengths = target_mask.sum(dim=1)
    last_target_ids = target_ids.gather(
        1, (target_lengths - 1).unsqueeze(1)
    ).squeeze(1)
    if not torch.all(last_target_ids == config["end_token_id"]):
        raise ValueError("every explanation target must end with end_token_id")

    decoder_input_ids = torch.full_like(target_ids, config["pad_token_id"])
    decoder_input_ids[:, 0] = config["start_token_id"]
    if target_ids.shape[1] > 1:
        decoder_input_ids[:, 1:] = target_ids[:, :-1]
    decoder_input_ids.masked_fill_(~target_mask, config["pad_token_id"])
    return {
        "target_ids": target_ids,
        "target_mask": target_mask,
        "decoder_input_ids": decoder_input_ids,
    }


def _validate_training_config(
    config: Mapping[str, Any], seed: int, model_variant: str = "selr"
) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    _validate_seed(seed)
    if config.get("model_variant") != model_variant:
        raise ValueError(
            f"training config model_variant must be {model_variant!r}"
        )
    objective = config.get("objective")
    allowed_objectives = (
        {"answer_only", "answer_plus_explanation"}
        if model_variant == "selr"
        else {"answer_only"}
    )
    if objective not in allowed_objectives:
        choices = ", ".join(sorted(allowed_objectives))
        raise ValueError(f"training config objective must be one of: {choices}")
    if config.get("seed") != seed:
        raise ValueError("training config seed must match the seed argument")
    for name in (
        "max_epochs",
        "batch_size",
        "max_steps",
        "max_runtime_seconds",
        "num_threads",
    ):
        _positive_int(config.get(name), name)
    if not isinstance(config.get("deterministic"), bool):
        raise ValueError("deterministic must be a boolean")
    if not isinstance(config.get("device"), str) or config["device"] not in {
        "cpu",
        "cuda",
        "auto",
    }:
        raise ValueError("device must be cpu, cuda, or auto")
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping) or optimizer.get("name") != "adamw":
        raise ValueError("optimizer must configure AdamW")
    for field in ("learning_rate", "weight_decay"):
        value = optimizer.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"optimizer.{field} must be a number")
        if not math.isfinite(value) or (
            value <= 0 if field == "learning_rate" else value < 0
        ):
            raise ValueError(f"optimizer.{field} is outside its valid range")
    validation_config = config.get("validation")
    if not isinstance(validation_config, Mapping):
        raise ValueError("validation must be a mapping")
    _positive_int(validation_config.get("every_epochs"), "validation.every_epochs")
    _positive_int(
        validation_config.get("early_stopping_patience"),
        "validation.early_stopping_patience",
    )
    for field in ("checkpoint_dir",):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty path")
    if objective == "answer_plus_explanation":
        explanation = config.get("explanation")
        if not isinstance(explanation, Mapping):
            raise ValueError("explanation must configure decoder and loss settings")
        for field in ("vocab_size", "max_decode_steps"):
            _positive_int(explanation.get(field), f"explanation.{field}")
        for field in ("pad_token_id", "start_token_id", "end_token_id"):
            token_id = explanation.get(field)
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < explanation["vocab_size"]
            ):
                raise ValueError(
                    f"explanation.{field} must be an integer within the vocabulary"
                )
        if (
            explanation["pad_token_id"] == explanation["start_token_id"]
            or explanation["pad_token_id"] == explanation["end_token_id"]
            or explanation["start_token_id"] == explanation["end_token_id"]
        ):
            raise ValueError("explanation special token IDs must be different")
        loss_weight = explanation.get("loss_weight")
        if (
            isinstance(loss_weight, bool)
            or not isinstance(loss_weight, (int, float))
            or not math.isfinite(loss_weight)
            or loss_weight <= 0
        ):
            raise ValueError("explanation.loss_weight must be a finite positive number")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _validate_reusable(batches: Iterable[Any], name: str) -> None:
    try:
        iterator = iter(batches)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of batches") from exc
    if iterator is batches:
        raise ValueError(f"{name} must be reusable across epochs, not a one-shot iterator")
