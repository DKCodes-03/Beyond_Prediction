"""Answer and explanation losses for SELR."""

import torch
from torch import Tensor
from torch.nn import functional as F


def answer_loss(
    answer_logits: Tensor,
    answer_targets: Tensor,
    candidate_mask: Tensor,
) -> Tensor:
    """Compute mean binary cross-entropy over valid retrieved candidates."""
    if not isinstance(answer_logits, Tensor) or answer_logits.ndim != 2:
        raise ValueError("answer_logits must have shape [B, C]")
    if (
        not isinstance(answer_targets, Tensor)
        or answer_targets.shape != answer_logits.shape
    ):
        raise ValueError("answer_targets must have the same [B, C] shape as answer_logits")
    if (
        not isinstance(candidate_mask, Tensor)
        or candidate_mask.shape != answer_logits.shape
    ):
        raise ValueError("candidate_mask must have the same [B, C] shape as answer_logits")
    if answer_logits.shape[0] < 1 or answer_logits.shape[1] < 1:
        raise ValueError("answer_logits must have non-empty batch and candidate axes")
    if not torch.is_floating_point(answer_logits) or not torch.is_floating_point(
        answer_targets
    ):
        raise ValueError("answer_logits and answer_targets must be floating-point")
    if answer_targets.dtype != answer_logits.dtype:
        raise ValueError("answer_logits and answer_targets must share dtype")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must have dtype torch.bool")
    if (
        answer_targets.device != answer_logits.device
        or candidate_mask.device != answer_logits.device
    ):
        raise ValueError("answer tensors and candidate_mask must share a device")
    if not torch.all(candidate_mask.any(dim=1)):
        raise ValueError("every example must have at least one valid candidate")

    valid_targets = answer_targets[candidate_mask]
    valid_logits = answer_logits[candidate_mask]
    if not torch.isfinite(valid_logits).all():
        raise FloatingPointError("valid answer logits must be finite")
    if not torch.isfinite(valid_targets).all():
        raise ValueError("valid answer targets must be finite")
    if not torch.all((valid_targets == 0) | (valid_targets == 1)):
        raise ValueError("valid answer targets must be binary")
    if torch.any(answer_targets[~candidate_mask] != 0):
        raise ValueError("padded answer targets must be zero")
    if not torch.all((answer_targets == 1).logical_and(candidate_mask).any(dim=1)):
        raise ValueError("every example must have at least one positive answer target")

    return F.binary_cross_entropy_with_logits(valid_logits, valid_targets)


def explanation_loss(
    token_logits: Tensor,
    token_targets: Tensor,
    token_mask: Tensor,
) -> Tensor:
    """Compute mean next-token cross entropy over valid explanation tokens."""
    if not isinstance(token_logits, Tensor) or token_logits.ndim != 3:
        raise ValueError("token_logits must have shape [B, T, V]")
    if token_logits.shape[0] < 1 or token_logits.shape[1] < 1 or token_logits.shape[2] < 1:
        raise ValueError("token_logits must have non-empty B, T, and V axes")
    if (
        not isinstance(token_targets, Tensor)
        or token_targets.shape != token_logits.shape[:2]
    ):
        raise ValueError("token_targets must have shape [B, T]")
    if (
        not isinstance(token_mask, Tensor)
        or token_mask.shape != token_targets.shape
    ):
        raise ValueError("token_mask must have the same [B, T] shape as token_targets")
    if not torch.is_floating_point(token_logits):
        raise ValueError("token_logits must be floating-point")
    if token_targets.dtype != torch.long:
        raise ValueError("token_targets must have dtype torch.int64")
    if token_mask.dtype != torch.bool:
        raise ValueError("token_mask must have dtype torch.bool")
    if (
        token_targets.device != token_logits.device
        or token_mask.device != token_logits.device
    ):
        raise ValueError("explanation tensors and token_mask must share a device")
    if not torch.all(token_mask.any(dim=1)):
        raise ValueError("every example must have at least one valid explanation token")
    if token_mask.shape[1] > 1 and torch.any(
        ~token_mask[:, :-1] & token_mask[:, 1:]
    ):
        raise ValueError("valid explanation tokens must form a contiguous prefix")
    if torch.any(token_targets < 0) or torch.any(token_targets >= token_logits.shape[2]):
        raise ValueError("token_targets must be within the vocabulary")

    valid_logits = token_logits[token_mask]
    valid_targets = token_targets[token_mask]
    if not torch.isfinite(valid_logits).all():
        raise FloatingPointError("valid explanation logits must be finite")
    return F.cross_entropy(valid_logits, valid_targets)
