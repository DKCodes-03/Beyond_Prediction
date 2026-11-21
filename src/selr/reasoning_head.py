"""Score retrieved candidate answers using candidate-conditioned latent context."""

import math

import torch
from torch import Tensor, nn


class ReasoningHead(nn.Module):
    """Produce one logit per candidate from the valid latent reasoning states."""

    def __init__(
        self, latent_dim: int, candidate_dim: int, hidden_dim: int | None = None
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("candidate_dim", candidate_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim is None:
            hidden_dim = latent_dim
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim < 1
        ):
            raise ValueError("hidden_dim must be a positive integer")

        self.latent_dim = latent_dim
        self.candidate_dim = candidate_dim
        self.hidden_dim = hidden_dim
        self.latent_key_projection = nn.Linear(latent_dim, hidden_dim)
        self.latent_value_projection = nn.Linear(latent_dim, hidden_dim)
        self.candidate_projection = nn.Linear(candidate_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        latent_states: Tensor,
        latent_mask: Tensor,
        candidate_embeddings: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        """Return finite candidate logits with padded candidate positions set to zero."""
        if not isinstance(latent_states, Tensor) or latent_states.ndim != 3:
            raise ValueError("latent_states must have shape [B, S, L]")
        if (
            not isinstance(candidate_embeddings, Tensor)
            or candidate_embeddings.ndim != 3
        ):
            raise ValueError("candidate_embeddings must have shape [B, C, D]")
        batch_size, step_count, latent_dim = latent_states.shape
        candidate_batch, candidate_count, candidate_dim = candidate_embeddings.shape
        if (
            batch_size < 1
            or step_count < 1
            or latent_dim != self.latent_dim
            or candidate_batch != batch_size
            or candidate_count < 1
            or candidate_dim != self.candidate_dim
        ):
            raise ValueError(
                "latent_states and candidate_embeddings must have matching "
                f"batch axes and feature dimensions {self.latent_dim} and "
                f"{self.candidate_dim}"
            )
        if not torch.is_floating_point(latent_states) or not torch.is_floating_point(
            candidate_embeddings
        ):
            raise ValueError("latent states and candidates must be floating-point")
        if (
            latent_states.device != candidate_embeddings.device
            or latent_states.dtype != candidate_embeddings.dtype
        ):
            raise ValueError("latent states and candidates must share device and dtype")
        if not isinstance(latent_mask, Tensor) or latent_mask.shape != (
            batch_size,
            step_count,
        ):
            raise ValueError("latent_mask must have shape [B, S]")
        if not isinstance(candidate_mask, Tensor) or candidate_mask.shape != (
            batch_size,
            candidate_count,
        ):
            raise ValueError("candidate_mask must have shape [B, C]")
        if latent_mask.dtype != torch.bool or candidate_mask.dtype != torch.bool:
            raise ValueError("latent_mask and candidate_mask must have dtype torch.bool")
        if (
            latent_mask.device != latent_states.device
            or candidate_mask.device != candidate_embeddings.device
        ):
            raise ValueError("feature tensors and masks must share a device")
        if not torch.all(latent_mask.any(dim=1)):
            raise ValueError("every example must have at least one valid latent state")
        if not torch.all(candidate_mask.any(dim=1)):
            raise ValueError("every example must have at least one valid candidate")
        if step_count > 1 and torch.any(~latent_mask[:, :-1] & latent_mask[:, 1:]):
            raise ValueError("valid latent states must form a contiguous prefix")

        keys = self.latent_key_projection(latent_states)
        values = self.latent_value_projection(latent_states)
        queries = self.candidate_projection(candidate_embeddings)
        attention_scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(
            self.hidden_dim
        )
        attention_scores = attention_scores.masked_fill(
            ~latent_mask.unsqueeze(1), torch.finfo(attention_scores.dtype).min
        )
        latent_context = torch.matmul(
            torch.softmax(attention_scores, dim=-1), values
        )
        candidate_features = torch.cat(
            (queries, latent_context, queries * latent_context), dim=-1
        )
        logits = self.scorer(candidate_features).squeeze(-1)
        logits = torch.where(candidate_mask, logits, torch.zeros_like(logits))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("reasoning head produced non-finite answer logits")
        return logits
