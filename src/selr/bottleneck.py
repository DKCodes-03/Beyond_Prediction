"""Compress padded encoder memory into a bounded sequence of latent states."""

import math

import torch
from torch import Tensor, nn


class LatentBottleneck(nn.Module):
    """Pool encoder memory into learned, cross-attending latent slots."""

    def __init__(self, input_dim: int, latent_dim: int, max_steps: int) -> None:
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("latent_dim", latent_dim),
            ("max_steps", max_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.max_steps = max_steps
        self.latent_queries = nn.Parameter(torch.empty(max_steps, latent_dim))
        nn.init.normal_(self.latent_queries, std=latent_dim**-0.5)
        self.key_projection = nn.Linear(input_dim, latent_dim)
        self.value_projection = nn.Linear(input_dim, latent_dim)
        self.attention_norm = nn.LayerNorm(latent_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(
        self, encoder_memory: Tensor, encoder_memory_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``[B, max_steps, L]`` states and their valid-prefix mask."""
        if not isinstance(encoder_memory, Tensor) or encoder_memory.ndim != 3:
            raise ValueError("encoder_memory must have shape [B, M, D]")
        batch_size, memory_size, input_dim = encoder_memory.shape
        if batch_size < 1 or memory_size < 1 or input_dim != self.input_dim:
            raise ValueError(
                f"encoder_memory must have non-empty B/M axes and D={self.input_dim}"
            )
        if not torch.is_floating_point(encoder_memory):
            raise ValueError("encoder_memory must have a floating-point dtype")
        if (
            not isinstance(encoder_memory_mask, Tensor)
            or encoder_memory_mask.shape != (batch_size, memory_size)
        ):
            raise ValueError("encoder_memory_mask must have shape [B, M]")
        if encoder_memory_mask.dtype != torch.bool:
            raise ValueError("encoder_memory_mask must have dtype torch.bool")
        if encoder_memory_mask.device != encoder_memory.device:
            raise ValueError("encoder_memory_mask and encoder_memory must share a device")
        if not torch.all(encoder_memory_mask.any(dim=1)):
            raise ValueError("every example must have at least one valid memory vector")

        keys = self.key_projection(encoder_memory)
        values = self.value_projection(encoder_memory)
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(
            self.latent_dim
        )
        scores = scores.masked_fill(
            ~encoder_memory_mask.unsqueeze(1), torch.finfo(scores.dtype).min
        )
        attended = torch.matmul(torch.softmax(scores, dim=-1), values)
        latent_states = self.attention_norm(queries + attended)
        latent_states = self.output_norm(
            latent_states + self.feed_forward(latent_states)
        )
        latent_mask = torch.ones(
            (batch_size, self.max_steps),
            dtype=torch.bool,
            device=encoder_memory.device,
        )
        return latent_states, latent_mask
