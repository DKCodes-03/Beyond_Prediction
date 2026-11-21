"""Apply bounded recurrent updates to a sequence of latent reasoning states."""

import torch
from torch import Tensor, nn


class LatentReasoner(nn.Module):
    """Update latent states sequentially with a shared gated recurrent cell."""

    def __init__(self, latent_dim: int, max_steps: int) -> None:
        super().__init__()
        for name, value in (("latent_dim", latent_dim), ("max_steps", max_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.latent_dim = latent_dim
        self.max_steps = max_steps
        self.reasoning_cell = nn.GRUCell(latent_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(
        self, latent_states: Tensor, latent_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return recurrently updated states, zeroing positions marked as padding."""
        if not isinstance(latent_states, Tensor) or latent_states.ndim != 3:
            raise ValueError("latent_states must have shape [B, S, L]")
        batch_size, step_count, latent_dim = latent_states.shape
        if (
            batch_size < 1
            or step_count < 1
            or step_count > self.max_steps
            or latent_dim != self.latent_dim
        ):
            raise ValueError(
                f"latent_states must have shape [B, S, {self.latent_dim}] "
                f"with 1 <= S <= {self.max_steps}"
            )
        if not torch.is_floating_point(latent_states):
            raise ValueError("latent_states must have a floating-point dtype")
        if not isinstance(latent_mask, Tensor) or latent_mask.shape != (
            batch_size,
            step_count,
        ):
            raise ValueError("latent_mask must have shape [B, S]")
        if latent_mask.dtype != torch.bool:
            raise ValueError("latent_mask must have dtype torch.bool")
        if latent_mask.device != latent_states.device:
            raise ValueError("latent_mask and latent_states must share a device")
        if not torch.all(latent_mask.any(dim=1)):
            raise ValueError("every example must have at least one valid latent state")
        if step_count > 1 and torch.any(~latent_mask[:, :-1] & latent_mask[:, 1:]):
            raise ValueError("valid latent states must form a contiguous prefix")

        hidden = latent_states.new_zeros((batch_size, self.latent_dim))
        outputs: list[Tensor] = []
        for step in range(step_count):
            updated = self.reasoning_cell(latent_states[:, step], hidden)
            valid = latent_mask[:, step].unsqueeze(-1)
            hidden = torch.where(valid, updated, hidden)
            state = self.output_norm(hidden)
            outputs.append(torch.where(valid, state, torch.zeros_like(state)))

        return torch.stack(outputs, dim=1), latent_mask
