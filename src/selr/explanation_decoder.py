"""Generate bounded explanations conditioned on SELR latent reasoning states."""

import math

import torch
from torch import Tensor, nn


class ExplanationDecoder(nn.Module):
    """Autoregressively decode tokens while attending to valid latent states.

    ``forward`` accepts teacher-forced decoder inputs ``[B, T]`` (typically
    ``BOS`` followed by the target shifted right) and returns next-token logits
    ``[B, T, V]``. ``generate`` performs greedy decoding and returns token IDs
    including the start token, stopping at the end token or after
    ``max_decode_steps`` generated tokens. In mixed batches, rows that finish
    early are padded through the batch's stopping step. Both methods condition
    every step on the masked latent memory ``[B, S, L]``.
    """

    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        max_decode_steps: int,
        embedding_dim: int | None = None,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("latent_dim", latent_dim),
            ("vocab_size", vocab_size),
            ("max_decode_steps", max_decode_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if embedding_dim is None:
            embedding_dim = latent_dim
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, int)
            or embedding_dim < 1
        ):
            raise ValueError("embedding_dim must be a positive integer")
        if (
            isinstance(pad_token_id, bool)
            or not isinstance(pad_token_id, int)
            or not 0 <= pad_token_id < vocab_size
        ):
            raise ValueError("pad_token_id must be an integer within the vocabulary")

        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_decode_steps = max_decode_steps
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.latent_key_projection = nn.Linear(latent_dim, latent_dim, bias=False)
        self.latent_value_projection = nn.Linear(latent_dim, latent_dim)
        self.initial_state_projection = nn.Linear(latent_dim, latent_dim)
        self.decoder_cell = nn.GRUCell(embedding_dim + latent_dim, latent_dim)
        self.output_projection = nn.Linear(latent_dim, vocab_size)

    def forward(
        self,
        latent_states: Tensor,
        latent_mask: Tensor,
        decoder_input_ids: Tensor,
    ) -> Tensor:
        """Return next-token logits for teacher-forced decoder input IDs."""
        batch_size, keys, values = self._project_latent_memory(
            latent_states, latent_mask
        )
        if (
            not isinstance(decoder_input_ids, Tensor)
            or decoder_input_ids.ndim != 2
        ):
            raise ValueError("decoder_input_ids must have shape [B, T]")
        input_batch, input_steps = decoder_input_ids.shape
        if (
            input_batch != batch_size
            or input_steps < 1
            or input_steps > self.max_decode_steps
        ):
            raise ValueError(
                "decoder_input_ids must have matching batch size and "
                f"1 <= T <= {self.max_decode_steps}"
            )
        if decoder_input_ids.dtype != torch.long:
            raise ValueError("decoder_input_ids must have dtype torch.int64")
        if decoder_input_ids.device != latent_states.device:
            raise ValueError("decoder_input_ids and latent_states must share a device")
        if torch.any(decoder_input_ids < 0) or torch.any(
            decoder_input_ids >= self.vocab_size
        ):
            raise ValueError("decoder_input_ids must be within the vocabulary")

        valid = latent_mask.unsqueeze(-1)
        latent_mean = latent_states.masked_fill(~valid, 0).sum(dim=1) / (
            latent_mask.sum(dim=1, keepdim=True).to(latent_states.dtype)
        )
        hidden = torch.tanh(self.initial_state_projection(latent_mean))
        outputs: list[Tensor] = []
        for step in range(input_steps):
            hidden, logits = self._decode_step(
                decoder_input_ids[:, step], hidden, keys, values, latent_mask
            )
            outputs.append(logits)

        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def generate(
        self,
        latent_states: Tensor,
        latent_mask: Tensor,
        start_token_id: int,
        end_token_id: int,
    ) -> Tensor:
        """Greedily decode a bounded token sequence, including its start token."""
        batch_size, keys, values = self._project_latent_memory(
            latent_states, latent_mask
        )
        self._validate_special_token_id(start_token_id, "start_token_id")
        self._validate_special_token_id(end_token_id, "end_token_id")
        if start_token_id == end_token_id:
            raise ValueError("start_token_id and end_token_id must be different")

        valid = latent_mask.unsqueeze(-1)
        latent_mean = latent_states.masked_fill(~valid, 0).sum(dim=1) / (
            latent_mask.sum(dim=1, keepdim=True).to(latent_states.dtype)
        )
        hidden = torch.tanh(self.initial_state_projection(latent_mean))
        current_ids = torch.full(
            (batch_size,),
            start_token_id,
            dtype=torch.long,
            device=latent_states.device,
        )
        active = torch.ones(batch_size, dtype=torch.bool, device=latent_states.device)
        output_ids = [current_ids]

        for _ in range(self.max_decode_steps):
            hidden, logits = self._decode_step(
                current_ids, hidden, keys, values, latent_mask
            )
            logits[:, self.pad_token_id] = torch.finfo(logits.dtype).min
            next_ids = logits.argmax(dim=-1)
            current_ids = torch.where(
                active, next_ids, torch.full_like(next_ids, self.pad_token_id)
            )
            output_ids.append(current_ids)
            active = active & current_ids.ne(end_token_id)
            if not active.any():
                break

        return torch.stack(output_ids, dim=1)

    def _project_latent_memory(
        self, latent_states: Tensor, latent_mask: Tensor
    ) -> tuple[int, Tensor, Tensor]:
        if not isinstance(latent_states, Tensor) or latent_states.ndim != 3:
            raise ValueError("latent_states must have shape [B, S, L]")
        batch_size, step_count, latent_dim = latent_states.shape
        if batch_size < 1 or step_count < 1 or latent_dim != self.latent_dim:
            raise ValueError(
                f"latent_states must have non-empty B/S axes and L={self.latent_dim}"
            )
        if not torch.is_floating_point(latent_states):
            raise ValueError("latent_states must have a floating-point dtype")
        if not torch.isfinite(latent_states).all():
            raise ValueError("latent_states must contain only finite values")
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
        if latent_states.device != self.latent_key_projection.weight.device:
            raise ValueError("latent_states and decoder parameters must share a device")

        return (
            batch_size,
            self.latent_key_projection(latent_states),
            self.latent_value_projection(latent_states),
        )

    def _decode_step(
        self,
        token_ids: Tensor,
        hidden: Tensor,
        keys: Tensor,
        values: Tensor,
        latent_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        scores = torch.matmul(keys, hidden.unsqueeze(-1)).squeeze(-1) / math.sqrt(
            self.latent_dim
        )
        scores = scores.masked_fill(
            ~latent_mask, torch.finfo(scores.dtype).min
        )
        context = torch.matmul(torch.softmax(scores, dim=-1).unsqueeze(1), values)
        context = context.squeeze(1)
        decoder_input = torch.cat((self.token_embedding(token_ids), context), dim=-1)
        hidden = self.decoder_cell(decoder_input, hidden)
        logits = self.output_projection(hidden)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("explanation decoder produced non-finite logits")
        return hidden, logits

    def _validate_special_token_id(self, token_id: int, name: str) -> None:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < self.vocab_size
        ):
            raise ValueError(f"{name} must be an integer within the vocabulary")
        if token_id == self.pad_token_id:
            raise ValueError(f"{name} must not equal pad_token_id")
