"""Tests for bounded latent-conditioned explanation decoding."""

import pytest
import torch

from src.selr.explanation_decoder import ExplanationDecoder


def test_explanation_decoder_conditions_on_latents_and_backpropagates():
    torch.manual_seed(41)
    decoder = ExplanationDecoder(
        latent_dim=4, vocab_size=9, max_decode_steps=3, embedding_dim=5
    )
    latent_states = torch.randn(2, 3, 4, requires_grad=True)
    latent_mask = torch.tensor([[True, True, False], [True, False, False]])
    decoder_input_ids = torch.tensor([[1, 4, 5], [1, 3, 0]])

    logits = decoder(latent_states, latent_mask, decoder_input_ids)

    assert logits.shape == (2, 3, 9)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert latent_states.grad is not None
    assert torch.isfinite(latent_states.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
        if parameter.requires_grad
    )


def test_explanation_decoder_ignores_padded_latent_states():
    torch.manual_seed(43)
    decoder = ExplanationDecoder(latent_dim=3, vocab_size=7, max_decode_steps=2)
    latent_states = torch.randn(1, 3, 3)
    latent_mask = torch.tensor([[True, False, False]])
    changed_padding = latent_states.clone()
    changed_padding[:, 1:] = 1_000
    decoder_input_ids = torch.tensor([[1, 2]])

    logits = decoder(latent_states, latent_mask, decoder_input_ids)
    changed_logits = decoder(changed_padding, latent_mask, decoder_input_ids)

    torch.testing.assert_close(logits, changed_logits)


def test_generation_stops_at_end_token_and_never_exceeds_budget():
    decoder = ExplanationDecoder(
        latent_dim=3, vocab_size=6, max_decode_steps=4, pad_token_id=0
    )
    with torch.no_grad():
        decoder.output_projection.weight.zero_()
        decoder.output_projection.bias.zero_()
        decoder.output_projection.bias[2] = 2
    latent_states = torch.randn(2, 2, 3)
    latent_mask = torch.ones(2, 2, dtype=torch.bool)

    stopped = decoder.generate(latent_states, latent_mask, 1, 2)
    assert torch.equal(stopped, torch.tensor([[1, 2], [1, 2]]))

    with torch.no_grad():
        decoder.output_projection.bias[2] = -1
        decoder.output_projection.bias[3] = 4
    bounded = decoder.generate(latent_states, latent_mask, 1, 2)
    assert bounded.shape == (2, decoder.max_decode_steps + 1)
    assert torch.equal(bounded, torch.tensor([[1, 3, 3, 3, 3], [1, 3, 3, 3, 3]]))


def test_explanation_decoder_rejects_invalid_masks_and_unbounded_inputs():
    decoder = ExplanationDecoder(latent_dim=2, vocab_size=5, max_decode_steps=2)
    states = torch.randn(1, 3, 2)
    input_ids = torch.tensor([[1]])

    with pytest.raises(ValueError, match="contiguous prefix"):
        decoder(states, torch.tensor([[True, False, True]]), input_ids)
    with pytest.raises(ValueError, match="1 <= T <= 2"):
        decoder(states, torch.ones(1, 3, dtype=torch.bool), torch.tensor([[1, 2, 3]]))
    with pytest.raises(ValueError, match="within the vocabulary"):
        decoder(states, torch.ones(1, 3, dtype=torch.bool), torch.tensor([[5]]))


@pytest.mark.parametrize(
    ("start_id", "end_id", "error"),
    [
        (0, 2, "start_token_id must not equal pad_token_id"),
        (1, 1, "must be different"),
    ],
)
def test_generation_rejects_invalid_special_tokens(start_id, end_id, error):
    decoder = ExplanationDecoder(latent_dim=2, vocab_size=5, max_decode_steps=2)
    states = torch.randn(1, 2, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match=error):
        decoder.generate(states, mask, start_id, end_id)
