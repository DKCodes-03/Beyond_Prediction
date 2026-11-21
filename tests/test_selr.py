"""Tests for the bounded answer-only SELR modules."""

import pytest
import torch

from src.selr.bottleneck import LatentBottleneck
from src.selr.latent_reasoner import LatentReasoner
from src.selr.losses import answer_loss, explanation_loss
from src.selr.reasoning_head import ReasoningHead
from src.training.trainer import SELRAnswerModel
from src.utils.checkpoint import load_checkpoint, save_checkpoint


def test_answer_loss_masks_padding_and_backpropagates_only_valid_candidates():
    logits = torch.tensor(
        [[0.0, 1.0, 50.0], [-1.0, 2.0, -50.0]], requires_grad=True
    )
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    candidate_mask = torch.tensor(
        [[True, True, False], [True, True, False]]
    )

    loss = answer_loss(logits, targets, candidate_mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[candidate_mask], targets[candidate_mask]
    )
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[~candidate_mask]) == 0
    assert torch.count_nonzero(logits.grad[candidate_mask]) > 0


@pytest.mark.parametrize(
    ("targets", "mask", "error"),
    [
        (
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[False, False]]),
            "at least one valid candidate",
        ),
        (
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[True, True]]),
            "at least one positive answer target",
        ),
        (
            torch.tensor([[1.0, 0.5]]),
            torch.tensor([[True, True]]),
            "must be binary",
        ),
        (
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[True, False]]),
            "padded answer targets must be zero",
        ),
    ],
)
def test_answer_loss_rejects_invalid_candidate_targets(targets, mask, error):
    with pytest.raises(ValueError, match=error):
        answer_loss(torch.zeros_like(targets), targets, mask)


def test_explanation_loss_masks_padding_and_backpropagates_only_valid_tokens():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    targets = torch.tensor([[1, 2, 3], [4, 1, 0]])
    token_mask = torch.tensor([[True, True, True], [True, True, False]])

    loss = explanation_loss(logits, targets, token_mask)
    expected = torch.nn.functional.cross_entropy(
        logits[token_mask], targets[token_mask]
    )
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[~token_mask]) == 0
    assert torch.count_nonzero(logits.grad[token_mask]) > 0


def test_explanation_loss_rejects_non_prefix_token_masks():
    with pytest.raises(ValueError, match="contiguous prefix"):
        explanation_loss(
            torch.zeros(1, 3, 4),
            torch.tensor([[1, 0, 2]]),
            torch.tensor([[True, False, True]]),
        )


def test_selr_modules_preserve_masks_shapes_and_backpropagate():
    torch.manual_seed(17)
    bottleneck = LatentBottleneck(input_dim=6, latent_dim=5, max_steps=3)
    reasoner = LatentReasoner(latent_dim=5, max_steps=3)
    answer_head = ReasoningHead(latent_dim=5, candidate_dim=6)
    memory = torch.randn(2, 4, 6)
    memory_mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    candidates = torch.randn(2, 3, 6)
    candidate_mask = torch.tensor(
        [[True, True, True], [True, True, False]]
    )

    latent_states, latent_mask = bottleneck(memory, memory_mask)
    reasoned_states, reasoned_mask = reasoner(latent_states, latent_mask)
    logits = answer_head(
        reasoned_states, reasoned_mask, candidates, candidate_mask
    )

    assert latent_states.shape == (2, 3, 5)
    assert latent_mask.dtype == torch.bool
    assert latent_mask.all()
    assert reasoned_states.shape == (2, 3, 5)
    assert reasoned_mask.equal(latent_mask)
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert logits[1, 2].item() == 0

    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    loss = answer_loss(logits, targets, candidate_mask)
    loss.backward()
    assert torch.isfinite(loss)
    for module in (bottleneck, reasoner, answer_head):
        assert any(parameter.grad is not None for parameter in module.parameters())
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
            if parameter.grad is not None
        )


def test_optional_explanation_decoder_preserves_answer_only_forward():
    torch.manual_seed(29)
    model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    batch = {
        "encoder_memory": torch.randn(2, 3, 4),
        "encoder_memory_mask": torch.tensor(
            [[True, True, False], [True, True, True]]
        ),
        "candidate_embeddings": torch.randn(2, 2, 4),
        "candidate_mask": torch.tensor([[True, True], [True, False]]),
        "answer_targets": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    }
    decoder_input_ids = torch.tensor([[1, 4], [1, 5]])
    model.eval()

    assert model.explanation_decoder is None
    with torch.no_grad():
        answer_only_logits = model(batch)
    with pytest.raises(ValueError, match="decoder is not enabled"):
        model.forward_with_explanations(batch, decoder_input_ids)

    model.enable_explanation_decoder(
        vocab_size=8, max_decode_steps=3, pad_token_id=0
    )
    with torch.no_grad():
        enabled_answer_logits = model(batch)
        combined_answer_logits, explanation_logits = model.forward_with_explanations(
            batch, decoder_input_ids
        )

    torch.testing.assert_close(enabled_answer_logits, answer_only_logits)
    torch.testing.assert_close(combined_answer_logits, answer_only_logits)
    assert explanation_logits.shape == (2, 2, 8)
    assert torch.isfinite(explanation_logits).all()


def test_bottleneck_ignores_padded_memory_and_rejects_empty_rows():
    torch.manual_seed(23)
    bottleneck = LatentBottleneck(input_dim=4, latent_dim=4, max_steps=2)
    memory = torch.randn(1, 3, 4)
    mask = torch.tensor([[True, False, False]])
    changed_padding = memory.clone()
    changed_padding[:, 1:] = 1_000

    states, _ = bottleneck(memory, mask)
    changed_states, _ = bottleneck(changed_padding, mask)

    torch.testing.assert_close(states, changed_states)
    with pytest.raises(ValueError, match="at least one valid memory vector"):
        bottleneck(memory, torch.zeros_like(mask))


def test_reasoner_rejects_non_prefix_latent_mask():
    reasoner = LatentReasoner(latent_dim=3, max_steps=3)
    states = torch.randn(2, 3, 3)
    prefix_mask = torch.tensor([[True, True, False], [True, False, False]])
    reasoned_states, returned_mask = reasoner(states, prefix_mask)

    assert returned_mask.equal(prefix_mask)
    assert torch.count_nonzero(reasoned_states[0, 2]) == 0
    assert torch.count_nonzero(reasoned_states[1, 1:]) == 0

    invalid_mask = torch.tensor([[True, False, True], [True, True, False]])
    with pytest.raises(ValueError, match="contiguous prefix"):
        reasoner(states, invalid_mask)


def test_selr_checkpoint_restores_model_optimizer_and_progress(tmp_path):
    torch.manual_seed(31)
    model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = {
        "encoder_memory": torch.randn(2, 3, 4),
        "encoder_memory_mask": torch.tensor(
            [[True, True, False], [True, True, True]]
        ),
        "candidate_embeddings": torch.randn(2, 2, 4),
        "candidate_mask": torch.tensor([[True, True], [True, False]]),
        "answer_targets": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    }
    optimizer.zero_grad(set_to_none=True)
    loss = answer_loss(
        model(batch), batch["answer_targets"], batch["candidate_mask"]
    )
    loss.backward()
    optimizer.step()
    expected_logits = model(batch).detach()
    config = {"model_variant": "selr", "objective": "answer_only"}
    checkpoint_path = save_checkpoint(
        tmp_path / "selr.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        seed=17,
        epoch=2,
        global_step=5,
        best_validation_exact_match=0.75,
        stale_epochs=1,
    )

    restored_model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.001)
    progress = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        expected_config=config,
        expected_seed=17,
    )

    assert progress == {
        "epoch": 2,
        "global_step": 5,
        "best_validation_exact_match": 0.75,
        "stale_epochs": 1,
    }
    assert restored_optimizer.state
    torch.testing.assert_close(restored_model(batch), expected_logits)
