"""Controlled perturbation tests for explanation conditioning."""

import torch

from src.training.trainer import SELRAnswerModel


def _batch():
    return {
        "encoder_memory": torch.tensor(
            [[[0.2, -0.5, 0.7, 0.1], [0.8, 0.3, -0.4, 0.6], [-0.2, 0.9, 0.5, -0.7]]]
        ),
        "encoder_memory_mask": torch.tensor([[True, True, True]]),
        "candidate_embeddings": torch.tensor([[[0.4, 0.2, -0.1, 0.6]]]),
        "candidate_mask": torch.tensor([[True]]),
        "answer_targets": torch.tensor([[1.0]]),
    }


def test_explanations_respond_to_evidence_and_latent_perturbations():
    torch.manual_seed(53)
    model = SELRAnswerModel(encoder_dim=4, latent_dim=3, max_steps=2)
    model.enable_explanation_decoder(
        vocab_size=8, max_decode_steps=2, pad_token_id=0
    )
    model.eval()
    batch = _batch()
    decoder_input_ids = torch.tensor([[1, 2]])

    with torch.no_grad():
        _, original_logits = model.forward_with_explanations(
            batch, decoder_input_ids
        )

        ablated_batch = dict(batch)
        ablated_batch["encoder_memory"] = batch["encoder_memory"].clone()
        ablated_batch["encoder_memory"][:, 2] = 0
        _, evidence_ablated_logits = model.forward_with_explanations(
            ablated_batch, decoder_input_ids
        )

        latent_states, latent_mask = model.bottleneck(
            batch["encoder_memory"], batch["encoder_memory_mask"]
        )
        reasoned_states, reasoned_mask = model.reasoner(latent_states, latent_mask)
        perturbed_states = reasoned_states.clone()
        perturbed_states[:, 0] = 0
        latent_perturbed_logits = model.explanation_decoder(
            perturbed_states, reasoned_mask, decoder_input_ids
        )

    assert torch.isfinite(original_logits).all()
    assert not torch.allclose(original_logits, evidence_ablated_logits)
    assert not torch.allclose(original_logits, latent_perturbed_logits)
