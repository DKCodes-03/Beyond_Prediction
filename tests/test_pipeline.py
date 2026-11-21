"""Tiny CPU integration coverage for the answer-only SELR pipeline."""

import json
from pathlib import Path

import torch

from src.encoder.graph_encoder import GraphQuestionEncoder
from src.encoder.node_embeddings import build_encoder_vocabulary
from src.evaluation.metrics import compute_answer_metrics
from src.graph.graph_builder import build_temporal_graph
from src.ingestion.preprocess import preprocess_facts
from src.llm.answer_decoder import CandidateAnswerDecoder, CandidateScores
from src.training.train_selr import EncodedQADataset, collate_encoded_examples
from src.training.trainer import BaselineAnswerModel, SELRAnswerModel


def test_tiny_cpu_selr_pipeline_from_facts_to_answer_metrics():
    fixture_path = Path(__file__).parent / "fixtures" / "temporal_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    facts = preprocess_facts(fixture["facts"])
    graph = build_temporal_graph(facts)
    fixture_query = fixture["queries"][1]
    qa_record = {
        "qa_id": fixture_query["query_id"],
        "split": "train",
        "question": "Which entity follows relation_1 then relation_2?",
        "seed_entity_ids": [fixture_query["seed_entity_id"]],
        "relation_sequence": fixture_query["relation_sequence"],
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": fixture_query["cutoff_date"],
        },
        "accepted_answer_ids": fixture_query["expected_answer_ids"],
    }

    vocabulary = build_encoder_vocabulary(graph, [qa_record])
    encoder = GraphQuestionEncoder(vocabulary, embedding_dim=8, dropout=0)
    encoder.eval()
    dataset = EncodedQADataset(
        graph,
        encoder,
        [qa_record],
        split="train",
        graph_id=fixture["fixture_id"],
        retrieval_settings={"max_hops": 2, "max_nodes": 8, "max_edges": 8},
        max_candidates=8,
    )
    example = dataset[0]
    batch = collate_encoded_examples([example])

    assert graph.fact_count == len(fixture["facts"])
    assert batch["encoder_memory"].device.type == "cpu"
    assert batch["encoder_memory_type_ids"].tolist() == [[0, 1, 1, 1, 2, 2]]
    assert batch["candidate_embeddings"].shape == (1, 1, 8)
    assert batch["candidate_mask"].tolist() == [[True]]
    assert batch["answer_targets"].tolist() == [[1.0]]

    baseline = BaselineAnswerModel(encoder_dim=8)
    baseline.eval()
    with torch.no_grad():
        baseline_logits = baseline(batch)
    assert baseline_logits.shape == (1, 1)
    assert torch.isfinite(baseline_logits).all()

    torch.manual_seed(17)
    model = SELRAnswerModel(encoder_dim=8, latent_dim=4, max_steps=2)
    model.eval()
    with torch.no_grad():
        answer_only_logits = model(batch)

    candidate_ids = tuple(fixture_query["expected_answer_ids"])
    model.enable_explanation_decoder(
        vocab_size=8, max_decode_steps=3, pad_token_id=0
    )
    explanation_decoder = model.explanation_decoder
    assert explanation_decoder is not None
    decoder_input_ids = torch.tensor([[1, 3]])
    with torch.no_grad():
        answer_logits, explanation_logits = model.forward_with_explanations(
            batch, decoder_input_ids
        )
        latent_states, latent_mask = model.bottleneck(
            batch["encoder_memory"], batch["encoder_memory_mask"]
        )
        reasoned_states, reasoned_mask = model.reasoner(
            latent_states, latent_mask
        )
        generated_explanation = explanation_decoder.generate(
            reasoned_states, reasoned_mask, start_token_id=1, end_token_id=2
        )

    torch.testing.assert_close(answer_logits, answer_only_logits)
    assert explanation_logits.shape == (1, 2, 8)
    assert torch.isfinite(explanation_logits).all()
    assert generated_explanation.shape[0] == 1
    assert generated_explanation.shape[1] <= (
        explanation_decoder.max_decode_steps + 1
    )
    assert generated_explanation[0, 0].item() == 1
    assert generated_explanation[0, -1].item() == 2 or (
        generated_explanation.shape[1]
        == model.explanation_decoder.max_decode_steps + 1
    )

    assert answer_logits.shape == (1, len(candidate_ids))
    assert torch.isfinite(answer_logits).all()

    decoder = CandidateAnswerDecoder(embedding_dim=8)
    prediction = decoder.decode(
        CandidateScores(candidate_ids, answer_logits[0]),
        qa_id=qa_record["qa_id"],
        model_variant="selr",
    )
    metrics = compute_answer_metrics([qa_record], [prediction])

    assert prediction["status"] == "ok"
    assert prediction["predicted_answer_ids"] == fixture_query["expected_answer_ids"]
    assert metrics["example_count"] == 1
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["micro_f1"] == 1.0
