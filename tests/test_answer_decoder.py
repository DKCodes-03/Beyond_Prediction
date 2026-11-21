"""Tests for the non-SELR candidate-scoring answer decoder."""

import pytest
import torch

from src.encoder.graph_encoder import GraphQuestionEncoder
from src.encoder.node_embeddings import build_encoder_vocabulary
from src.evaluation.path_baseline import graph_path_sanity_baseline
from src.graph.graph_builder import build_temporal_graph
from src.llm.answer_decoder import CandidateAnswerDecoder, CandidateScores
from src.retrieval.subgraph_retriever import retrieve_subgraph


@pytest.fixture
def graph():
    return build_temporal_graph(
        [
            {
                "fact_id": "f_ab",
                "subject_id": "a",
                "relation_id": "r1",
                "object_id": "b",
                "timestamp": "2014-01-01",
                "source_id": "fixture",
            },
            {
                "fact_id": "f_ac",
                "subject_id": "a",
                "relation_id": "r1",
                "object_id": "c",
                "timestamp": "2014-01-02",
                "source_id": "fixture",
            },
            {
                "fact_id": "f_ad",
                "subject_id": "a",
                "relation_id": "r1",
                "object_id": "d",
                "timestamp": "2014-01-02",
                "source_id": "fixture",
            },
        ]
    )


@pytest.fixture
def query():
    return {
        "qa_id": "multi-answer",
        "question": "Which entity follows r1?",
        "seed_entity_ids": ["a"],
        "relation_sequence": ["r1"],
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2014-01-02",
        },
    }


@pytest.fixture
def encoded(graph, query):
    training_record = {**query, "split": "train"}
    vocabulary = build_encoder_vocabulary(graph, [training_record])
    encoder = GraphQuestionEncoder(vocabulary, embedding_dim=8, dropout=0)
    retrieved = retrieve_subgraph(graph, query, graph_id="fixture-v1")
    return encoder(retrieved)


def test_decoder_scores_candidates_and_trains_on_all_correct_answers(encoded):
    decoder = CandidateAnswerDecoder(embedding_dim=8)
    scores = decoder(encoded)
    batched_scores = decoder.score_embeddings(
        encoded.context_embedding.unsqueeze(0),
        encoded.candidate_embeddings.unsqueeze(0),
    )

    assert scores.candidate_ids == ("b", "c", "d")
    assert scores.logits.shape == (3,)
    torch.testing.assert_close(batched_scores[0], scores.logits)
    loss = decoder.loss(scores, ["b", "c"])
    assert torch.isfinite(loss)
    loss.backward()
    assert decoder.scorer[0].weight.grad is not None


def test_decoder_rejects_incomplete_candidate_set(encoded):
    decoder = CandidateAnswerDecoder(embedding_dim=8)
    scores = decoder(encoded)

    with pytest.raises(ValueError, match="missing accepted answers"):
        decoder.loss(scores, ["b", "outside-retrieval"])


def test_decoder_returns_all_thresholded_answers_and_top_candidate_fallback(encoded):
    decoder = CandidateAnswerDecoder(embedding_dim=8)
    scores = CandidateScores(
        candidate_ids=encoded.candidate_ids,
        logits=torch.tensor([1.0, 1.0, -1.0]),
    )

    result = decoder.decode(scores, qa_id="multi-answer")
    assert result["status"] == "ok"
    assert result["predicted_answer_ids"] == ["b", "c"]
    assert [row["answer_id"] for row in result["answer_scores"]] == ["b", "c", "d"]

    fallback = decoder.decode(
        CandidateScores(encoded.candidate_ids, torch.tensor([-1.0, -2.0, -3.0])),
        qa_id="multi-answer",
    )
    assert fallback["predicted_answer_ids"] == ["b"]


def test_decoder_explicitly_reports_empty_candidates(graph, query):
    record = {
        **query,
        "qa_id": "empty-query",
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2013-12-31",
        },
    }
    vocabulary = build_encoder_vocabulary(
        graph, [{**record, "split": "train"}]
    )
    encoder = GraphQuestionEncoder(vocabulary, embedding_dim=8, dropout=0)
    retrieved = retrieve_subgraph(graph, record, graph_id="fixture-v1")
    scores = CandidateAnswerDecoder(embedding_dim=8)(encoder(retrieved))

    assert scores.candidate_ids == ()
    result = CandidateAnswerDecoder(embedding_dim=8).decode(scores, qa_id="empty-query")
    assert result["status"] == "no_candidates"
    assert result["predicted_answer_ids"] == []
    assert result["failure_reason"]
    assert result["answer_scores"] == []


def test_decoder_exact_policy_matches_graph_oracle_answer_set(graph, query):
    decoder = CandidateAnswerDecoder(embedding_dim=8)
    encoder = GraphQuestionEncoder(
        build_encoder_vocabulary(graph, [{**query, "split": "train"}]),
        embedding_dim=8,
        dropout=0,
    )
    retrieved = retrieve_subgraph(graph, query, graph_id="fixture-v1")
    with torch.no_grad():
        decoder.scorer[-1].weight.zero_()
        decoder.scorer[-1].bias.fill_(1)

    decoded = decoder.decode(decoder(encoder(retrieved)), qa_id=query["qa_id"])
    oracle = graph_path_sanity_baseline(graph, query)
    assert set(decoded["predicted_answer_ids"]) == set(
        oracle["predicted_answer_ids"]
    )
