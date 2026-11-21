"""Tests for train-fitted question vocabulary and temporal graph encoding."""

import pytest
import torch

from src.encoder.graph_encoder import GraphQuestionEncoder
from src.encoder.node_embeddings import (
    NodeRelationTimeEmbeddings,
    build_encoder_vocabulary,
)
from src.graph.graph_builder import build_temporal_graph
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
                "fact_id": "f_bc",
                "subject_id": "b",
                "relation_id": "r2",
                "object_id": "c",
                "timestamp": "2014-01-02",
                "source_id": "fixture",
            },
        ]
    )


@pytest.fixture
def train_record():
    return {
        "qa_id": "train-1",
        "question": "Which agent acted before 2014?",
        "split": "train",
    }


@pytest.fixture
def retrieved_subgraph(graph):
    query = {
        "qa_id": "query-1",
        "question": "Which entity follows r1 then r2 on or before 2014-01-02?",
        "seed_entity_ids": ["a"],
        "relation_sequence": ["r1", "r2"],
        "temporal_constraint": {
            "operator": "on_or_before",
            "cutoff_date": "2014-01-02",
        },
    }
    return retrieve_subgraph(graph, query, graph_id="fixture-v1")


def test_vocabulary_uses_all_graph_ids_but_only_train_question_tokens(
    graph, train_record
):
    vocabulary = build_encoder_vocabulary(graph, [train_record])

    assert vocabulary.entity_ids == ("a", "b", "c")
    assert vocabulary.relation_ids == ("r1", "r2")
    assert "agent" in vocabulary.question_tokens
    assert "validationonly" not in vocabulary.question_tokens
    with pytest.raises(ValueError, match="split='train'"):
        build_encoder_vocabulary(
            graph,
            [{**train_record, "split": "validation", "question": "validationonly"}],
        )


def test_graph_question_encoder_returns_aligned_representations_and_gradients(
    graph, train_record, retrieved_subgraph
):
    vocabulary = build_encoder_vocabulary(graph, [train_record])
    model = GraphQuestionEncoder(vocabulary, embedding_dim=16, dropout=0)
    model.eval()

    output = model(retrieved_subgraph)

    assert output.question_embedding.shape == (16,)
    assert output.graph_embedding.shape == (16,)
    assert output.context_embedding.shape == (16,)
    assert output.node_ids == ("a", "b", "c")
    assert output.node_embeddings.shape == (3, 16)
    assert output.edge_embeddings.shape == (2, 16)
    assert output.candidate_ids == ("c",)
    assert output.candidate_embeddings.shape == (1, 16)
    assert torch.isfinite(output.context_embedding).all()

    output.candidate_embeddings.square().sum().backward()
    assert model.embeddings.entity_embeddings.weight.grad is not None
    assert model.embeddings.relation_embeddings.weight.grad is not None
    assert model.embeddings.time_encoder[0].weight.grad is not None


def test_time_encoder_accepts_unseen_dates_and_question_tokens(
    graph, train_record
):
    vocabulary = build_encoder_vocabulary(graph, [train_record])
    embeddings = NodeRelationTimeEmbeddings(vocabulary, embedding_dim=8)

    question = embeddings.question("An unseen phrase dated 2020-06-01?")
    timestamp = embeddings.time("2020-06-01")

    assert question.shape == (8,)
    assert timestamp.shape == (8,)
    assert torch.isfinite(question).all()
    assert torch.isfinite(timestamp).all()

