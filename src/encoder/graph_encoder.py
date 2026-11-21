"""Encode a question and its retrieved temporal subgraph with mean aggregation."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.encoder.node_embeddings import EncoderVocabulary, NodeRelationTimeEmbeddings
from src.graph.graph_utils import canonical_id, validate_fact


@dataclass(frozen=True)
class EncoderOutput:
    """Question, graph, node, edge, and retrieved-candidate representations."""

    question_embedding: Tensor
    graph_embedding: Tensor
    context_embedding: Tensor
    node_ids: tuple[str, ...]
    node_embeddings: Tensor
    candidate_ids: tuple[str, ...]
    candidate_embeddings: Tensor
    edge_embeddings: Tensor


class GraphQuestionEncoder(nn.Module):
    """One-layer, directed message-passing encoder for one retrieved example."""

    def __init__(
        self,
        vocabulary: EncoderVocabulary,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError("dropout must be a number between 0 and 1")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be a number between 0 and 1")
        self.embedding_dim = embedding_dim
        self.embeddings = NodeRelationTimeEmbeddings(vocabulary, embedding_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.edge_encoder = nn.Sequential(
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(embedding_dim),
        )
        self.subject_message = nn.Linear(embedding_dim, embedding_dim)
        self.object_message = nn.Linear(embedding_dim, embedding_dim)
        self.node_update = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(embedding_dim),
        )
        self.graph_projection = nn.Linear(2 * embedding_dim, embedding_dim)
        self.context_projection = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(self, retrieved_subgraph: Mapping[str, Any]) -> EncoderOutput:
        """Validate and encode one record returned by ``retrieve_subgraph``."""
        if not isinstance(retrieved_subgraph, Mapping):
            raise ValueError("retrieved_subgraph must be a mapping")
        query_context = retrieved_subgraph.get("query_context")
        if not isinstance(query_context, Mapping):
            raise ValueError("retrieved_subgraph query_context must be a mapping")
        question = query_context.get("question")
        question_embedding = self.embeddings.question(question)

        seed_values = query_context.get("seed_entity_ids")
        if not isinstance(seed_values, (list, tuple)) or not seed_values:
            raise ValueError("query_context seed_entity_ids must be a non-empty list")
        node_ids = {
            canonical_id(entity_id, "seed_entity_id") for entity_id in seed_values
        }

        entity_id_map = retrieved_subgraph.get("entity_id_map")
        if not isinstance(entity_id_map, Mapping):
            raise ValueError("retrieved_subgraph entity_id_map must be a mapping")
        node_ids.update(
            canonical_id(entity_id, "entity_id")
            for entity_id in entity_id_map.values()
        )

        raw_facts = retrieved_subgraph.get("facts")
        if not isinstance(raw_facts, (list, tuple)):
            raise ValueError("retrieved_subgraph facts must be a list")
        facts = [validate_fact(record, index) for index, record in enumerate(raw_facts)]
        for fact in facts:
            node_ids.update((fact["subject_id"], fact["object_id"]))
        ordered_node_ids = tuple(sorted(node_ids))
        node_index = {entity_id: index for index, entity_id in enumerate(ordered_node_ids)}

        base_node_embeddings = torch.stack(
            [self.embeddings.entity(entity_id) for entity_id in ordered_node_ids]
        )
        edge_embeddings, subject_indices, object_indices = self._encode_edges(
            facts, node_index
        )
        node_embeddings = self._aggregate_messages(
            base_node_embeddings,
            edge_embeddings,
            subject_indices,
            object_indices,
        )

        candidate_values = retrieved_subgraph.get("candidate_answer_ids")
        if not isinstance(candidate_values, (list, tuple)):
            raise ValueError("candidate_answer_ids must be a list")
        candidate_ids = tuple(
            canonical_id(value, "candidate_answer_id") for value in candidate_values
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_answer_ids must be unique")
        missing_candidates = sorted(set(candidate_ids) - set(node_index))
        if missing_candidates:
            raise ValueError(
                "candidate answers must be present in the retrieved subgraph: "
                f"{missing_candidates!r}"
            )
        candidate_indices = torch.tensor(
            [node_index[entity_id] for entity_id in candidate_ids],
            dtype=torch.long,
            device=node_embeddings.device,
        )
        candidate_embeddings = node_embeddings.index_select(0, candidate_indices)

        node_pool = node_embeddings.mean(dim=0)
        edge_pool = (
            edge_embeddings.mean(dim=0)
            if edge_embeddings.shape[0]
            else node_pool.new_zeros(self.embedding_dim)
        )
        graph_embedding = torch.tanh(
            self.graph_projection(torch.cat((node_pool, edge_pool), dim=-1))
        )
        context_embedding = torch.tanh(
            self.context_projection(
                torch.cat(
                    (self.dropout(question_embedding), graph_embedding), dim=-1
                )
            )
        )
        return EncoderOutput(
            question_embedding=question_embedding,
            graph_embedding=graph_embedding,
            context_embedding=context_embedding,
            node_ids=ordered_node_ids,
            node_embeddings=node_embeddings,
            candidate_ids=candidate_ids,
            candidate_embeddings=candidate_embeddings,
            edge_embeddings=edge_embeddings,
        )

    def _encode_edges(
        self, facts: list[dict[str, str]], node_index: Mapping[str, int]
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = self.embeddings.entity_embeddings.weight.device
        if not facts:
            empty_indices = torch.empty(0, dtype=torch.long, device=device)
            empty_edges = self.embeddings.entity_embeddings.weight.new_empty(
                (0, self.embedding_dim)
            )
            return empty_edges, empty_indices, empty_indices

        edge_inputs = [
            torch.cat(
                (
                    self.embeddings.entity(fact["subject_id"]),
                    self.embeddings.relation(fact["relation_id"]),
                    self.embeddings.entity(fact["object_id"]),
                    self.embeddings.time(fact["timestamp"]),
                )
            )
            for fact in facts
        ]
        edge_embeddings = self.edge_encoder(torch.stack(edge_inputs))
        subject_indices = torch.tensor(
            [node_index[fact["subject_id"]] for fact in facts],
            dtype=torch.long,
            device=device,
        )
        object_indices = torch.tensor(
            [node_index[fact["object_id"]] for fact in facts],
            dtype=torch.long,
            device=device,
        )
        return edge_embeddings, subject_indices, object_indices

    def _aggregate_messages(
        self,
        base_node_embeddings: Tensor,
        edge_embeddings: Tensor,
        subject_indices: Tensor,
        object_indices: Tensor,
    ) -> Tensor:
        if not edge_embeddings.shape[0]:
            messages = base_node_embeddings.new_zeros(base_node_embeddings.shape)
            degrees = base_node_embeddings.new_zeros(
                (base_node_embeddings.shape[0], 1)
            )
        else:
            subject_messages = self.subject_message(edge_embeddings)
            object_messages = self.object_message(edge_embeddings)
            messages = base_node_embeddings.new_zeros(base_node_embeddings.shape)
            messages.index_add_(0, subject_indices, subject_messages)
            messages.index_add_(0, object_indices, object_messages)
            degrees = base_node_embeddings.new_zeros(
                (base_node_embeddings.shape[0], 1)
            )
            ones = base_node_embeddings.new_ones((edge_embeddings.shape[0], 1))
            degrees.index_add_(0, subject_indices, ones)
            degrees.index_add_(0, object_indices, ones)

        mean_messages = messages / degrees.clamp_min(1)
        updates = self.node_update(
            torch.cat((base_node_embeddings, mean_messages), dim=-1)
        )
        return base_node_embeddings + updates
