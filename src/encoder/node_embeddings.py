"""Vocabulary construction and learnable question/entity/relation/time vectors."""

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import torch
from torch import Tensor, nn

from src.graph.graph_utils import canonical_id, normalize_date
from src.graph.temporal_graph import TemporalGraph


_UNKNOWN_TOKEN = "<unk>"
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class EncoderVocabulary:
    """Transductive graph IDs and a question-token vocabulary fitted on train."""

    entity_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    question_tokens: tuple[str, ...]


def tokenize_question(question: str) -> tuple[str, ...]:
    """Split a question into lowercase word, number, and punctuation tokens."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return tuple(_TOKEN_PATTERN.findall(question.casefold()))


def build_encoder_vocabulary(
    graph: TemporalGraph,
    training_qa_records: Iterable[Mapping[str, Any]],
) -> EncoderVocabulary:
    """Use all graph IDs but fit lexical tokens exclusively from train questions."""
    question_tokens: set[str] = set()
    record_count = 0
    for index, record in enumerate(training_qa_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"training QA record {index} must be a mapping")
        if record.get("split") != "train":
            raise ValueError(
                f"training QA record {index} must have split='train'"
            )
        question_tokens.update(tokenize_question(record.get("question")))
        record_count += 1
    if record_count == 0:
        raise ValueError("training_qa_records must not be empty")

    return EncoderVocabulary(
        entity_ids=tuple(sorted(graph.entity_ids)),
        relation_ids=tuple(sorted(graph.relation_ids)),
        question_tokens=tuple(sorted(question_tokens)),
    )


class NodeRelationTimeEmbeddings(nn.Module):
    """Learnable entity/relation/token embeddings with a continuous date encoder."""

    def __init__(
        self,
        vocabulary: EncoderVocabulary,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise ValueError("embedding_dim must be a positive integer")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")

        self.embedding_dim = embedding_dim
        self.entity_index = self._make_index(vocabulary.entity_ids, "entity")
        self.relation_index = self._make_index(vocabulary.relation_ids, "relation")
        tokens = tuple(
            token.casefold()
            for token in vocabulary.question_tokens
            if token.casefold() != _UNKNOWN_TOKEN
        )
        if len(tokens) != len(vocabulary.question_tokens):
            raise ValueError(f"question_tokens must not contain {_UNKNOWN_TOKEN!r}")
        self.question_index = self._make_index(tokens, "question token")

        self.entity_embeddings = nn.Embedding(
            len(self.entity_index) + 1, embedding_dim
        )
        self.relation_embeddings = nn.Embedding(
            len(self.relation_index) + 1, embedding_dim
        )
        self.question_embeddings = nn.Embedding(
            len(self.question_index) + 1, embedding_dim
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @staticmethod
    def _make_index(values: Iterable[str], field: str) -> dict[str, int]:
        identifiers = tuple(canonical_id(value, field) for value in values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{field} vocabulary entries must be unique")
        return {value: index + 1 for index, value in enumerate(identifiers)}

    def entity(self, entity_id: str) -> Tensor:
        """Return the entity embedding, or the shared unknown vector for OOV IDs."""
        index = self.entity_index.get(canonical_id(entity_id, "entity_id"), 0)
        return self.entity_embeddings.weight[index]

    def relation(self, relation_id: str) -> Tensor:
        """Return the relation embedding, or the shared unknown vector for OOV IDs."""
        index = self.relation_index.get(
            canonical_id(relation_id, "relation_id"), 0
        )
        return self.relation_embeddings.weight[index]

    def question(self, text: str) -> Tensor:
        """Mean-pool token embeddings; unseen tokens use the unknown vector."""
        tokens = tokenize_question(text)
        indices = [self.question_index.get(token, 0) for token in tokens]
        token_indices = torch.tensor(
            indices,
            dtype=torch.long,
            device=self.question_embeddings.weight.device,
        )
        return self.question_embeddings(token_indices).mean(dim=0)

    def time(self, timestamp: str) -> Tensor:
        """Encode a calendar date with annual phase and a smoothly varying year."""
        normalized = normalize_date(timestamp)
        parsed_date = date.fromisoformat(normalized)
        annual_phase = (
            2.0 * math.pi * (parsed_date.timetuple().tm_yday - 1) / 365.2425
        )
        features = torch.tensor(
            [
                math.sin(annual_phase),
                math.cos(annual_phase),
                parsed_date.year / 100.0,
            ],
            dtype=self.time_encoder[0].weight.dtype,
            device=self.time_encoder[0].weight.device,
        )
        return self.time_encoder(features)
