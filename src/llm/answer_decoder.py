"""Candidate-scoring answer decoder shared by the baseline and later SELR."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.encoder.graph_encoder import EncoderOutput
from src.graph.graph_utils import canonical_id


@dataclass(frozen=True)
class CandidateScores:
    """Unnormalized answer scores aligned with the retrieved candidate IDs."""

    candidate_ids: tuple[str, ...]
    logits: Tensor


class CandidateAnswerDecoder(nn.Module):
    """Score retrieved final-hop entities conditioned on encoded question context.

    Training treats every accepted answer as positive and minimizes binary
    cross-entropy over the full candidate set. Inference returns every candidate
    whose sigmoid score meets the threshold, or the top candidate if none do.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise ValueError("embedding_dim must be a positive integer")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        resolved_hidden_dim = embedding_dim if hidden_dim is None else hidden_dim
        if (
            isinstance(resolved_hidden_dim, bool)
            or not isinstance(resolved_hidden_dim, int)
            or resolved_hidden_dim < 1
        ):
            raise ValueError("hidden_dim must be a positive integer")

        self.embedding_dim = embedding_dim
        self.scorer = nn.Sequential(
            nn.Linear(2 * embedding_dim, resolved_hidden_dim),
            nn.GELU(),
            nn.Linear(resolved_hidden_dim, 1),
        )

    def forward(self, encoded: EncoderOutput) -> CandidateScores:
        """Return one logit per candidate in the encoder's candidate order."""
        if not isinstance(encoded, EncoderOutput):
            raise ValueError("encoded must be an EncoderOutput")
        context = encoded.context_embedding
        candidates = encoded.candidate_embeddings
        if candidates.ndim != 2 or candidates.shape[0] != len(encoded.candidate_ids):
            raise ValueError(
                "candidate_embeddings must have one embedding per candidate ID "
                f"with width {self.embedding_dim}"
            )
        if len(set(encoded.candidate_ids)) != len(encoded.candidate_ids):
            raise ValueError("candidate IDs must be unique")
        logits = self.score_embeddings(context, candidates)
        return CandidateScores(candidate_ids=encoded.candidate_ids, logits=logits)

    def score_embeddings(self, context: Tensor, candidates: Tensor) -> Tensor:
        """Score one or batched contexts against aligned candidate embeddings."""
        if not isinstance(context, Tensor) or context.shape[-1:] != (self.embedding_dim,):
            raise ValueError(
                f"context embeddings must end in dimension {self.embedding_dim}"
            )
        if (
            not isinstance(candidates, Tensor)
            or candidates.shape[-1:] != (self.embedding_dim,)
            or candidates.ndim not in {2, 3}
        ):
            raise ValueError(
                "candidate embeddings must have shape [C, D] or [B, C, D]"
            )
        if context.ndim == 1 and candidates.ndim == 2:
            if context.shape != (self.embedding_dim,):
                raise ValueError("context embedding has an incompatible shape")
            expanded_context = context.unsqueeze(0).expand(candidates.shape[0], -1)
        elif context.ndim == 2 and candidates.ndim == 3:
            if context.shape[0] != candidates.shape[0]:
                raise ValueError("context and candidate batch sizes must match")
            expanded_context = context.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        else:
            raise ValueError("context and candidate embeddings must have matching ranks")
        if context.dtype != candidates.dtype or context.device != candidates.device:
            raise ValueError("context and candidate embeddings must share dtype and device")
        return self.scorer(
            torch.cat((expanded_context, candidates), dim=-1)
        ).squeeze(-1)

    def loss(
        self,
        scores: CandidateScores,
        accepted_answer_ids: Sequence[str],
    ) -> Tensor:
        """Compute multi-label BCE, rejecting incomplete retrieved answer sets."""
        if not isinstance(scores, CandidateScores):
            raise ValueError("scores must be CandidateScores")
        if scores.logits.ndim != 1 or scores.logits.shape[0] != len(scores.candidate_ids):
            raise ValueError("logits must contain one score per candidate ID")
        if not isinstance(accepted_answer_ids, (list, tuple)) or not accepted_answer_ids:
            raise ValueError("accepted_answer_ids must be a non-empty list")
        accepted = tuple(
            canonical_id(answer_id, "accepted_answer_id")
            for answer_id in accepted_answer_ids
        )
        if len(set(accepted)) != len(accepted):
            raise ValueError("accepted_answer_ids must be unique")
        candidate_ids = tuple(
            canonical_id(candidate_id, "candidate_answer_id")
            for candidate_id in scores.candidate_ids
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        accepted_set = set(accepted)
        missing_answers = sorted(accepted_set - set(candidate_ids))
        if missing_answers:
            raise ValueError(
                "retrieved candidates are missing accepted answers: "
                f"{missing_answers!r}"
            )
        if not candidate_ids:
            raise ValueError("cannot train answer decoder with no candidates")
        targets = scores.logits.new_tensor(
            [float(candidate_id in accepted_set) for candidate_id in candidate_ids]
        )
        return F.binary_cross_entropy_with_logits(scores.logits, targets)

    def decode(
        self,
        scores: CandidateScores,
        *,
        qa_id: str,
        model_variant: str = "baseline",
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Return a JSON-ready prediction record and candidate probabilities."""
        if not isinstance(scores, CandidateScores):
            raise ValueError("scores must be CandidateScores")
        if not isinstance(qa_id, str) or not qa_id.strip():
            raise ValueError("qa_id must be a non-empty string")
        if not isinstance(model_variant, str) or not model_variant.strip():
            raise ValueError("model_variant must be a non-empty string")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a number between 0 and 1")
        resolved_threshold = float(threshold)
        if not math.isfinite(resolved_threshold) or not 0 <= resolved_threshold <= 1:
            raise ValueError("threshold must be a number between 0 and 1")
        if scores.logits.ndim != 1 or scores.logits.shape[0] != len(scores.candidate_ids):
            raise ValueError("logits must contain one score per candidate ID")
        if len(set(scores.candidate_ids)) != len(scores.candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if not torch.isfinite(scores.logits).all():
            raise ValueError("candidate logits must all be finite")

        if not scores.candidate_ids:
            return {
                "qa_id": canonical_id(qa_id, "qa_id"),
                "model_variant": model_variant.strip(),
                "status": "no_candidates",
                "predicted_answer_ids": [],
                "answer_scores": [],
                "failure_reason": "retrieval returned no candidate answers",
            }

        probabilities = torch.sigmoid(scores.logits).detach().cpu().tolist()
        predicted_ids = [
            candidate_id
            for candidate_id, probability in zip(
                scores.candidate_ids, probabilities, strict=True
            )
            if probability >= resolved_threshold
        ]
        if not predicted_ids:
            predicted_ids = [
                scores.candidate_ids[int(torch.argmax(scores.logits).item())]
            ]
        return {
            "qa_id": canonical_id(qa_id, "qa_id"),
            "model_variant": model_variant.strip(),
            "status": "ok",
            "predicted_answer_ids": predicted_ids,
            "answer_scores": [
                {"answer_id": candidate_id, "score": probability}
                for candidate_id, probability in zip(
                    scores.candidate_ids, probabilities, strict=True
                )
            ],
        }
