"""Prompt rendering helpers for optional Qwen answer/comparison runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


SYSTEM_PROMPT = (
    "You are a precise temporal knowledge-graph reasoning assistant. "
    "Answer only from the supplied evidence. Do not invent entities or facts."
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_answer_prompt(
    question: str,
    evidence: Sequence[Mapping[str, Any]] | None = None,
    *,
    answer_ids: Sequence[str] | None = None,
) -> str:
    """Render a bounded answer prompt with optional retrieved evidence."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    evidence_rows = [] if evidence is None else list(evidence)
    if not isinstance(evidence_rows, list) or any(
        not isinstance(row, Mapping) for row in evidence_rows
    ):
        raise ValueError("evidence must be a sequence of mappings")

    lines = [
        SYSTEM_PROMPT,
        "",
        "Question:",
        question.strip(),
        "",
        "Evidence:",
        _json(evidence_rows),
    ]
    if answer_ids:
        if any(not isinstance(answer_id, str) or not answer_id.strip() for answer_id in answer_ids):
            raise ValueError("answer_ids must contain only non-empty strings")
        lines.extend(["", "Candidate answer IDs:", _json(list(answer_ids))])
    lines.extend(["", "Respond with a JSON object containing `answer_ids` as a JSON array."])
    return "\n".join(lines)


def build_explanation_prompt(
    question: str,
    evidence: Sequence[Mapping[str, Any]],
    answer_ids: Sequence[str],
) -> str:
    """Render a prompt that asks for an answer and evidence-linked explanation."""
    if not isinstance(answer_ids, (list, tuple)) or not answer_ids:
        raise ValueError("answer_ids must be a non-empty sequence")
    if any(not isinstance(answer_id, str) or not answer_id.strip() for answer_id in answer_ids):
        raise ValueError("answer_ids must contain only non-empty strings")
    base = build_answer_prompt(question, evidence, answer_ids=answer_ids)
    return (
        base
        + "\nAlso include `explanation` as one or more evidence-linked sentences "
        "that justify the answer."
    )
