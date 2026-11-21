"""Focused tests for the opt-in Qwen and QLoRA integration."""

import pytest

from src.llm.prompt_templates import (
    build_answer_prompt,
    build_explanation_prompt,
)
from src.llm.qwen_wrapper import QwenWrapper
from src.training.train_qlora import run_qlora_training


def test_prompt_templates_render_bounded_evidence_and_answer_ids():
    prompt = build_answer_prompt(
        "Which entity follows A?",
        [{"subject": "A", "relation": "visited", "object": "B"}],
        answer_ids=["B"],
    )
    assert "Which entity follows A?" in prompt
    assert '"object": "B"' in prompt
    assert "answer_ids" in prompt

    explanation = build_explanation_prompt(
        "Which entity follows A?",
        [{"subject": "A", "relation": "visited", "object": "B"}],
        ["B"],
    )
    assert "`explanation`" in explanation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_name": ""}, "model_name"),
        ({"load_in_4bit": "yes"}, "load_in_4bit"),
        ({"bnb_4bit_quant_type": "int8"}, "bnb_4bit_quant_type"),
        ({"bnb_4bit_compute_dtype": "float32"}, "bnb_4bit_compute_dtype"),
    ],
)
def test_qwen_wrapper_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        QwenWrapper(**kwargs)


def test_answer_prompt_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="question"):
        build_answer_prompt(" ")
    with pytest.raises(ValueError, match="evidence"):
        build_answer_prompt("Question?", ["not-a-mapping"])
    with pytest.raises(ValueError, match="answer_ids"):
        build_explanation_prompt("Question?", [], [])


def test_qlora_training_validates_records_before_optional_dependencies(tmp_path):
    with pytest.raises(ValueError, match="records"):
        run_qlora_training([], output_dir=tmp_path)
    with pytest.raises(ValueError, match="accepted_answer_ids"):
        run_qlora_training(
            [{"question": "Which entity?", "accepted_answer_ids": "B"}],
            output_dir=tmp_path,
        )

