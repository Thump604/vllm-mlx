# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Gemma 4 reasoning/control-token parsing."""

from vllm_mlx.api.utils import clean_output_text
from vllm_mlx.reasoning.gemma4_parser import (
    END_TOKEN,
    Gemma4ReasoningParser,
    START_TOKEN,
    TURN_END_TOKEN,
)


def test_extract_reasoning_strips_trailing_turn_and_channel_leak():
    parser = Gemma4ReasoningParser()

    reasoning, content = parser.extract_reasoning(
        "READY<turn|><turn|><turn|><turn|><turn|>//thought\n"
        "<channel|>READY<turn|><turn|><turn|>\n"
        "thought"
    )

    assert reasoning is None
    assert content == "READY"


def test_extract_reasoning_strips_empty_thought_tail_without_turn_token():
    parser = Gemma4ReasoningParser()

    reasoning, content = parser.extract_reasoning("READY//thought\n<channel|>READY\nthought")

    assert reasoning is None
    assert content == "READY"


def test_extract_reasoning_handles_empty_thought_channel_prefix():
    parser = Gemma4ReasoningParser()

    reasoning, content = parser.extract_reasoning(
        f"{START_TOKEN}{END_TOKEN}BLUE{TURN_END_TOKEN}"
    )

    assert reasoning is None
    assert content == "BLUE"


def test_extract_reasoning_handles_explicit_reasoning_block():
    parser = Gemma4ReasoningParser()

    reasoning, content = parser.extract_reasoning(
        f"{START_TOKEN}plan carefully{END_TOKEN}VISIBLE{TURN_END_TOKEN}"
    )

    assert reasoning == "plan carefully"
    assert content == "VISIBLE"


def test_extract_reasoning_handles_incomplete_reasoning_block():
    parser = Gemma4ReasoningParser()

    reasoning, content = parser.extract_reasoning(f"{START_TOKEN}step 1")

    assert reasoning == "step 1"
    assert content is None


def test_streaming_suppresses_partial_turn_fragments():
    parser = Gemma4ReasoningParser()
    previous_text = ""
    current_text = ""
    content_parts = []
    reasoning_parts = []

    for chunk in ("REA", "DY", "<tu", "rn|>", "<turn|>", "//thought\n", "<channel|>"):
        previous_text = current_text
        current_text += chunk
        delta = parser.extract_reasoning_streaming(previous_text, current_text, chunk)
        if delta and delta.content:
            content_parts.append(delta.content)
        if delta and delta.reasoning:
            reasoning_parts.append(delta.reasoning)

    assert "".join(content_parts) == "READY"
    assert reasoning_parts == []


def test_streaming_emits_reasoning_then_content():
    parser = Gemma4ReasoningParser()
    previous_text = ""
    current_text = ""
    reasoning_parts = []
    content_parts = []

    for chunk in (START_TOKEN, "step", END_TOKEN, "FINAL", TURN_END_TOKEN):
        previous_text = current_text
        current_text += chunk
        delta = parser.extract_reasoning_streaming(previous_text, current_text, chunk)
        if delta and delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta and delta.content:
            content_parts.append(delta.content)

    assert "".join(reasoning_parts) == "step"
    assert "".join(content_parts) == "FINAL"


def test_clean_output_text_strips_gemma_control_tokens():
    assert clean_output_text("READY<turn|><channel|><|think|>") == "READY"
