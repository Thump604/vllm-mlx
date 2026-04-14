# SPDX-License-Identifier: Apache-2.0
"""Regression tests for tokenizer fallback helpers."""

from __future__ import annotations

import sys


def test_load_model_with_fallback_returns_success_path(monkeypatch):
    """The happy path must return the (model, tokenizer) tuple from mlx_lm.load."""
    from vllm_mlx.utils import tokenizer as tok

    expected_model = object()
    expected_tokenizer = object()

    monkeypatch.setattr(tok, "_needs_tokenizer_fallback", lambda model_name: False)

    class FakeMlxLm:
        @staticmethod
        def load(model_name, tokenizer_config=None):
            assert model_name == "mlx-community/Qwen3-0.6B-8bit"
            assert tokenizer_config == {"eos_token": "<|im_end|>"}
            return expected_model, expected_tokenizer

    monkeypatch.setitem(sys.modules, "mlx_lm", FakeMlxLm)

    model, tokenizer = tok.load_model_with_fallback(
        "mlx-community/Qwen3-0.6B-8bit",
        tokenizer_config={"eos_token": "<|im_end|>"},
    )

    assert model is expected_model
    assert tokenizer is expected_tokenizer
