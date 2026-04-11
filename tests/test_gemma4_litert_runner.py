# SPDX-License-Identifier: Apache-2.0
"""Tests for the Gemma 4 LiteRT runtime runner wrapper."""

from __future__ import annotations

import numpy as np

from vllm_mlx.patches.gemma4_litert_runner import (
    Gemma4LiteRTMTPRunner,
    Gemma4LiteRTMTPState,
)


class _FakeSignatureRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        activations = kwargs["activations"]
        projected = activations[..., :1536] * 0.5
        logits = np.ones((1, 1, 32), dtype=np.float32) * activations.sum()
        return {
            "logits": logits,
            "projected_activations": projected.astype(np.float32),
        }


def test_build_activations_and_run_step():
    fake = _FakeSignatureRunner()
    runner = Gemma4LiteRTMTPRunner(
        signature_runner=fake,
        contract={
            "activations_formula": "concat(hidden_states, projected_activations)",
            "model_hidden_size": 1536,
            "projected_activations_size": 1536,
            "param_tensor_shape": [1, 1, 1, 7],
        },
        input_details={
            "kv_cache_k_13": {"shape": [1, 1, 8, 256]},
            "kv_cache_k_14": {"shape": [1, 1, 8, 512]},
            "kv_cache_v_13": {"shape": [1, 1, 256, 8]},
            "kv_cache_v_14": {"shape": [1, 1, 512, 8]},
            "mask": {"shape": [1, 1, 1, 8]},
        },
    )

    state = runner.make_cache()
    hidden = np.ones((1, 1, 1536), dtype=np.float32)
    result = runner.run_step(hidden, input_pos=3, state=state)

    assert fake.calls, "runner must invoke the signature runner"
    activations = fake.calls[0]["activations"]
    assert activations.shape == (1, 1, 3072)
    assert np.allclose(activations[..., :1536], 1.0)
    assert np.allclose(activations[..., 1536:], 0.0)
    assert result["logits"].shape == (1, 1, 32)
    assert state.projected_activations.shape == (1, 1, 1536)
    assert np.allclose(state.projected_activations, 0.5)


def test_build_activations_rejects_bad_hidden_width():
    runner = Gemma4LiteRTMTPRunner(
        signature_runner=_FakeSignatureRunner(),
        contract={
            "activations_formula": "concat(hidden_states, projected_activations)",
            "model_hidden_size": 1536,
            "projected_activations_size": 1536,
            "param_tensor_shape": [1, 1, 1, 7],
        },
        input_details={
            "kv_cache_k_13": {"shape": [1, 1, 8, 256]},
            "kv_cache_k_14": {"shape": [1, 1, 8, 512]},
            "kv_cache_v_13": {"shape": [1, 1, 256, 8]},
            "kv_cache_v_14": {"shape": [1, 1, 512, 8]},
            "mask": {"shape": [1, 1, 1, 8]},
        },
    )
    state = Gemma4LiteRTMTPState(
        projected_activations=np.zeros((1, 1, 1536), dtype=np.float32),
        kv_cache_k_13=np.zeros((1, 1, 8, 256), dtype=np.int8),
        kv_cache_k_14=np.zeros((1, 1, 8, 512), dtype=np.int8),
        kv_cache_v_13=np.zeros((1, 1, 256, 8), dtype=np.int8),
        kv_cache_v_14=np.zeros((1, 1, 512, 8), dtype=np.int8),
    )

    hidden = np.zeros((1, 1, 256), dtype=np.float32)
    try:
        runner.build_activations(hidden, state)
    except ValueError as exc:
        assert "Expected hidden size 1536" in str(exc)
    else:
        raise AssertionError("build_activations must reject the wrong hidden size")


def test_state_copy_is_deep():
    state = Gemma4LiteRTMTPState(
        projected_activations=np.ones((1, 1, 4), dtype=np.float32),
        kv_cache_k_13=np.ones((1, 1, 2, 2), dtype=np.int8),
        kv_cache_k_14=np.ones((1, 1, 2, 4), dtype=np.int8),
        kv_cache_v_13=np.ones((1, 1, 2, 2), dtype=np.int8),
        kv_cache_v_14=np.ones((1, 1, 4, 2), dtype=np.int8),
    )

    copied = state.copy()
    copied.projected_activations[0, 0, 0] = 99.0
    copied.kv_cache_k_13[0, 0, 0, 0] = 7

    assert state.projected_activations[0, 0, 0] == 1.0
    assert state.kv_cache_k_13[0, 0, 0, 0] == 1
