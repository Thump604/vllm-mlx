# SPDX-License-Identifier: Apache-2.0
"""Architecture-faithful scalar Gemma target integration tests."""

from __future__ import annotations

import pytest
from mlx.utils import tree_map

mx = pytest.importorskip("mlx.core")

import vllm_mlx.specprefill_target_executor as target_executor
from mlx_lm.models.gemma4_text import Model as LmGemmaModel
from mlx_lm.models.gemma4_text import ModelArgs as LmGemmaArgs
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.gemma4.config import TextConfig as VlmGemmaArgs
from mlx_vlm.models.gemma4.language import LanguageModel as VlmGemmaModel
from vllm_mlx.specprefill_cache import (
    SparseCacheIdentity,
    SparseCacheState,
    SparsePolicyTuning,
)
from vllm_mlx.specprefill_gemma_cache import (
    GEMMA4_26B_A4B,
    GEMMA4_31B,
    GEMMA4_E2B,
    GemmaCacheBackend,
)
from vllm_mlx.specprefill_positions import (
    GEMMA4_A4B_TARGET,
    GEMMA4_DENSE_TARGET,
    PositionPhase,
)
from vllm_mlx.specprefill_target_executor import (
    SparseTargetPrefillError,
    SparseTargetPrefillSession,
    execute_sparse_target_prefill,
)
from vllm_mlx.specprefill_target_hooks import (
    TargetPositionHooks,
    TargetPositionSession,
)


_CASES = (
    (GEMMA4_E2B, GEMMA4_DENSE_TARGET, False),
    (GEMMA4_31B, GEMMA4_DENSE_TARGET, False),
    (GEMMA4_26B_A4B, GEMMA4_A4B_TARGET, True),
)


def _args(spec, backend, a4b):
    cls = LmGemmaArgs if backend is GemmaCacheBackend.MLX_LM else VlmGemmaArgs
    return cls(
        hidden_size=8,
        num_hidden_layers=spec.layer_count,
        intermediate_size=16,
        num_attention_heads=2,
        head_dim=4,
        global_head_dim=8,
        global_partial_rotary_factor=0.25,
        vocab_size=32,
        vocab_size_per_layer_input=32,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        num_kv_shared_layers=spec.layer_count - spec.owner_count,
        hidden_size_per_layer_input=0,
        sliding_window=spec.sliding_window,
        layer_types=list(spec.layer_types),
        final_logit_softcapping=None,
        use_double_wide_mlp=False,
        enable_moe_block=a4b,
        num_experts=2 if a4b else None,
        top_k_experts=1 if a4b else None,
        moe_intermediate_size=4 if a4b else None,
        attention_k_eq_v=spec is not GEMMA4_E2B,
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        },
    )


def _model(spec, backend, a4b=False):
    model = (
        LmGemmaModel(_args(spec, backend, a4b))
        if backend is GemmaCacheBackend.MLX_LM
        else VlmGemmaModel(_args(spec, backend, a4b))
    )
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    return model


def _logits(output):
    return output.logits if type(output) is LanguageModelOutput else output


def _state(positions):
    full_length = max(positions) + 1
    identity = SparseCacheIdentity.from_tokens(
        target_id="gemma@sha256:target",
        tokenizer_id="tokenizer@sha256:tokenizer",
        scorer_id="scorer@sha256:scorer",
        selector_version="hybrid-v1",
        tuning=SparsePolicyTuning(1.0, 0.0, 1, 1, 64),
        tokens=tuple(range(full_length)),
        selection_fingerprint="a" * 64,
    )
    return SparseCacheState.from_selection(
        identity, (tuple(positions),), (full_length,)
    )


def _dense(model, tokens, cache, step_size):
    logits = None
    for start in range(0, tokens.shape[1], step_size):
        logits = _logits(model(tokens[:, start : start + step_size], cache=cache))
        target_executor._eval_forward(logits, cache)
    return logits


def _active_kv(cache):
    active = []
    for entry in cache:
        length = min(entry.offset, getattr(entry, "max_size", entry.offset))
        active.append(
            (
                entry.keys[..., :length, :],
                entry.values[..., :length, :],
            )
        )
    return active


def _assert_close(actual, expected):
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=1e-2, atol=1e-2).item()


@pytest.mark.parametrize("backend", tuple(GemmaCacheBackend))
@pytest.mark.parametrize(("spec", "adapter", "a4b"), _CASES)
def test_keep_one_actual_layout_matches_bf16_logits_and_active_kv(
    spec, adapter, a4b, backend
):
    model = _model(spec, backend, a4b)
    # q_norm and proportional partial RoPE are real architecture modules here.
    full = next(
        layer.self_attn
        for layer in model.layers
        if layer.self_attn.layer_type == "full_attention"
    )
    assert full.q_norm is not None
    mx.eval(full.rope._freqs)
    assert mx.isinf(full.rope._freqs).any().item()
    assert (~mx.isinf(full.rope._freqs)).sum().item() == 1

    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    dense_cache = model.make_cache()
    dense_logits = _dense(model, tokens, dense_cache, 3)
    sparse_cache = model.make_cache()
    result = execute_sparse_target_prefill(
        model,
        tokens,
        sparse_cache,
        _state((0, 1, 2, 3)),
        adapter,
        step_size=3,
    )
    assert result.logits.dtype == mx.bfloat16
    _assert_close(result.logits, dense_logits)
    for sparse_pair, dense_pair in zip(
        _active_kv(sparse_cache), _active_kv(dense_cache), strict=True
    ):
        for sparse_value, dense_value in zip(sparse_pair, dense_pair, strict=True):
            _assert_close(sparse_value, dense_value)


@pytest.mark.parametrize("backend", tuple(GemmaCacheBackend))
def test_real_proportional_rope_noncontiguous_quantum_matches_one_token_oracle(
    backend,
):
    model = _model(GEMMA4_E2B, backend)
    adapter = GEMMA4_DENSE_TARGET
    positions = (0, 3, 4)
    tokens = mx.array([[1, 4, 5]], dtype=mx.int32)

    oracle_cache = model.make_cache()
    hooks = TargetPositionHooks.for_model(model, adapter)
    oracle_logits = None
    for physical, logical in enumerate(positions):
        session = TargetPositionSession(
            logical_positions=((logical,),),
            physical_starts=(physical,),
            phase=PositionPhase.SPARSE_PREFILL,
        )
        with hooks.session(session):
            oracle_logits = _logits(
                model(tokens[:, physical : physical + 1], cache=oracle_cache)
            )
            target_executor._eval_forward(oracle_logits, oracle_cache)

    sparse_cache = model.make_cache()
    result = execute_sparse_target_prefill(
        model,
        tokens,
        sparse_cache,
        _state(positions),
        adapter,
        step_size=2,
    )
    assert oracle_logits is not None
    assert result.logits.dtype == oracle_logits.dtype == mx.bfloat16
    _assert_close(result.logits, oracle_logits)
    for sparse_pair, oracle_pair in zip(
        _active_kv(sparse_cache), _active_kv(oracle_cache), strict=True
    ):
        for sparse_value, oracle_value in zip(sparse_pair, oracle_pair, strict=True):
            _assert_close(sparse_value, oracle_value)


@pytest.mark.parametrize("backend", tuple(GemmaCacheBackend))
def test_real_proportional_rope_hook_matches_independent_native_offsets(backend):
    model = _model(GEMMA4_E2B, backend)
    attention = next(
        layer.self_attn
        for layer in model.layers
        if layer.self_attn.layer_type == "full_attention"
    )
    rope = attention.rope
    positions = (0, 3, 7)
    x = mx.random.normal((1, 2, len(positions), attention.head_dim)).astype(
        mx.bfloat16
    )
    expected = mx.concatenate(
        [
            rope(x[:, :, index : index + 1], offset=logical)
            for index, logical in enumerate(positions)
        ],
        axis=2,
    )
    hooks = TargetPositionHooks.for_model(model, GEMMA4_DENSE_TARGET)
    session = TargetPositionSession(
        logical_positions=(positions,),
        physical_starts=(0,),
        phase=PositionPhase.SPARSE_PREFILL,
    )
    with hooks.session(session):
        actual = rope(x, offset=0)
        mx.eval(actual)
    assert actual.dtype == expected.dtype == mx.bfloat16
    _assert_close(actual, expected)


@pytest.mark.parametrize("backend", tuple(GemmaCacheBackend))
def test_oversized_concat_tail_normalizes_once_then_decodes(backend):
    model = _model(GEMMA4_E2B, backend)
    count = GEMMA4_E2B.sliding_window + 3
    cache = model.make_cache()
    execute_sparse_target_prefill(
        model,
        mx.arange(count, dtype=mx.int32)[None, :] % 31,
        cache,
        _state(tuple(range(count))),
        GEMMA4_DENSE_TARGET,
        step_size=257,
    )
    rotating = [entry for entry in cache if hasattr(entry, "max_size")]
    assert all(
        entry.offset == count
        and entry.keys.shape[2] == entry.max_size
        and entry._idx == entry.max_size
        for entry in rotating
    )
    logits = _logits(model(mx.array([[1]], dtype=mx.int32), cache=cache))
    target_executor._eval_forward(logits, cache)
    assert all(entry.offset == count + 1 for entry in cache)


@pytest.mark.parametrize("backend", tuple(GemmaCacheBackend))
@pytest.mark.parametrize("failure", ("lazy", "cancel"))
def test_saturated_failure_or_cancel_restores_then_safe_decode(
    backend, failure, monkeypatch
):
    model = _model(GEMMA4_E2B, backend)
    count = GEMMA4_E2B.sliding_window + 3
    cache = model.make_cache()
    session = SparseTargetPrefillSession(
        model,
        mx.arange(count, dtype=mx.int32)[None, :] % 31,
        cache,
        _state(tuple(range(count))),
        GEMMA4_DENSE_TARGET,
        step_size=257,
    )
    session.step()
    session.step()
    if failure == "lazy":
        real_eval = target_executor._eval_forward
        monkeypatch.setattr(
            target_executor,
            "_eval_forward",
            lambda *args: (_ for _ in ()).throw(RuntimeError("lazy failure")),
        )
        with pytest.raises(RuntimeError, match="lazy failure"):
            session.step()
        monkeypatch.setattr(target_executor, "_eval_forward", real_eval)
    else:
        session.cancel()
    assert all(entry.offset == 0 and entry.keys is None for entry in cache)
    logits = _logits(model(mx.array([[1]], dtype=mx.int32), cache=cache))
    target_executor._eval_forward(logits, cache)
    assert all(entry.offset == 1 for entry in cache)


@pytest.mark.parametrize("model_backend", tuple(GemmaCacheBackend))
def test_crossed_model_cache_backends_fail_before_forward(model_backend):
    cache_backend = (
        GemmaCacheBackend.MLX_VLM
        if model_backend is GemmaCacheBackend.MLX_LM
        else GemmaCacheBackend.MLX_LM
    )
    model = _model(GEMMA4_E2B, model_backend)
    crossed_cache = _model(GEMMA4_E2B, cache_backend).make_cache()
    with pytest.raises(SparseTargetPrefillError, match="topology"):
        SparseTargetPrefillSession(
            model,
            [[1]],
            crossed_cache,
            _state((0,)),
            GEMMA4_DENSE_TARGET,
            step_size=2,
        )


def test_pathological_one_token_chunking_and_batch_rows_fail_closed():
    model = _model(GEMMA4_E2B, GemmaCacheBackend.MLX_LM)
    with pytest.raises(SparseTargetPrefillError, match="chunk_size > 1"):
        SparseTargetPrefillSession(
            model,
            [[1, 2]],
            model.make_cache(),
            _state((0, 1)),
            GEMMA4_DENSE_TARGET,
            step_size=1,
        )
    state = SparseCacheState.from_selection(
        _state((0,)).identities[0], ((0,), (0,)), (1, 1)
    )
    with pytest.raises(SparseTargetPrefillError, match="scalar-only"):
        SparseTargetPrefillSession(
            model,
            [[1], [1]],
            model.make_cache(),
            state,
            GEMMA4_DENSE_TARGET,
            step_size=2,
        )
