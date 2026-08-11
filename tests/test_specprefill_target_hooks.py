# SPDX-License-Identifier: Apache-2.0
"""Synthetic, no-weight tests for request-local target RoPE transport."""

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import pytest
import vllm_mlx.specprefill_target_hooks as target_hooks

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from vllm_mlx.specprefill_positions import (
    GEMMA4_DENSE_TARGET,
    QWEN_DENSE_TARGET,
    PositionPhase,
)
from vllm_mlx.specprefill_target_hooks import (
    TargetPositionHookError,
    TargetPositionHooks,
    TargetPositionSession,
)


class Attention:
    def __init__(self, rope):
        self.rope = rope


class Model:
    def __init__(self, ropes):
        self.layers = [SimpleNamespace(self_attn=Attention(rope)) for rope in ropes]


class ModuleAttention(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.rope = rope
        self.proj = nn.Linear(8, 8, bias=False)


class ModuleDecoder(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.self_attn = ModuleAttention(rope)


class ModuleModel(nn.Module):
    def __init__(self, ropes):
        super().__init__()
        self.layers = [ModuleDecoder(rope) for rope in ropes]


class CustomScaledRoPE(nn.Module):
    """Small MLX-like partial RoPE with `_freqs` and amplitude mscale."""

    def __init__(self):
        super().__init__()
        self._dims = 4
        self._freqs = mx.array([1.0, 2.0])
        self.mscale = 0.5

    def __call__(self, x, offset=0):
        positions = mx.arange(offset, offset + x.shape[2], dtype=mx.float32)
        inv_freq = 1.0 / self._freqs
        angles = positions[None, None, :, None] * inv_freq[None, None, None, :]
        cos_a = mx.cos(angles).astype(x.dtype)
        sin_a = mx.sin(angles).astype(x.dtype)
        x_rot, x_pass = x[..., : self._dims] * self.mscale, x[..., self._dims :]
        first, second = x_rot[..., :2], x_rot[..., 2:]
        rotated = mx.concatenate(
            (first * cos_a - second * sin_a, first * sin_a + second * cos_a),
            axis=-1,
        )
        return mx.concatenate((rotated, x_pass), axis=-1)


class RecordingRope:
    """A native-only rope used to prove ContextVar isolation across threads."""

    dims = 8
    base = 10000.0

    def __init__(self):
        self.calls = []

    def __call__(self, x, offset=0):
        self.calls.append((x, offset))
        return ("native", x, offset)


def _session(logical_positions, physical_starts, phase=PositionPhase.SPARSE_PREFILL):
    return TargetPositionSession(
        logical_positions=logical_positions,
        physical_starts=physical_starts,
        phase=phase,
    )


def _assert_close(actual, expected):
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()


def _state_paths(value, prefix=()):
    if isinstance(value, dict):
        return tuple(
            path
            for key, child in value.items()
            for path in _state_paths(child, (*prefix, str(key)))
        )
    if isinstance(value, list):
        return tuple(
            path
            for index, child in enumerate(value)
            for path in _state_paths(child, (*prefix, str(index)))
        )
    return (prefix,)


@pytest.mark.parametrize("traditional", (False, True))
def test_keep_ratio_one_matches_native_rope_without_model_weights(traditional):
    rope = nn.RoPE(8, traditional=traditional, base=10000.0)
    model = Model([rope])
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
    x = mx.random.normal((1, 2, 4, 8))
    expected = rope(x, offset=0)

    with hooks.session(_session(((0, 1, 2, 3),), (0,))):
        actual = model.layers[0].self_attn.rope(x, offset=0)
        # The caller must realize any lazy target work before session release.
        _assert_close(actual, expected)

    assert hooks.active_session is None
    _assert_close(model.layers[0].self_attn.rope(x, offset=0), expected)


def test_position_acknowledgement_occurs_once_during_rope_model_call():
    first = nn.RoPE(8, traditional=False, base=10000.0)
    second = nn.RoPE(8, traditional=False, base=10000.0)
    model = Model([first, second])
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
    events = []

    class Ack:
        active = False

        def acknowledge(self, positions):
            assert self.active is True
            events.append(positions)

    ack = Ack()
    session = TargetPositionSession(
        logical_positions=((0, 3),),
        physical_starts=(0,),
        phase=PositionPhase.SPARSE_PREFILL,
        logical_position_ack=ack,
    )
    x = mx.random.normal((1, 2, 2, 8))

    with hooks.session(session):
        assert events == []
        ack.active = True
        mx.eval(
            model.layers[0].self_attn.rope(x, offset=0),
            model.layers[1].self_attn.rope(x, offset=0),
        )
        ack.active = False

    assert events == [(0, 3)]


def test_supported_native_mtp_rope_is_owned_by_same_request_local_hooks():
    backbone_rope = nn.RoPE(8, traditional=False, base=10000.0)
    mtp_rope = nn.RoPE(8, traditional=False, base=10000.0)
    model = Model([backbone_rope])
    model.mtp_capability = SimpleNamespace(supported=True)
    model.mtp = SimpleNamespace(layers=[SimpleNamespace(self_attn=Attention(mtp_rope))])

    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)

    assert tuple(wrapper[2] for wrapper in hooks._wrappers) == (
        backbone_rope,
        mtp_rope,
    )


def test_install_preserves_real_mlx_module_rope_identity_and_parameter_topology():
    rope = nn.RoPE(8, traditional=False, base=10000.0)
    model = ModuleModel([rope])
    before_rope = model.layers[0].self_attn.rope
    before_parameters = model.parameters()
    before_children = model.children()
    before_state_paths = _state_paths(before_parameters)

    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)

    assert model.layers[0].self_attn.rope is before_rope
    assert model.parameters().keys() == before_parameters.keys()
    assert model.children().keys() == before_children.keys()
    assert _state_paths(model.parameters()) == before_state_paths
    assert hooks._wrappers[0][2] is before_rope

    model.update(before_parameters)
    assert model.layers[0].self_attn.rope is before_rope
    assert _state_paths(model.parameters()) == before_state_paths


@pytest.mark.parametrize("traditional", (False, True))
def test_noncontiguous_positions_match_native_one_token_calls(traditional):
    rope = nn.RoPE(8, traditional=traditional, base=10000.0)
    model = Model([rope])
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
    x = mx.random.normal((1, 2, 4, 8))
    positions = (0, 3, 7, 8)
    expected = mx.concatenate(
        [
            rope(x[:, :, index : index + 1], offset=position)
            for index, position in enumerate(positions)
        ],
        axis=2,
    )

    with hooks.session(_session((positions,), (0,))):
        actual = model.layers[0].self_attn.rope(x, offset=0)
        _assert_close(actual, expected)


def test_custom_frequency_prescale_partial_rope_matches_native_and_keeps_tail():
    rope = CustomScaledRoPE()
    model = Model([rope])
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
    x = mx.random.normal((1, 2, 3, 8))
    positions = (2, 5, 11)
    expected = mx.concatenate(
        [
            rope(x[:, :, index : index + 1], offset=position)
            for index, position in enumerate(positions)
        ],
        axis=2,
    )

    with hooks.session(_session((positions,), (0,))):
        actual = model.layers[0].self_attn.rope(x, offset=0)
        _assert_close(actual, expected)
        _assert_close(actual[..., 4:], x[..., 4:])


def test_per_row_heterogeneous_positions_broadcast_as_batch_heads_sequence_dims():
    rope = nn.RoPE(8, traditional=False, base=10000.0)
    model = Model([rope])
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
    x = mx.random.normal((2, 2, 3, 8))
    logical = ((0, 2, 5), (10, 12, 17))
    expected_rows = []
    for row, row_positions in enumerate(logical):
        expected_rows.append(
            mx.concatenate(
                [
                    rope(x[row : row + 1, :, index : index + 1], offset=position)
                    for index, position in enumerate(row_positions)
                ],
                axis=2,
            )
        )
    expected = mx.concatenate(expected_rows, axis=0)

    with hooks.session(_session(logical, (0, 4))):
        actual = model.layers[0].self_attn.rope(x, offset=mx.array([0, 4]))
        _assert_close(actual, expected)


def test_physical_offset_selects_a_chunk_without_mutating_cache_coordinate():
    rope = nn.RoPE(8, traditional=False, base=10000.0)
    model = Model([rope])
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
    x = mx.random.normal((1, 2, 2, 8))
    positions = (5, 9)
    expected = mx.concatenate(
        [
            rope(x[:, :, index : index + 1], offset=position)
            for index, position in enumerate((5, 9))
        ],
        axis=2,
    )

    with hooks.session(_session((positions,), (13,))):
        actual = model.layers[0].self_attn.rope(x, offset=13)
        _assert_close(actual, expected)


def test_dispatch_has_no_per_layer_host_sync_for_batched_offsets():
    source = inspect.getsource(target_hooks._validate_scalar_offset)
    dispatcher = inspect.getsource(target_hooks._install_rope_dispatch)
    assert ".tolist(" not in source
    assert ".item(" not in source
    assert "mx.eval(" not in source
    assert ".tolist(" not in dispatcher
    assert ".item(" not in dispatcher
    assert "mx.eval(" not in dispatcher


def test_position_tensor_is_materialized_once_for_all_rope_layers(monkeypatch):
    first = nn.RoPE(8, traditional=False, base=10000.0)
    second = nn.RoPE(8, traditional=False, base=10000.0)
    model = Model([first, second])
    hooks = TargetPositionHooks.for_model(model, GEMMA4_DENSE_TARGET)
    real_array = target_hooks.mx.array
    calls = 0

    def recording_array(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_array(*args, **kwargs)

    monkeypatch.setattr(target_hooks.mx, "array", recording_array)
    # Per-RoPE execution metadata is captured at installation.  A dispatcher
    # must not rebuild device frequencies after every layer invocation.
    monkeypatch.setattr(
        target_hooks,
        "_inverse_frequencies",
        lambda *_args: pytest.fail("dispatcher recalculated RoPE frequencies"),
    )
    session = _session(((0, 3, 7),), (0,))
    assert calls == 1
    x = mx.random.normal((1, 2, 3, 8))
    with hooks.session(session):
        first_out = model.layers[0].self_attn.rope(x, offset=0)
        second_out = model.layers[1].self_attn.rope(x, offset=0)
        mx.eval(first_out, second_out)
    assert calls == 1


def test_gemma_shared_kv_rope_layers_observe_the_same_logical_session():
    first = nn.RoPE(8, traditional=False, base=10000.0)
    second = nn.RoPE(8, traditional=False, base=10000.0)
    model = Model([first, second])
    hooks = TargetPositionHooks.for_model(model, GEMMA4_DENSE_TARGET)
    x = mx.random.normal((1, 2, 3, 8))
    positions = (1, 4, 9)
    expected = mx.concatenate(
        [
            first(x[:, :, index : index + 1], offset=position)
            for index, position in enumerate(positions)
        ],
        axis=2,
    )

    with hooks.session(_session((positions,), (5,))):
        owner = model.layers[0].self_attn.rope(x, offset=5)
        # Gemma shared-KV followers receive the owner offset; both wrappers
        # must use the same request-local coordinates without changing it.
        follower = model.layers[1].self_attn.rope(x, offset=5)
        _assert_close(owner, expected)
        _assert_close(follower, expected)


def test_concurrent_context_sessions_are_isolated_and_foreign_thread_delegates():
    rope = nn.RoPE(8, traditional=False, base=10000.0)
    model = Model([rope])
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
    first = _session(((0,),), (0,))
    second = _session(((9,),), (0,))
    barrier = threading.Barrier(2)
    observed = []

    def worker(session):
        with hooks.session(session):
            barrier.wait(timeout=2)
            observed.append(hooks.active_session)

    threads = [
        threading.Thread(target=worker, args=(session,)) for session in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert set(observed) == {first, second}
    assert hooks.active_session is None

    foreign_rope = RecordingRope()
    foreign_model = Model([foreign_rope])
    foreign_hooks = TargetPositionHooks.for_model(foreign_model, QWEN_DENSE_TARGET)
    foreign_sessions = []
    foreign_result = []
    with hooks.session(first):
        thread = threading.Thread(
            target=lambda: (
                foreign_sessions.append(hooks.active_session),
                foreign_result.append(
                    foreign_model.layers[0].self_attn.rope("foreign", offset=17)
                ),
            )
        )
        thread.start()
        thread.join(timeout=2)
    assert foreign_sessions == [None]
    assert foreign_result == [("native", "foreign", 17)]
    assert foreign_rope.calls == [("foreign", 17)]
    assert foreign_hooks.active_session is None


def test_vector_offset_requires_a_matching_precomputed_batch_shape():
    rope = nn.RoPE(8, traditional=False, base=10000.0)
    model = Model([rope])
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
    x = mx.random.normal((1, 2, 2, 8))

    with hooks.session(_session(((0, 2), (10, 12)), (0, 7))):
        with pytest.raises(
            TargetPositionHookError, match="positions must have host shape"
        ):
            model.layers[0].self_attn.rope(x, offset=mx.array([0, 7]))


def test_nested_session_and_topology_tamper_fail_closed():
    rope = nn.RoPE(8, traditional=False, base=10000.0)
    model = Model([rope])
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
    session = _session(((0,),), (0,))
    with hooks.session(session):
        with pytest.raises(
            TargetPositionHookError, match="already has an active session"
        ):
            with hooks.session(session):
                pass

    model.layers[0].self_attn.rope = nn.RoPE(8, traditional=False, base=10000.0)
    with pytest.raises(TargetPositionHookError, match="topology was modified"):
        TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
    with pytest.raises(TargetPositionHookError, match="topology was modified"):
        with hooks.session(session):
            pass


def test_class_dispatch_tamper_fails_closed_without_replacing_rope_instance():
    rope = nn.RoPE(8, traditional=False, base=10000.0)
    model = Model([rope])
    hooks = TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
    installed_dispatch = type(rope).__call__
    type(rope).__call__ = lambda self, *args, **kwargs: args[0]
    try:
        with pytest.raises(
            TargetPositionHookError, match="class dispatcher was modified"
        ):
            TargetPositionHooks.for_model(model, QWEN_DENSE_TARGET)
        assert model.layers[0].self_attn.rope is rope
    finally:
        type(rope).__call__ = installed_dispatch
    hooks._verify_installed()


def test_partial_install_failure_rolls_back_prior_dispatch_registration():
    first = nn.RoPE(8, traditional=False, base=10000.0)

    class UnweakrefableRope:
        __slots__ = ()

        dims = 8
        base = 10000.0

        def __call__(self, x, offset=0):
            return x

    second = UnweakrefableRope()

    model = Model([first, second])
    with pytest.raises(TargetPositionHookError, match="must support weak references"):
        TargetPositionHooks(model, QWEN_DENSE_TARGET)
    assert model.layers[0].self_attn.rope is first
    assert target_hooks._ROPE_HOOKS.get(first) is None
