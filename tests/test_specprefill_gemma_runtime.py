# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only Gemma prepared runtime and scheduler bridge contracts."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

import vllm_mlx.specprefill_runtime as runtime
from mlx.utils import tree_map
from mlx_vlm.models.gemma4.config import TextConfig as VlmGemmaArgs
from mlx_vlm.models.gemma4.language import LanguageModel as VlmGemmaModel
from mlx_vlm.models.cache import KVCache, RotatingKVCache
from vllm_mlx.cooperative_specprefill import CooperativeSpecPrefillConfig
from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.model_registry import _specprefill_capabilities
from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler, MLLMSchedulerConfig
from vllm_mlx.specprefill import SpecPrefillCoverage, SpecPrefillPolicy
from vllm_mlx.specprefill_gemma_cache import (
    FULL,
    GEMMA4_26B_A4B,
    GEMMA4_31B,
    GEMMA4_E2B,
)
from vllm_mlx.specprefill_positions import (
    GEMMA4_A4B_TARGET,
    GEMMA4_DENSE_TARGET,
)
from vllm_mlx.specprefill_profiles import (
    SpecPrefillCalibration,
    SpecPrefillCell,
    SpecPrefillEngine,
    SpecPrefillProfile,
    SpecPrefillProfileKey,
    SpecPrefillProfileRegistry,
    SpecPrefillProfileTier,
    SpecPrefillTuning,
)
from vllm_mlx.specprefill_scorer_session import (
    ScorerSessionPhase,
    ScorerSessionProgress,
)
from vllm_mlx.specprefill_selection import (
    SPECPREFILL_SELECTOR_VERSION,
    RotatingTailRequirement,
)
from vllm_mlx.scheduler import SchedulerConfig


def _artifact(tmp_path, name, contents):
    path = tmp_path / name
    path.write_bytes(contents)
    return str(path), hashlib.sha256(contents).hexdigest()


def _registry(
    target_hash,
    scorer_hash,
    *,
    artifact=GEMMA4_E2B,
    adapter=GEMMA4_DENSE_TARGET,
):
    tuning = SpecPrefillTuning(0.5, 0.0, 0, 1, 2)
    key = SpecPrefillProfileKey(
        target_artifact_id=artifact.artifact_id,
        target_artifact_hash=target_hash,
        tokenizer_artifact_hash="b" * 64,
        scorer_artifact_id="gemma-scorer",
        scorer_artifact_hash=scorer_hash,
        adapter_id=adapter.adapter_id,
        engine=SpecPrefillEngine.CONTINUOUS_BATCHING,
        cell=SpecPrefillCell.SPARSE_ONLY,
    )
    calibration = SpecPrefillCalibration(
        selector_version=SPECPREFILL_SELECTOR_VERSION,
        tuning=tuning,
        crossover_tokens=1,
        max_context_tokens=128,
        residency_limit_bytes=1024,
        min_ttft_improvement_pct=1.0,
        max_total_latency_regression_pct=1.0,
        max_decode_throughput_regression_pct=1.0,
        required_context_tokens=(1,),
        required_concurrency_levels=(1, 2, 4, 8),
        max_p95_inter_token_latency_regression_pct=1.0,
        min_prefill_heavy_throughput_improvement_pct=1.0,
    )
    profile = SpecPrefillProfile(
        key,
        SpecPrefillProfileTier.DIAGNOSTIC,
        calibration,
    )
    return SpecPrefillProfileRegistry((profile,)), key, tuning


def _target():
    text = SimpleNamespace(
        config=SimpleNamespace(
            model_type="gemma4_text",
            layer_types=list(GEMMA4_E2B.layer_types),
            num_hidden_layers=GEMMA4_E2B.layer_count,
            num_kv_shared_layers=(GEMMA4_E2B.layer_count - GEMMA4_E2B.owner_count),
            sliding_window=GEMMA4_E2B.sliding_window,
        ),
        model=SimpleNamespace(previous_kvs=list(GEMMA4_E2B.previous_kvs)),
    )
    return SimpleNamespace(language_model=text, vision_tower=object()), text


def _scalar_cache(artifact=GEMMA4_E2B):
    cache = []
    for layer_type in artifact.layer_types[: artifact.owner_count]:
        cache.append(
            KVCache()
            if layer_type == FULL
            else RotatingKVCache(artifact.sliding_window, keep=0)
        )
    return cache


def _real_target(artifact, *, a4b=False):
    args = VlmGemmaArgs(
        hidden_size=8,
        num_hidden_layers=artifact.layer_count,
        intermediate_size=16,
        num_attention_heads=2,
        head_dim=4,
        global_head_dim=8,
        global_partial_rotary_factor=0.25,
        vocab_size=32,
        vocab_size_per_layer_input=32,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        num_kv_shared_layers=artifact.layer_count - artifact.owner_count,
        hidden_size_per_layer_input=0,
        sliding_window=artifact.sliding_window,
        layer_types=list(artifact.layer_types),
        final_logit_softcapping=None,
        use_double_wide_mlp=False,
        enable_moe_block=a4b,
        num_experts=2 if a4b else None,
        top_k_experts=1 if a4b else None,
        moe_intermediate_size=4 if a4b else None,
        attention_k_eq_v=artifact is not GEMMA4_E2B,
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
    model = VlmGemmaModel(args)
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    return model


class _Processor:
    eos_token_id = 31
    clean_up_tokenization_spaces = False

    def encode(self, _prompt):
        return [1, 2, 3, 4]

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


class _OneQuantumScorerSession:
    phase = ScorerSessionPhase.PREFILL
    importance = mx.array([0.0, 0.25, 0.5, 1.0])

    def step(self):
        self.phase = ScorerSessionPhase.COMPLETE
        return ScorerSessionProgress(
            ScorerSessionPhase.COMPLETE,
            4,
            1,
            1,
            True,
        )

    def cancel(self):
        pass


def _real_prepare(
    monkeypatch,
    tmp_path,
    artifact,
    adapter,
    *,
    a4b=False,
    cache_factory=None,
):
    scorer_path, scorer_hash = _artifact(
        tmp_path, f"{artifact.artifact_id}-scorer", b"synthetic scorer"
    )
    target_path, target_hash = _artifact(
        tmp_path, f"{artifact.artifact_id}-target", b"synthetic target"
    )
    registry, key, tuning = _registry(
        target_hash,
        scorer_hash,
        artifact=artifact,
        adapter=adapter,
    )
    target = _real_target(artifact, a4b=a4b)
    processor = _Processor()
    monkeypatch.setattr(runtime.SpecPrefillScorer, "for_model", lambda _model: object())
    monkeypatch.setattr(
        runtime,
        "SpecPrefillScorerSession",
        lambda *_a, **_k: _OneQuantumScorerSession(),
    )
    factory = cache_factory or (lambda model: model.make_cache())
    prepare = runtime.build_gemma_cb_specprefill_prepare(
        scorer_artifact_path=scorer_path,
        scorer_artifact_hash=scorer_hash,
        target_artifact_path=target_path,
        target_artifact_hash=target_hash,
        tokenizer_artifact_hash=key.tokenizer_artifact_hash,
        gemma_artifact=artifact,
        profile_registry=registry,
        profile_key=key,
        calibrated_tuning=tuning,
        estimated_residency_bytes=64,
        target_identity_attestor=lambda actual_target, actual_processor: (
            runtime.TargetProcessorAttestation(
                actual_target,
                actual_processor,
                target_hash,
                key.tokenizer_artifact_hash,
            )
        ),
        scorer_loader=lambda *_args: runtime.LoadedSpecPrefillScorer(
            object(), lambda: None
        ),
        target_cache_factory=factory,
        target_prefill_step_size=2,
    )
    return prepare, target, processor


def _cooperative_config(prepared):
    execution = prepared.gemma_batch_config.execution_config
    return CooperativeSpecPrefillConfig(
        target_id=execution.target_id,
        tokenizer_id=execution.tokenizer_id,
        scorer_id=execution.scorer_id,
        tuning=execution.tuning,
        rotating_tail_requirement=RotatingTailRequirement(
            prepared.gemma_batch_config.attestation.artifact.sliding_window
        ),
    )


def _scheduler_config(prepared, *, session_factory=None):
    return MLLMSchedulerConfig(
        enable_prefix_cache=False,
        specprefill_enabled=True,
        specprefill_profile_registry=prepared.profile_registry,
        specprefill_profile_key=prepared.profile_key,
        specprefill_estimated_residency_bytes=64,
        specprefill_session_factory=session_factory or prepared.session_factory,
        specprefill_target_forward_context=prepared.target_forward_context,
        specprefill_cache_capability=prepared.cache_capability,
        specprefill_gemma_batch_config=prepared.gemma_batch_config,
        specprefill_diagnostic=True,
        specprefill_advertisable=False,
    )


@pytest.mark.parametrize(
    ("artifact", "adapter", "a4b"),
    (
        (GEMMA4_E2B, GEMMA4_DENSE_TARGET, False),
        (GEMMA4_31B, GEMMA4_DENSE_TARGET, False),
        (GEMMA4_26B_A4B, GEMMA4_A4B_TARGET, True),
    ),
)
def test_real_gemma_layouts_probe_and_create_fresh_scalar_target_session(
    monkeypatch, tmp_path, artifact, adapter, a4b
):
    prepare, target, processor = _real_prepare(
        monkeypatch, tmp_path, artifact, adapter, a4b=a4b
    )
    prepared = prepare(target, processor)
    admission = prepared.session_factory(
        SimpleNamespace(request_id=artifact.artifact_id),
        (1, 2, 3, 4),
        _cooperative_config(prepared),
    )

    progress = admission.session.step()

    assert progress.quantum_committed is True
    assert admission.session.phase.value == "sparse_target_prefill"
    assert admission.session.outcome.value == "active"
    target_session = admission.session._target_session
    assert target_session is not None
    assert target_session.model is target
    assert all(entry.offset == 0 for entry in target_session.cache)
    assert prepared.cache_capability.layout == artifact.artifact_id
    prepared.cleanup()


def test_second_fresh_gemma_cache_topology_drift_fails_before_target_quantum(
    monkeypatch, tmp_path
):
    calls = 0

    def drifting_cache(model):
        nonlocal calls
        calls += 1
        cache = model.make_cache()
        return cache if calls == 1 else cache[:-1]

    prepare, target, processor = _real_prepare(
        monkeypatch,
        tmp_path,
        GEMMA4_E2B,
        GEMMA4_DENSE_TARGET,
        cache_factory=drifting_cache,
    )
    prepared = prepare(target, processor)
    admission = prepared.session_factory(
        SimpleNamespace(request_id="drift"),
        (1, 2, 3, 4),
        _cooperative_config(prepared),
    )

    admission.session.step()

    assert calls == 2
    assert admission.session.outcome.value == "fallback"
    assert admission.session.fallback_reason == "target_setup_failed"
    prepared.cleanup()


def test_real_gemma_scheduler_runs_one_quantum_to_adoption_then_decode_first(
    monkeypatch, tmp_path
):
    prepare, target, processor = _real_prepare(
        monkeypatch, tmp_path, GEMMA4_E2B, GEMMA4_DENSE_TARGET
    )
    prepared = prepare(target, processor)
    events = []
    admissions = []

    def tracking_factory(request, tokens, config):
        admission = prepared.session_factory(request, tokens, config)
        admissions.append(admission)
        original_step = admission.session.step

        def tracked_step():
            events.append("quantum")
            return original_step()

        admission.session.step = tracked_step
        return admission

    scheduler = MLLMScheduler(
        target,
        processor,
        _scheduler_config(prepared, session_factory=tracking_factory),
    )
    scheduler._ensure_batch_generator()
    assert scheduler.batch_generator is not None
    original_next = scheduler.batch_generator.next

    def tracked_next():
        events.append("decode")
        return original_next()

    scheduler.batch_generator.next = tracked_next
    scheduler.add_request(
        "prompt",
        request_id="integrated",
        max_tokens=2,
        temperature=0.0,
        specprefill_policy="sparse",
        specprefill_coverage="selective",
    )

    first = scheduler.step()
    second = scheduler.step()
    third = scheduler.step()

    assert events[:2] == ["decode", "quantum"]
    assert first.specprefill_progress["integrated"].scorer_quanta == 1
    assert first.specprefill_progress["integrated"].target_quanta == 0
    assert second.specprefill_progress["integrated"].target_quanta == 1, admissions[
        0
    ].session.failure
    assert third.specprefill_progress["integrated"].target_quanta == 2
    assert scheduler.request_id_to_uid["integrated"] == 0
    active = scheduler.batch_generator.active_batch
    assert active is not None
    assert active.gemma_sparse_config is prepared.gemma_batch_config
    assert active.sparse_row_states[0].logical_positions == (0, 1, 2, 3)
    assert active.sparse_row_states[0].next_logical_position == 4

    before = len(events)
    decode_output = scheduler.step()

    assert events[before:] == ["decode"]
    assert len(decode_output.outputs) == 1
    assert decode_output.outputs[0].request_id == "integrated"
    assert active.sparse_row_states[0].next_logical_position == 5
    assert active.sparse_row_states[0].physical_valid_length == 5
    status = scheduler.get_stats()["specprefill"]
    assert status == {
        "enabled": False,
        "advertisable": False,
        "diagnostic": True,
    }
    assert _specprefill_capabilities({"state": "diagnostic", **status}) == []
    prepared.cleanup()


def test_batched_engine_start_propagates_identical_gemma_config_to_generator(
    monkeypatch, tmp_path
):
    import vllm_mlx.mllm_scheduler as scheduler_module

    prepare, target, processor = _real_prepare(
        monkeypatch, tmp_path, GEMMA4_E2B, GEMMA4_DENSE_TARGET
    )
    captured = []
    real_scheduler = scheduler_module.MLLMScheduler

    class CapturingScheduler:
        def __init__(self, model, processor, config):
            self.inner = real_scheduler(model, processor, config)
            self.inner._ensure_batch_generator()
            self.config = config
            self.batch_generator = self.inner.batch_generator
            captured.append(self)

        async def start(self):
            pass

        async def stop(self):
            if self.inner.batch_generator is not None:
                self.inner.batch_generator.close()
                self.inner.batch_generator = None

        def get_stats(self):
            return self.inner.get_stats()

    monkeypatch.setattr(scheduler_module, "MLLMScheduler", CapturingScheduler)
    engine = BatchedEngine(
        "gemma-diagnostic",
        scheduler_config=SchedulerConfig(
            enable_prefix_cache=False,
            specprefill_enabled=True,
            specprefill_prepare=prepare,
        ),
        force_mllm=True,
    )
    engine._model = target
    engine._processor = processor

    asyncio.run(engine._start_mllm())

    assert len(captured) == 1
    wired = captured[0]
    prepared = engine._prepared_specprefill_runtime
    assert prepared is not None
    assert wired.config.specprefill_gemma_batch_config is prepared.gemma_batch_config
    assert wired.batch_generator._expected_gemma_sparse_config is (
        prepared.gemma_batch_config
    )
    assert engine.get_stats()["specprefill"] == {
        "enabled": False,
        "advertisable": False,
        "diagnostic": True,
    }
    assert (
        _specprefill_capabilities(
            {"state": "diagnostic", **engine.get_stats()["specprefill"]}
        )
        == []
    )
    assert asyncio.run(engine._cleanup_mllm_runtime_ownership()) == []


def _prepare(monkeypatch, tmp_path, *, cache_factory=_scalar_cache):
    scorer_path, scorer_hash = _artifact(
        tmp_path, "scorer.safetensors", b"gemma scorer"
    )
    target_path, target_hash = _artifact(
        tmp_path, "target.safetensors", b"gemma target"
    )
    registry, key, tuning = _registry(target_hash, scorer_hash)
    target, text = _target()
    processor = object()
    events = []

    class FakeScorerSession:
        phase = ScorerSessionPhase.PREFILL

        def cancel(self):
            events.append("scorer_cancel")

    class FakeHooks:
        @contextmanager
        def session_for_plan(self, plan):
            events.append(("plan", plan.logical_positions))
            yield

    class FakeTargetSession:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def step(self):
            raise AssertionError("not stepped in runtime preparation test")

        def cancel(self):
            events.append("target_cancel")

    monkeypatch.setattr(runtime.SpecPrefillScorer, "for_model", lambda _model: object())
    monkeypatch.setattr(
        runtime.TargetPositionHooks,
        "for_model",
        lambda actual, adapter: (
            events.append(("hooks", actual, adapter)) or FakeHooks()
        ),
    )
    monkeypatch.setattr(
        runtime, "resolve_target_position_adapter", lambda _model: GEMMA4_DENSE_TARGET
    )
    monkeypatch.setattr(
        runtime, "SpecPrefillScorerSession", lambda *_a, **_k: FakeScorerSession()
    )
    monkeypatch.setattr(runtime, "SparseTargetPrefillSession", FakeTargetSession)

    prepare = runtime.build_gemma_cb_specprefill_prepare(
        scorer_artifact_path=scorer_path,
        scorer_artifact_hash=scorer_hash,
        target_artifact_path=target_path,
        target_artifact_hash=target_hash,
        tokenizer_artifact_hash=key.tokenizer_artifact_hash,
        gemma_artifact=GEMMA4_E2B,
        profile_registry=registry,
        profile_key=key,
        calibrated_tuning=tuning,
        estimated_residency_bytes=64,
        target_identity_attestor=lambda actual_target, actual_processor: (
            runtime.TargetProcessorAttestation(
                actual_target,
                actual_processor,
                target_hash,
                key.tokenizer_artifact_hash,
            )
        ),
        scorer_loader=lambda *_args: runtime.LoadedSpecPrefillScorer(
            object(), lambda: events.append("cleanup")
        ),
        target_cache_factory=lambda _model: cache_factory(),
        target_prefill_step_size=2,
    )
    return prepare(target, processor), target, text, processor, events


def test_diagnostic_gemma_runtime_prepares_real_cache_and_scheduler_bridge(
    monkeypatch, tmp_path
):
    prepared, target, text, processor, _events = _prepare(monkeypatch, tmp_path)

    assert prepared.target_model is target
    assert prepared.processor is processor
    assert prepared.diagnostic is True
    assert prepared.advertisable is False
    assert prepared.cache_capability.layout == GEMMA4_E2B.artifact_id
    assert prepared.cache_capability.backend == "mlx_vlm"
    assert prepared.cache_capability.rotating is True
    assert prepared.gemma_batch_config.attestation.text_model is text

    config = MLLMSchedulerConfig(
        enable_prefix_cache=False,
        specprefill_enabled=True,
        specprefill_profile_registry=prepared.profile_registry,
        specprefill_profile_key=prepared.profile_key,
        specprefill_estimated_residency_bytes=64,
        specprefill_session_factory=prepared.session_factory,
        specprefill_target_forward_context=prepared.target_forward_context,
        specprefill_cache_capability=prepared.cache_capability,
        specprefill_gemma_batch_config=prepared.gemma_batch_config,
        specprefill_diagnostic=True,
        specprefill_advertisable=False,
    )
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = config
    request = MLLMRequest(
        request_id="gemma-diagnostic",
        prompt="prompt",
        prompt_token_ids=(1, 2, 3, 4),
        specprefill_policy=SpecPrefillPolicy.SPARSE,
        specprefill_coverage=SpecPrefillCoverage.SELECTIVE,
    )
    cooperative, reason = scheduler._cooperative_config_for(request)

    assert reason is None
    assert cooperative is not None
    assert cooperative.rotating_tail_requirement == RotatingTailRequirement(
        GEMMA4_E2B.sliding_window
    )
    admission = prepared.session_factory(request, request.prompt_token_ids, cooperative)
    assert admission.session.config is cooperative
    scheduler.model = target
    scheduler.processor = processor
    scheduler.mm_processor = None
    scheduler.stop_tokens = set()
    scheduler.batch_generator = None
    scheduler._ensure_batch_generator()
    assert scheduler.batch_generator._expected_gemma_sparse_config is (
        prepared.gemma_batch_config
    )
    assert scheduler.batch_generator._expected_sparse_execution_config == (
        prepared.gemma_batch_config.execution_config
    )


def test_gemma_runtime_fails_closed_for_forged_bytes_and_cross_backend(
    monkeypatch, tmp_path
):
    prepared, *_ = _prepare(monkeypatch, tmp_path)
    prepared.cleanup()

    scorer_path, scorer_hash = _artifact(tmp_path, "scorer2", b"scorer2")
    target_path, target_hash = _artifact(tmp_path, "target2", b"target2")
    registry, key, tuning = _registry(target_hash, scorer_hash)
    target, _text = _target()
    processor = object()
    base = dict(
        scorer_artifact_path=scorer_path,
        scorer_artifact_hash=scorer_hash,
        target_artifact_path=target_path,
        target_artifact_hash=target_hash,
        tokenizer_artifact_hash=key.tokenizer_artifact_hash,
        gemma_artifact=GEMMA4_E2B,
        profile_registry=registry,
        profile_key=key,
        calibrated_tuning=tuning,
        estimated_residency_bytes=64,
        scorer_loader=lambda *_args: runtime.LoadedSpecPrefillScorer(
            object(), lambda: None
        ),
    )
    wrong_identity = lambda actual_target, actual_processor: (
        runtime.TargetProcessorAttestation(
            actual_target,
            actual_processor,
            "a" * 64,
            key.tokenizer_artifact_hash,
        )
    )
    prepare = runtime.build_gemma_cb_specprefill_prepare(
        **base,
        target_identity_attestor=wrong_identity,
        target_cache_factory=lambda _model: _scalar_cache(),
    )
    with pytest.raises(Exception, match="artifact bytes"):
        prepare(target, processor)

    from mlx_lm.models.cache import KVCache as LmKVCache

    prepare = runtime.build_gemma_cb_specprefill_prepare(
        **base,
        target_identity_attestor=lambda actual_target, actual_processor: (
            runtime.TargetProcessorAttestation(
                actual_target,
                actual_processor,
                target_hash,
                key.tokenizer_artifact_hash,
            )
        ),
        target_cache_factory=lambda _model: [
            LmKVCache() for _ in range(GEMMA4_E2B.owner_count)
        ],
    )
    with pytest.raises(Exception, match="topology|backend|cache"):
        prepare(target, processor)


@pytest.mark.parametrize(
    "override",
    (
        {"enable_prefix_cache": True},
        {"enable_prefix_cache": False, "enable_mtp": True},
        {"enable_prefix_cache": False, "max_kv_size": 16},
    ),
)
def test_gemma_scheduler_compositions_fail_closed(monkeypatch, tmp_path, override):
    prepared, *_ = _prepare(monkeypatch, tmp_path)
    kwargs = dict(
        enable_prefix_cache=False,
        specprefill_enabled=True,
        specprefill_profile_registry=prepared.profile_registry,
        specprefill_profile_key=prepared.profile_key,
        specprefill_estimated_residency_bytes=64,
        specprefill_session_factory=prepared.session_factory,
        specprefill_target_forward_context=prepared.target_forward_context,
        specprefill_cache_capability=prepared.cache_capability,
        specprefill_gemma_batch_config=prepared.gemma_batch_config,
        specprefill_diagnostic=True,
        specprefill_advertisable=False,
    )
    kwargs.update(override)
    with pytest.raises(ValueError, match="Gemma CB SpecPrefill"):
        MLLMSchedulerConfig(**kwargs)
