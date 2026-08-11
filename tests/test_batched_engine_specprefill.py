# SPDX-License-Identifier: Apache-2.0
"""Synthetic BatchedEngine SpecPrefill launch and transport contracts."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from vllm_mlx.engine.batched import (
    BatchedEngine,
    PreparedMLLMSpecPrefillRuntime,
)
from vllm_mlx.mllm_scheduler import MLLMSpecPrefillCacheCapability
from vllm_mlx.scheduler import SchedulerConfig
from vllm_mlx.specprefill_profiles import (
    SpecPrefillCalibration,
    SpecPrefillCell,
    SpecPrefillEngine,
    SpecPrefillProfile,
    SpecPrefillProfileKey,
    SpecPrefillProfileRegistry,
    SpecPrefillProfileTier,
    SpecPrefillQualificationEvidence,
    SpecPrefillTuning,
)


def _hash(letter: str) -> str:
    return letter * 64


def _certified_registry(
    *,
    target_hash=None,
    tokenizer_hash=None,
    scorer_hash=None,
    adapter_id="qwen35_text_hybrid",
):
    key = SpecPrefillProfileKey(
        target_artifact_id="qwen-target",
        target_artifact_hash=target_hash or _hash("a"),
        tokenizer_artifact_hash=tokenizer_hash or _hash("b"),
        scorer_artifact_id="qwen-scorer",
        scorer_artifact_hash=scorer_hash or _hash("c"),
        adapter_id=adapter_id,
        engine=SpecPrefillEngine.CONTINUOUS_BATCHING,
        cell=SpecPrefillCell.SPARSE_ONLY,
    )
    calibration = SpecPrefillCalibration(
        selector_version="hybrid-v1",
        tuning=SpecPrefillTuning(0.5, 0.1, 1, 1, 32),
        crossover_tokens=8,
        max_context_tokens=128,
        residency_limit_bytes=1024,
        min_ttft_improvement_pct=10.0,
        max_total_latency_regression_pct=5.0,
        max_decode_throughput_regression_pct=5.0,
        required_context_tokens=(8, 16, 32, 64, 128),
        required_concurrency_levels=(1, 2, 4, 8),
        max_p95_inter_token_latency_regression_pct=5.0,
        min_prefill_heavy_throughput_improvement_pct=10.0,
    )
    evidence = SpecPrefillQualificationEvidence(
        report_id="run/specprefill/cb.json",
        report_sha256=_hash("d"),
        key=key,
        selector_version="hybrid-v1",
        tested_context_tokens=(8, 16, 32, 64, 128),
        tested_concurrency_levels=(1, 2, 4, 8),
        deterministic_baseline_successes=1,
        preserved_baseline_successes=1,
        fabricated_or_source_corruption_count=0,
        quality_noninferiority_ci_lower_points=-1.0,
        median_ttft_improvement_pct=20.0,
        median_total_latency_regression_pct=1.0,
        decode_throughput_regression_pct=1.0,
        oom_count=0,
        swap_escalation_count=0,
        unbounded_retry_count=0,
        peak_resident_bytes=512,
        admission_safety_reserve_pct=10.0,
        cb_p95_inter_token_latency_regression_pct=1.0,
        cb_aggregate_throughput_regression_pct=0.0,
        cb_prefill_heavy_throughput_improvement_pct=20.0,
        mtp_evidence_id=None,
        mtp_evidence_sha256=None,
        mtp_drafts=0,
        mtp_accepted=0,
    )
    profile = SpecPrefillProfile(
        key,
        SpecPrefillProfileTier.PRODUCTION,
        calibration,
        evidence,
    )
    return SpecPrefillProfileRegistry((profile,)), key


def _scorer_artifact(tmp_path, contents=b"synthetic scorer weights"):
    path = tmp_path / "scorer.safetensors"
    path.write_bytes(contents)
    return str(path), hashlib.sha256(contents).hexdigest()


def _bundle(cleanup, target_model, processor):
    registry, key = _certified_registry()
    return PreparedMLLMSpecPrefillRuntime(
        profile_registry=registry,
        profile_key=key,
        estimated_residency_bytes=512,
        session_factory=lambda **_kwargs: None,
        cache_capability=MLLMSpecPrefillCacheCapability(
            adapter_id=key.adapter_id,
            layout="qwen3_5_nonrotating_hybrid",
        ),
        target_forward_context=lambda _forward: None,
        target_model=target_model,
        processor=processor,
        target_artifact_hash=key.target_artifact_hash,
        tokenizer_artifact_hash=key.tokenizer_artifact_hash,
        scorer_artifact_hash=key.scorer_artifact_hash,
        cleanup=cleanup,
    )


class _Scheduler:
    instances = []
    fail_starts = 0

    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.config = config
        self.stopped = False
        self.__class__.instances.append(self)

    async def start(self):
        if self.__class__.fail_starts:
            self.__class__.fail_starts -= 1
            raise RuntimeError("scheduler start failed")

    async def stop(self):
        self.stopped = True


def _engine(config):
    engine = BatchedEngine("test", scheduler_config=config, force_mllm=True)
    engine._model = object()
    engine._processor = object()
    return engine


def test_prepared_launch_wires_exact_scheduler_dependencies(monkeypatch):
    import vllm_mlx.mllm_scheduler as scheduler_module

    _Scheduler.instances = []
    _Scheduler.fail_starts = 0
    prepare_calls = []
    cleanup_calls = []
    engine = _engine(SchedulerConfig())
    bundle = _bundle(
        lambda: cleanup_calls.append("cleanup"), engine._model, engine._processor
    )
    config = SchedulerConfig(
        specprefill_enabled=True,
        specprefill_prepare=lambda model, processor: (
            prepare_calls.append((model, processor)) or bundle
        ),
    )
    engine._scheduler_config = config
    monkeypatch.setattr(scheduler_module, "MLLMScheduler", _Scheduler)

    asyncio.run(engine._start_mllm())

    assert prepare_calls == [(engine._model, engine._processor)]
    wired = _Scheduler.instances[0].config
    assert wired.specprefill_enabled is True
    assert wired.specprefill_profile_registry is bundle.profile_registry
    assert wired.specprefill_profile_key is bundle.profile_key
    assert wired.specprefill_session_factory is bundle.session_factory
    assert wired.specprefill_cache_capability is bundle.cache_capability
    assert wired.specprefill_target_forward_context is bundle.target_forward_context
    assert wired.specprefill_advertisable is True
    assert wired.specprefill_diagnostic is False
    assert cleanup_calls == []

    asyncio.run(engine.stop())
    assert cleanup_calls == ["cleanup"]


def test_failed_start_cleans_prepared_runtime_and_retry_prepares_again(monkeypatch):
    import vllm_mlx.mllm_scheduler as scheduler_module

    _Scheduler.instances = []
    _Scheduler.fail_starts = 1
    prepare_calls = []
    cleanup_calls = []

    def prepare(_model, _processor):
        attempt = len(prepare_calls) + 1
        prepare_calls.append(attempt)
        return _bundle(
            lambda: cleanup_calls.append(attempt), _model, _processor
        )

    engine = _engine(
        SchedulerConfig(specprefill_enabled=True, specprefill_prepare=prepare)
    )
    monkeypatch.setattr(scheduler_module, "MLLMScheduler", _Scheduler)

    with pytest.raises(RuntimeError, match="scheduler start failed"):
        asyncio.run(engine._start_mllm())
    assert engine._mllm_scheduler is None
    assert engine._prepared_specprefill_runtime is None
    assert cleanup_calls == [1]

    asyncio.run(engine._start_mllm())
    assert prepare_calls == [1, 2]
    assert engine._prepared_specprefill_runtime is not None
    assert cleanup_calls == [1]


def test_prepare_failure_or_incomplete_bundle_never_constructs_scheduler(monkeypatch):
    import vllm_mlx.mllm_scheduler as scheduler_module

    _Scheduler.instances = []
    cleaned = []
    monkeypatch.setattr(scheduler_module, "MLLMScheduler", _Scheduler)
    raising = _engine(
        SchedulerConfig(
            specprefill_enabled=True,
            specprefill_prepare=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("scorer prepare failed")
            ),
        )
    )

    with pytest.raises(RuntimeError, match="scorer prepare failed"):
        asyncio.run(raising._start_mllm())
    assert _Scheduler.instances == []
    assert raising._prepared_specprefill_runtime is None

    incomplete = _engine(
        SchedulerConfig(
            specprefill_enabled=True,
            specprefill_prepare=lambda *_args: SimpleNamespace(
                cleanup=lambda: cleaned.append("incomplete")
            ),
        )
    )
    with pytest.raises(TypeError, match="PreparedMLLMSpecPrefillRuntime"):
        asyncio.run(incomplete._start_mllm())
    assert cleaned == ["incomplete"]
    assert _Scheduler.instances == []
    assert incomplete._prepared_specprefill_runtime is None


def test_prepared_bundle_rejects_gemma_or_rotating_adapter():
    registry, key = _certified_registry()

    with pytest.raises(ValueError, match="not admitted"):
        PreparedMLLMSpecPrefillRuntime(
            profile_registry=registry,
            profile_key=key,
            estimated_residency_bytes=512,
            session_factory=lambda **_kwargs: None,
            cache_capability=MLLMSpecPrefillCacheCapability(
                adapter_id=key.adapter_id,
                layout="gemma4_dense",
            ),
            target_forward_context=lambda _forward: None,
            target_model=object(),
            processor=object(),
            target_artifact_hash=key.target_artifact_hash,
            tokenizer_artifact_hash=key.tokenizer_artifact_hash,
            scorer_artifact_hash=key.scorer_artifact_hash,
            cleanup=lambda: None,
        )


def test_prepared_bundle_rejects_model_processor_or_hash_mismatch(monkeypatch):
    import vllm_mlx.mllm_scheduler as scheduler_module

    _Scheduler.instances = []
    cleaned = []
    engine = _engine(SchedulerConfig())
    mismatched = _bundle(lambda: cleaned.append("identity"), object(), object())
    engine._scheduler_config = SchedulerConfig(
        specprefill_enabled=True,
        specprefill_prepare=lambda *_args: mismatched,
    )
    monkeypatch.setattr(scheduler_module, "MLLMScheduler", _Scheduler)

    with pytest.raises(ValueError, match="target model identity mismatch"):
        asyncio.run(engine._start_mllm())
    assert cleaned == ["identity"]
    assert _Scheduler.instances == []

    registry, key = _certified_registry()
    with pytest.raises(ValueError, match="artifact hashes"):
        PreparedMLLMSpecPrefillRuntime(
            profile_registry=registry,
            profile_key=key,
            estimated_residency_bytes=512,
            session_factory=lambda **_kwargs: None,
            cache_capability=MLLMSpecPrefillCacheCapability(
                adapter_id=key.adapter_id,
                layout="qwen3_5_nonrotating_hybrid",
            ),
            target_forward_context=lambda _forward: None,
            target_model=object(),
            processor=object(),
            target_artifact_hash=_hash("e"),
            tokenizer_artifact_hash=key.tokenizer_artifact_hash,
            scorer_artifact_hash=key.scorer_artifact_hash,
            cleanup=lambda: None,
        )


def test_startup_preserves_original_when_both_cleanups_fail_and_retries(monkeypatch):
    import vllm_mlx.mllm_scheduler as scheduler_module

    state = {
        "start_fails": True,
        "scheduler_cleanup_fails": True,
        "prepared_cleanup_fails": True,
    }
    events = []
    prepare_calls = 0

    class Scheduler(_Scheduler):
        async def start(self):
            if state["start_fails"]:
                raise RuntimeError("original startup failure")

        async def stop(self):
            events.append(("stop", self))
            if state["scheduler_cleanup_fails"]:
                raise asyncio.CancelledError("scheduler cleanup cancelled")
            self.stopped = True

    engine = _engine(SchedulerConfig())

    def cleanup():
        events.append(("cleanup", engine._prepared_specprefill_runtime))
        if state["prepared_cleanup_fails"]:
            raise RuntimeError("prepared cleanup failed")

    def prepare(model, processor):
        nonlocal prepare_calls
        prepare_calls += 1
        events.append(("prepare", prepare_calls))
        return _bundle(cleanup, model, processor)

    engine._scheduler_config = SchedulerConfig(
        specprefill_enabled=True,
        specprefill_prepare=prepare,
    )
    monkeypatch.setattr(scheduler_module, "MLLMScheduler", Scheduler)

    with pytest.raises(RuntimeError, match="original startup failure"):
        asyncio.run(engine._start_mllm())
    old_scheduler = engine._mllm_scheduler
    old_prepared = engine._prepared_specprefill_runtime
    assert old_scheduler is not None
    assert old_prepared is not None
    assert prepare_calls == 1
    assert events == [
        ("prepare", 1),
        ("stop", old_scheduler),
        ("cleanup", old_prepared),
    ]

    with pytest.raises(RuntimeError, match="cleanup remains incomplete"):
        asyncio.run(engine._start_mllm())
    assert engine._mllm_scheduler is old_scheduler
    assert engine._prepared_specprefill_runtime is old_prepared
    assert prepare_calls == 1
    assert events[-2:] == [("stop", old_scheduler), ("cleanup", old_prepared)]

    state["scheduler_cleanup_fails"] = False
    state["prepared_cleanup_fails"] = False
    state["start_fails"] = False
    asyncio.run(engine._start_mllm())
    assert engine._mllm_scheduler is not None
    assert engine._prepared_specprefill_runtime is not None
    assert engine._mllm_scheduler is not old_scheduler
    assert engine._prepared_specprefill_runtime is not old_prepared
    assert prepare_calls == 2
    assert events[-3:] == [
        ("stop", old_scheduler),
        ("cleanup", old_prepared),
        ("prepare", 2),
    ]


def test_mismatch_cleanup_failure_retains_candidate_until_retry(monkeypatch):
    import vllm_mlx.mllm_scheduler as scheduler_module

    _Scheduler.instances = []
    events = []
    cleanup_fails = True
    prepare_calls = 0
    engine = _engine(SchedulerConfig())

    def invalid_cleanup():
        events.append("invalid_cleanup")
        if cleanup_fails:
            raise RuntimeError("invalid cleanup failed")

    invalid = _bundle(invalid_cleanup, object(), object())

    def prepare(model, processor):
        nonlocal prepare_calls
        prepare_calls += 1
        events.append(f"prepare_{prepare_calls}")
        if prepare_calls == 1:
            return invalid
        return _bundle(lambda: None, model, processor)

    engine._scheduler_config = SchedulerConfig(
        specprefill_enabled=True,
        specprefill_prepare=prepare,
    )
    monkeypatch.setattr(scheduler_module, "MLLMScheduler", _Scheduler)

    with pytest.raises(ValueError, match="target model identity mismatch"):
        asyncio.run(engine._start_mllm())
    assert engine._prepared_specprefill_runtime is invalid
    assert prepare_calls == 1
    assert events == ["prepare_1", "invalid_cleanup"]

    cleanup_fails = False
    asyncio.run(engine._start_mllm())

    assert events[:3] == ["prepare_1", "invalid_cleanup", "invalid_cleanup"]
    assert events[3] == "prepare_2"
    assert prepare_calls == 2
    assert engine._prepared_specprefill_runtime is not invalid
    assert len(_Scheduler.instances) == 1


@pytest.mark.parametrize(
    "config",
    [
        SchedulerConfig(specprefill_enabled=True),
        SchedulerConfig(
            specprefill_enabled=True,
            enable_mtp=True,
            specprefill_prepare=lambda *_args: None,
        ),
        SchedulerConfig(
            specprefill_enabled=True,
            max_kv_size=64,
            specprefill_prepare=lambda *_args: None,
        ),
    ],
)
def test_enabled_launch_fails_closed_without_admitted_preparation(config):
    engine = _engine(config)

    with pytest.raises(RuntimeError):
        asyncio.run(engine._start_mllm())

    assert engine._mllm_scheduler is None
    assert engine._prepared_specprefill_runtime is None


def test_server_enable_switch_copies_scheduler_launch_config(monkeypatch):
    from vllm_mlx.server import _configure_batched_specprefill
    import vllm_mlx.specprefill_runtime as runtime

    prepare = lambda *_args: None
    original = SchedulerConfig(
        specprefill_enabled=False,
        specprefill_prepare=prepare,
    )
    configured = _configure_batched_specprefill(original, enabled=True)

    assert configured is not original
    assert original.specprefill_enabled is False
    assert configured.specprefill_enabled is True

    with pytest.raises(ValueError, match="unavailable"):
        _configure_batched_specprefill(SchedulerConfig(), enabled=True)

    builder_calls = []
    built_prepare = lambda *_args: None
    monkeypatch.setattr(
        runtime,
        "build_qwen_cb_specprefill_prepare",
        lambda **kwargs: builder_calls.append(kwargs) or built_prepare,
    )
    internal = SchedulerConfig(
        specprefill_runtime_builder_inputs={"internal": "inputs"}
    )
    built = _configure_batched_specprefill(internal, enabled=True)

    assert builder_calls == [{"internal": "inputs"}]
    assert built.specprefill_prepare is built_prepare
    assert internal.specprefill_prepare is None


def test_batched_stats_promote_exact_specprefill_status():
    engine = _engine(SchedulerConfig())
    exact = {"enabled": False, "advertisable": False, "diagnostic": True}
    engine._mllm_scheduler = SimpleNamespace(
        get_stats=lambda: {"specprefill": exact}
    )

    stats = engine.get_stats()

    assert stats["specprefill"] is exact


def test_public_status_and_model_discovery_separate_diagnostic_and_production(
    monkeypatch,
):
    import vllm_mlx.server as server

    class Engine:
        def __init__(self, specprefill):
            self.specprefill = specprefill

        def get_stats(self):
            return {
                "running": True,
                "specprefill": self.specprefill,
                "requests": [],
            }

    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_model_name", "qwen")
    monkeypatch.setattr(server, "_residency_manager", None)
    monkeypatch.setattr(server, "_default_model_key", None)
    monkeypatch.setattr(server, "_embedding_engine", None)
    monkeypatch.setattr(server, "_rerank_engine", None)
    diagnostic = {"enabled": False, "advertisable": False, "diagnostic": True}
    monkeypatch.setattr(server, "_engine", Engine(diagnostic))

    status_payload = asyncio.run(server.status())
    diagnostic_models = asyncio.run(server.list_models())

    assert status_payload["specprefill"] == diagnostic
    assert diagnostic_models.data[0].capabilities == []

    production = {"enabled": True, "advertisable": True, "diagnostic": False}
    monkeypatch.setattr(server, "_engine", Engine(production))
    production_models = asyncio.run(server.list_models())

    assert production_models.data[0].capabilities == ["specprefill"]

    class ThrowingEngine:
        def get_stats(self):
            raise RuntimeError("stats unavailable")

    monkeypatch.setattr(server, "_engine", ThrowingEngine())
    unavailable_models = asyncio.run(server.list_models())
    assert unavailable_models.data[0].capabilities == []


def test_registry_status_keeps_diagnostic_state_out_of_model_discovery(monkeypatch):
    import vllm_mlx.server as server

    diagnostic = {
        "state": "diagnostic",
        "enabled": False,
        "advertisable": False,
        "diagnostic": True,
    }

    class Manager:
        memory_budget_bytes = 1024**3

        def list_models(self):
            return [
                {
                    "id": "diagnostic-qwen",
                    "capabilities": [],
                    "specprefill": diagnostic,
                }
            ]

    monkeypatch.setattr(server, "_model_manager", Manager())
    monkeypatch.setattr(server, "_model_name", None)

    status_payload = asyncio.run(server.status())
    models_payload = asyncio.run(server.list_models())

    assert status_payload["model_manager"]["models"][0]["specprefill"] == diagnostic
    assert models_payload.data[0].capabilities == []


def test_request_controls_and_terminal_metadata_cross_batched_engine():
    terminal = SimpleNamespace(
        output_text="ok",
        output_token_ids=[7],
        prompt_tokens=10,
        completion_tokens=1,
        finish_reason="stop",
        mtp_drafts=4,
        mtp_accepted=3,
        specprefill_requested_policy="auto",
        specprefill_effective_policy="sparse",
        specprefill_coverage="selective",
        specprefill_engaged=True,
        specprefill_selector_version="hybrid-v1",
        specprefill_fallback_reason=None,
        specprefill_total_tokens=10,
        specprefill_selected_tokens=5,
        specprefill_scorer_ms=1.0,
        specprefill_target_prefill_ms=2.0,
    )

    class Scheduler:
        seen = None

        async def generate(self, **kwargs):
            self.seen = kwargs
            return terminal

    engine = _engine(SchedulerConfig())
    engine._loaded = True
    engine._mllm_scheduler = Scheduler()
    output = asyncio.run(
        engine.generate(
            "prompt",
            specprefill_policy="auto",
            specprefill_coverage="selective",
            specprefill_control_token_indices=[4, 1],
        )
    )

    assert engine._mllm_scheduler.seen["specprefill_policy"] == "auto"
    assert engine._mllm_scheduler.seen["specprefill_coverage"] == "selective"
    assert engine._mllm_scheduler.seen["specprefill_control_token_indices"] == (4, 1)
    assert "specprefill_keep_pct" not in engine._mllm_scheduler.seen
    assert output.mtp_drafts == 4
    assert output.mtp_accepted == 3
    assert output.specprefill_selected_tokens == 5
    assert output.specprefill_target_prefill_ms == 2.0

    with pytest.raises(ValueError, match="does not accept"):
        asyncio.run(
            engine.generate(
                "prompt",
                specprefill_policy="auto",
                specprefill_keep_pct=0.9,
            )
        )

    with pytest.raises(ValueError, match="conflicts with specprefill_policy"):
        asyncio.run(
            engine.generate(
                "prompt",
                specprefill=False,
                specprefill_policy="sparse",
            )
        )


def test_streaming_terminal_preserves_complete_independent_metadata():
    terminal = SimpleNamespace(
        output_text="ok",
        new_text="ok",
        prompt_tokens=10,
        completion_tokens=1,
        finished=True,
        finish_reason="stop",
        mtp_drafts=2,
        mtp_accepted=1,
        specprefill_requested_policy="auto",
        specprefill_effective_policy="dense",
        specprefill_coverage="exhaustive",
        specprefill_engaged=False,
        specprefill_selector_version="hybrid-v1",
        specprefill_fallback_reason="coverage_not_selective",
        specprefill_total_tokens=10,
        specprefill_selected_tokens=0,
        specprefill_scorer_ms=0.0,
        specprefill_target_prefill_ms=0.0,
    )

    class Scheduler:
        seen = None

        async def add_request_async(self, **kwargs):
            self.seen = kwargs
            return "request"

        async def stream_outputs(self, _request_id):
            yield terminal

    async def collect(engine):
        return [
            item
            async for item in engine.stream_generate(
                "prompt",
                specprefill_policy="auto",
                specprefill_coverage="exhaustive",
                specprefill_control_token_indices=[3],
            )
        ]

    engine = _engine(SchedulerConfig())
    engine._loaded = True
    engine._mllm_scheduler = Scheduler()
    output = asyncio.run(collect(engine))[0]

    assert engine._mllm_scheduler.seen["specprefill_control_token_indices"] == (3,)
    assert output.finished is True
    assert (output.mtp_drafts, output.mtp_accepted) == (2, 1)
    assert output.specprefill_effective_policy == "dense"
    assert output.specprefill_fallback_reason == "coverage_not_selective"


def test_explicit_media_bit_survives_empty_extraction_arrays():
    class Scheduler:
        seen = None

        async def generate(self, **kwargs):
            self.seen = kwargs
            return SimpleNamespace(
                output_text="",
                output_token_ids=[],
                prompt_tokens=1,
                completion_tokens=0,
                finish_reason="stop",
            )

    engine = _engine(SchedulerConfig())
    engine._loaded = True
    engine._mllm_scheduler = Scheduler()
    asyncio.run(
        engine.generate(
            "prompt",
            images=[],
            specprefill_policy="auto",
            specprefill_coverage="selective",
            specprefill_has_media=True,
        )
    )

    assert engine._mllm_scheduler.seen["specprefill_has_media"] is True


def test_real_runtime_builder_eagerly_installs_and_builds_request_sessions(
    monkeypatch, tmp_path,
):
    from contextlib import contextmanager

    import vllm_mlx.specprefill_runtime as runtime
    from vllm_mlx.cooperative_specprefill import CooperativeSpecPrefillConfig
    from vllm_mlx.specprefill_cache import (
        SparseCacheIdentity,
        SparseCacheState,
        SparsePolicyTuning,
    )
    from vllm_mlx.specprefill_positions import QWEN35_TEXT_HYBRID_TARGET
    from vllm_mlx.specprefill_scorer_session import ScorerSessionPhase

    scorer_path, scorer_hash = _scorer_artifact(tmp_path)
    registry, key = _certified_registry(scorer_hash=scorer_hash)
    profile = registry.profiles[0]
    events = []
    scorer_model = object()
    target_text = SimpleNamespace(layers=[SimpleNamespace(is_linear=False)])
    target_outer = SimpleNamespace(language_model=target_text)
    processor = object()

    class FakeScorerSession:
        phase = ScorerSessionPhase.PREFILL

        def cancel(self):
            events.append("scorer_cancel")

    class FakeHooks:
        @contextmanager
        def session_for_plan(self, plan):
            events.append(("decode_plan", plan.logical_positions))
            yield

    class FakeTargetSession:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def step(self):
            raise AssertionError("not stepped in builder test")

        def cancel(self):
            events.append("target_cancel")

    monkeypatch.setattr(
        runtime.SpecPrefillScorer,
        "for_model",
        lambda model: events.append(("install_scorer", model)) or object(),
    )
    monkeypatch.setattr(
        runtime.TargetPositionHooks,
        "for_model",
        lambda model, adapter: (
            events.append(("install_hooks", model, adapter)) or FakeHooks()
        ),
    )
    monkeypatch.setattr(
        runtime,
        "resolve_target_position_adapter",
        lambda _model: QWEN35_TEXT_HYBRID_TARGET,
    )
    monkeypatch.setattr(
        runtime,
        "SpecPrefillScorerSession",
        lambda *_a, **_k: FakeScorerSession(),
    )
    monkeypatch.setattr(runtime, "SparseTargetPrefillSession", FakeTargetSession)
    caches = []

    def cache_factory(model):
        from mlx_lm.models.cache import KVCache

        cache = [KVCache()]
        caches.append((model, cache))
        return cache

    def loader(path, expected_hash):
        events.append(("load", path, expected_hash))
        return runtime.LoadedSpecPrefillScorer(
            scorer_model,
            lambda: events.append("loader_cleanup"),
        )

    prepare = runtime.build_qwen_cb_specprefill_prepare(
        scorer_artifact_path=scorer_path,
        scorer_artifact_hash=key.scorer_artifact_hash,
        target_artifact_hash=key.target_artifact_hash,
        tokenizer_artifact_hash=key.tokenizer_artifact_hash,
        profile_registry=registry,
        profile_key=key,
        calibrated_tuning=profile.calibration.tuning,
        estimated_residency_bytes=512,
        target_identity_attestor=lambda target, actual_processor: (
            runtime.TargetProcessorAttestation(
                target,
                actual_processor,
                key.target_artifact_hash,
                key.tokenizer_artifact_hash,
            )
        ),
        scorer_loader=loader,
        target_cache_factory=cache_factory,
    )
    prepared = prepare(target_outer, processor)

    assert events[:3] == [
        ("load", scorer_path, key.scorer_artifact_hash),
        ("install_scorer", scorer_model),
        ("install_hooks", target_text, QWEN35_TEXT_HYBRID_TARGET),
    ]
    tuning = SparsePolicyTuning(0.5, 0.1, 1, 1, 32)
    config = CooperativeSpecPrefillConfig(
        target_id=f"{key.target_artifact_id}@sha256:{key.target_artifact_hash}",
        tokenizer_id=f"tokenizer@sha256:{key.tokenizer_artifact_hash}",
        scorer_id=f"{key.scorer_artifact_id}@sha256:{key.scorer_artifact_hash}",
        tuning=tuning,
    )
    admission = prepared.session_factory(
        SimpleNamespace(request_id="request"),
        (1, 2, 3),
        config,
    )
    identity = SparseCacheIdentity.from_tokens(
        target_id=config.target_id,
        tokenizer_id=config.tokenizer_id,
        scorer_id=config.scorer_id,
        selector_version="hybrid-v1",
        tuning=tuning,
        tokens=(1, 2, 3),
        selection_fingerprint="a" * 64,
    )
    state = SparseCacheState.from_selection(identity, ((0, 2),), (3,))
    target_session = admission.session._target_session_factory((1, 3), state)
    assert isinstance(target_session, FakeTargetSession)
    assert len(caches) == 2
    assert caches[0][0] is target_text
    assert tuple(caches[1][1]) == target_session.args[2]
    second_target_session = admission.session._target_session_factory((1, 3), state)
    assert isinstance(second_target_session, FakeTargetSession)
    assert len(caches) == 3
    assert caches[1][1] is not caches[2][1]
    assert target_session.args[2] is not second_target_session.args[2]
    forward = SimpleNamespace(
        phase=runtime.MLLMTargetForwardPhase.DECODE,
        sparse_row_states=state.rows,
    )
    with prepared.target_forward_context(forward):
        events.append("decode")
    assert events[-2:] == [("decode_plan", ((3,),)), "decode"]
    with pytest.raises(
        runtime.SpecPrefillRuntimePreparationError,
        match="sparse and dense rows",
    ):
        prepared.target_forward_context(
            SimpleNamespace(
                phase=runtime.MLLMTargetForwardPhase.DECODE,
                sparse_row_states=(state.rows[0], None),
            )
        )

    prepared.cleanup()
    assert events[-1] == "loader_cleanup"
    with pytest.raises(runtime.SpecPrefillRuntimePreparationError, match="closed"):
        prepared.session_factory(
            SimpleNamespace(request_id="later"), (1,), config
        )


def test_real_runtime_builder_cleanup_is_fail_atomic_on_hook_failure(
    monkeypatch, tmp_path
):
    import vllm_mlx.specprefill_runtime as runtime
    from vllm_mlx.specprefill_positions import QWEN35_TEXT_HYBRID_TARGET

    scorer_path, scorer_hash = _scorer_artifact(tmp_path)
    registry, key = _certified_registry(scorer_hash=scorer_hash)
    profile = registry.profiles[0]
    cleanup_calls = []
    monkeypatch.setattr(runtime.SpecPrefillScorer, "for_model", lambda _model: object())
    monkeypatch.setattr(
        runtime,
        "resolve_target_position_adapter",
        lambda _model: QWEN35_TEXT_HYBRID_TARGET,
    )
    monkeypatch.setattr(
        runtime.TargetPositionHooks,
        "for_model",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("hook install failed")),
    )
    prepare = runtime.build_qwen_cb_specprefill_prepare(
        scorer_artifact_path=scorer_path,
        scorer_artifact_hash=key.scorer_artifact_hash,
        target_artifact_hash=key.target_artifact_hash,
        tokenizer_artifact_hash=key.tokenizer_artifact_hash,
        profile_registry=registry,
        profile_key=key,
        calibrated_tuning=profile.calibration.tuning,
        estimated_residency_bytes=512,
        target_identity_attestor=lambda target, processor: (
            runtime.TargetProcessorAttestation(
                target,
                processor,
                key.target_artifact_hash,
                key.tokenizer_artifact_hash,
            )
        ),
        scorer_loader=lambda *_args: runtime.LoadedSpecPrefillScorer(
            object(), lambda: cleanup_calls.append("cleanup")
        ),
    )

    with pytest.raises(RuntimeError, match="hook install failed"):
        prepare(SimpleNamespace(language_model=object()), object())

    assert cleanup_calls == ["cleanup"]


def test_real_runtime_builder_rejects_wrong_scorer_bytes_before_loading(tmp_path):
    import vllm_mlx.specprefill_runtime as runtime

    scorer_path, _actual_hash = _scorer_artifact(tmp_path, b"wrong bytes")
    registry, key = _certified_registry()
    profile = registry.profiles[0]
    load_calls = []
    prepare = runtime.build_qwen_cb_specprefill_prepare(
        scorer_artifact_path=scorer_path,
        scorer_artifact_hash=key.scorer_artifact_hash,
        target_artifact_hash=key.target_artifact_hash,
        tokenizer_artifact_hash=key.tokenizer_artifact_hash,
        profile_registry=registry,
        profile_key=key,
        calibrated_tuning=profile.calibration.tuning,
        estimated_residency_bytes=512,
        target_identity_attestor=lambda target, processor: (
            runtime.TargetProcessorAttestation(
                target,
                processor,
                key.target_artifact_hash,
                key.tokenizer_artifact_hash,
            )
        ),
        scorer_loader=lambda *_args: load_calls.append("load"),
    )

    with pytest.raises(ValueError, match="artifact bytes"):
        prepare(object(), object())

    assert load_calls == []


@pytest.mark.parametrize("symlink_kind", ["directory", "shard"])
def test_scorer_artifact_manifest_rejects_symlinks(tmp_path, symlink_kind):
    import vllm_mlx.specprefill_runtime as runtime

    real = tmp_path / "real"
    real.mkdir()
    (real / "weights.safetensors").write_bytes(b"weights")
    if symlink_kind == "directory":
        artifact = tmp_path / "artifact-link"
        artifact.symlink_to(real, target_is_directory=True)
    else:
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        (artifact / "weights.safetensors").symlink_to(
            real / "weights.safetensors"
        )

    with pytest.raises(
        runtime.SpecPrefillRuntimePreparationError,
        match="symlink",
    ):
        runtime.sha256_artifact_path(artifact)


@pytest.mark.parametrize("wrong_object", ["target", "processor"])
def test_real_runtime_builder_rejects_unbound_target_attestation(
    tmp_path, wrong_object
):
    import vllm_mlx.specprefill_runtime as runtime

    scorer_path, scorer_hash = _scorer_artifact(tmp_path)
    registry, key = _certified_registry(scorer_hash=scorer_hash)
    profile = registry.profiles[0]

    def attest(target, processor):
        return runtime.TargetProcessorAttestation(
            object() if wrong_object == "target" else target,
            object() if wrong_object == "processor" else processor,
            key.target_artifact_hash,
            key.tokenizer_artifact_hash,
        )

    prepare = runtime.build_qwen_cb_specprefill_prepare(
        scorer_artifact_path=scorer_path,
        scorer_artifact_hash=scorer_hash,
        target_artifact_hash=key.target_artifact_hash,
        tokenizer_artifact_hash=key.tokenizer_artifact_hash,
        profile_registry=registry,
        profile_key=key,
        calibrated_tuning=profile.calibration.tuning,
        estimated_residency_bytes=1,
        target_identity_attestor=attest,
        scorer_loader=lambda *_args: pytest.fail("loader must not run"),
    )

    with pytest.raises(ValueError, match="loaded objects"):
        prepare(object(), object())


@pytest.mark.parametrize("cache_failure", ["raise", "rotating"])
def test_real_runtime_builder_eager_cache_probe_fails_and_cleans(
    monkeypatch, tmp_path, cache_failure
):
    import vllm_mlx.specprefill_runtime as runtime
    from mlx_lm.models.cache import RotatingKVCache
    from vllm_mlx.specprefill_positions import QWEN35_TEXT_HYBRID_TARGET

    scorer_path, scorer_hash = _scorer_artifact(tmp_path)
    registry, key = _certified_registry(scorer_hash=scorer_hash)
    profile = registry.profiles[0]
    cleanup_calls = []
    monkeypatch.setattr(runtime.SpecPrefillScorer, "for_model", lambda _model: object())
    monkeypatch.setattr(
        runtime,
        "resolve_target_position_adapter",
        lambda _model: QWEN35_TEXT_HYBRID_TARGET,
    )
    monkeypatch.setattr(
        runtime.TargetPositionHooks,
        "for_model",
        lambda *_args: SimpleNamespace(),
    )

    def cache_factory(_model):
        if cache_failure == "raise":
            raise RuntimeError("cache allocation failed")
        return [RotatingKVCache(max_size=32)]

    prepare = runtime.build_qwen_cb_specprefill_prepare(
        scorer_artifact_path=scorer_path,
        scorer_artifact_hash=scorer_hash,
        target_artifact_hash=key.target_artifact_hash,
        tokenizer_artifact_hash=key.tokenizer_artifact_hash,
        profile_registry=registry,
        profile_key=key,
        calibrated_tuning=profile.calibration.tuning,
        estimated_residency_bytes=1,
        target_identity_attestor=lambda target, processor: (
            runtime.TargetProcessorAttestation(
                target,
                processor,
                key.target_artifact_hash,
                key.tokenizer_artifact_hash,
            )
        ),
        scorer_loader=lambda *_args: runtime.LoadedSpecPrefillScorer(
            object(), lambda: cleanup_calls.append("cleanup")
        ),
        target_cache_factory=cache_factory,
    )
    target = SimpleNamespace(
        language_model=SimpleNamespace(layers=[SimpleNamespace(is_linear=False)])
    )

    expected = RuntimeError if cache_failure == "raise" else runtime.SpecPrefillRuntimePreparationError
    with pytest.raises(expected):
        prepare(target, object())

    assert cleanup_calls == ["cleanup"]


def test_real_runtime_builder_rejects_qwen_vlm_adapter(monkeypatch, tmp_path):
    import vllm_mlx.specprefill_runtime as runtime
    from vllm_mlx.specprefill_positions import QWEN35_VLM_HYBRID_TARGET

    scorer_path, scorer_hash = _scorer_artifact(tmp_path)
    registry, key = _certified_registry(
        scorer_hash=scorer_hash,
        adapter_id=QWEN35_VLM_HYBRID_TARGET.adapter_id,
    )
    profile = registry.profiles[0]
    cleanup_calls = []
    monkeypatch.setattr(runtime.SpecPrefillScorer, "for_model", lambda _model: object())
    monkeypatch.setattr(
        runtime,
        "resolve_target_position_adapter",
        lambda _model: QWEN35_VLM_HYBRID_TARGET,
    )
    prepare = runtime.build_qwen_cb_specprefill_prepare(
        scorer_artifact_path=scorer_path,
        scorer_artifact_hash=scorer_hash,
        target_artifact_hash=key.target_artifact_hash,
        tokenizer_artifact_hash=key.tokenizer_artifact_hash,
        profile_registry=registry,
        profile_key=key,
        calibrated_tuning=profile.calibration.tuning,
        estimated_residency_bytes=1,
        target_identity_attestor=lambda target, processor: (
            runtime.TargetProcessorAttestation(
                target,
                processor,
                key.target_artifact_hash,
                key.tokenizer_artifact_hash,
            )
        ),
        scorer_loader=lambda *_args: runtime.LoadedSpecPrefillScorer(
            object(), lambda: cleanup_calls.append("cleanup")
        ),
    )

    with pytest.raises(runtime.SpecPrefillRuntimePreparationError, match="text"):
        prepare(object(), object())

    assert cleanup_calls == ["cleanup"]


def test_runtime_builder_requires_certification_and_marks_diagnostic_nonadvertisable(
    monkeypatch, tmp_path,
):
    from dataclasses import replace

    import vllm_mlx.specprefill_runtime as runtime
    from mlx_lm.models.cache import KVCache
    from vllm_mlx.specprefill_positions import QWEN35_TEXT_HYBRID_TARGET
    from vllm_mlx.specprefill_profiles import (
        SpecPrefillProfileRegistry,
        SpecPrefillProfileTier,
    )

    scorer_path, scorer_hash = _scorer_artifact(tmp_path)
    certified_registry, key = _certified_registry(scorer_hash=scorer_hash)
    production = certified_registry.profiles[0]
    uncertified = replace(production, qualification_evidence=None)
    with pytest.raises(ValueError, match="certified evidence"):
        runtime.build_qwen_cb_specprefill_prepare(
            scorer_artifact_path=scorer_path,
            scorer_artifact_hash=key.scorer_artifact_hash,
            target_artifact_hash=key.target_artifact_hash,
            tokenizer_artifact_hash=key.tokenizer_artifact_hash,
            profile_registry=SpecPrefillProfileRegistry((uncertified,)),
            profile_key=key,
            calibrated_tuning=production.calibration.tuning,
            estimated_residency_bytes=1,
            target_identity_attestor=lambda *_args: None,
        )

    diagnostic_profile = replace(
        production,
        tier=SpecPrefillProfileTier.DIAGNOSTIC,
        qualification_evidence=None,
    )
    monkeypatch.setattr(runtime.SpecPrefillScorer, "for_model", lambda _model: object())
    monkeypatch.setattr(
        runtime,
        "resolve_target_position_adapter",
        lambda _model: QWEN35_TEXT_HYBRID_TARGET,
    )
    monkeypatch.setattr(
        runtime.TargetPositionHooks,
        "for_model",
        lambda *_args: SimpleNamespace(session_for_plan=lambda _plan: None),
    )
    prepare = runtime.build_qwen_cb_specprefill_prepare(
        scorer_artifact_path=scorer_path,
        scorer_artifact_hash=key.scorer_artifact_hash,
        target_artifact_hash=key.target_artifact_hash,
        tokenizer_artifact_hash=key.tokenizer_artifact_hash,
        profile_registry=SpecPrefillProfileRegistry((diagnostic_profile,)),
        profile_key=key,
        calibrated_tuning=production.calibration.tuning,
        estimated_residency_bytes=1,
        target_identity_attestor=lambda target, processor: (
            runtime.TargetProcessorAttestation(
                target,
                processor,
                key.target_artifact_hash,
                key.tokenizer_artifact_hash,
            )
        ),
        diagnostic=True,
        scorer_loader=lambda *_args: runtime.LoadedSpecPrefillScorer(
            object(), lambda: None
        ),
        target_cache_factory=lambda _model: [KVCache()],
    )
    target = SimpleNamespace(
        language_model=SimpleNamespace(layers=[SimpleNamespace(is_linear=False)])
    )
    processor = object()
    prepared = prepare(target, processor)

    assert prepared.diagnostic is True
    assert prepared.advertisable is False
