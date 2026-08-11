"""Continuous-batching native-Qwen MTP capability gates."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


def _model(
    model_type: str,
    *,
    capability: object | None = None,
    mtp: object | None = None,
) -> SimpleNamespace:
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type), mtp=mtp)
    if capability is not None:
        model.mtp_capability = capability
    return model


@pytest.mark.parametrize(
    ("model_type", "capability", "expected"),
    [
        (
            "qwen3_5",
            SimpleNamespace(supported=True, reason=None),
            "native_mtp_continuous_batching_unsupported",
        ),
        (
            "qwen3_6_moe",
            SimpleNamespace(
                supported=False,
                reason="native_mtp_weights_not_loaded",
            ),
            "native_mtp_weights_not_loaded",
        ),
        (
            "qwen3_5_moe",
            SimpleNamespace(supported=False, reason=None),
            "native_mtp_unsupported",
        ),
        ("qwen3_6", None, "native_mtp_capability_missing"),
    ],
)
def test_native_qwen_cb_gate_has_stable_fail_closed_reasons(
    model_type, capability, expected
):
    from vllm_mlx.scheduler import _continuous_batching_mtp_capability

    with pytest.raises(RuntimeError, match=f"^{expected}$"):
        _continuous_batching_mtp_capability(
            _model(model_type, capability=capability),
            enabled=True,
        )


def test_native_qwen_cb_gate_rejects_malformed_capability():
    from vllm_mlx.scheduler import _continuous_batching_mtp_capability

    capability = SimpleNamespace(supported="yes", reason=None)
    with pytest.raises(RuntimeError, match="^native_mtp_capability_invalid$"):
        _continuous_batching_mtp_capability(
            _model("qwen3_5", capability=capability),
            enabled=True,
        )


def test_qwen3_next_without_native_capability_retains_legacy_gate_result():
    from vllm_mlx.scheduler import _continuous_batching_mtp_capability

    model = _model("qwen3_next", mtp=object())
    assert _continuous_batching_mtp_capability(model, enabled=True) is None


def test_standard_scheduler_never_installs_native_qwen_mtp(monkeypatch):
    import vllm_mlx.scheduler as scheduler_mod

    model = _model(
        "qwen3_5",
        capability=SimpleNamespace(supported=True, reason=None),
        mtp=object(),
    )
    installed = []
    constructed = []
    hooked = []

    class FakeBatchGenerator:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.kwargs = kwargs

    monkeypatch.setattr(scheduler_mod, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(
        scheduler_mod,
        "_install_mtp",
        lambda *args, **kwargs: installed.append((args, kwargs)),
    )
    monkeypatch.setattr(
        scheduler_mod,
        "_install_chunked_prefill",
        lambda *args, **kwargs: hooked.append((args, kwargs)),
    )
    scheduler = scheduler_mod.Scheduler(
        model,
        SimpleNamespace(),
        scheduler_mod.SchedulerConfig(
            enable_prefix_cache=False,
            enable_mtp=True,
            chunked_prefill_tokens=32,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="^native_mtp_continuous_batching_unsupported$",
    ):
        scheduler._create_batch_generator(scheduler_mod.SamplingParams())
    assert constructed == []
    assert hooked == []
    assert installed == []
    assert scheduler.batch_generator is None


def test_standard_scheduler_preserves_qwen3_next_legacy_install(monkeypatch):
    import vllm_mlx.scheduler as scheduler_mod

    model = _model("qwen3_next", mtp=object())
    installed = []

    class FakeBatchGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(scheduler_mod, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(
        scheduler_mod,
        "_install_mtp",
        lambda *args, **kwargs: installed.append((args, kwargs)),
    )
    config = scheduler_mod.SchedulerConfig(
        enable_prefix_cache=False,
        enable_mtp=True,
        mtp_num_draft_tokens=2,
        mtp_optimistic=True,
    )
    scheduler = scheduler_mod.Scheduler(model, SimpleNamespace(), config)

    batch_generator = scheduler._create_batch_generator(scheduler_mod.SamplingParams())

    assert isinstance(batch_generator, FakeBatchGenerator)
    assert len(installed) == 1
    install_kwargs = installed[0][1]
    stats_state = install_kwargs.pop("stats_state")
    assert stats_state is scheduler._mtp_stats_state
    assert install_kwargs == {
        "model": model,
        "num_draft_tokens": 2,
        "optimistic": True,
    }
    assert stats_state.counters == {
        "attempted": 0,
        "accepted": 0,
        "rejected": 0,
        "errors": 0,
    }
    assert stats_state.bypass_counts == {
        "prefill": 0,
        "no_active_batch": 0,
        "cache_mismatch": 0,
    }


def test_preloaded_mllm_startup_fails_before_injection_or_scheduler(monkeypatch):
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.scheduler import SchedulerConfig

    model = _model(
        "qwen3_5",
        capability=SimpleNamespace(supported=True, reason=None),
        mtp=object(),
    )
    engine = BatchedEngine(
        model_name="fixture-qwen3.5",
        scheduler_config=SchedulerConfig(enable_mtp=True),
        force_mllm=True,
    )
    engine._model = model
    engine._processor = object()
    injected = []
    monkeypatch.setattr(
        engine,
        "_inject_mtp_mllm",
        lambda: injected.append(True),
    )

    with pytest.raises(
        RuntimeError,
        match="^native_mtp_continuous_batching_unsupported$",
    ):
        asyncio.run(engine._start_mllm())
    assert injected == []
    assert engine._mllm_scheduler is None


def test_preloaded_mllm_startup_propagates_unsupported_capability_reason():
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.scheduler import SchedulerConfig

    model = _model(
        "qwen3_6",
        capability=SimpleNamespace(
            supported=False,
            reason="native_mtp_aux_head_missing",
        ),
    )
    engine = BatchedEngine(
        model_name="fixture-qwen3.6",
        scheduler_config=SchedulerConfig(enable_mtp=True),
        force_mllm=True,
    )
    engine._model = model
    engine._processor = object()

    with pytest.raises(RuntimeError, match="^native_mtp_aux_head_missing$"):
        asyncio.run(engine._start_mllm())
    assert engine._mllm_scheduler is None


def test_lazy_mllm_native_gate_does_not_publish_loaded_ownership(monkeypatch):
    import vllm_mlx.models.mllm as model_mod
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.scheduler import SchedulerConfig

    instances = []

    class FakeMLXMultimodalLM:
        def __init__(self, *args, **kwargs):
            self.model = _model(
                "qwen3_5",
                capability=SimpleNamespace(supported=True, reason=None),
                mtp=object(),
            )
            self.processor = object()
            instances.append(self)

        def load(self):
            return None

    monkeypatch.setattr(model_mod, "MLXMultimodalLM", FakeMLXMultimodalLM)
    engine = BatchedEngine(
        model_name="fixture-qwen3.5",
        scheduler_config=SchedulerConfig(enable_mtp=True),
        force_mllm=True,
    )

    with pytest.raises(
        RuntimeError,
        match="^native_mtp_continuous_batching_unsupported$",
    ):
        asyncio.run(engine.start())

    assert len(instances) == 1
    assert engine._mllm_instance is None
    assert engine._model is None
    assert engine._processor is None
    assert engine._mllm_scheduler is None
    assert engine._loaded is False


def _bare_mllm_scheduler(monkeypatch, language_model):
    import mlx_lm.sample_utils as sample_utils
    import vllm_mlx.mllm_scheduler as scheduler_mod

    constructed = []

    class FakeBatchGenerator:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.language_model = language_model
            self.prefix_cache = None

    monkeypatch.setattr(sample_utils, "make_sampler", lambda **kwargs: object())
    monkeypatch.setattr(scheduler_mod, "MLLMBatchGenerator", FakeBatchGenerator)
    scheduler = object.__new__(scheduler_mod.MLLMScheduler)
    scheduler.batch_generator = None
    scheduler.model = SimpleNamespace(language_model=language_model)
    scheduler.processor = object()
    scheduler.mm_processor = object()
    scheduler.stop_tokens = set()
    scheduler.config = scheduler_mod.MLLMSchedulerConfig(
        enable_prefix_cache=False,
        enable_mtp=True,
    )
    return scheduler, scheduler_mod, constructed


def test_mllm_scheduler_never_installs_native_qwen_mtp(monkeypatch):
    import vllm_mlx.mllm_batch_generator as batch_generator_mod

    language_model = _model(
        "qwen3_5",
        capability=SimpleNamespace(supported=True, reason=None),
        mtp=object(),
    )
    scheduler, _, constructed = _bare_mllm_scheduler(monkeypatch, language_model)
    installed = []
    hooked = []
    ssd_started = []
    scheduler.config.chunked_prefill_tokens = 32
    scheduler.config.ssd_cache_dir = "/not-used"
    monkeypatch.setattr(
        batch_generator_mod,
        "install_mtp_mllm",
        lambda *args, **kwargs: installed.append((args, kwargs)),
    )
    monkeypatch.setattr(
        batch_generator_mod,
        "install_chunked_prefill_mllm",
        lambda *args, **kwargs: hooked.append((args, kwargs)),
    )

    class UnexpectedSSDTier:
        def __init__(self, *args, **kwargs):
            ssd_started.append("constructed")

    import vllm_mlx.ssd_cache as ssd_cache_mod

    monkeypatch.setattr(ssd_cache_mod, "SSDCacheTier", UnexpectedSSDTier)

    with pytest.raises(
        RuntimeError,
        match="^native_mtp_continuous_batching_unsupported$",
    ):
        scheduler._ensure_batch_generator()
    assert constructed == []
    assert hooked == []
    assert ssd_started == []
    assert installed == []
    assert scheduler.batch_generator is None


def test_mllm_scheduler_preserves_qwen3_next_legacy_install(monkeypatch):
    import vllm_mlx.mllm_batch_generator as batch_generator_mod

    language_model = _model("qwen3_next", mtp=object())
    scheduler, _, _ = _bare_mllm_scheduler(monkeypatch, language_model)
    installed = []
    monkeypatch.setattr(
        batch_generator_mod,
        "install_mtp_mllm",
        lambda *args, **kwargs: installed.append((args, kwargs)),
    )

    scheduler._ensure_batch_generator()

    assert len(installed) == 1
    assert installed[0][0][0] is scheduler.batch_generator
    assert installed[0][0][1] is language_model
    assert installed[0][1] == {"num_draft_tokens": 1}
