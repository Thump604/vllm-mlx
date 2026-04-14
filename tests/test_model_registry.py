# SPDX-License-Identifier: Apache-2.0
"""Tests for registry-backed multi-model serving."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from vllm_mlx.engine.base import BaseEngine, GenerationOutput
from vllm_mlx.model_registry import (
    ContentionPolicy,
    LoadedModel,
    ModelManager,
    RegisteredModel,
    RegistryManagerConfig,
    RegistryServeDefaults,
    ResolvedModelConfig,
    ServingProfile,
    load_registry_config,
)
from vllm_mlx.utils.download import DownloadConfig


class FakeEngine(BaseEngine):
    """Small test double for model lifecycle behaviour."""

    def __init__(
        self, config: ResolvedModelConfig, start_gate: asyncio.Event | None = None
    ):
        self._config = config
        self._start_gate = start_gate
        self.started = 0
        self.stopped = 0

    @property
    def model_name(self) -> str:
        return self._config.resolved_source

    @property
    def is_mllm(self) -> bool:
        return False

    @property
    def tokenizer(self) -> Any:
        return None

    async def start(self) -> None:
        if self._start_gate is not None:
            await self._start_gate.wait()
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def generate(self, *args, **kwargs) -> GenerationOutput:
        return GenerationOutput(text="ok")

    async def stream_generate(self, *args, **kwargs):
        yield GenerationOutput(text="ok", new_text="ok", finished=True)

    async def chat(self, *args, **kwargs) -> GenerationOutput:
        return GenerationOutput(text="ok")

    async def stream_chat(self, *args, **kwargs):
        yield GenerationOutput(text="ok", new_text="ok", finished=True)


def _defaults() -> RegistryServeDefaults:
    return RegistryServeDefaults(
        continuous_batching=False,
        force_mllm=False,
        enable_mtp=False,
        prefill_step_size=2048,
        specprefill_enabled=False,
        specprefill_threshold=8192,
        specprefill_keep_pct=0.3,
        specprefill_draft_model=None,
        stream_interval=1,
        gpu_memory_utilization=0.9,
        scheduler_config=None,
        max_tokens=32768,
        download_config=DownloadConfig(),
        serving_profile=ServingProfile(),
    )


def _manager_config(
    *,
    budget_gb: float,
    strategy: str = "wait_then_fail",
    wait_timeout_s: float | None = 1.0,
    preempt_after_s: float | None = None,
) -> RegistryManagerConfig:
    return RegistryManagerConfig(
        memory_budget_bytes=int(budget_gb * (1024**3)),
        policy=ContentionPolicy(
            strategy=strategy,
            wait_timeout_s=wait_timeout_s,
            preempt_after_s=preempt_after_s,
        ),
    )


def _registry(tmp_path: Path, sizes_gb: dict[str, float]) -> dict[str, RegisteredModel]:
    registry = {}
    for name, size_gb in sizes_gb.items():
        source = tmp_path / name
        source.mkdir()
        registry[name] = RegisteredModel(
            name=name,
            source=str(source),
            estimated_memory_bytes=int(size_gb * (1024**3)),
        )
    return registry


@pytest.mark.asyncio
async def test_acquire_shares_single_inflight_load(tmp_path):
    registry = _registry(tmp_path, {"alpha": 4})
    gate = asyncio.Event()
    created: list[FakeEngine] = []

    def engine_factory(config: ResolvedModelConfig) -> FakeEngine:
        engine = FakeEngine(config, start_gate=gate)
        created.append(engine)
        return engine

    manager = ModelManager(
        _manager_config(budget_gb=8),
        registry,
        _defaults(),
        engine_factory=engine_factory,
    )

    first = asyncio.create_task(manager.acquire("alpha"))
    await asyncio.sleep(0)
    second = asyncio.create_task(manager.acquire("alpha"))
    await asyncio.sleep(0)

    assert len(created) == 1
    gate.set()

    lease_a = await first
    lease_b = await second
    assert lease_a.engine is lease_b.engine
    assert created[0].started == 1

    await lease_a.release()
    await lease_b.release()


@pytest.mark.asyncio
async def test_idle_lru_eviction_preserves_budget(tmp_path):
    registry = _registry(tmp_path, {"alpha": 4, "beta": 4, "gamma": 5})
    created: dict[str, FakeEngine] = {}

    def engine_factory(config: ResolvedModelConfig) -> FakeEngine:
        engine = FakeEngine(config)
        created[config.entry.name] = engine
        return engine

    manager = ModelManager(
        _manager_config(budget_gb=9),
        registry,
        _defaults(),
        engine_factory=engine_factory,
    )

    lease = await manager.acquire("alpha")
    await lease.release()
    await asyncio.sleep(0.01)

    lease = await manager.acquire("beta")
    await lease.release()
    await asyncio.sleep(0.01)

    lease = await manager.acquire("gamma")
    await lease.release()

    assert "alpha" not in manager._loaded
    assert "beta" in manager._loaded
    assert "gamma" in manager._loaded
    assert created["alpha"].stopped == 1
    assert created["beta"].stopped == 0


@pytest.mark.asyncio
async def test_preempt_policy_cancels_active_request_and_loads_waiting_model(tmp_path):
    registry = _registry(tmp_path, {"alpha": 8, "beta": 8})
    created: dict[str, FakeEngine] = {}

    def engine_factory(config: ResolvedModelConfig) -> FakeEngine:
        engine = FakeEngine(config)
        created[config.entry.name] = engine
        return engine

    manager = ModelManager(
        _manager_config(
            budget_gb=10,
            strategy="preempt",
            wait_timeout_s=2.0,
        ),
        registry,
        _defaults(),
        engine_factory=engine_factory,
    )

    acquired = asyncio.Event()
    cancelled = asyncio.Event()

    async def hold_alpha() -> None:
        lease = await manager.acquire("alpha")
        acquired.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            await lease.release()

    active_task = asyncio.create_task(hold_alpha())
    await acquired.wait()

    beta_lease = await manager.acquire("beta")
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    await beta_lease.release()

    with pytest.raises(asyncio.CancelledError):
        await active_task

    assert "beta" in manager._loaded
    assert "alpha" not in manager._loaded
    assert created["alpha"].stopped == 1
    assert created["beta"].started == 1


@pytest.mark.asyncio
async def test_different_models_do_not_run_concurrently_by_default(tmp_path):
    registry = _registry(tmp_path, {"alpha": 4, "beta": 4})
    created: dict[str, FakeEngine] = {}

    def engine_factory(config: ResolvedModelConfig) -> FakeEngine:
        engine = FakeEngine(config)
        created[config.entry.name] = engine
        return engine

    manager = ModelManager(
        _manager_config(budget_gb=12, wait_timeout_s=1.0),
        registry,
        _defaults(),
        engine_factory=engine_factory,
    )

    acquired = asyncio.Event()
    released = asyncio.Event()

    async def hold_alpha() -> None:
        lease = await manager.acquire("alpha")
        acquired.set()
        try:
            await released.wait()
        finally:
            await lease.release()

    alpha_task = asyncio.create_task(hold_alpha())
    await acquired.wait()

    beta_task = asyncio.create_task(manager.acquire("beta"))
    await asyncio.sleep(0.05)
    assert not beta_task.done()

    released.set()
    beta_lease = await asyncio.wait_for(beta_task, timeout=1.0)
    await beta_lease.release()
    await alpha_task

    assert created["alpha"].started == 1
    assert created["beta"].started == 1


def test_load_registry_config_accepts_phase1_contract_schema(tmp_path):
    config_path = tmp_path / "registry.yaml"
    config_path.write_text("""
schema_version: 1
policy_defaults:
  memory_budget_gb: 96
  contention_policy:
    strategy: wait_then_preempt
    wait_timeout_s: 15
models:
  - id: qwen3.5-27b
    display_name: Qwen 3.5 27B
    source: /models/qwen27
    family: qwen3.5
    architecture: dense
    execution_class: shared_candidate
    estimated_memory_gb: 31
    supports_mtp: true
    multimodal: true
    serving_profile:
      force_mllm: true
      continuous_batching: true
      prefill_step_size: 256
      tool_call_parser: qwen3_coder
      reasoning_parser: qwen3
      enable_auto_tool_choice: true
      enable_thinking_default: true
      specprefill:
        enabled: true
        threshold: 8192
        keep_pct: 0.3
    draft_model:
      id: qwen3.5-2b-draft
      source: /models/qwen2b
      estimated_memory_gb: 3
model_presets:
  - id: coding-quality
    display_name: Coding (Quality)
    model_id: qwen3.5-27b
    priority_class: interactive
    sampling_profile:
      temperature: 0.6
      top_p: 0.95
      enable_thinking: true
    request_policy:
      max_tokens: 32768
      timeout_s: 2400
service_presets: []
discussion_profiles: []
""")

    manager, registry = load_registry_config(config_path, _defaults())

    assert manager.memory_budget_bytes == 96 * (1024**3)
    assert manager.policy.strategy == "wait_then_preempt"
    assert registry["qwen3.5-27b"].tool_call_parser == "qwen3_coder"
    assert registry["qwen3.5-27b"].reasoning_parser == "qwen3"
    assert registry["qwen3.5-27b"].enable_mtp is True
    assert registry["qwen3.5-27b"].force_mllm is True
    assert registry["qwen3.5-27b"].specprefill_draft_model == "/models/qwen2b"


@pytest.mark.asyncio
async def test_non_local_registry_entry_requires_explicit_memory_estimate():
    registry = {
        "remote": RegisteredModel(
            name="remote",
            source="mlx-community/some-remote-model",
        )
    }
    manager = ModelManager(
        _manager_config(budget_gb=8),
        registry,
        _defaults(),
        engine_factory=lambda config: FakeEngine(config),
    )

    with pytest.raises(ValueError, match="estimated_memory_gb"):
        await manager.acquire("remote")


def test_resolve_model_config_merges_per_model_serving_profile(tmp_path):
    registry = _registry(tmp_path, {"gemma": 4, "qwen": 5})
    registry["gemma"] = RegisteredModel(
        name="gemma",
        source=registry["gemma"].source,
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        enable_auto_tool_choice=True,
    )
    registry["qwen"] = RegisteredModel(
        name="qwen",
        source=registry["qwen"].source,
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        enable_auto_tool_choice=True,
        enable_thinking_default=True,
    )

    manager = ModelManager(
        _manager_config(budget_gb=12),
        registry,
        _defaults(),
        engine_factory=lambda config: FakeEngine(config),
    )

    gemma = manager._resolve_model_config(registry["gemma"], registry["gemma"].source)
    qwen = manager._resolve_model_config(registry["qwen"], registry["qwen"].source)

    assert gemma.serving_profile.tool_call_parser == "gemma4"
    assert gemma.serving_profile.reasoning_parser == "gemma4"
    assert gemma.serving_profile.enable_auto_tool_choice is True
    assert gemma.serving_profile.enable_thinking_default is None

    assert qwen.serving_profile.tool_call_parser == "qwen3_coder"
    assert qwen.serving_profile.reasoning_parser == "qwen3"
    assert qwen.serving_profile.enable_auto_tool_choice is True
    assert qwen.serving_profile.enable_thinking_default is True
