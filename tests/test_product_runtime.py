# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vllm_mlx.control.runtime import (
    ManagedProductRuntime,
    ManagedRuntimeError,
    profile_to_model_spec,
    verify_profile_artifact,
)
from vllm_mlx.lifecycle import ModelSpec, ResidencyManager
from vllm_mlx.model_profile import compute_subject_digest


class FakeEngine:
    def __init__(self, spec):
        self.spec = spec
        self.started = 0
        self.stopped = 0

    async def start(self):
        self.started += 1

    async def stop(self):
        self.stopped += 1


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(root: Path, *, converted=False):
    repository_root = Path(__file__).parents[1]
    profile = json.loads(
        (repository_root / "schemas/examples/model-profile-v1.example.json").read_text()
    )
    files = {
        "config.json": b'{"model_type":"test"}',
        "tokenizer.json": b'{"version":"1"}',
        "chat_template.jinja": b"{{ messages }}",
        "generation_config.json": b'{"temperature":1.0}',
        "model.safetensors.index.json": b'{"weight_map":{}}',
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    profile["profile_id"] = "release-model"
    profile["profile_revision"] = 1
    profile["identity"]["artifact_id"] = "release-model"
    profile["artifact"]["hashes"] = {
        "config_sha256": _hash(root / "config.json"),
        "tokenizer_sha256": _hash(root / "tokenizer.json"),
        "chat_template_sha256": _hash(root / "chat_template.jinja"),
        "generation_config_sha256": _hash(root / "generation_config.json"),
        "weights_manifest_sha256": _hash(root / "model.safetensors.index.json"),
    }
    profile["artifact"]["quantization"]["source"] = (
        "conversion_manifest" if converted else "local_artifact_config"
    )
    profile["subject_digest"] = compute_subject_digest(profile)
    return profile


def test_artifact_verification_is_content_bound(tmp_path):
    profile = _profile(tmp_path)
    evidence = verify_profile_artifact(profile, tmp_path)
    assert evidence["verified_hashes"] == profile["artifact"]["hashes"]

    (tmp_path / "config.json").write_text("changed")
    with pytest.raises(ManagedRuntimeError, match="does not match"):
        verify_profile_artifact(profile, tmp_path)


def test_profile_to_model_spec_applies_only_declared_overrides(tmp_path):
    profile = _profile(tmp_path)
    profile["serving"]["features"]["continuous_batching"].update(
        mode="available_on_activation",
        settings={"max_concurrency": 3},
    )
    profile["subject_digest"] = compute_subject_digest(profile)

    spec = profile_to_model_spec(
        profile,
        tmp_path,
        overrides={
            "features.continuous_batching": True,
            "limits.max_output_tokens": 2048,
        },
    )

    assert spec.use_batching is True
    assert spec.scheduler_config.max_num_seqs == 3
    assert spec.scheduler_config.max_kv_size == 32768
    assert spec.max_tokens == 2048
    assert spec.profile_id == "release-model"


@pytest.mark.anyio
async def test_bound_converted_artifact_activates_and_external_remove_is_refused(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "external"
    artifact.mkdir()
    profile = _profile(artifact, converted=True)
    monkeypatch.setenv(
        "VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "lifecycle.json")
    )
    engines = []

    async def factory(spec):
        engine = FakeEngine(spec)
        engines.append(engine)
        return engine

    manager = ResidencyManager(factory)
    manager.register_model(ModelSpec("default", "placeholder"))
    applied = []
    runtime = ManagedProductRuntime(
        manager,
        model_root=tmp_path / "managed",
        endpoint="http://127.0.0.1:8080",
        artifact_bindings={"release-model": artifact},
        apply_profile=lambda selected, spec: applied.append(
            (selected["profile_id"], spec.profile_id)
        ),
    )

    installed = await runtime.install(profile)
    activated = await runtime.activate(profile, {})

    assert installed["managed"] is False
    assert activated["active_profile"]["profile_id"] == "release-model"
    assert activated["healthy"] is True
    assert applied == [("release-model", "release-model")]
    with pytest.raises(ManagedRuntimeError, match="externally bound"):
        await runtime.remove(profile)
    await manager.shutdown()


@pytest.mark.anyio
async def test_first_activation_requires_no_placeholder_resident(tmp_path, monkeypatch):
    artifact = tmp_path / "external"
    artifact.mkdir()
    profile = _profile(artifact, converted=True)
    monkeypatch.setenv(
        "VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "lifecycle.json")
    )

    async def factory(spec):
        return FakeEngine(spec)

    manager = ResidencyManager(factory)
    runtime = ManagedProductRuntime(
        manager,
        model_root=tmp_path / "managed",
        endpoint="http://127.0.0.1:8080",
        artifact_bindings={"release-model": artifact},
    )

    assert runtime.status()["state"] == "unloaded"
    await runtime.install(profile)
    activated = await runtime.activate(profile, {})
    assert activated["state"] == "loaded"
    assert manager.get_spec("default").profile_id == "release-model"
    await manager.shutdown()


@pytest.mark.anyio
async def test_failed_first_activation_clears_persisted_candidate(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "external"
    artifact.mkdir()
    profile = _profile(artifact, converted=True)
    state_path = tmp_path / "lifecycle.json"
    monkeypatch.setenv("VLLM_MLX_LIFECYCLE_STATE_PATH", str(state_path))

    async def failing_factory(spec):
        raise RuntimeError("engine start failed")

    manager = ResidencyManager(failing_factory)
    runtime = ManagedProductRuntime(
        manager,
        model_root=tmp_path / "managed",
        endpoint="http://127.0.0.1:8080",
        artifact_bindings={"release-model": artifact},
    )
    await runtime.install(profile)

    with pytest.raises(RuntimeError, match="engine start failed"):
        await runtime.activate(profile, {})

    assert runtime.status()["active_profile"] is None
    assert manager._control.model_key is None
    manager._control.close()

    restarted = ResidencyManager(failing_factory)
    assert restarted._control.model_key is None
    assert restarted.list_status() == []
    restarted._control.close()


@pytest.mark.anyio
async def test_converted_profile_requires_explicit_artifact_binding(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "fixture"
    artifact.mkdir()
    profile = _profile(artifact, converted=True)
    monkeypatch.setenv(
        "VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "lifecycle.json")
    )
    manager = ResidencyManager(lambda spec: FakeEngine(spec))
    manager.register_model(ModelSpec("default", "placeholder"))
    runtime = ManagedProductRuntime(
        manager,
        model_root=tmp_path / "managed",
        endpoint="http://127.0.0.1:8080",
    )

    with pytest.raises(ManagedRuntimeError, match="converted-artifact binding"):
        await runtime.install(profile)
    manager._control.close()
