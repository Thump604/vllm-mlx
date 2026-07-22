# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from vllm_mlx.catalog import load_catalog
from vllm_mlx.control import ProductControlError, ProductControlService
from vllm_mlx.lifecycle import ModelSpec, ResidencyManager
from vllm_mlx.model_profile import compute_subject_digest


class FakeEngine:
    async def start(self):
        return None

    async def stop(self):
        return None


class RecordingRuntime:
    def __init__(self):
        self.calls = []
        self.block = None
        self.failure = None

    async def install(self, profile):
        self.calls.append(("install", profile["profile_id"]))
        if self.block is not None:
            await self.block.wait()
        return {"artifact_path": "/managed/model"}

    async def activate(self, profile, overrides):
        self.calls.append(("activate", profile["profile_id"], dict(overrides)))
        if self.failure is not None:
            raise self.failure
        return {"state": "loaded"}

    async def stop(self):
        self.calls.append(("stop",))
        return {"state": "unloaded"}

    async def remove(self, profile):
        self.calls.append(("remove", profile["profile_id"]))
        return {"removed": True}

    def status(self):
        return {"state": "loaded", "healthy": True}

    def diagnostics(self):
        return {"runtime": "ready"}

    def operation_is_cancellable(self, operation_kind):
        return operation_kind != "model.install"


@pytest.fixture
def product_service(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    profile = json.loads(
        (root / "schemas/examples/model-profile-v1.example.json").read_text()
    )
    profile["profile_id"] = "release-model"
    profile["profile_revision"] = 2
    profile["subject_digest"] = compute_subject_digest(profile)
    profile["qualification"] = {
        "status": "qualified",
        "reason": None,
        "evidence": [
            {
                "evidence_id": "test-pass",
                "kind": "unit_test",
                "location": "tests/test_product_control_service.py",
                "artifact_sha256": "1" * 64,
                "result": "pass",
                "hardware_fingerprint": "test-host",
                "workload_id": "service-contract",
                "subject_digest": profile["subject_digest"],
                "created_at": "2026-07-21T00:00:00Z",
            }
        ],
    }
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir()
    (catalog_root / "release.json").write_text(json.dumps(profile))

    monkeypatch.setenv(
        "VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "lifecycle.json")
    )
    manager = ResidencyManager(lambda spec: FakeEngine())
    manager.register_model(ModelSpec("default", "placeholder"))
    runtime = RecordingRuntime()
    service = ProductControlService(load_catalog(catalog_root), manager, runtime)
    yield service, manager, runtime, profile
    if manager._control is not None and manager._control.is_open:
        manager._control.close()


async def _terminal(service, operation_id):
    for _ in range(20):
        record = service.get_operation(operation_id)
        if record["status"] in {"succeeded", "failed", "cancelled"}:
            return record
        await asyncio.sleep(0)
    raise AssertionError("operation did not reach a terminal state")


@pytest.mark.anyio
async def test_install_binds_exact_profile_and_replays_idempotently(product_service):
    service, _manager, runtime, profile = product_service
    reference = {
        "profile_id": profile["profile_id"],
        "profile_revision": profile["profile_revision"],
        "subject_digest": profile["subject_digest"],
    }
    request = {"profile": reference, "idempotency_key": "install-release"}

    first = service.install(request, route_profile_id="release-model")
    terminal = await _terminal(service, first["operation_id"])
    replay = service.install(request, route_profile_id="release-model")

    assert terminal["status"] == "succeeded"
    assert terminal["result"] == {"artifact_path": "/managed/model"}
    assert replay == terminal
    assert runtime.calls == [("install", "release-model")]


@pytest.mark.anyio
async def test_activation_rejects_stale_subject_and_disallowed_override(
    product_service,
):
    service, _manager, runtime, profile = product_service
    reference = {
        "profile_id": profile["profile_id"],
        "profile_revision": profile["profile_revision"],
        "subject_digest": "0" * 64,
    }
    with pytest.raises(ProductControlError) as stale:
        service.activate(
            {
                "profile": reference,
                "idempotency_key": "activate-release",
                "overrides": {},
            }
        )
    assert stale.value.code == "profile_subject_mismatch"

    reference["subject_digest"] = profile["subject_digest"]
    with pytest.raises(ProductControlError) as disallowed:
        service.activate(
            {
                "profile": reference,
                "idempotency_key": "activate-release",
                "overrides": {"features.mtp": True},
            }
        )
    assert disallowed.value.code == "invalid_request"
    assert runtime.calls == []


def test_activation_rejects_unqualified_profile(product_service):
    service, _manager, runtime, profile = product_service
    profile["qualification"] = {
        "status": "not_qualified",
        "reason": "qualification pending",
        "evidence": [],
    }
    service.catalog._profiles = (profile,)
    reference = {
        "profile_id": profile["profile_id"],
        "profile_revision": profile["profile_revision"],
        "subject_digest": profile["subject_digest"],
    }

    with pytest.raises(ProductControlError) as caught:
        service.activate(
            {
                "profile": reference,
                "idempotency_key": "activate-unqualified",
                "overrides": {},
            }
        )

    assert caught.value.code == "profile_not_qualified"
    assert runtime.calls == []


@pytest.mark.anyio
async def test_idempotency_conflict_does_not_start_second_operation(product_service):
    service, _manager, runtime, profile = product_service
    reference = {
        "profile_id": profile["profile_id"],
        "profile_revision": profile["profile_revision"],
        "subject_digest": profile["subject_digest"],
    }
    first = service.activate(
        {
            "profile": reference,
            "idempotency_key": "activate-release",
            "overrides": {},
        }
    )
    await _terminal(service, first["operation_id"])

    with pytest.raises(ProductControlError) as conflict:
        service.activate(
            {
                "profile": reference,
                "idempotency_key": "activate-release",
                "overrides": {"limits.max_output_tokens": 1024},
            }
        )
    assert conflict.value.code == "idempotency_conflict"
    assert runtime.calls == [("activate", "release-model", {})]


@pytest.mark.anyio
async def test_non_cancellable_install_stays_running(product_service):
    service, _manager, runtime, profile = product_service
    runtime.block = asyncio.Event()
    reference = {
        "profile_id": profile["profile_id"],
        "profile_revision": profile["profile_revision"],
        "subject_digest": profile["subject_digest"],
    }
    operation = service.install(
        {"profile": reference, "idempotency_key": "install-blocked"},
        route_profile_id="release-model",
    )
    await asyncio.sleep(0)

    with pytest.raises(ProductControlError) as caught:
        await service.cancel(
            operation["operation_id"], {"idempotency_key": "cancel-install"}
        )
    assert caught.value.code == "operation_not_cancellable"
    assert service.get_operation(operation["operation_id"])["status"] == "running"
    runtime.block.set()
    await _terminal(service, operation["operation_id"])


@pytest.mark.anyio
async def test_operation_failure_response_does_not_leak_runtime_path(product_service):
    service, _manager, runtime, profile = product_service
    runtime.failure = RuntimeError("failed under /private/secret/model")
    operation = service.activate(
        {
            "profile": {
                "profile_id": profile["profile_id"],
                "profile_revision": profile["profile_revision"],
                "subject_digest": profile["subject_digest"],
            },
            "idempotency_key": "activate-failure",
            "overrides": {},
        }
    )

    terminal = await _terminal(service, operation["operation_id"])

    assert terminal["status"] == "failed"
    assert terminal["error"] == {
        "code": "runtime_unavailable",
        "message": "operation failed; inspect protected runtime diagnostics",
    }
    assert "/private/secret" not in str(terminal)


@pytest.mark.anyio
async def test_cancel_is_idempotent_and_rejects_key_reuse(product_service):
    service, _manager, runtime, profile = product_service
    runtime.block = asyncio.Event()
    runtime.operation_is_cancellable = lambda operation_kind: True
    operation = service.install(
        {
            "profile": {
                "profile_id": profile["profile_id"],
                "profile_revision": profile["profile_revision"],
                "subject_digest": profile["subject_digest"],
            },
            "idempotency_key": "install-for-cancel",
        },
        route_profile_id=profile["profile_id"],
    )
    first = await service.cancel(
        operation["operation_id"], {"idempotency_key": "cancel-once"}
    )
    replay = await service.cancel(
        operation["operation_id"], {"idempotency_key": "cancel-once"}
    )

    assert first["status"] == "cancelled"
    assert replay == first

    stop = service.stop({"idempotency_key": "stop-after-cancel"})
    await _terminal(service, stop["operation_id"])
    with pytest.raises(ProductControlError) as conflict:
        await service.cancel(stop["operation_id"], {"idempotency_key": "cancel-once"})
    assert conflict.value.code == "idempotency_conflict"
