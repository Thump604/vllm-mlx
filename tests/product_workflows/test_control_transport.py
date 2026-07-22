# SPDX-License-Identifier: Apache-2.0

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.control import ProductControlService
from vllm_mlx.control.routes import create_control_router
from vllm_mlx.control_client import ControlClient
from vllm_mlx.lifecycle import ModelSpec, ResidencyManager
from vllm_mlx.product_workflows import install_to_code


class FakeEngine:
    async def start(self):
        return None

    async def stop(self):
        return None


class WorkflowRuntime:
    def __init__(self):
        self.active = None

    async def install(self, profile):
        return {"installed": profile["profile_id"]}

    async def activate(self, profile, overrides):
        self.active = {
            "profile_id": profile["profile_id"],
            "profile_revision": profile["profile_revision"],
            "subject_digest": profile["subject_digest"],
        }
        return {"loaded": True, "overrides": dict(overrides)}

    async def stop(self):
        self.active = None
        return {"loaded": False}

    async def remove(self, profile):
        return {"removed": profile["profile_id"]}

    def status(self):
        return {
            "active_profile": self.active,
            "state": "loaded" if self.active else "unloaded",
            "healthy": self.active is not None,
            "endpoint": "http://127.0.0.1:8080" if self.active else None,
        }

    def diagnostics(self):
        return {"active_profile": self.active}

    def operation_is_cancellable(self, operation_kind):
        return False


class RequestsCompatibleSession:
    """Expose the requests.Response properties used by ControlClient."""

    def __init__(self, client):
        self.client = client

    def request(self, method, url, **kwargs):
        response = self.client.request(method, url, **kwargs)
        response.ok = 200 <= response.status_code < 300
        return response


def test_install_to_code_runs_through_real_control_transport(
    product_catalog, tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "VLLM_MLX_LIFECYCLE_STATE_PATH", str(tmp_path / "lifecycle.json")
    )
    manager = ResidencyManager(lambda spec: FakeEngine())
    manager.register_model(ModelSpec("default", "placeholder"))
    service = ProductControlService(product_catalog, manager, WorkflowRuntime())
    app = FastAPI()
    app.include_router(create_control_router(lambda: service))

    with TestClient(app) as transport:
        client = ControlClient(
            "http://testserver", session=RequestsCompatibleSession(transport)
        )
        result = install_to_code(
            client,
            product_catalog,
            profile_id="golden-model",
            profile_revision=1,
            install_idempotency_key="install-golden",
            activate_idempotency_key="activate-golden",
            coding_client="openai",
            max_operation_polls=20,
        )

    assert result["install_operation"]["status"] == "succeeded"
    assert result["activate_operation"]["status"] == "succeeded"
    assert result["runtime_status"]["state"] == "loaded"
    assert result["coding_configuration"]["environment"]["OPENAI_BASE_URL"] == (
        "http://127.0.0.1:8080/v1"
    )
    manager._control.close()
