# SPDX-License-Identifier: Apache-2.0

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.control import ProductControlError
from vllm_mlx.control_api import CONTROL_API_VERSION
from vllm_mlx.control.routes import (
    create_control_router,
    product_control_error_handler,
)


class RouteService:
    def __init__(self):
        self.calls = []

    def list_catalog(self):
        return [{"profile_id": "laguna"}]

    def get_profile(self, profile_id):
        if profile_id != "laguna":
            raise ProductControlError("profile_not_found", "missing")
        return {"profile_id": profile_id}

    def install(self, request, *, route_profile_id):
        self.calls.append(("install", request, route_profile_id))
        return {"operation_id": "install-1", "status": "queued"}

    def activate(self, request):
        self.calls.append(("activate", request))
        return {"operation_id": "activate-1", "status": "queued"}

    def stop(self, request):
        return {"operation_id": "stop-1", "status": "queued"}

    def remove(self, request, *, route_profile_id):
        return {"operation_id": "remove-1", "status": "queued"}

    def get_operation(self, operation_id):
        return {"operation_id": operation_id, "status": "succeeded"}

    async def cancel(self, operation_id, request):
        return {"operation_id": operation_id, "status": "cancelled"}

    def status(self):
        return {"state": "unloaded"}

    def diagnostics(self):
        return {"runtime": "ready"}


def _client():
    service = RouteService()
    app = FastAPI()
    app.include_router(create_control_router(lambda: service))
    return TestClient(app), service


def _reference():
    return {
        "profile_id": "laguna",
        "profile_revision": 1,
        "subject_digest": "a" * 64,
    }


def test_control_routes_publish_capability_and_catalog_envelopes():
    client, _service = _client()

    capabilities = client.get("/api/v1/control/capabilities").json()
    catalog = client.get("/api/v1/control/catalog").json()

    assert capabilities["data"]["kind"] == "vllm-mlx-control-api"
    assert catalog["data"] == [{"profile_id": "laguna"}]
    assert capabilities["error"] is None
    assert catalog["error"] is None


def test_install_route_parses_exact_request_before_service():
    client, service = _client()
    response = client.post(
        "/api/v1/control/models/laguna/install",
        json={"profile": _reference(), "idempotency_key": "install-laguna"},
    )

    assert response.status_code == 202
    assert response.json()["data"]["operation_id"] == "install-1"
    assert service.calls == [
        (
            "install",
            {"profile": _reference(), "idempotency_key": "install-laguna"},
            "laguna",
        )
    ]


def test_install_route_rejects_route_identity_mismatch_before_service():
    client, service = _client()
    response = client.post(
        "/api/v1/control/models/other/install",
        json={"profile": _reference(), "idempotency_key": "install-laguna"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert service.calls == []


def test_activation_route_rejects_hidden_fields():
    client, service = _client()
    response = client.put(
        "/api/v1/control/active",
        json={
            "profile": _reference(),
            "idempotency_key": "activate-laguna",
            "hidden": True,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert service.calls == []


def test_control_routes_preserve_stable_service_error():
    client, _service = _client()
    response = client.get("/api/v1/control/catalog/missing")

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "profile_not_found",
        "message": "missing",
    }


def test_control_auth_failure_preserves_versioned_error_envelope():
    async def reject_credentials():
        raise ProductControlError("authentication_failed", "API key required")

    service = RouteService()
    app = FastAPI()
    app.add_exception_handler(ProductControlError, product_control_error_handler)
    app.include_router(
        create_control_router(lambda: service),
        dependencies=[Depends(reject_credentials)],
    )

    response = TestClient(app).get("/api/v1/control/catalog")

    assert response.status_code == 401
    assert response.json()["api_version"] == CONTROL_API_VERSION
    assert response.json()["data"] is None
    assert response.json()["error"] == {
        "code": "authentication_failed",
        "message": "API key required",
    }
