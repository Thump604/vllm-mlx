# SPDX-License-Identifier: Apache-2.0
"""FastAPI transport for the versioned product control service."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from vllm_mlx.control_api import (
    CONTROL_API_VERSION,
    build_control_api_descriptor,
    parse_activation_request,
    parse_idempotent_request,
    parse_profile_mutation_request,
)

from .service import ProductControlError, ProductControlService


def _envelope(data: Any = None, error: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "api_version": CONTROL_API_VERSION,
        "request_id": uuid.uuid4().hex,
        "data": data,
        "error": error,
    }


def _error_response(exc: ProductControlError) -> JSONResponse:
    statuses = {
        "invalid_request": 400,
        "authentication_failed": 401,
        "profile_not_qualified": 409,
        "profile_not_found": 404,
        "operation_not_found": 404,
        "profile_revision_stale": 409,
        "profile_subject_mismatch": 409,
        "lifecycle_conflict": 409,
        "idempotency_conflict": 409,
        "operation_not_cancellable": 409,
        "runtime_unavailable": 503,
    }
    return JSONResponse(
        status_code=statuses.get(exc.code, 500),
        content=_envelope(error={"code": exc.code, "message": str(exc)}),
    )


async def product_control_error_handler(
    _request: Request, exc: Exception
) -> JSONResponse:
    """Preserve the control envelope for dependency and service failures."""
    if not isinstance(exc, ProductControlError):
        raise exc
    return _error_response(exc)


def _invalid_request(exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=_envelope(error={"code": "invalid_request", "message": str(exc)}),
    )


def create_control_router(
    get_service: Callable[[], ProductControlService],
) -> APIRouter:
    """Create the stable control router around one injected service provider."""
    router = APIRouter(prefix="/api/v1/control", tags=["product-control"])

    @router.get("/capabilities")
    async def capabilities():
        return _envelope(build_control_api_descriptor())

    @router.get("/catalog")
    async def catalog():
        try:
            return _envelope(get_service().list_catalog())
        except ProductControlError as exc:
            return _error_response(exc)

    @router.get("/catalog/{profile_id}")
    async def profile(profile_id: str):
        try:
            return _envelope(get_service().get_profile(profile_id))
        except ProductControlError as exc:
            return _error_response(exc)

    @router.post("/models/{profile_id}/install")
    async def install(profile_id: str, request: Request):
        try:
            body = await request.json()
            parsed = parse_profile_mutation_request(body, route_profile_id=profile_id)
            return JSONResponse(
                status_code=202,
                content=_envelope(
                    get_service().install(parsed, route_profile_id=profile_id)
                ),
            )
        except ProductControlError as exc:
            return _error_response(exc)
        except (TypeError, ValueError) as exc:
            return _invalid_request(exc)

    @router.put("/active")
    async def activate(request: Request):
        try:
            parsed = parse_activation_request(await request.json())
            return JSONResponse(
                status_code=202,
                content=_envelope(get_service().activate(parsed)),
            )
        except ProductControlError as exc:
            return _error_response(exc)
        except (TypeError, ValueError) as exc:
            return _invalid_request(exc)

    @router.post("/active/stop")
    async def stop(request: Request):
        try:
            parsed = parse_idempotent_request(await request.json())
            return JSONResponse(
                status_code=202,
                content=_envelope(get_service().stop(parsed)),
            )
        except ProductControlError as exc:
            return _error_response(exc)
        except (TypeError, ValueError) as exc:
            return _invalid_request(exc)

    @router.post("/models/{profile_id}/remove")
    async def remove(profile_id: str, request: Request):
        try:
            body = await request.json()
            parsed = parse_profile_mutation_request(body, route_profile_id=profile_id)
            return JSONResponse(
                status_code=202,
                content=_envelope(
                    get_service().remove(parsed, route_profile_id=profile_id)
                ),
            )
        except ProductControlError as exc:
            return _error_response(exc)
        except (TypeError, ValueError) as exc:
            return _invalid_request(exc)

    @router.get("/operations/{operation_id}")
    async def operation(operation_id: str):
        try:
            return _envelope(get_service().get_operation(operation_id))
        except ProductControlError as exc:
            return _error_response(exc)

    @router.post("/operations/{operation_id}/cancel")
    async def cancel(operation_id: str, request: Request):
        try:
            parsed = parse_idempotent_request(await request.json())
            return _envelope(await get_service().cancel(operation_id, parsed))
        except ProductControlError as exc:
            return _error_response(exc)
        except (TypeError, ValueError) as exc:
            return _invalid_request(exc)

    @router.get("/status")
    async def status():
        try:
            return _envelope(get_service().status())
        except ProductControlError as exc:
            return _error_response(exc)

    @router.get("/diagnostics")
    async def diagnostics():
        try:
            return _envelope(get_service().diagnostics())
        except ProductControlError as exc:
            return _error_response(exc)

    return router
