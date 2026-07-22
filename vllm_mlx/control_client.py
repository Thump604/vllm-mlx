# SPDX-License-Identifier: Apache-2.0
"""Synchronous client for the versioned product control and inference APIs."""

from __future__ import annotations

import json
from typing import Any, Iterator, Mapping
from urllib.parse import quote

import requests  # type: ignore[import-untyped]

from .control_api import CONTROL_API_VERSION, parse_api_version


class ControlClientError(RuntimeError):
    """Stable control API error returned to a product client."""

    def __init__(self, code: str, message: str, *, status_code: int | None = None):
        self.code = code
        self.status_code = status_code
        super().__init__(message)


class ControlClient:
    """Thin HTTP client that never owns runtime or lifecycle state."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        api_key: str | None = None,
        timeout: float | None = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.headers = {"Accept": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def capabilities(self) -> dict[str, Any]:
        return self._request_dict("GET", "/api/v1/control/capabilities")

    def catalog(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/v1/control/catalog")
        if not isinstance(data, list):
            raise ControlClientError("invalid_response", "catalog data must be a list")
        return data

    def profile(self, profile_id: str) -> dict[str, Any]:
        return self._request_dict(
            "GET", f"/api/v1/control/catalog/{quote(profile_id, safe='')}"
        )

    def install(
        self, profile: Mapping[str, Any], idempotency_key: str
    ) -> dict[str, Any]:
        profile_id = quote(str(profile["profile_id"]), safe="")
        return self._request_dict(
            "POST",
            f"/api/v1/control/models/{profile_id}/install",
            body={"profile": dict(profile), "idempotency_key": idempotency_key},
        )

    def activate(
        self,
        profile: Mapping[str, Any],
        idempotency_key: str,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request_dict(
            "PUT",
            "/api/v1/control/active",
            body={
                "profile": dict(profile),
                "idempotency_key": idempotency_key,
                "overrides": dict(overrides or {}),
            },
        )

    def stop(self, idempotency_key: str) -> dict[str, Any]:
        return self._request_dict(
            "POST",
            "/api/v1/control/active/stop",
            body={"idempotency_key": idempotency_key},
        )

    def status(self) -> dict[str, Any]:
        return self._request_dict("GET", "/api/v1/control/status")

    def operation(self, operation_id: str) -> dict[str, Any]:
        return self._request_dict(
            "GET", f"/api/v1/control/operations/{quote(operation_id, safe='')}"
        )

    def cancel_operation(
        self, operation_id: str, idempotency_key: str
    ) -> dict[str, Any]:
        return self._request_dict(
            "POST",
            f"/api/v1/control/operations/{quote(operation_id, safe='')}/cancel",
            body={"idempotency_key": idempotency_key},
        )

    def diagnostics(self) -> dict[str, Any]:
        return self._request_dict("GET", "/api/v1/control/diagnostics")

    def chat(
        self,
        *,
        model: str,
        message: str,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any] | Iterator[str]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": stream,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            json=body,
            timeout=self.timeout,
            stream=stream,
        )
        response.raise_for_status()
        if not stream:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ControlClientError("invalid_response", "chat response is invalid")
            return payload
        return self._stream_chat(response)

    def _stream_chat(self, response: requests.Response) -> Iterator[str]:
        complete = False
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    complete = True
                    return
                try:
                    payload = json.loads(data)
                    content = payload["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                    raise ControlClientError(
                        "invalid_stream", "chat stream event is invalid"
                    ) from exc
                if isinstance(content, str) and content:
                    yield content
            if not complete:
                raise ControlClientError(
                    "incomplete_stream", "chat stream ended before the done event"
                )
        finally:
            response.close()

    def _request(
        self, method: str, path: str, *, body: Mapping[str, Any] | None = None
    ) -> Any:
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            headers=self.headers,
            json=dict(body) if body is not None else None,
            timeout=self.timeout,
        )
        try:
            envelope = response.json()
        except ValueError as exc:
            raise ControlClientError(
                "invalid_response",
                "control response is not JSON",
                status_code=response.status_code,
            ) from exc
        if not isinstance(envelope, dict):
            raise ControlClientError("invalid_response", "control envelope is invalid")
        api_version = envelope.get("api_version")
        self._require_compatible_server(str(api_version))
        error = envelope.get("error")
        if error is not None:
            if not isinstance(error, dict):
                raise ControlClientError("invalid_response", "control error is invalid")
            raise ControlClientError(
                str(error.get("code", "unknown_error")),
                str(error.get("message", "control operation failed")),
                status_code=response.status_code,
            )
        if not response.ok:
            raise ControlClientError(
                "http_error",
                f"control request failed with HTTP {response.status_code}",
                status_code=response.status_code,
            )
        if "data" not in envelope or "request_id" not in envelope:
            raise ControlClientError(
                "invalid_response", "control envelope is incomplete"
            )
        return envelope["data"]

    @staticmethod
    def _require_compatible_server(server_version: str) -> None:
        server = parse_api_version(server_version)
        client = parse_api_version(CONTROL_API_VERSION)
        if server[0] != client[0]:
            raise ControlClientError(
                "incompatible_api_version",
                f"server control API {server_version} is incompatible with client {CONTROL_API_VERSION}",
            )

    def _request_dict(
        self, method: str, path: str, *, body: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        data = self._request(method, path, body=body)
        if not isinstance(data, dict):
            raise ControlClientError(
                "invalid_response", "control data must be an object"
            )
        return data
