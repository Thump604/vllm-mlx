# SPDX-License-Identifier: Apache-2.0
"""Real HTTP/stream routes with deterministic scheduler-capacity failures."""

import asyncio
import json

import httpx
import pytest

from vllm_mlx.admission import AdmissionController


@pytest.fixture
def capacity_server(monkeypatch):
    import vllm_mlx.server as server

    class CapacityEngine:
        model_name = "admission-test"
        is_mllm = False
        tokenizer = None
        preserve_native_tool_format = False

        def __init__(self):
            self.admission = AdmissionController(max_requests=1)
            self.admission.reserve("occupied", 1)
            self.prompt_tokens = 1

        def submit(self):
            self.admission.reserve("rejected", self.prompt_tokens)
            raise AssertionError("Expected the real admission controller to reject")

        async def generate(self, **kwargs):
            self.submit()

        async def chat(self, **kwargs):
            self.submit()

        async def stream_generate(self, **kwargs):
            self.submit()
            yield  # Makes the engine boundary an async generator.

        async def stream_chat(self, **kwargs):
            self.submit()
            yield

    engine = CapacityEngine()
    releases = []

    async def acquire(*args, **kwargs):
        return engine

    async def release(*args, **kwargs):
        releases.append(True)

    for name, value in {
        "_engine": engine,
        "_model_name": engine.model_name,
        "_model_manager": None,
        "_api_key": None,
        "_reasoning_parser": None,
        "_tool_parser_instance": None,
        "_tool_call_parser": None,
        "_enable_auto_tool_choice": False,
        "_rate_limiter": server.RateLimiter(enabled=False),
    }.items():
        monkeypatch.setattr(server, name, value)
    monkeypatch.setattr(server, "_acquire_default_engine_for_request", acquire)
    monkeypatch.setattr(server, "_release_engine_for_request", release)
    monkeypatch.setattr(server, "_release_default_engine", release)
    return server, engine, releases


def _request(server, route, stream):
    body = {"model": "admission-test", "stream": stream, "max_tokens": 8}
    if route == "completions":
        body["prompt"] = "hello"
    elif route == "responses":
        body.pop("max_tokens")
        body["input"] = "hello"
    else:
        body["messages"] = [{"role": "user", "content": "hello"}]

    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            return await client.post(f"/v1/{route}", json=body)

    return asyncio.run(send())


@pytest.mark.parametrize(
    "route", ["completions", "chat/completions", "responses", "messages"]
)
def test_nonstream_capacity_has_retryable_http_error(capacity_server, route):
    server, engine, releases = capacity_server
    response = _request(server, route, False)
    assert response.status_code == 429, response.text
    assert response.headers["retry-after"] == "1"
    error = response.json()["error"]
    assert error["code"] == "scheduler_capacity_exceeded"
    assert "unavailable" in error["message"]
    assert engine.admission.snapshot().num_requests == 1
    if route != "responses":
        assert releases


@pytest.mark.parametrize(
    "route", ["completions", "chat/completions", "responses", "messages"]
)
def test_stream_capacity_is_error_not_empty_success(capacity_server, route):
    server, engine, _ = capacity_server
    response = _request(server, route, True)
    assert response.status_code == 200, response.text
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    errors = [
        event for event in events if "error" in event or event.get("type") == "error"
    ]
    assert len(errors) == 1, response.text
    error = errors[0].get("error", errors[0])
    assert error["code"] == "scheduler_capacity_exceeded"
    assert "unavailable" in error["message"]
    assert engine.admission.snapshot().num_requests == 1
    if route in ("completions", "chat/completions"):
        assert response.text.count("data: [DONE]") == 1
        assert response.text.index("scheduler_capacity_exceeded") < response.text.index(
            "data: [DONE]"
        )
    if route == "responses":
        assert not any(event.get("type") == "response.completed" for event in events)


def test_individually_oversized_prompt_is_not_retryable(capacity_server):
    server, engine, _ = capacity_server
    engine.admission = AdmissionController(max_prompt_tokens=1)
    engine.prompt_tokens = 2
    response = _request(server, "chat/completions", False)
    assert response.status_code == 413, response.text
    assert "retry-after" not in response.headers
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert engine.admission.snapshot().num_requests == 0


def test_responses_wrapper_closes_inner_stream_on_disconnect():
    from vllm_mlx.server import _responses_admission_errors

    closed = []

    async def inner():
        try:
            yield "event: response.created\ndata: {}\n\n"
        finally:
            closed.append(True)

    async def run():
        stream = _responses_admission_errors(inner())
        await anext(stream)
        await stream.aclose()

    asyncio.run(run())
    assert closed == [True]
