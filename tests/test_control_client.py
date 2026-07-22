# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from vllm_mlx.control_client import ControlClient, ControlClientError


class FakeResponse:
    def __init__(self, payload, *, status_code=200, lines=()):
        self.payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.lines = lines
        self.closed = False

    def json(self):
        return self.payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=False):
        return iter(self.lines)

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return next(self.responses)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return next(self.responses)


def envelope(data=None, error=None):
    return {"api_version": "1.0", "request_id": "req-1", "data": data, "error": error}


def test_control_client_serializes_exact_profile_mutation():
    session = FakeSession(
        [FakeResponse(envelope({"operation_id": "op-1"}), status_code=202)]
    )
    client = ControlClient("http://runtime", session=session)
    profile = {
        "profile_id": "laguna/s",
        "profile_revision": 3,
        "subject_digest": "a" * 64,
    }

    assert client.install(profile, "install-laguna") == {"operation_id": "op-1"}
    method, url, request = session.calls[0]
    assert method == "POST"
    assert url == "http://runtime/api/v1/control/models/laguna%2Fs/install"
    assert request["json"] == {"profile": profile, "idempotency_key": "install-laguna"}


def test_control_client_preserves_stable_error_code():
    session = FakeSession(
        [
            FakeResponse(
                envelope(error={"code": "profile_revision_stale", "message": "stale"}),
                status_code=409,
            )
        ]
    )
    with pytest.raises(ControlClientError, match="stale") as caught:
        ControlClient(session=session).status()
    assert caught.value.code == "profile_revision_stale"
    assert caught.value.status_code == 409


def test_control_client_rejects_incompatible_or_incomplete_envelope():
    session = FakeSession([FakeResponse({"api_version": "2.0", "data": {}})])
    with pytest.raises(ControlClientError, match="incompatible"):
        ControlClient(session=session).status()


def test_control_client_accepts_additive_server_minor_version():
    payload = envelope({"state": "unloaded"})
    payload["api_version"] = "1.1"
    assert ControlClient(session=FakeSession([FakeResponse(payload)])).status() == {
        "state": "unloaded"
    }


def test_streaming_chat_collects_only_content_events():
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": "hello"}}]}),
        "data: " + json.dumps({"choices": [{"delta": {}}]}),
        "data: [DONE]",
    ]
    response = FakeResponse({}, lines=lines)
    session = FakeSession([response])
    result = ControlClient(session=session).chat(
        model="laguna", message="hi", stream=True
    )
    assert list(result) == ["hello"]
    assert session.calls[0][2]["json"]["stream"] is True
    assert response.closed is True


def test_streaming_chat_rejects_truncated_stream_and_closes_response():
    response = FakeResponse(
        {},
        lines=["data: " + json.dumps({"choices": [{"delta": {"content": "partial"}}]})],
    )
    result = ControlClient(session=FakeSession([response])).chat(
        model="laguna", message="hi", stream=True
    )
    with pytest.raises(ControlClientError, match="before the done event"):
        list(result)
    assert response.closed is True
