# SPDX-License-Identifier: Apache-2.0
"""Focused MCP SDK compatibility and failed-connect cleanup tests."""

from types import SimpleNamespace

import pytest

from vllm_mlx.mcp.client import MCPClient
from vllm_mlx.mcp.types import MCPServerState, MCPTransport


def _client() -> MCPClient:
    config = SimpleNamespace(
        name="test",
        transport=MCPTransport.STDIO,
        enabled=True,
        command="python",
        args=[],
        env=None,
        timeout=1.0,
    )
    return MCPClient(config)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(
            protocol_version="2025-11-25",
            server_info=SimpleNamespace(name="sdk2"),
        ),
        SimpleNamespace(
            protocolVersion="2024-11-05",
            serverInfo=SimpleNamespace(name="sdk1"),
        ),
    ],
)
async def test_initialize_session_accepts_both_sdk_field_styles(result, caplog):
    client = _client()
    caplog.set_level("DEBUG")
    client._session = SimpleNamespace(initialize=lambda: result)

    async def initialize():
        return result

    client._session.initialize = initialize
    await client._initialize_session()
    assert "initialized" in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    "tool",
    [
        SimpleNamespace(
            name="snake",
            description="snake schema",
            input_schema={"type": "object"},
        ),
        SimpleNamespace(
            name="camel",
            description="camel schema",
            inputSchema={"type": "object"},
        ),
    ],
)
async def test_discover_tools_accepts_both_sdk_field_styles(tool):
    client = _client()
    client._session = SimpleNamespace(list_tools=lambda: SimpleNamespace(tools=[tool]))

    async def list_tools():
        return SimpleNamespace(tools=[tool])

    client._session.list_tools = list_tools
    await client._discover_tools()
    assert client.tools[0].input_schema == {"type": "object"}


@pytest.mark.anyio
@pytest.mark.parametrize("field", ["is_error", "isError"])
async def test_call_tool_accepts_both_sdk_field_styles(field):
    client = _client()
    client._state = MCPServerState.CONNECTED
    result = SimpleNamespace(content=[], **{field: True})

    async def call_tool(_name, _arguments):
        return result

    client._session = SimpleNamespace(call_tool=call_tool)
    outcome = await client.call_tool("tool", {})
    assert outcome.is_error is True


@pytest.mark.anyio
async def test_connect_closes_partial_contexts_and_returns_promptly(monkeypatch):
    client = _client()

    class Context:
        def __init__(self):
            self.exited = 0

        async def __aexit__(self, *_args):
            self.exited += 1

    session_context = Context()
    stdio_context = Context()

    async def connect_stdio():
        client._stdio_client = stdio_context
        client._session = session_context

    async def initialize_session():
        raise AttributeError("protocolVersion")

    monkeypatch.setattr(client, "_connect_stdio", connect_stdio)
    monkeypatch.setattr(client, "_initialize_session", initialize_session)

    assert await client.connect() is False
    assert client.state is MCPServerState.ERROR
    assert client._session is None
    assert client._stdio_client is None
    assert session_context.exited == 1
    assert stdio_context.exited == 1
