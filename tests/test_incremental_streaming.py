#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Integration test: verify incremental streaming delivers tokens progressively.

Gated by VLLM_MLX_INTEGRATION_SERVER env var (set to the server URL).
Validates that TTFT < total_generation_time, proving tokens stream
progressively rather than being batched then yielded.

    VLLM_MLX_INTEGRATION_SERVER=http://127.0.0.1:8080 pytest tests/test_incremental_streaming.py -v
"""

import json
import os
import time

import pytest
import urllib.request

SERVER = os.environ.get("VLLM_MLX_INTEGRATION_SERVER")

pytestmark = pytest.mark.skipif(
    not SERVER, reason="VLLM_MLX_INTEGRATION_SERVER not set"
)


def _get_model_name():
    """Discover the active model from the server's /v1/models endpoint."""
    req = urllib.request.Request(f"{SERVER}/v1/models")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
    return data["data"][0]["id"]


def _stream_chat(messages, max_tokens=256, thinking_budget=None):
    """Make a streaming chat request, return (ttft, total_time, chunks)."""
    body = {
        "model": _get_model_name(),
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if thinking_budget is not None:
        body["enable_thinking"] = True
        body["thinking_token_budget"] = thinking_budget

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft = None
    chunks = []

    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data:") or "[DONE]" in line:
                continue
            chunk = json.loads(line[5:].strip())
            delta = chunk["choices"][0]["delta"]
            has_content = bool(delta.get("content") or delta.get("reasoning_content"))
            if has_content and ttft is None:
                ttft = time.perf_counter() - t0
            chunks.append(chunk)

    total_time = time.perf_counter() - t0
    return ttft, total_time, chunks


class TestIncrementalStreaming:
    """Verify tokens stream progressively (not batch-then-yield)."""

    def test_ttft_less_than_total(self):
        """TTFT must be significantly less than total generation time."""
        ttft, total, chunks = _stream_chat(
            [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
            max_tokens=128,
        )
        assert ttft is not None, "No content chunks received"
        assert len(chunks) > 5, f"Too few chunks ({len(chunks)}), not streaming"
        # TTFT should be <50% of total time for a multi-token generation
        assert ttft < total * 0.5, (
            f"TTFT ({ttft:.2f}s) is not significantly less than total ({total:.2f}s) "
            f"-- tokens may not be streaming progressively"
        )

    def test_thinking_streams_progressively(self):
        """Thinking tokens must stream, not batch-then-yield."""
        ttft, total, chunks = _stream_chat(
            [{"role": "user", "content": "What is 17 * 23? Think step by step."}],
            max_tokens=512,
            thinking_budget=256,
        )
        assert ttft is not None, "No content/reasoning chunks received"
        # With thinking, TTFT is the first reasoning token -- should arrive
        # well before the full 256-token thinking budget completes
        assert ttft < total * 0.5, (
            f"TTFT ({ttft:.2f}s) is not significantly less than total ({total:.2f}s) "
            f"-- thinking may not be streaming progressively"
        )
        # Should have many chunks (at least budget/2 reasoning + some content)
        assert len(chunks) > 20, f"Too few chunks ({len(chunks)})"

    def test_no_stop_token_in_content(self):
        """Stop tokens must not appear in streamed content."""
        _, _, chunks = _stream_chat(
            [{"role": "user", "content": "Say hello."}],
            max_tokens=32,
        )
        content = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks
        )
        assert "<|im_end|>" not in content, "Stop token leaked into content"
        assert "<|endoftext|>" not in content, "EOS token leaked into content"

    def test_finish_reason_on_last_chunk(self):
        """Last chunk must have finish_reason set."""
        _, _, chunks = _stream_chat(
            [{"role": "user", "content": "Say hi."}],
            max_tokens=32,
        )
        assert chunks, "No chunks received"
        last = chunks[-1]
        fr = last["choices"][0].get("finish_reason")
        assert fr in ("stop", "length"), f"Unexpected finish_reason: {fr}"

    def test_second_request_still_works(self):
        """Server must handle sequential requests without degradation."""
        _, _, chunks1 = _stream_chat(
            [{"role": "user", "content": "Say 'first'."}],
            max_tokens=16,
        )
        _, _, chunks2 = _stream_chat(
            [{"role": "user", "content": "Say 'second'."}],
            max_tokens=16,
        )
        content1 = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks1
        )
        content2 = "".join(
            c["choices"][0]["delta"].get("content", "") for c in chunks2
        )
        assert content1.strip(), "First request empty"
        assert content2.strip(), "Second request empty"
