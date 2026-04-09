"""Fast suite for BatchedEngine non-MLLM text SpecPrefill path.

Real Nemotron-H Nano 4B as both target and draft. Tiny specprefill_threshold
(16) so short prompts can exercise the SpecPrefill engagement path without
needing 8K+ test inputs. No mocks except the env-gated
VLLM_MLX_TEST_FORCE_SPECPREFILL_FAIL=pre_emit|post_emit hook documented in
CLAUDE.md Golden Rule 5.

Spec: docs/superpowers/specs/2026-04-08-batched-engine-text-specprefill-design.md
"""
import asyncio
import os
from pathlib import Path

import pytest
import pytest_asyncio

# Make sure runtime patches are applied so the Nano 4B nemotron_h "-" pattern
# fix is in place. The fixture below loads the model via mlx_lm; without this
# patch, the load fails with KeyError('-').
import sys
_RUNTIME_LIB = "/opt/ai-runtime/lib"
if _RUNTIME_LIB not in sys.path:
    sys.path.insert(0, _RUNTIME_LIB)
import runtime_patches
runtime_patches._patch_nemotron_h_mlp_pattern()

from vllm_mlx.engine.batched import BatchedEngine

NANO_4B_PATH = "/Users/David/ai-models/mlx_models/NVIDIA-Nemotron-3-Nano-4B-4bit"
NANO_4B_DIR_NAME = "NVIDIA-Nemotron-3-Nano-4B-4bit"


@pytest_asyncio.fixture
async def nano4b_engine():
    """Real BatchedEngine with Nemotron-H Nano 4B as both target and draft.

    Tiny specprefill_threshold (16) so short prompts in the fast suite
    trigger SpecPrefill engagement. Generous keep_pct (0.5) — tests don't
    care about paper-comparable speedups, only correctness of dispatch and
    serialization.

    Function-scoped (not session-scoped) because pytest-asyncio's default
    loop scope is function; a session-scoped async fixture on a
    function-scoped event loop hangs. Nano 4B loads in ~1.5s so per-test
    reload cost is acceptable (~15s total overhead for the 10-test suite).
    """
    # Sanity check that the model is on disk before constructing the engine
    assert Path(NANO_4B_PATH).exists(), (
        f"Test fixture requires Nemotron-H Nano 4B at {NANO_4B_PATH}; "
        "run the download in /tmp/test_load_nano4b.py from Session 89."
    )

    engine = BatchedEngine(
        model_name=NANO_4B_PATH,
        specprefill_enabled=True,
        specprefill_threshold=16,
        specprefill_keep_pct=0.5,
        specprefill_draft_model_path=NANO_4B_DIR_NAME,
    )
    await engine.start()
    try:
        yield engine
    finally:
        await engine.stop()


def _has_log(caplog, substring: str) -> bool:
    """True iff caplog captured at least one log message containing substring."""
    return any(substring in rec.getMessage() for rec in caplog.records)


def _count_log(caplog, substring: str) -> int:
    return sum(1 for rec in caplog.records if substring in rec.getMessage())


@pytest.mark.asyncio
async def test_below_threshold_takes_scheduler_path(nano4b_engine, caplog):
    """A 5-token prompt is below the threshold (16) and should take the
    scheduler path. No SpecPrefill text begin/end logs should appear, and
    the response should be non-empty (proves the scheduler path still works
    even though SpecPrefill is enabled by default in the fixture).
    """
    caplog.set_level("INFO")

    last = None
    async for chunk in nano4b_engine.stream_generate(
        prompt="What is 2+2?",
        max_tokens=8,
        temperature=1.0,
        top_p=0.95,
    ):
        last = chunk

    assert last is not None and last.text, "scheduler path produced no output"
    assert not _has_log(caplog, "SpecPrefill text begin"), (
        "Below-threshold prompt unexpectedly engaged SpecPrefill"
    )


@pytest.mark.asyncio
async def test_above_threshold_takes_specprefill_path(nano4b_engine, caplog):
    """A 100-token prompt is above the threshold (16) and should engage
    SpecPrefill. The text begin/phases/end logs should all appear.
    """
    caplog.set_level("INFO")

    # Build a ~100-token prompt by repeating a phrase
    prompt = "The quick brown fox jumps over the lazy dog. " * 12

    last = None
    async for chunk in nano4b_engine.stream_generate(
        prompt=prompt,
        max_tokens=8,
        temperature=1.0,
        top_p=0.95,
    ):
        last = chunk

    assert last is not None and last.text, "SpecPrefill path produced no output"
    assert _has_log(caplog, "SpecPrefill text begin"), (
        "Above-threshold prompt did not engage SpecPrefill"
    )
    assert _has_log(caplog, "SpecPrefill text phases"), (
        "SpecPrefill phase timing log missing"
    )
    assert _has_log(caplog, "SpecPrefill text end"), (
        "SpecPrefill end log missing"
    )


@pytest.mark.asyncio
async def test_force_enable_bypasses_threshold(nano4b_engine, caplog):
    """A 5-token prompt with specprefill=True should engage SpecPrefill
    despite being below the threshold (force-enable bypasses threshold).
    """
    caplog.set_level("INFO")

    last = None
    async for chunk in nano4b_engine.stream_generate(
        prompt="What is 2+2?",
        max_tokens=8,
        temperature=1.0,
        top_p=0.95,
        specprefill=True,
    ):
        last = chunk

    assert last is not None and last.text
    assert _has_log(caplog, "SpecPrefill text begin"), (
        "specprefill=True did not bypass threshold for short prompt"
    )


@pytest.mark.asyncio
async def test_force_disable_hard_disables(nano4b_engine, caplog):
    """A 500-token prompt with specprefill=False should take the scheduler
    path even though the prompt is above threshold and the engine default
    is enabled. Force-disable always wins.
    """
    caplog.set_level("INFO")

    prompt = "The quick brown fox jumps over the lazy dog. " * 60  # ~500 tokens

    last = None
    async for chunk in nano4b_engine.stream_generate(
        prompt=prompt,
        max_tokens=8,
        temperature=1.0,
        top_p=0.95,
        specprefill=False,
    ):
        last = chunk

    assert last is not None and last.text
    assert not _has_log(caplog, "SpecPrefill text begin"), (
        "specprefill=False did not hard-disable for above-threshold prompt"
    )


@pytest.mark.asyncio
async def test_stop_sequences_force_scheduler_path(nano4b_engine, caplog):
    """An above-threshold prompt with non-empty stop sequences should take
    the scheduler path. No SpecPrefill engagement, no warning (because the
    user did not force-enable).
    """
    caplog.set_level("INFO")

    prompt = "The quick brown fox jumps over the lazy dog. " * 12

    last = None
    async for chunk in nano4b_engine.stream_generate(
        prompt=prompt,
        max_tokens=8,
        temperature=1.0,
        top_p=0.95,
        stop=["END"],
    ):
        last = chunk

    assert last is not None
    assert not _has_log(caplog, "SpecPrefill text begin"), (
        "stop sequences did not exclude request from SpecPrefill"
    )
    assert not _has_log(caplog, "stop sequences unsupported"), (
        "Unexpected WARNING for non-force-enabled stop request"
    )


@pytest.mark.asyncio
async def test_force_enable_with_stop_warns_and_falls_back(nano4b_engine, caplog):
    """An above-threshold prompt with specprefill=True AND stop=["END"]
    should bypass SpecPrefill (stop wins over force-enable), and emit
    exactly one WARNING log saying stop sequences are unsupported.
    """
    caplog.set_level("INFO")

    prompt = "The quick brown fox jumps over the lazy dog. " * 12

    last = None
    async for chunk in nano4b_engine.stream_generate(
        prompt=prompt,
        max_tokens=8,
        temperature=1.0,
        top_p=0.95,
        specprefill=True,
        stop=["END"],
    ):
        last = chunk

    assert last is not None
    assert not _has_log(caplog, "SpecPrefill text begin"), (
        "Force-enable with stop should not engage SpecPrefill"
    )
    warnings = _count_log(caplog, "stop sequences unsupported")
    assert warnings == 1, (
        f"Expected exactly 1 'stop sequences unsupported' WARNING, got {warnings}"
    )


@pytest.mark.asyncio
async def test_concurrent_eligible_requests_serialize(nano4b_engine, caplog):
    """3 concurrent eligible requests must serialize under
    self._text_generation_lock. Proven by parsing captured begin/end logs:
    for any pair of requests A and B, if begin(A) was logged before begin(B),
    then end(A) must also be logged before begin(B).
    """
    import re
    caplog.set_level("INFO")

    prompt = "The quick brown fox jumps over the lazy dog. " * 12

    async def _one_request():
        last = None
        async for chunk in nano4b_engine.stream_generate(
            prompt=prompt,
            max_tokens=8,
            temperature=1.0,
            top_p=0.95,
        ):
            last = chunk
        return last

    results = await asyncio.gather(
        _one_request(),
        _one_request(),
        _one_request(),
    )
    assert all(r is not None and r.text for r in results), (
        "Concurrent SpecPrefill requests did not all complete"
    )

    begin_re = re.compile(r"SpecPrefill text begin req=(\w+)")
    end_re = re.compile(r"SpecPrefill text end req=(\w+)")

    begins = []  # list of (timestamp, req_id)
    ends = []
    for rec in caplog.records:
        msg = rec.getMessage()
        m = begin_re.search(msg)
        if m:
            begins.append((rec.created, m.group(1)))
            continue
        m = end_re.search(msg)
        if m:
            ends.append((rec.created, m.group(1)))

    assert len(begins) == 3, f"expected 3 begin logs, got {len(begins)}"
    assert len(ends) == 3, f"expected 3 end logs, got {len(ends)}"

    # For every pair (A, B) with begin(A).ts < begin(B).ts, end(A).ts must
    # also be < begin(B).ts. This proves serialization: A fully finishes
    # before B starts.
    end_ts = {req_id: ts for ts, req_id in ends}
    begins_sorted = sorted(begins)
    for i in range(len(begins_sorted) - 1):
        ts_a, req_a = begins_sorted[i]
        ts_b, req_b = begins_sorted[i + 1]
        assert end_ts[req_a] < ts_b, (
            f"Serialization violated: req={req_a} ended at {end_ts[req_a]} "
            f"which is NOT before req={req_b} began at {ts_b}"
        )


@pytest.mark.asyncio
async def test_cancellation_under_lock_releases_cleanly(nano4b_engine, caplog):
    """Cancel an eligible request mid-generation, then immediately fire a
    second eligible request. The second must complete cleanly. The lock
    must be released only after the first request's worker thread has
    fully exited (CLAUDE.md Golden Rule 4: shield + await pattern).
    """
    caplog.set_level("INFO")

    prompt = "The quick brown fox jumps over the lazy dog. " * 12

    async def _gen():
        last = None
        async for chunk in nano4b_engine.stream_generate(
            prompt=prompt,
            max_tokens=64,
            temperature=1.0,
            top_p=0.95,
        ):
            last = chunk
        return last

    task1 = asyncio.create_task(_gen())
    # Let the first request actually start, get the lock, and begin Phase 1
    await asyncio.sleep(2.0)
    task1.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        pass

    # Second request — should complete cleanly with no Metal errors
    result2 = await _gen()
    assert result2 is not None and result2.text, (
        "Second request after cancellation did not complete"
    )

    # Verify the second request's begin log appears AFTER the first
    # request's end log (proving the lock was actually held until task1's
    # worker fully released).
    import re
    begin_re = re.compile(r"SpecPrefill text begin req=(\w+)")
    end_re = re.compile(r"SpecPrefill text end req=(\w+)")
    events = []  # list of (ts, kind, req_id)
    for rec in caplog.records:
        msg = rec.getMessage()
        m = begin_re.search(msg)
        if m:
            events.append((rec.created, "begin", m.group(1)))
            continue
        m = end_re.search(msg)
        if m:
            events.append((rec.created, "end", m.group(1)))

    events.sort()
    # The 4 events should appear as: begin(A), end(A), begin(B), end(B)
    # NOT as: begin(A), begin(B), ... (which would prove the lock was
    # released before task1's worker finished).
    assert len(events) >= 4, f"expected at least 4 begin/end events, got {events}"
    seq = [(kind, req_id) for _, kind, req_id in events]
    # First two events should be begin/end of the same req_id
    assert seq[0][0] == "begin" and seq[1][0] == "end" and seq[0][1] == seq[1][1], (
        f"Lock-release ordering violated: events were {seq}"
    )


@pytest.mark.asyncio
async def test_fallback_on_pre_emission_failure(nano4b_engine, caplog, monkeypatch):
    """With VLLM_MLX_TEST_FORCE_SPECPREFILL_FAIL=pre_emit set, the worker
    raises BEFORE the first chunk is enqueued. The dispatch should log an
    ERROR and fall back to the scheduler path. The user-visible response
    should still be a complete answer (from the scheduler).
    """
    caplog.set_level("INFO")
    monkeypatch.setenv("VLLM_MLX_TEST_FORCE_SPECPREFILL_FAIL", "pre_emit")

    prompt = "The quick brown fox jumps over the lazy dog. " * 12

    last = None
    async for chunk in nano4b_engine.stream_generate(
        prompt=prompt,
        max_tokens=8,
        temperature=1.0,
        top_p=0.95,
    ):
        last = chunk

    assert last is not None and last.text, (
        "Pre-emission fallback did not produce a response"
    )
    # ERROR log with traceback should be present
    assert any(
        "SpecPrefill (non-MLLM text) failed before first chunk" in rec.getMessage()
        for rec in caplog.records if rec.levelname == "ERROR"
    ), "Expected ERROR log for pre-emission fallback"


@pytest.mark.asyncio
async def test_post_emission_failure_propagates(nano4b_engine, caplog, monkeypatch):
    """With VLLM_MLX_TEST_FORCE_SPECPREFILL_FAIL=post_emit set, the worker
    raises AFTER the first chunk has been enqueued. The dispatch must
    propagate the exception (cannot safely fall back mid-stream). The
    consumer should receive at least one chunk before the exception
    surfaces.
    """
    caplog.set_level("INFO")
    monkeypatch.setenv("VLLM_MLX_TEST_FORCE_SPECPREFILL_FAIL", "post_emit")

    prompt = "The quick brown fox jumps over the lazy dog. " * 12

    chunks_received = 0
    raised = False
    try:
        async for chunk in nano4b_engine.stream_generate(
            prompt=prompt,
            max_tokens=16,
            temperature=1.0,
            top_p=0.95,
        ):
            chunks_received += 1
    except RuntimeError as e:
        if "VLLM_MLX_TEST_FORCE_SPECPREFILL_FAIL=post_emit" in str(e):
            raised = True
        else:
            raise

    assert raised, "Expected RuntimeError to propagate after post_emit injection"
    assert chunks_received >= 1, (
        "Expected at least one chunk before post-emission failure; "
        f"got {chunks_received}"
    )
    # ERROR log with the propagation message should be present
    assert any(
        "after first chunk emission; propagating error" in rec.getMessage()
        for rec in caplog.records if rec.levelname == "ERROR"
    ), "Expected ERROR log for post-emission propagation"


def _build_needle_haystack(target_tokens: int, needle: str) -> str:
    """Build a haystack of approximately `target_tokens` tokens with the
    needle inserted near the middle."""
    para = (
        "The lunar module sits on the plain. The regolith is fine and grey. "
        "Ancient craters mark the horizon. Radiation counters tick steadily. "
        "Nothing lives here except the instruments. "
    )
    est_per_para = 35
    n_paras = max(1, target_tokens // est_per_para - 2)
    insert_at = n_paras // 2
    parts = []
    for i in range(n_paras):
        parts.append(para)
        if i == insert_at:
            parts.append(f"\n\n{needle}\n\n")
    return "".join(parts)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("VLLM_MLX_TEST_NEMOTRON_SPECPREFILL_INTEGRATION"),
    reason="Set VLLM_MLX_TEST_NEMOTRON_SPECPREFILL_INTEGRATION=1 to run",
)
async def test_nemotron_specprefill_end_to_end_with_needle(nano4b_engine, caplog):
    """Real long-prompt end-to-end SpecPrefill on Nemotron-H Nano 4B.

    Uses a realistic threshold (8192) and a 16K needle-in-haystack prompt
    that actually exercises Phase 1 scoring on meaningful token counts.
    Verifies the needle is retrieved and SpecPrefill engaged.
    """
    caplog.set_level("INFO")
    # Override the fixture's tiny threshold for this realistic test
    nano4b_engine._specprefill_threshold = 8192
    nano4b_engine._specprefill_keep_pct = 0.2

    needle = "The hidden activation code is NX-4271-NANO. Remember it."
    haystack = _build_needle_haystack(target_tokens=16384, needle=needle)
    question = (
        "What is the hidden activation code? Answer with just the code, "
        "nothing else."
    )
    prompt = f"{haystack}\n\n{question}"

    last = None
    async for chunk in nano4b_engine.stream_generate(
        prompt=prompt,
        max_tokens=32,
        temperature=1.0,
        top_p=0.95,
    ):
        last = chunk

    assert last is not None
    assert "NX-4271-NANO" in last.text, (
        f"Needle not retrieved: {last.text!r}"
    )
    assert _has_log(caplog, "sparse="), (
        "SpecPrefill phase log did not include sparse= field"
    )
