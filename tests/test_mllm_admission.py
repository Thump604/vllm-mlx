# SPDX-License-Identifier: Apache-2.0
"""Source-contract tests for MLLM scheduler admission wiring.

These tests use lightweight constructor and generator seams; they are not
live model or inference proof.
"""

import asyncio
from collections import deque
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_scheduler(*, max_requests=None, max_prompt_tokens=None):
    from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

    class Tokenizer:
        eos_token_id = None
        eos_token_ids = None
        name_or_path = None

        def encode(self, prompt):
            return prompt.split()

        def decode(self, tokens):
            return ""

    scheduler = MLLMScheduler(
        model=SimpleNamespace(config=None),
        processor=SimpleNamespace(tokenizer=Tokenizer()),
        config=MLLMSchedulerConfig(
            max_num_seqs=16,
            enable_vision_cache=False,
            enable_prefix_cache=False,
            max_inflight_requests=max_requests,
            max_inflight_prompt_tokens=max_prompt_tokens,
        ),
    )

    assert scheduler._admission is not None
    scheduler._ensure_batch_generator = lambda: None
    scheduler.vision_cache = None
    scheduler._step_count = 0
    scheduler._clear_cache_interval = 100
    return scheduler


def _set_generator(scheduler, **overrides):
    defaults = {
        "abort_prefill": MagicMock(),
        "schedule_removal": MagicMock(),
        "has_pending_removal": lambda request_id: False,
        "has_pending_removals": lambda: False,
        "process_pending_removals": MagicMock(),
        "next": MagicMock(return_value=[]),
        "close": MagicMock(),
    }
    defaults.update(overrides)
    generator = SimpleNamespace(**defaults)
    scheduler.batch_generator = generator
    return generator


def _move_to_running(scheduler, request_id, uid):
    from vllm_mlx.request import RequestStatus

    request = scheduler.requests[request_id]
    scheduler.waiting.remove(request)
    request.status = RequestStatus.RUNNING
    request.batch_uid = uid
    scheduler.running[request_id] = request
    scheduler.request_id_to_uid[request_id] = uid
    scheduler.uid_to_request_id[uid] = request_id
    return request


def _stop_response(uid, request_id):
    return SimpleNamespace(
        uid=uid,
        request_id=request_id,
        token=1,
        finish_reason="stop",
        mtp_attempted=False,
        mtp_attempted_count=0,
        from_draft=False,
    )


def test_admission_fields_are_appended_and_unlimited_by_default():
    from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig
    from vllm_mlx.scheduler import SchedulerConfig, SchedulingPolicy

    assert list(SchedulerConfig.__dataclass_fields__)[-2:] == [
        "mllm_max_inflight_requests",
        "mllm_max_inflight_prompt_tokens",
    ]
    assert list(MLLMSchedulerConfig.__dataclass_fields__)[-2:] == [
        "max_inflight_requests",
        "max_inflight_prompt_tokens",
    ]

    standard = SchedulerConfig(3, 12, SchedulingPolicy.PRIORITY, 4, 5, 6)
    mllm = MLLMSchedulerConfig(3, 4, 5, 6)
    assert standard.mllm_max_inflight_requests is None
    assert standard.mllm_max_inflight_prompt_tokens is None
    assert mllm.max_inflight_requests is None
    assert mllm.max_inflight_prompt_tokens is None


def test_request_limit_rejects_before_scheduler_state_publication():
    from vllm_mlx.admission import AdmissionCapacityError

    scheduler = _make_scheduler(max_requests=1)
    scheduler.add_request("one two", request_id="first")
    original_request = scheduler.requests["first"]
    original_waiting = list(scheduler.waiting)

    with pytest.raises(AdmissionCapacityError) as excinfo:
        scheduler.add_request("three", request_id="second")

    assert excinfo.value.resource == "request"
    assert scheduler.requests == {"first": original_request}
    assert list(scheduler.waiting) == original_waiting
    assert scheduler._admission.snapshot().num_requests == 1
    assert scheduler._admission.snapshot().num_prompt_tokens == 2


def test_prompt_token_limit_covers_waiting_and_running_lifetime():
    from vllm_mlx.admission import AdmissionCapacityError

    scheduler = _make_scheduler(max_prompt_tokens=3)
    scheduler.add_request("one two", request_id="first")
    scheduler.add_request("three", request_id="second")
    _move_to_running(scheduler, "first", uid=7)

    assert scheduler._admission.snapshot().num_requests == 2
    assert scheduler._admission.snapshot().num_prompt_tokens == 3

    with pytest.raises(AdmissionCapacityError) as excinfo:
        scheduler.add_request("four", request_id="third")

    assert excinfo.value.resource == "prompt_token"
    assert set(scheduler.requests) == {"first", "second"}
    assert [request.request_id for request in scheduler.waiting] == ["second"]


def test_duplicate_id_rejects_without_changing_reservation_or_queue():
    scheduler = _make_scheduler(max_requests=2, max_prompt_tokens=5)
    scheduler.add_request("one", request_id="same")
    original_request = scheduler.requests["same"]
    original_waiting = list(scheduler.waiting)

    with pytest.raises(ValueError, match="duplicate MLLM request ID"):
        scheduler.add_request("two two", request_id="same")

    assert scheduler.requests["same"] is original_request
    assert list(scheduler.waiting) == original_waiting
    assert scheduler._admission.snapshot().num_requests == 1
    assert scheduler._admission.snapshot().num_prompt_tokens == 1


def test_tokenization_failure_is_rejected_when_prompt_budget_enabled():
    scheduler = _make_scheduler(max_prompt_tokens=8)
    scheduler.processor.tokenizer.encode = MagicMock(
        side_effect=RuntimeError("tokenizer failed")
    )

    with pytest.raises(
        ValueError,
        match="Prompt token count is required when token admission is enabled",
    ):
        scheduler.add_request("one", request_id="failed")

    assert scheduler.requests == {}
    assert list(scheduler.waiting) == []
    assert scheduler._admission.snapshot().num_requests == 0
    assert scheduler._admission.snapshot().num_prompt_tokens == 0


def test_abort_releases_once_even_when_generator_cleanup_fails():
    scheduler = _make_scheduler(max_requests=1, max_prompt_tokens=2)
    scheduler.add_request("one two", request_id="first")
    request = _move_to_running(scheduler, "first", uid=7)
    cleanup_error = RuntimeError("schedule removal failed")
    generator = _set_generator(
        scheduler, schedule_removal=MagicMock(side_effect=cleanup_error)
    )

    with pytest.raises(RuntimeError) as excinfo:
        scheduler.abort_request("first")

    assert excinfo.value is cleanup_error
    assert request.status.name == "FINISHED_ABORTED"
    assert scheduler.requests == {}
    assert scheduler.running == {}
    assert scheduler.request_id_to_uid == {}
    assert scheduler.uid_to_request_id == {}
    assert scheduler._admission.snapshot().num_requests == 0
    assert scheduler._admission.snapshot().num_prompt_tokens == 0
    assert scheduler.abort_request("first") is False
    generator.schedule_removal.assert_called_once_with([7], request_ids=["first"])


def test_completion_releases_and_allows_same_id_reuse(monkeypatch):
    import vllm_mlx.mllm_scheduler as scheduler_module

    scheduler = _make_scheduler(max_requests=1, max_prompt_tokens=2)
    scheduler.add_request("one two", request_id="same")
    _move_to_running(scheduler, "same", uid=7)
    generator = _set_generator(
        scheduler, next=MagicMock(return_value=[_stop_response(7, "same")])
    )
    monkeypatch.setattr(scheduler_module.mx, "clear_cache", lambda: None)

    output = scheduler.step()

    assert output.finished_request_ids == {"same"}
    assert scheduler.requests == {}
    assert scheduler.running == {}
    assert scheduler._admission.snapshot().num_requests == 0
    assert scheduler._admission.snapshot().num_prompt_tokens == 0
    assert scheduler.add_request("replacement", request_id="same") == "same"
    assert scheduler._admission.snapshot().num_requests == 1
    generator.next.assert_called_once_with()


def test_stale_completion_cannot_release_replacement_reservation(monkeypatch):
    import vllm_mlx.mllm_scheduler as scheduler_module

    scheduler = _make_scheduler(max_requests=1, max_prompt_tokens=2)
    scheduler.add_request("old", request_id="same")
    old_request = scheduler.requests["same"]
    assert scheduler.abort_request("same") is True

    scheduler.add_request("new words", request_id="same")
    replacement = _move_to_running(scheduler, "same", uid=8)
    monkeypatch.setattr(scheduler_module.mx, "clear_cache", lambda: None)

    scheduler._cleanup_finished({"same"}, expected_owners={"same": (old_request, 7)})

    assert scheduler.requests["same"] is replacement
    assert scheduler.running["same"] is replacement
    assert scheduler.request_id_to_uid == {"same": 8}
    assert scheduler.uid_to_request_id == {8: "same"}
    assert scheduler._admission.snapshot().num_requests == 1
    assert scheduler._admission.snapshot().num_prompt_tokens == 2


def test_step_error_uses_abort_path_to_release_admission():
    scheduler = _make_scheduler(max_requests=1, max_prompt_tokens=1)
    scheduler.add_request("one", request_id="first")
    _move_to_running(scheduler, "first", uid=21)
    generator = _set_generator(scheduler)

    scheduler._fail_requests_after_step_error(RuntimeError("forward failed"))

    assert scheduler.requests == {}
    assert scheduler.running == {}
    assert scheduler._admission.snapshot().num_requests == 0
    assert scheduler._admission.snapshot().num_prompt_tokens == 0
    generator.process_pending_removals.assert_called_once_with()


def test_reset_uses_abort_path_to_release_all_reservations():
    scheduler = _make_scheduler(max_requests=2, max_prompt_tokens=3)
    scheduler.add_request("one", request_id="first")
    scheduler.add_request("two two", request_id="second")
    requests = list(scheduler.requests.values())
    generator = _set_generator(scheduler)

    scheduler.reset()

    assert scheduler.requests == {}
    assert scheduler.waiting == deque()
    assert scheduler.running == {}
    assert scheduler._admission.snapshot().num_requests == 0
    assert scheduler._admission.snapshot().num_prompt_tokens == 0
    assert all(request._admission_reserved is False for request in requests)
    generator.process_pending_removals.assert_called_once_with()
    generator.close.assert_called_once_with()
    assert scheduler.batch_generator is None


def test_batched_engine_propagates_only_mllm_admission_fields(monkeypatch):
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

    captured = {}

    class FakeMLLMScheduler:
        def __init__(self, model, processor, config):
            captured["scheduler_config"] = config

        async def start(self):
            return None

    fake_module = SimpleNamespace(
        MLLMScheduler=FakeMLLMScheduler,
        MLLMSchedulerConfig=MLLMSchedulerConfig,
    )
    monkeypatch.setitem(sys.modules, "vllm_mlx.mllm_scheduler", fake_module)

    config = SimpleNamespace(
        mllm_max_inflight_requests=3,
        mllm_max_inflight_prompt_tokens=17,
        enable_mtp=False,
    )
    engine = BatchedEngine(
        model_name="fake-qwen",
        scheduler_config=config,
        force_mllm=True,
    )
    engine._model = object()
    engine._processor = object()
    engine._mllm_instance = SimpleNamespace()

    asyncio.run(engine._start_mllm())

    assert captured["scheduler_config"].max_inflight_requests == 3
    assert captured["scheduler_config"].max_inflight_prompt_tokens == 17
