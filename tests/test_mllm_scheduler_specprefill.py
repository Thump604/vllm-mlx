# SPDX-License-Identifier: Apache-2.0
"""Synthetic scheduler contracts for cooperative MLLM SpecPrefill."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.cooperative_specprefill import (
    CooperativeSpecPrefillOutcome,
    CooperativeSpecPrefillSession,
)
from vllm_mlx.mllm_batch_generator import MLLMBatchResponse, SparseAdoptionError
from vllm_mlx.mllm_scheduler import (
    MLLMScheduler,
    MLLMSchedulerConfig,
    MLLMSpecPrefillAdmission,
    MLLMSpecPrefillCacheCapability,
)
from vllm_mlx.specprefill import (
    SpecPrefillCoverage,
    SpecPrefillPolicy,
    SpecPrefillScorerLaneBusy,
)
from vllm_mlx.specprefill_profiles import (
    SpecPrefillCell,
    SpecPrefillEngine,
    SpecPrefillProfileKey,
    SpecPrefillProfileRegistry,
    SpecPrefillTuning,
)
from vllm_mlx.specprefill_scorer_session import (
    ScorerSessionPhase,
    ScorerSessionProgress,
)
from vllm_mlx.specprefill_target_executor import (
    SparseTargetPrefillProgress,
    SparseTargetPrefillResult,
    SparseTargetPrefillTelemetry,
)

_TOKENS = (10, 11, 12, 13, 14, 15, 16, 17)
_IMPORTANCE = mx.array([0.0, 0.1, 0.2, 0.3, 0.9, 0.8, 0.4, 0.5])
_PROFILE_KEY = SpecPrefillProfileKey(
    target_artifact_id="target",
    target_artifact_hash="a" * 64,
    tokenizer_artifact_hash="b" * 64,
    scorer_artifact_id="scorer",
    scorer_artifact_hash="c" * 64,
    adapter_id="synthetic-adapter",
    engine=SpecPrefillEngine.CONTINUOUS_BATCHING,
    cell=SpecPrefillCell.SPARSE_ONLY,
)
_PROFILE_TUNING = SpecPrefillTuning(0.25, 0.0, 0, 1, 2)


class _Tokenizer:
    eos_token_id = 99
    name_or_path = None
    clean_up_tokenization_spaces = False

    def encode(self, _prompt, *, add_special_tokens=False):
        assert add_special_tokens is False
        return list(_TOKENS)

    def decode(self, tokens):
        return "".join(str(token) for token in tokens)


class _ProfileRegistry(SpecPrefillProfileRegistry):
    def resolve(self, *_args, **_kwargs):
        return SimpleNamespace(
            eligible=True,
            tuning=_PROFILE_TUNING,
            fallback_reason=None,
        )


class _ScorerSession:
    def __init__(
        self,
        *,
        busy_once=False,
        fail=False,
        cleanup_raises=False,
        events=None,
        name="",
    ):
        self.phase = ScorerSessionPhase.PREFILL
        self.importance = _IMPORTANCE
        self.busy_once = busy_once
        self.fail = fail
        self.cleanup_raises = cleanup_raises
        self.events = events if events is not None else []
        self.name = name
        self.calls = 0
        self.cancelled = False

    def step(self):
        self.events.append(f"score:{self.name}")
        if self.busy_once:
            self.busy_once = False
            raise SpecPrefillScorerLaneBusy("busy")
        self.calls += 1
        if self.fail:
            raise RuntimeError("scorer failed")
        self.phase = ScorerSessionPhase.COMPLETE
        return ScorerSessionProgress(
            ScorerSessionPhase.COMPLETE,
            len(_TOKENS),
            1,
            1,
            True,
        )

    def cancel(self):
        self.cancelled = True
        if self.cleanup_raises:
            raise RuntimeError("cleanup failed")


@dataclass
class _TargetSession:
    state: object

    def __post_init__(self):
        self.calls = 0
        self.cancelled = False
        self.cache = [object()]
        self.cache_reads = 0
        self.replace_on_second_read = False
        self.fail_cache_read = False
        self.result = SparseTargetPrefillResult(
            logits=mx.array([[[0.0, 8.0]]]),
            cache_state=self.state.clone(),
            telemetry=SparseTargetPrefillTelemetry(
                selected_tokens=self.state.rows[0].physical_valid_length,
                target_prefill_ms=1.0,
                chunk_count=1,
                physical_cache_starts=((0,),),
            ),
        )

    def step(self):
        self.calls += 1
        return SparseTargetPrefillProgress(
            selected_tokens_processed=self.state.rows[0].physical_valid_length,
            chunk_count=1,
            complete=True,
        )

    def cancel(self):
        self.cancelled = True

    @property
    def adoption_cache(self):
        if self.calls < 1:
            raise RuntimeError("cache not ready")
        if self.fail_cache_read:
            raise RuntimeError("cache seam failed")
        self.cache_reads += 1
        if self.replace_on_second_read and self.cache_reads == 2:
            self.cache[0] = object()
        return self.cache


class _AdmissionFactory:
    def __init__(self, scorer_by_request=None, events=None):
        self.scorer_by_request = scorer_by_request or {}
        self.events = events if events is not None else []
        self.sessions = {}

    def __call__(self, request, tokens, config):
        scorer = self.scorer_by_request.get(request.request_id)
        if scorer is None:
            scorer = _ScorerSession(events=self.events, name=request.request_id)

        def target_factory(_selected_tokens, sparse_state):
            return _TargetSession(sparse_state)

        session = CooperativeSpecPrefillSession(
            request.request_id,
            tokens,
            scorer,
            target_factory,
            config,
        )
        self.sessions[request.request_id] = session
        return MLLMSpecPrefillAdmission(session)


class _BatchGenerator:
    def __init__(self, events):
        self.events = events
        self.active_batch = None
        self.inserted = []
        self.next_uid = 100
        self.expected_config = None
        self.adoptions = []
        self.adoption_error = None
        self.responses = []
        self.on_adopt = None
        self.scheduled_removals = []
        self.pending_removals = []
        self.active_rows = {}
        self.active_cache_rows = {}

    def process_pending_removals(self):
        for uid in self.pending_removals:
            self.active_rows.pop(uid, None)
            self.active_cache_rows.pop(uid, None)
        self.pending_removals.clear()

    def insert(self, requests):
        self.inserted.extend(requests)
        result = list(range(self.next_uid, self.next_uid + len(requests)))
        self.next_uid += len(requests)
        return result

    def next(self):
        self.events.append("decode")
        responses, self.responses = self.responses, []
        return responses

    def set_expected_sparse_execution_config(self, config):
        self.expected_config = config
        self._expected_sparse_execution_config = config

    def adopt_prefilled_sparse_row(self, request, cache, logits, sparse_state):
        self.adoptions.append((request, cache, logits, sparse_state))
        if self.adoption_error is not None:
            raise self.adoption_error
        if self.on_adopt is not None:
            self.on_adopt()
        uid = self.next_uid
        self.next_uid += 1
        self.active_rows[uid] = request
        self.active_cache_rows[uid] = cache
        return uid

    def abort_prefill(self, _request_id):
        pass

    def schedule_removal(self, _uids):
        self.scheduled_removals.extend(_uids)
        self.pending_removals.extend(_uids)


def _scheduler(
    factory,
    *,
    enabled=True,
    mtp=False,
    cache_layout="qwen3_5_nonrotating_hybrid",
):
    events = factory.events
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.model = object()
    scheduler.processor = _Tokenizer()
    scheduler.config = MLLMSchedulerConfig(
        enable_mtp=mtp,
        specprefill_enabled=enabled,
        specprefill_profile_registry=_ProfileRegistry(),
        specprefill_profile_key=_PROFILE_KEY,
        specprefill_estimated_residency_bytes=1,
        specprefill_session_factory=factory,
        specprefill_target_forward_context=lambda _forward: None,
        specprefill_cache_capability=MLLMSpecPrefillCacheCapability(
            adapter_id=_PROFILE_KEY.adapter_id,
            layout=cache_layout,
        ),
    )
    scheduler.batch_generator = _BatchGenerator(events)
    scheduler.waiting = deque()
    scheduler.running = {}
    scheduler.requests = {}
    scheduler.finished_req_ids = set()
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler._specprefill_queue = deque()
    scheduler._specprefill_admissions = {}
    scheduler._specprefill_batch_requests = {}
    scheduler._specprefill_cancel_pending = set()
    scheduler._specprefill_cancel_lock = __import__("threading").RLock()
    scheduler._detokenizer_pool = {}
    scheduler.output_queues = {}
    scheduler._step_count = 0
    scheduler._clear_cache_interval = 10_000
    scheduler.num_requests_processed = 0
    scheduler.total_prompt_tokens = 0
    scheduler.total_completion_tokens = 0
    scheduler.stop_tokens = {99}
    return scheduler


def _add_selective(scheduler, request_id):
    return scheduler.add_request(
        "prompt",
        request_id=request_id,
        specprefill_policy=SpecPrefillPolicy.AUTO,
        specprefill_coverage=SpecPrefillCoverage.SELECTIVE,
    )


def test_active_decode_runs_before_one_round_robin_busy_quantum():
    events = []
    first = _ScorerSession(busy_once=True, events=events, name="first")
    second = _ScorerSession(events=events, name="second")
    factory = _AdmissionFactory({"first": first, "second": second}, events)
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "first")
    _add_selective(scheduler, "second")

    first_output = scheduler.step()

    assert events == ["decode", "score:first"]
    assert first_output.specprefill_progress["first"].busy is True
    assert first_output.specprefill_progress["first"].quantum_committed is False
    assert list(scheduler._specprefill_queue) == ["second", "first"]
    assert scheduler.batch_generator.inserted == []

    scheduler.step()
    assert events[-2:] == ["decode", "score:second"]
    assert first.calls == 0
    assert second.calls == 1


def test_preoutput_failure_requeues_dense_without_emitting_output():
    events = []
    scorer = _ScorerSession(fail=True, events=events, name="failed")
    factory = _AdmissionFactory({"failed": scorer}, events)
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "failed")

    output = scheduler.step()

    request = scheduler.running["failed"]
    assert output.outputs == []
    assert request.specprefill_effective_policy is SpecPrefillPolicy.DENSE
    assert request.specprefill_fallback_reason == "scorer_failed"
    assert len(scheduler.batch_generator.inserted) == 1
    assert "failed" not in scheduler._specprefill_admissions


def test_ready_session_adopts_only_after_bounded_target_quantum():
    events = []
    factory = _AdmissionFactory(events=events)
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "ready")

    first = scheduler.step()
    assert first.specprefill_progress["ready"].scorer_quanta == 1
    assert scheduler.batch_generator.adoptions == []

    second = scheduler.step()

    session = factory.sessions["ready"]
    assert second.specprefill_progress["ready"].target_quanta == 1
    assert session.outcome is CooperativeSpecPrefillOutcome.DECODE
    assert len(scheduler.batch_generator.adoptions) == 1
    assert scheduler.request_id_to_uid["ready"] == 100
    assert "ready" not in scheduler._specprefill_admissions


@pytest.mark.parametrize(
    ("failure", "expected_reason", "dense_replay"),
    [
        (
            SparseAdoptionError("pre", sampling_consumed=False),
            "adoption_failed",
            True,
        ),
        (
            SparseAdoptionError("post", sampling_consumed=True),
            "adoption_failed_after_sampling",
            False,
        ),
        (
            SparseAdoptionError(
                "rollback",
                sampling_consumed=False,
                rollback_succeeded=False,
            ),
            "adoption_rollback_failed",
            False,
        ),
        (RuntimeError("unknown"), "adoption_failed_unknown_boundary", False),
    ],
)
def test_adoption_replay_obeys_typed_sampling_boundary(
    failure, expected_reason, dense_replay
):
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "boundary")
    scheduler.step()
    scheduler.batch_generator.adoption_error = failure

    output = scheduler.step()

    assert bool(scheduler.batch_generator.inserted) is dense_replay
    if dense_replay:
        assert output.outputs == []
        assert scheduler.running["boundary"].specprefill_fallback_reason == expected_reason
    else:
        assert output.outputs[0].finished is True
        assert output.outputs[0].specprefill_fallback_reason == expected_reason
        assert "boundary" not in scheduler.running


def test_cleanup_failure_is_terminal_and_never_enqueues_dense():
    scorer = _ScorerSession(fail=True, cleanup_raises=True, name="cleanup")
    factory = _AdmissionFactory({"cleanup": scorer})
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "cleanup")

    output = scheduler.step()

    assert scheduler.batch_generator.inserted == []
    assert output.outputs[0].finished is True
    assert output.outputs[0].specprefill_fallback_reason == "resource_cleanup_failed"
    assert "cleanup" not in scheduler.running


def test_scheduler_step_removes_every_row_failed_by_poisoned_batch():
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory, enabled=False)
    _add_selective(scheduler, "poisoned-1")
    _add_selective(scheduler, "poisoned-2")
    scheduler.step()
    owned = dict(scheduler.request_id_to_uid)
    scheduler.batch_generator.responses = [
        MLLMBatchResponse(
            uid=uid,
            request_id=request_id,
            token=0,
            logprobs=mx.zeros(1),
            finish_reason="error",
        )
        for request_id, uid in owned.items()
    ]

    output = scheduler.step()

    assert output.finished_request_ids == {"poisoned-1", "poisoned-2"}
    assert {item.request_id for item in output.outputs} == output.finished_request_ids
    assert all(item.finish_reason == "error" for item in output.outputs)
    assert scheduler.running == {}
    assert scheduler.requests == {}
    assert scheduler.request_id_to_uid == {}
    assert scheduler.uid_to_request_id == {}


def test_terminal_output_preserves_complete_specprefill_metadata_and_mtp_independence():
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "metadata")
    scheduler.step()
    scheduler.step()
    uid = scheduler.request_id_to_uid["metadata"]
    scheduler.running["metadata"].mtp_drafts = 3
    scheduler.running["metadata"].mtp_accepted = 2
    scheduler.batch_generator.responses = [
        MLLMBatchResponse(
            uid=uid,
            request_id="metadata",
            token=1,
            logprobs=mx.zeros(2),
            finish_reason="length",
        )
    ]

    output = scheduler.step().outputs[0]

    assert output.specprefill_requested_policy == "auto"
    assert output.specprefill_effective_policy == "sparse"
    assert output.specprefill_coverage == "selective"
    assert output.specprefill_fallback_reason is None
    assert output.specprefill_selector_version is not None
    assert output.specprefill_total_tokens == len(_TOKENS)
    assert output.specprefill_selected_tokens > 0
    assert output.specprefill_scorer_ms >= 0.0
    assert output.specprefill_target_prefill_ms == 1.0
    assert (output.mtp_drafts, output.mtp_accepted) == (3, 2)


def test_dense_or_unavailable_policy_never_enters_cooperative_queue():
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory, enabled=False)
    scheduler.add_request(
        "prompt",
        request_id="dense",
        specprefill_coverage=SpecPrefillCoverage.SELECTIVE,
    )

    scheduler.step()

    request = scheduler.running["dense"]
    assert request.specprefill_effective_policy is SpecPrefillPolicy.DENSE
    assert request.specprefill_fallback_reason == "admission_denied"
    assert factory.sessions == {}
    assert len(scheduler.batch_generator.inserted) == 1
    assert scheduler._specprefill_queue == deque()


def test_media_request_uses_separate_dense_fallback_path():
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory)
    scheduler.add_request(
        "prompt",
        images=["image.png"],
        request_id="media",
        specprefill_coverage=SpecPrefillCoverage.SELECTIVE,
    )

    scheduler.step()

    request = scheduler.running["media"]
    assert request.specprefill_effective_policy is SpecPrefillPolicy.DENSE
    assert request.specprefill_fallback_reason == "media_request"
    assert factory.sessions == {}


def test_ready_sparse_waits_behind_incompatible_dense_active_batch():
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "waiter")
    scheduler.step()
    scheduler.batch_generator.active_batch = SimpleNamespace(
        has_sparse_rows=False,
        sparse_execution_config=None,
    )

    scheduler.step()

    assert scheduler.batch_generator.adoptions == []
    assert list(scheduler._specprefill_queue) == ["waiter"]
    assert scheduler.running["waiter"].output_tokens == []


def test_cancellation_is_terminal_and_never_requeues_dense():
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "cancelled")
    scheduler._schedule_waiting()
    session = factory.sessions["cancelled"]

    assert scheduler.abort_request("cancelled") is True
    assert session.outcome is CooperativeSpecPrefillOutcome.ACTIVE
    assert "cancelled" in scheduler._specprefill_admissions
    assert scheduler.has_requests() is True

    scheduler.step()

    assert session.outcome is CooperativeSpecPrefillOutcome.CANCELLED
    assert "cancelled" not in scheduler.running
    assert "cancelled" not in scheduler.requests
    assert scheduler.batch_generator.inserted == []
    assert scheduler._specprefill_queue == deque()


def test_mtp_composition_fails_closed_to_dense():
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory, mtp=True)
    _add_selective(scheduler, "mtp")

    scheduler.step()

    request = scheduler.running["mtp"]
    assert request.specprefill_effective_policy is SpecPrefillPolicy.DENSE
    assert request.specprefill_fallback_reason == "mtp_composition_unavailable"
    assert factory.sessions == {}


@pytest.mark.parametrize(
    "layout", ("gemma4_dense", "gemma4_a4b", "mixed_rotating")
)
def test_unqualified_adapter_cache_layouts_fail_closed_to_dense(layout):
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory, cache_layout=layout)
    _add_selective(scheduler, layout)

    scheduler.step()

    request = scheduler.running[layout]
    assert request.specprefill_effective_policy is SpecPrefillPolicy.DENSE
    assert request.specprefill_fallback_reason == "cache_capability_unsupported"
    assert factory.sessions == {}


def test_cache_object_identity_change_is_terminal_before_adoption():
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "identity")
    scheduler.step()
    target = factory.sessions["identity"]._target_session
    target.replace_on_second_read = True

    output = scheduler.step()

    assert scheduler.batch_generator.adoptions == []
    assert scheduler.batch_generator.inserted == []
    assert output.outputs[0].specprefill_fallback_reason == (
        "adoption_failed_unknown_boundary"
    )
    assert "identity" not in scheduler.running


def test_abort_during_generator_adoption_removes_published_orphan_row():
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "coordinated-abort")
    scheduler.step()
    scheduler.batch_generator.on_adopt = lambda: scheduler.abort_request(
        "coordinated-abort"
    )

    output = scheduler.step()

    assert output.outputs == []
    assert scheduler.batch_generator.scheduled_removals == [100]
    assert scheduler.batch_generator.pending_removals == []
    assert scheduler.batch_generator.active_rows == {}
    assert scheduler.batch_generator.active_cache_rows == {}
    assert "coordinated-abort" not in scheduler.running
    assert "coordinated-abort" not in scheduler._specprefill_admissions
    assert 100 not in scheduler.uid_to_request_id


def test_mark_adopted_failure_removes_published_row_before_terminal_failure():
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "mark-failure")
    scheduler.step()
    session = factory.sessions["mark-failure"]

    def fail_mark():
        raise RuntimeError("mark failed")

    session.mark_adopted = fail_mark
    output = scheduler.step()

    assert scheduler.batch_generator.scheduled_removals == [100]
    assert scheduler.batch_generator.pending_removals == []
    assert scheduler.batch_generator.active_rows == {}
    assert scheduler.batch_generator.active_cache_rows == {}
    assert output.outputs[0].specprefill_fallback_reason == (
        "adoption_transition_failed"
    )
    assert "mark-failure" not in scheduler.running
    assert 100 not in scheduler.uid_to_request_id


def test_authoritative_cache_property_failure_is_unknown_boundary_terminal():
    factory = _AdmissionFactory()
    scheduler = _scheduler(factory)
    _add_selective(scheduler, "cache-seam")
    scheduler.step()
    target = factory.sessions["cache-seam"]._target_session
    target.fail_cache_read = True
    output = scheduler.step()

    assert scheduler.batch_generator.adoptions == []
    assert scheduler.batch_generator.inserted == []
    assert output.outputs[0].specprefill_fallback_reason == (
        "adoption_failed_unknown_boundary"
    )
