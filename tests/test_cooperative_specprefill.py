# SPDX-License-Identifier: Apache-2.0
"""Request-local cooperative SpecPrefill orchestration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.cooperative_specprefill import (
    CooperativeSpecPrefillCleanupError,
    CooperativeSpecPrefillConfig,
    CooperativeSpecPrefillError,
    CooperativeSpecPrefillOutcome,
    CooperativeSpecPrefillPhase,
    CooperativeSpecPrefillSession,
)
from vllm_mlx.specprefill import SpecPrefillScorerLaneBusy, build_selection_plan
from vllm_mlx.specprefill_cache import SparseCacheState, SparsePolicyTuning
from vllm_mlx.specprefill_scorer_session import (
    ScorerSessionPhase,
    ScorerSessionProgress,
)
from vllm_mlx.specprefill_selection import RotatingTailRequirement
from vllm_mlx.specprefill_target_executor import (
    SparseTargetPrefillLaneBusy,
    SparseTargetPrefillProgress,
    SparseTargetPrefillResult,
    SparseTargetPrefillTelemetry,
)

_TOKENS = (10, 11, 12, 13, 14, 15, 16, 17)
_IMPORTANCE = mx.array([0.0, 0.1, 0.2, 0.3, 0.9, 0.8, 0.4, 0.5])
_TUNING = SparsePolicyTuning(
    keep_pct=0.25,
    backbone_pct=0.0,
    halo_chunks=0,
    anchor_chunks=1,
    chunk_size=2,
)
_CONFIG = CooperativeSpecPrefillConfig(
    target_id="target@sha256:target",
    tokenizer_id="tokenizer@sha256:tokenizer",
    scorer_id="scorer@sha256:scorer",
    tuning=_TUNING,
)


class _FakeScorerSession:
    def __init__(
        self,
        *,
        importance=_IMPORTANCE,
        fail_at: int | None = None,
        cleanup_raises: bool = False,
    ):
        self.phase = ScorerSessionPhase.PREFILL
        self.importance = importance
        self.fail_at = fail_at
        self.calls = 0
        self.cancelled = False
        self.cleanup_raises = cleanup_raises
        self.active = False

    def step(self):
        self.calls += 1
        if self.fail_at == self.calls:
            raise RuntimeError("scorer failed")
        self.active = True
        try:
            schedule = (
                ScorerSessionProgress(ScorerSessionPhase.PREFILL, 4, 0, 0, False),
                ScorerSessionProgress(ScorerSessionPhase.LOOKAHEAD, 8, 0, 0, False),
                ScorerSessionProgress(ScorerSessionPhase.IMPORTANCE, 8, 1, 0, False),
                ScorerSessionProgress(ScorerSessionPhase.IMPORTANCE, 8, 1, 1, False),
                ScorerSessionProgress(ScorerSessionPhase.COMPLETE, 8, 1, 1, True),
            )
            progress = schedule[self.calls - 1]
            self.phase = progress.phase
            return progress
        finally:
            self.active = False

    def cancel(self):
        self.cancelled = True
        if self.cleanup_raises:
            raise RuntimeError("scorer cleanup failed")


class _BusyOnceScorer(_FakeScorerSession):
    def __init__(self):
        super().__init__()
        self._busy_returned = False

    def step(self):
        if not self._busy_returned:
            self._busy_returned = True
            raise SpecPrefillScorerLaneBusy("busy")
        return super().step()


@dataclass
class _FakeTargetSession:
    state: SparseCacheState
    fail_at: int | None = None
    busy_once: bool = False
    cleanup_raises: bool = False

    def __post_init__(self):
        self.calls = 0
        self.cancelled = False
        self.active = False
        self._busy_returned = False
        self.result = SparseTargetPrefillResult(
            logits=mx.array([[[1.0, 0.0]]]),
            cache_state=self.state.clone(),
            telemetry=SparseTargetPrefillTelemetry(
                selected_tokens=self.state.rows[0].physical_valid_length,
                target_prefill_ms=1.0,
                chunk_count=2,
                physical_cache_starts=((0,), (1,)),
            ),
        )

    def step(self):
        if self.busy_once and not self._busy_returned:
            self._busy_returned = True
            raise SparseTargetPrefillLaneBusy("busy")
        self.calls += 1
        if self.fail_at == self.calls:
            raise RuntimeError("target failed")
        self.active = True
        try:
            return SparseTargetPrefillProgress(
                selected_tokens_processed=min(self.calls, 2),
                chunk_count=self.calls,
                complete=self.calls == 2,
            )
        finally:
            self.active = False

    def cancel(self):
        self.cancelled = True
        if self.cleanup_raises:
            raise RuntimeError("target cleanup failed")


class _TargetFactory:
    def __init__(self, *, setup_raises=False, **target_options):
        self.calls = []
        self.setup_raises = setup_raises
        self.target_options = target_options
        self.session = None

    def __call__(self, selected_tokens, sparse_state):
        self.calls.append((selected_tokens, sparse_state))
        if self.setup_raises:
            raise RuntimeError("target setup failed")
        self.session = _FakeTargetSession(sparse_state, **self.target_options)
        return self.session


def _cooperative(
    *,
    scorer=None,
    factory=None,
    tokens=_TOKENS,
    config=_CONFIG,
    selection_builder=build_selection_plan,
):
    scorer = scorer or _FakeScorerSession()
    factory = factory or _TargetFactory()
    session = CooperativeSpecPrefillSession(
        "request-1",
        tokens,
        scorer,
        factory,
        config,
        selection_builder=selection_builder,
    )
    return session, scorer, factory


def _run_until_ready(session):
    progress = []
    while not session.ready_for_adoption:
        progress.append(session.step())
    return progress


def test_exact_phase_trace_commits_only_one_model_quantum_per_step():
    session, scorer, factory = _cooperative()

    progress = _run_until_ready(session)

    assert [item.attempted_phase for item in progress] == [
        CooperativeSpecPrefillPhase.SCORE_PREFILL,
        CooperativeSpecPrefillPhase.SCORE_PREFILL,
        CooperativeSpecPrefillPhase.LOOKAHEAD,
        CooperativeSpecPrefillPhase.IMPORTANCE,
        CooperativeSpecPrefillPhase.IMPORTANCE,
        CooperativeSpecPrefillPhase.SPARSE_TARGET_PREFILL,
        CooperativeSpecPrefillPhase.SPARSE_TARGET_PREFILL,
    ]
    assert all(item.quantum_committed for item in progress)
    assert scorer.calls == 5
    assert factory.session.calls == 2
    assert len(progress) == scorer.calls + factory.session.calls
    assert all(not scorer.active for _ in progress)
    assert not factory.session.active
    assert session.phase is CooperativeSpecPrefillPhase.SPARSE_TARGET_PREFILL
    assert session.outcome is CooperativeSpecPrefillOutcome.READY_FOR_ADOPTION
    assert session.telemetry.scorer_quanta == 5
    assert session.telemetry.target_quanta == 2
    assert session.telemetry.importance_layers == 1
    assert session.telemetry.target_chunks == 2
    assert session.telemetry.target_prefill_ms == pytest.approx(1.0)


def test_selection_and_exact_identity_are_deterministic_and_request_owned():
    first, _, first_factory = _cooperative()
    second, _, second_factory = _cooperative()

    _run_until_ready(first)
    _run_until_ready(second)

    assert first.selection_plan == second.selection_plan
    assert first.identity == second.identity
    assert first.identity.selection_fingerprint == first.selection_plan.fingerprint
    assert first.identity.full_token_hash == second.identity.full_token_hash
    assert first.sparse_state is first_factory.calls[0][1]
    assert second.sparse_state is second_factory.calls[0][1]
    expected_tokens = tuple(
        _TOKENS[index] for index in first.selection_plan.selected_indices
    )
    assert first_factory.calls[0][0] == expected_tokens
    assert first.sparse_state.logical_positions == (
        first.selection_plan.selected_indices,
    )
    assert first.sparse_state.next_logical_positions == (len(_TOKENS),)


def test_different_full_prompt_cannot_share_exact_identity():
    first, _, _ = _cooperative()
    changed, _, _ = _cooperative(tokens=(*_TOKENS[:-1], 99))

    _run_until_ready(first)
    _run_until_ready(changed)

    assert first.selection_plan == changed.selection_plan
    assert (
        first.identity.selection_fingerprint == changed.identity.selection_fingerprint
    )
    assert first.identity.full_token_hash != changed.identity.full_token_hash
    assert first.identity != changed.identity


def test_control_anchors_and_rotating_tail_reach_selection_and_identity():
    requirement = RotatingTailRequirement(2)
    config = CooperativeSpecPrefillConfig(
        target_id=_CONFIG.target_id,
        tokenizer_id=_CONFIG.tokenizer_id,
        scorer_id=_CONFIG.scorer_id,
        tuning=_TUNING,
        control_token_indices=(5, 0, 5),
        rotating_tail_requirement=requirement,
    )
    observed = {}

    def recording_builder(importance, **kwargs):
        observed.update(kwargs)
        return build_selection_plan(importance, **kwargs)

    session, _, _ = _cooperative(config=config, selection_builder=recording_builder)
    _run_until_ready(session)

    assert config.control_token_indices == (0, 5)
    assert observed["control_token_indices"] == (0, 5)
    assert observed["rotating_tail_requirement"] is requirement
    assert session.selection_plan.control_anchor_indices == (0, 5)
    assert session.selection_plan.rotating_tail_requirement is requirement
    assert session.selection_plan.rotating_tail_chunks == (3,)
    assert {0, 2, 3}.issubset(session.selection_plan.selected_chunks)
    assert session.identity.selection_fingerprint == session.selection_plan.fingerprint


def test_selector_cannot_drop_mandatory_request_semantics():
    config = CooperativeSpecPrefillConfig(
        target_id=_CONFIG.target_id,
        tokenizer_id=_CONFIG.tokenizer_id,
        scorer_id=_CONFIG.scorer_id,
        tuning=_TUNING,
        control_token_indices=(0,),
        rotating_tail_requirement=RotatingTailRequirement(2),
    )

    def dropping_builder(importance, **kwargs):
        kwargs.pop("control_token_indices")
        kwargs.pop("rotating_tail_requirement")
        return build_selection_plan(importance, **kwargs)

    session, _, factory = _cooperative(
        config=config, selection_builder=dropping_builder
    )

    while session.outcome is CooperativeSpecPrefillOutcome.ACTIVE:
        session.step()

    assert session.outcome is CooperativeSpecPrefillOutcome.FALLBACK
    assert session.fallback_reason == "selection_failed"
    assert factory.calls == []


def test_config_validates_tuning_controls_tail_and_prompt_bounds():
    with pytest.raises(TypeError, match="CooperativeSpecPrefillConfig"):
        _cooperative(config=object())
    with pytest.raises(TypeError, match="SparsePolicyTuning"):
        CooperativeSpecPrefillConfig(
            _CONFIG.target_id, _CONFIG.tokenizer_id, _CONFIG.scorer_id, object()
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        CooperativeSpecPrefillConfig(
            _CONFIG.target_id,
            _CONFIG.tokenizer_id,
            _CONFIG.scorer_id,
            _TUNING,
            control_token_indices=(True,),
        )
    with pytest.raises(TypeError, match="RotatingTailRequirement"):
        CooperativeSpecPrefillConfig(
            _CONFIG.target_id,
            _CONFIG.tokenizer_id,
            _CONFIG.scorer_id,
            _TUNING,
            rotating_tail_requirement=object(),
        )
    config = CooperativeSpecPrefillConfig(
        _CONFIG.target_id,
        _CONFIG.tokenizer_id,
        _CONFIG.scorer_id,
        _TUNING,
        control_token_indices=(len(_TOKENS),),
    )
    with pytest.raises(ValueError, match="within the request prompt"):
        _cooperative(config=config)


def test_lane_busy_is_retryable_and_does_not_commit_a_quantum():
    scorer = _BusyOnceScorer()
    factory = _TargetFactory(busy_once=True)
    session, _, _ = _cooperative(scorer=scorer, factory=factory)

    scorer_busy = session.step()
    assert scorer_busy.busy
    assert not scorer_busy.quantum_committed
    assert session.phase is CooperativeSpecPrefillPhase.SCORE_PREFILL
    assert session.outcome is CooperativeSpecPrefillOutcome.ACTIVE

    while session.phase is not CooperativeSpecPrefillPhase.SPARSE_TARGET_PREFILL:
        session.step()
    target_busy = session.step()
    assert target_busy.busy
    assert not target_busy.quantum_committed
    assert session.outcome is CooperativeSpecPrefillOutcome.ACTIVE
    assert session.telemetry.busy_retries == 2

    _run_until_ready(session)
    assert session.outcome is CooperativeSpecPrefillOutcome.READY_FOR_ADOPTION


@pytest.mark.parametrize(
    ("scorer", "factory", "reason"),
    (
        (_FakeScorerSession(fail_at=1), _TargetFactory(), "scorer_failed"),
        (_FakeScorerSession(), _TargetFactory(fail_at=1), "target_prefill_failed"),
    ),
)
def test_execution_failure_becomes_pre_adoption_fallback(scorer, factory, reason):
    session, _, _ = _cooperative(scorer=scorer, factory=factory)

    while session.outcome is CooperativeSpecPrefillOutcome.ACTIVE:
        session.step()

    assert session.outcome is CooperativeSpecPrefillOutcome.FALLBACK
    assert session.fallback_reason == reason
    assert session.failure.error_type == "RuntimeError"
    assert scorer.cancelled
    if factory.session is not None:
        assert factory.session.cancelled
    with pytest.raises(CooperativeSpecPrefillError, match="not publishable"):
        _ = session.prepared_result


def test_target_setup_failure_is_distinct_and_preserves_immutable_identity():
    factory = _TargetFactory(setup_raises=True)
    session, scorer, _ = _cooperative(factory=factory)

    while session.outcome is CooperativeSpecPrefillOutcome.ACTIVE:
        session.step()

    assert session.outcome is CooperativeSpecPrefillOutcome.FALLBACK
    assert session.fallback_reason == "target_setup_failed"
    assert session.selection_plan is not None
    assert session.identity is not None
    assert session.sparse_state is not None
    assert scorer.cancelled
    assert session._target_session is None
    assert session._target_session_factory is None


@pytest.mark.parametrize(
    ("malformed_target", "expected_outcome", "expected_reason"),
    (
        (
            None,
            CooperativeSpecPrefillOutcome.FALLBACK,
            "target_setup_failed",
        ),
        (
            SimpleNamespace(cancel=lambda: None, result=object()),
            CooperativeSpecPrefillOutcome.FALLBACK,
            "target_setup_failed",
        ),
        (
            SimpleNamespace(step=lambda: None, result=object()),
            CooperativeSpecPrefillOutcome.FAILED,
            "resource_cleanup_failed",
        ),
        (
            SimpleNamespace(step=lambda: None, cancel=lambda: None),
            CooperativeSpecPrefillOutcome.FALLBACK,
            "target_setup_failed",
        ),
    ),
)
def test_malformed_target_factory_result_becomes_target_setup_fallback(
    malformed_target, expected_outcome, expected_reason
):
    factory = lambda _tokens, _state: malformed_target
    session, scorer, _ = _cooperative(factory=factory)

    while session.outcome is CooperativeSpecPrefillOutcome.ACTIVE:
        session.step()

    assert session.outcome is expected_outcome
    assert session.fallback_reason == expected_reason
    if expected_outcome is CooperativeSpecPrefillOutcome.FAILED:
        assert isinstance(session.failure, CooperativeSpecPrefillCleanupError)
    else:
        assert session.failure.error_type == "CooperativeSpecPrefillError"
    assert scorer.cancelled
    assert session._target_session is None
    assert session._target_session_factory is None


def test_invalid_importance_falls_back_before_target_construction():
    scorer = _FakeScorerSession(importance=mx.array([0.0, float("nan")] * 4))
    session, _, factory = _cooperative(scorer=scorer)

    while session.outcome is CooperativeSpecPrefillOutcome.ACTIVE:
        session.step()

    assert session.outcome is CooperativeSpecPrefillOutcome.FALLBACK
    assert session.fallback_reason == "selection_failed"
    assert factory.calls == []
    assert scorer.cancelled


def test_cancel_hides_prepared_result_and_never_enters_decode():
    session, scorer, factory = _cooperative()
    _run_until_ready(session)
    assert session.prepared_result is factory.session.result

    session.cancel()

    assert session.outcome is CooperativeSpecPrefillOutcome.CANCELLED
    assert session.phase is CooperativeSpecPrefillPhase.SPARSE_TARGET_PREFILL
    assert scorer.cancelled
    assert factory.session.cancelled
    assert session._target_session is None
    assert session._target_session_factory is None
    assert session._prepared_result is None
    with pytest.raises(CooperativeSpecPrefillError, match="not publishable"):
        _ = session.prepared_result
    with pytest.raises(CooperativeSpecPrefillError, match="adoption requires"):
        session.mark_adopted()


@pytest.mark.parametrize("resource", ("scorer", "target"))
def test_cleanup_failure_is_terminal_failed_and_drops_target_references(resource):
    scorer = _FakeScorerSession(cleanup_raises=resource == "scorer")
    factory = _TargetFactory(cleanup_raises=resource == "target")
    session, _, _ = _cooperative(scorer=scorer, factory=factory)

    if resource == "target":
        _run_until_ready(session)
    session.cancel()

    assert session.outcome is CooperativeSpecPrefillOutcome.FAILED
    assert session.fallback_reason == "resource_cleanup_failed"
    assert isinstance(session.failure, CooperativeSpecPrefillCleanupError)
    assert session.failure.cleanup_failures
    assert session._scorer_session is None
    assert session._target_session is None
    assert session._target_session_factory is None
    assert session._prepared_result is None
    with pytest.raises(CooperativeSpecPrefillError, match="cannot step"):
        session.step()


def test_decode_transition_requires_explicit_successful_adoption():
    session, _, factory = _cooperative()
    with pytest.raises(CooperativeSpecPrefillError, match="adoption requires"):
        session.mark_adopted()

    _run_until_ready(session)
    result = session.mark_adopted()

    assert result is factory.session.result
    assert session.phase is CooperativeSpecPrefillPhase.DECODE
    assert session.outcome is CooperativeSpecPrefillOutcome.DECODE
    with pytest.raises(CooperativeSpecPrefillError, match="cannot step"):
        session.step()
    with pytest.raises(CooperativeSpecPrefillError, match="unsafe after"):
        session.fallback("adoption_failed")
    with pytest.raises(CooperativeSpecPrefillError, match="unavailable after"):
        session.cancel()
