# SPDX-License-Identifier: Apache-2.0
"""Request-owned orchestration for cooperative SpecPrefill.

This module deliberately contains no scheduler policy and no model wrappers.
One call to :meth:`CooperativeSpecPrefillSession.step` attempts at most one
bounded scorer or target quantum.  Selection and exact cache metadata are
prepared only after the scorer has yielded its final quantum, and target
results remain non-publishable until the caller explicitly marks adoption by
the normal decoder.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Protocol, Sequence

from .specprefill import (
    SpecPrefillScorerLaneBusy,
    build_selection_plan,
)
from .specprefill_cache import (
    SparseCacheExecutionConfig,
    SparseCacheIdentity,
    SparseCacheState,
    SparsePolicyTuning,
)
from .specprefill_scorer_session import (
    ScorerSessionPhase,
    ScorerSessionProgress,
    SpecPrefillScorerSession,
)
from .specprefill_selection import RotatingTailRequirement, SelectionPlan
from .specprefill_target_executor import (
    SparseTargetPrefillLaneBusy,
    SparseTargetPrefillProgress,
    SparseTargetPrefillResult,
    SparseTargetPrefillSession,
)


class CooperativeSpecPrefillError(RuntimeError):
    """A cooperative request is used outside its safe lifecycle."""


@dataclass(frozen=True)
class CooperativeSpecPrefillFailure:
    """Traceback-free failure facts that cannot retain model/cache frames."""

    error_type: str
    message: str


class CooperativeSpecPrefillCleanupError(CooperativeSpecPrefillError):
    """Request resources could not be completely cleaned before adoption."""

    def __init__(
        self,
        cleanup_failures: Sequence[Exception],
        prior_failure: Exception | None = None,
    ) -> None:
        self.cleanup_failures = tuple(
            _summarize_failure(failure) for failure in cleanup_failures
        )
        self.prior_failure = (
            None if prior_failure is None else _summarize_failure(prior_failure)
        )
        super().__init__(
            "cooperative SpecPrefill resource cleanup failed: "
            + ", ".join(failure.error_type for failure in self.cleanup_failures)
        )


class CooperativeSpecPrefillPhase(str, Enum):
    """Externally visible model-work phases for one sparse request."""

    SCORE_PREFILL = "score_prefill"
    LOOKAHEAD = "lookahead"
    IMPORTANCE = "importance"
    SPARSE_TARGET_PREFILL = "sparse_target_prefill"
    DECODE = "decode"


class CooperativeSpecPrefillOutcome(str, Enum):
    """Request outcomes kept separate from model-work phase."""

    ACTIVE = "active"
    READY_FOR_ADOPTION = "ready_for_adoption"
    DECODE = "decode"
    FALLBACK = "fallback"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class CooperativeSpecPrefillConfig:
    """Exact artifacts and selector controls owned by one admitted request."""

    target_id: str
    tokenizer_id: str
    scorer_id: str
    tuning: SparsePolicyTuning
    control_token_indices: tuple[int, ...] = ()
    rotating_tail_requirement: RotatingTailRequirement | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tuning, SparsePolicyTuning):
            raise TypeError("tuning must be a SparsePolicyTuning")
        try:
            control_indices = tuple(self.control_token_indices)
        except TypeError as exc:
            raise TypeError(
                "control_token_indices must be an iterable of integers"
            ) from exc
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in control_indices
        ):
            raise ValueError(
                "control_token_indices must contain non-negative integer positions"
            )
        object.__setattr__(
            self, "control_token_indices", tuple(sorted(set(control_indices)))
        )
        if self.rotating_tail_requirement is not None and not isinstance(
            self.rotating_tail_requirement, RotatingTailRequirement
        ):
            raise TypeError(
                "rotating_tail_requirement must be a RotatingTailRequirement or None"
            )
        # Validate artifact identifiers at admission rather than after scoring.
        SparseCacheExecutionConfig(
            target_id=self.target_id,
            tokenizer_id=self.tokenizer_id,
            scorer_id=self.scorer_id,
            selector_version="pending-selection",
            tuning=self.tuning,
        )


@dataclass(frozen=True)
class CooperativeSpecPrefillTelemetry:
    """Cumulative bounded-work facts; scheduler wait time is excluded."""

    scorer_quanta: int = 0
    target_quanta: int = 0
    busy_retries: int = 0
    scorer_prefill_tokens: int = 0
    lookahead_steps: int = 0
    importance_layers: int = 0
    selected_tokens: int = 0
    target_chunks: int = 0
    scorer_ms: float = 0.0
    target_prefill_ms: float = 0.0


@dataclass(frozen=True)
class CooperativeSpecPrefillProgress:
    """State after one call that attempted no more than one model quantum."""

    phase: CooperativeSpecPrefillPhase
    outcome: CooperativeSpecPrefillOutcome
    attempted_phase: CooperativeSpecPrefillPhase | None
    quantum_committed: bool
    busy: bool
    telemetry: CooperativeSpecPrefillTelemetry
    fallback_reason: str | None


class SparseTargetSessionFactory(Protocol):
    """Capture a fresh target cache without giving this layer model ownership."""

    def __call__(
        self,
        selected_tokens: tuple[int, ...],
        sparse_state: SparseCacheState,
    ) -> SparseTargetPrefillSession: ...


SelectionBuilder = Callable[..., SelectionPlan]


class CooperativeSpecPrefillSession:
    """A request-local scorer/selector/target state machine.

    The caller retains policy, queueing, and decoder ownership.  In
    particular, target completion changes the outcome to
    ``READY_FOR_ADOPTION`` while the phase remains ``SPARSE_TARGET_PREFILL``.
    Only :meth:`mark_adopted` may cross the first-output boundary into
    ``DECODE``.
    """

    def __init__(
        self,
        request_id: str,
        tokens: Sequence[int],
        scorer_session: SpecPrefillScorerSession,
        target_session_factory: SparseTargetSessionFactory,
        config: CooperativeSpecPrefillConfig,
        *,
        selection_builder: SelectionBuilder = build_selection_plan,
    ) -> None:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(config, CooperativeSpecPrefillConfig):
            raise TypeError("config must be a CooperativeSpecPrefillConfig")
        token_values = tuple(tokens)
        if not token_values:
            raise ValueError("cooperative SpecPrefill needs at least one token")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_values
        ):
            raise ValueError("tokens must be non-negative integer IDs")
        if any(index >= len(token_values) for index in config.control_token_indices):
            raise ValueError(
                "control_token_indices must be positions within the request prompt"
            )
        if not callable(target_session_factory):
            raise TypeError("target_session_factory must be callable")
        if not callable(selection_builder):
            raise TypeError("selection_builder must be callable")

        try:
            phase = _scorer_phase(scorer_session.phase)
        except AttributeError as exc:
            raise TypeError("scorer_session must expose a scorer phase") from exc
        if phase not in (
            CooperativeSpecPrefillPhase.SCORE_PREFILL,
            CooperativeSpecPrefillPhase.LOOKAHEAD,
            CooperativeSpecPrefillPhase.IMPORTANCE,
        ):
            raise ValueError("scorer_session must be active at admission")

        self.request_id = request_id
        self.tokens = token_values
        self.config = config
        self._scorer_session: SpecPrefillScorerSession | None = scorer_session
        self._target_session_factory: SparseTargetSessionFactory | None = (
            target_session_factory
        )
        self._selection_builder = selection_builder
        self._target_session: SparseTargetPrefillSession | None = None
        self._selection_plan: SelectionPlan | None = None
        self._identity: SparseCacheIdentity | None = None
        self._sparse_state: SparseCacheState | None = None
        self._prepared_result: SparseTargetPrefillResult | None = None
        self._failure: (
            CooperativeSpecPrefillFailure | CooperativeSpecPrefillCleanupError | None
        ) = None
        self._fallback_reason: str | None = None
        self._outcome = CooperativeSpecPrefillOutcome.ACTIVE
        self._phase: CooperativeSpecPrefillPhase = phase
        self._telemetry = CooperativeSpecPrefillTelemetry()

    @property
    def phase(self) -> CooperativeSpecPrefillPhase:
        return self._phase

    @property
    def outcome(self) -> CooperativeSpecPrefillOutcome:
        return self._outcome

    @property
    def telemetry(self) -> CooperativeSpecPrefillTelemetry:
        return self._telemetry

    @property
    def selection_plan(self) -> SelectionPlan | None:
        return self._selection_plan

    @property
    def identity(self) -> SparseCacheIdentity | None:
        return self._identity

    @property
    def sparse_state(self) -> SparseCacheState | None:
        return self._sparse_state

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    @property
    def failure(
        self,
    ) -> CooperativeSpecPrefillFailure | CooperativeSpecPrefillCleanupError | None:
        return self._failure

    @property
    def ready_for_adoption(self) -> bool:
        return self._outcome is CooperativeSpecPrefillOutcome.READY_FOR_ADOPTION

    @property
    def prepared_result(self) -> SparseTargetPrefillResult:
        if (
            self._outcome
            not in (
                CooperativeSpecPrefillOutcome.READY_FOR_ADOPTION,
                CooperativeSpecPrefillOutcome.DECODE,
            )
            or self._prepared_result is None
        ):
            raise CooperativeSpecPrefillError(
                "target result is not publishable before decoder adoption readiness"
            )
        return self._prepared_result

    def step(self) -> CooperativeSpecPrefillProgress:
        """Attempt at most one scorer or target model quantum."""
        if self._outcome is not CooperativeSpecPrefillOutcome.ACTIVE:
            raise CooperativeSpecPrefillError(
                f"cannot step cooperative request in outcome {self._outcome.value}"
            )
        attempted_phase = self._phase
        if attempted_phase is CooperativeSpecPrefillPhase.SPARSE_TARGET_PREFILL:
            return self._step_target(attempted_phase)
        return self._step_scorer(attempted_phase)

    def mark_adopted(self) -> SparseTargetPrefillResult:
        """Cross the output boundary only after decoder adoption succeeds."""
        if not self.ready_for_adoption or self._prepared_result is None:
            raise CooperativeSpecPrefillError(
                "decoder adoption requires a complete sparse target result"
            )
        self._phase = CooperativeSpecPrefillPhase.DECODE
        self._outcome = CooperativeSpecPrefillOutcome.DECODE
        # The decoder/cache owner now holds the adopted target resources.
        self._target_session = None
        self._target_session_factory = None
        return self._prepared_result

    def fallback(self, reason: str, failure: Exception | None = None) -> None:
        """Discard cooperative ownership before decoder adoption/output."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("fallback reason must be a non-empty string")
        if self._outcome is CooperativeSpecPrefillOutcome.DECODE:
            raise CooperativeSpecPrefillError(
                "dense fallback is unsafe after decoder adoption"
            )
        if self._outcome in (
            CooperativeSpecPrefillOutcome.FALLBACK,
            CooperativeSpecPrefillOutcome.CANCELLED,
            CooperativeSpecPrefillOutcome.FAILED,
        ):
            return
        cleanup_failures = self._close_request_resources()
        if cleanup_failures:
            self._fail_closed(cleanup_failures, failure)
            return
        self._fallback_reason = reason
        self._failure = None if failure is None else _summarize_failure(failure)
        self._outcome = CooperativeSpecPrefillOutcome.FALLBACK

    def cancel(self) -> None:
        """Cancel before adoption; ordinary decode owns later cancellation."""
        if self._outcome is CooperativeSpecPrefillOutcome.DECODE:
            raise CooperativeSpecPrefillError(
                "cooperative cancellation is unavailable after decoder adoption"
            )
        if self._outcome in (
            CooperativeSpecPrefillOutcome.FALLBACK,
            CooperativeSpecPrefillOutcome.CANCELLED,
            CooperativeSpecPrefillOutcome.FAILED,
        ):
            return
        cleanup_failures = self._close_request_resources()
        if cleanup_failures:
            self._fail_closed(cleanup_failures)
            return
        self._outcome = CooperativeSpecPrefillOutcome.CANCELLED

    def _step_scorer(
        self, attempted_phase: CooperativeSpecPrefillPhase
    ) -> CooperativeSpecPrefillProgress:
        if self._scorer_session is None:  # pragma: no cover - invariant guard.
            raise CooperativeSpecPrefillError("scorer phase has no scorer session")
        started_at = time.perf_counter()
        try:
            scorer_progress = self._scorer_session.step()
        except SpecPrefillScorerLaneBusy:
            self._record_busy()
            return self._progress(attempted_phase, committed=False, busy=True)
        except Exception as exc:
            self.fallback("scorer_failed", exc)
            return self._progress(attempted_phase, committed=False, busy=False)

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        self._record_scorer(scorer_progress, elapsed_ms)
        if scorer_progress.complete:
            try:
                plan = self._build_selection()
            except Exception as exc:
                self.fallback("selection_failed", exc)
                return self._progress(attempted_phase, committed=True, busy=False)
            self._selection_plan = plan
            try:
                self._setup_target(plan)
            except Exception as exc:
                self.fallback("target_setup_failed", exc)
                return self._progress(attempted_phase, committed=True, busy=False)
            scorer_cleanup_failure = self._release_scorer_after_selection()
            if scorer_cleanup_failure is not None:
                cleanup_failures = self._close_request_resources(
                    (scorer_cleanup_failure,)
                )
                self._fail_closed(cleanup_failures)
        else:
            self._phase = _scorer_phase(scorer_progress.phase)
        return self._progress(attempted_phase, committed=True, busy=False)

    def _step_target(
        self, attempted_phase: CooperativeSpecPrefillPhase
    ) -> CooperativeSpecPrefillProgress:
        if self._target_session is None:  # pragma: no cover - invariant guard.
            raise CooperativeSpecPrefillError("target phase has no target session")
        try:
            target_progress = self._target_session.step()
            result = self._target_session.result if target_progress.complete else None
        except SparseTargetPrefillLaneBusy:
            self._record_busy()
            return self._progress(attempted_phase, committed=False, busy=True)
        except Exception as exc:
            self.fallback("target_prefill_failed", exc)
            return self._progress(attempted_phase, committed=False, busy=False)

        self._record_target(target_progress)
        if result is not None:
            self._prepared_result = result
            self._telemetry = replace(
                self._telemetry,
                target_prefill_ms=(
                    self._telemetry.target_prefill_ms
                    + result.telemetry.target_prefill_ms
                ),
            )
            self._outcome = CooperativeSpecPrefillOutcome.READY_FOR_ADOPTION
        return self._progress(attempted_phase, committed=True, busy=False)

    def _build_selection(self) -> SelectionPlan:
        if self._scorer_session is None:  # pragma: no cover - invariant guard.
            raise CooperativeSpecPrefillError("selection has no scorer session")
        tuning = self.config.tuning
        plan = self._selection_builder(
            self._scorer_session.importance,
            keep_pct=tuning.keep_pct,
            chunk_size=tuning.chunk_size,
            backbone_pct=tuning.backbone_pct,
            halo_chunks=tuning.halo_chunks,
            anchor_chunks=tuning.anchor_chunks,
            control_token_indices=self.config.control_token_indices,
            rotating_tail_requirement=self.config.rotating_tail_requirement,
        )
        if plan.prompt_length != len(self.tokens):
            raise CooperativeSpecPrefillError(
                "selection importance length must match the full prompt"
            )
        if plan.control_anchor_indices != self.config.control_token_indices:
            raise CooperativeSpecPrefillError(
                "selection plan dropped request control-token anchors"
            )
        if plan.rotating_tail_requirement != self.config.rotating_tail_requirement:
            raise CooperativeSpecPrefillError(
                "selection plan dropped the rotating-cache tail requirement"
            )
        return plan

    def _setup_target(self, plan: SelectionPlan) -> None:
        tuning = self.config.tuning
        identity = SparseCacheIdentity.from_tokens(
            target_id=self.config.target_id,
            tokenizer_id=self.config.tokenizer_id,
            scorer_id=self.config.scorer_id,
            selector_version=plan.selector_version,
            tuning=tuning,
            tokens=self.tokens,
            selection_fingerprint=plan.fingerprint,
        )
        sparse_state = SparseCacheState.from_selection(
            identity,
            (plan.selected_indices,),
            (len(self.tokens),),
        )
        selected_tokens = tuple(self.tokens[index] for index in plan.selected_indices)
        if self._target_session_factory is None:  # pragma: no cover - invariant guard.
            raise CooperativeSpecPrefillError("target setup has no session factory")

        # Preserve immutable identity before a factory failure so diagnostics
        # can identify exactly which sparse cache was being constructed.
        self._identity = identity
        self._sparse_state = sparse_state
        target_session = self._target_session_factory(selected_tokens, sparse_state)
        if target_session is None:
            raise CooperativeSpecPrefillError(
                "target_session_factory must return a target session"
            )
        # Publish ownership before structural validation so every non-None
        # factory result is either cleaned or reported as a cleanup failure.
        self._target_session = target_session
        self._target_session_factory = None
        if not callable(getattr(target_session, "step", None)) or not callable(
            getattr(target_session, "cancel", None)
        ):
            raise CooperativeSpecPrefillError(
                "target session must expose callable step and cancel methods"
            )
        try:
            inspect.getattr_static(target_session, "result")
        except AttributeError as exc:
            raise CooperativeSpecPrefillError(
                "target session must expose a result when completion is reported"
            ) from exc
        self._phase = CooperativeSpecPrefillPhase.SPARSE_TARGET_PREFILL
        self._telemetry = replace(
            self._telemetry, selected_tokens=plan.selected_token_count
        )

    def _release_scorer_after_selection(self) -> Exception | None:
        scorer_session = self._scorer_session
        self._scorer_session = None
        if scorer_session is None:
            return None
        try:
            scorer_session.cancel()
        except Exception as exc:
            return exc
        return None

    def _close_request_resources(
        self, initial_failures: Sequence[Exception] = ()
    ) -> tuple[Exception, ...]:
        """Drop all device-owning references even when cleanup itself fails."""
        failures = list(initial_failures)
        scorer_session = self._scorer_session
        target_session = self._target_session
        self._scorer_session = None
        self._target_session = None
        self._target_session_factory = None
        self._prepared_result = None

        if scorer_session is not None:
            try:
                scorer_session.cancel()
            except Exception as exc:
                failures.append(exc)
        if target_session is not None:
            try:
                target_session.cancel()
            except Exception as exc:
                failures.append(exc)
        return tuple(failures)

    def _fail_closed(
        self,
        cleanup_failures: Sequence[Exception],
        prior_failure: Exception | None = None,
    ) -> None:
        self._prepared_result = None
        self._fallback_reason = "resource_cleanup_failed"
        self._failure = CooperativeSpecPrefillCleanupError(
            cleanup_failures, prior_failure
        )
        self._outcome = CooperativeSpecPrefillOutcome.FAILED

    def _record_busy(self) -> None:
        self._telemetry = replace(
            self._telemetry, busy_retries=self._telemetry.busy_retries + 1
        )

    def _record_scorer(
        self, progress: ScorerSessionProgress, elapsed_ms: float
    ) -> None:
        self._telemetry = replace(
            self._telemetry,
            scorer_quanta=self._telemetry.scorer_quanta + 1,
            scorer_prefill_tokens=progress.prefill_tokens,
            lookahead_steps=progress.lookahead_steps,
            importance_layers=progress.importance_layers,
            scorer_ms=self._telemetry.scorer_ms + elapsed_ms,
        )

    def _record_target(self, progress: SparseTargetPrefillProgress) -> None:
        self._telemetry = replace(
            self._telemetry,
            target_quanta=self._telemetry.target_quanta + 1,
            target_chunks=progress.chunk_count,
        )

    def _progress(
        self,
        attempted_phase: CooperativeSpecPrefillPhase | None,
        *,
        committed: bool,
        busy: bool,
    ) -> CooperativeSpecPrefillProgress:
        return CooperativeSpecPrefillProgress(
            phase=self._phase,
            outcome=self._outcome,
            attempted_phase=attempted_phase,
            quantum_committed=committed,
            busy=busy,
            telemetry=self._telemetry,
            fallback_reason=self._fallback_reason,
        )


def _scorer_phase(phase: ScorerSessionPhase) -> CooperativeSpecPrefillPhase:
    if phase is ScorerSessionPhase.PREFILL:
        return CooperativeSpecPrefillPhase.SCORE_PREFILL
    if phase is ScorerSessionPhase.LOOKAHEAD:
        return CooperativeSpecPrefillPhase.LOOKAHEAD
    if phase is ScorerSessionPhase.IMPORTANCE:
        return CooperativeSpecPrefillPhase.IMPORTANCE
    raise CooperativeSpecPrefillError(
        f"scorer phase {phase!r} has no active cooperative phase"
    )


def _summarize_failure(failure: Exception) -> CooperativeSpecPrefillFailure:
    """Copy diagnostics without retaining traceback frames or device owners."""
    return CooperativeSpecPrefillFailure(type(failure).__name__, str(failure))
