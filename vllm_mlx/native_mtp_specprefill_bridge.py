# SPDX-License-Identifier: Apache-2.0
"""Standard-text CB admission bridge for attested SpecPrefill native MTP.

This module owns no scheduler policy and never performs dense replay.  It
advances one request-owned cooperative SpecPrefill quantum at a time, then
turns the exact completed target receipts into the public mlx-lm sparse
bootstrap consumed by :class:`NativeMTPContinuousBatchAdapter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .cooperative_specprefill import (
    CooperativeSpecPrefillConfig,
    CooperativeSpecPrefillOutcome,
)


class NativeMTPSpecPrefillBridgeState(str, Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    BOOTSTRAP_READY = "bootstrap_ready"
    ADOPTED = "adopted"
    TERMINAL = "terminal"


class NativeMTPSpecPrefillBridgeError(RuntimeError):
    """A combined sparse admission cannot safely be continued."""


@dataclass(frozen=True)
class NativeMTPSpecPrefillBridgeProgress:
    state: NativeMTPSpecPrefillBridgeState
    busy: bool = False
    fallback_reason: str | None = None


class NativeMTPSpecPrefillBridge:
    """One request's attested sparse admission before cohort ownership.

    ``prepared`` is the eager runtime bundle built at engine start.  The
    optional factory is the sole route that can produce receipt-bearing target
    forwards; ordinary SpecPrefill sessions deliberately remain unusable here.
    """

    def __init__(self, prepared: Any, request: Any, config: CooperativeSpecPrefillConfig):
        factory = getattr(prepared, "native_mtp_session_factory", None)
        if not callable(factory):
            raise NativeMTPSpecPrefillBridgeError(
                "native_mtp_attested_session_factory_unavailable"
            )
        tokens = tuple(getattr(request, "prompt_token_ids", ()) or ())
        if not tokens:
            raise NativeMTPSpecPrefillBridgeError("native_mtp_prompt_required")
        self._request = request
        self._tokens = tokens
        self._session = factory(request, tokens, config)
        self._bootstrap = None
        self._state = NativeMTPSpecPrefillBridgeState.WAITING
        self._fallback_reason = None

    @property
    def state(self) -> NativeMTPSpecPrefillBridgeState:
        return self._state

    @property
    def bootstrap(self) -> Any:
        if self._state not in {
            NativeMTPSpecPrefillBridgeState.BOOTSTRAP_READY,
            NativeMTPSpecPrefillBridgeState.ADOPTED,
        }:
            raise NativeMTPSpecPrefillBridgeError("native_mtp_sparse_bootstrap_unready")
        return self._bootstrap

    def step(self) -> NativeMTPSpecPrefillBridgeProgress:
        if self._state is NativeMTPSpecPrefillBridgeState.TERMINAL:
            return NativeMTPSpecPrefillBridgeProgress(
                self._state, fallback_reason=self._fallback_reason
            )
        if self._state is NativeMTPSpecPrefillBridgeState.BOOTSTRAP_READY:
            return NativeMTPSpecPrefillBridgeProgress(self._state)
        if self._state is NativeMTPSpecPrefillBridgeState.ADOPTED:
            raise NativeMTPSpecPrefillBridgeError("native_mtp_sparse_bootstrap_already_adopted")
        self._state = NativeMTPSpecPrefillBridgeState.PREFILLING
        progress = self._session.step()
        if progress.busy:
            return NativeMTPSpecPrefillBridgeProgress(self._state, busy=True)
        if self._session.outcome is not CooperativeSpecPrefillOutcome.READY_FOR_ADOPTION:
            if self._session.outcome is not CooperativeSpecPrefillOutcome.ACTIVE:
                self._fallback_reason = self._session.fallback_reason or "sparse_prefill_failed"
                self._state = NativeMTPSpecPrefillBridgeState.TERMINAL
            return NativeMTPSpecPrefillBridgeProgress(self._state, fallback_reason=self._fallback_reason)
        try:
            self._bootstrap = self._build_bootstrap()
            self._state = NativeMTPSpecPrefillBridgeState.BOOTSTRAP_READY
            return NativeMTPSpecPrefillBridgeProgress(self._state)
        except BaseException:
            self.cancel()
            raise

    def mark_adopted(self) -> Any:
        if self._state is not NativeMTPSpecPrefillBridgeState.BOOTSTRAP_READY:
            raise NativeMTPSpecPrefillBridgeError("native_mtp_sparse_bootstrap_unready")
        try:
            self._session.mark_adopted()
        except BaseException:
            self.cancel()
            raise
        self._state = NativeMTPSpecPrefillBridgeState.ADOPTED
        return self._bootstrap

    def cancel(self) -> None:
        if self._state is NativeMTPSpecPrefillBridgeState.TERMINAL:
            return
        bootstrap = self._bootstrap
        self._bootstrap = None
        if bootstrap is not None:
            bootstrap.close()
        if self._state is not NativeMTPSpecPrefillBridgeState.ADOPTED:
            self._session.cancel()
        self._state = NativeMTPSpecPrefillBridgeState.TERMINAL

    def _build_bootstrap(self) -> Any:
        try:
            from mlx_lm.generate import NativeMTPSparseBootstrap
        except ImportError as exc:
            raise NativeMTPSpecPrefillBridgeError(
                "native_mtp_sparse_bootstrap_api_unavailable"
            ) from exc
        plan = self._session.selection_plan
        result = self._session.prepared_result
        cache = self._session.prepared_cache
        if plan is None or not isinstance(cache, list):
            raise NativeMTPSpecPrefillBridgeError("native_mtp_sparse_result_invalid")
        positions = tuple(plan.selected_indices)
        if not positions or positions[-1] != len(self._tokens) - 1:
            raise NativeMTPSpecPrefillBridgeError("native_mtp_sparse_final_anchor_missing")
        selected = tuple(self._tokens[position] for position in positions)
        successors = tuple(self._tokens[position + 1] for position in positions[:-1])
        receipts = tuple(result.forward_receipts)
        if not receipts:
            raise NativeMTPSpecPrefillBridgeError("native_mtp_sparse_receipts_missing")
        return NativeMTPSparseBootstrap(
            receipts=receipts,
            selected_logical_positions=positions,
            selected_token_ids=selected,
            immediate_successor_token_ids=successors,
            target_cache=cache,
            next_logical_position=len(self._tokens),
        )
