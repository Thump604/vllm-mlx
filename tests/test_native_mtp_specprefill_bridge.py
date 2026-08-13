"""Focused executable contracts for the standard-text sparse bridge.

These tests substitute only the public bootstrap constructor; the bridge uses
the real cooperative outcome enum and drives the same ready/adoption boundary
as the runtime owner.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from vllm_mlx.cooperative_specprefill import CooperativeSpecPrefillOutcome
from vllm_mlx.native_mtp_specprefill_bridge import (
    NativeMTPSpecPrefillBridge,
    NativeMTPSpecPrefillBridgeState,
)


class _Bootstrap:
    made = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.closed = False
        type(self).made.append(self)

    def close(self):
        self.closed = True


class _Session:
    def __init__(self, *, ready_after=2, fail_adoption=False):
        self.ready_after = ready_after
        self.fail_adoption = fail_adoption
        self.steps = 0
        self.cancelled = False
        self.outcome = CooperativeSpecPrefillOutcome.ACTIVE
        self.fallback_reason = None
        self.selection_plan = SimpleNamespace(selected_indices=(0, 2, 4))
        self.prepared_result = SimpleNamespace(forward_receipts=(object(), object()))
        self.prepared_cache = [object()]

    def step(self):
        self.steps += 1
        if self.steps >= self.ready_after:
            self.outcome = CooperativeSpecPrefillOutcome.READY_FOR_ADOPTION
        return SimpleNamespace(busy=False)

    def mark_adopted(self):
        if self.fail_adoption:
            raise RuntimeError("adoption_failed")
        self.outcome = CooperativeSpecPrefillOutcome.DECODE

    def cancel(self):
        self.cancelled = True
        self.outcome = CooperativeSpecPrefillOutcome.CANCELLED


def _bridge(monkeypatch, *, session=None):
    generate = importlib.import_module("mlx_lm.generate")

    monkeypatch.setattr(generate, "NativeMTPSparseBootstrap", _Bootstrap)
    session = session or _Session()
    request = SimpleNamespace(request_id="r", prompt_token_ids=[4, 5, 6, 7, 8])
    prepared = SimpleNamespace(
        native_mtp_session_factory=lambda request, tokens, config: session
    )
    return NativeMTPSpecPrefillBridge(prepared, request, object()), session


def test_multi_quantum_bridge_preserves_original_successors_and_cache(monkeypatch):
    bridge, session = _bridge(monkeypatch)
    assert bridge.step().state is NativeMTPSpecPrefillBridgeState.PREFILLING
    assert bridge.step().state is NativeMTPSpecPrefillBridgeState.BOOTSTRAP_READY
    bootstrap = bridge.bootstrap
    assert bootstrap.selected_token_ids == (4, 6, 8)
    assert bootstrap.immediate_successor_token_ids == (5, 7)
    assert bootstrap.target_cache is session.prepared_cache
    assert bootstrap.receipts is session.prepared_result.forward_receipts
    assert bridge.mark_adopted() is bootstrap
    bridge.cancel()
    assert bootstrap.closed
    assert not session.cancelled


def test_build_failure_cancels_unadopted_session(monkeypatch):
    bridge, session = _bridge(monkeypatch)
    session.prepared_result = SimpleNamespace(forward_receipts=())
    bridge.step()
    with pytest.raises(RuntimeError, match="native_mtp_sparse_receipts_missing"):
        bridge.step()
    assert session.cancelled
    assert bridge.state is NativeMTPSpecPrefillBridgeState.TERMINAL


def test_mark_adopt_failure_abandons_bootstrap_and_session(monkeypatch):
    bridge, session = _bridge(
        monkeypatch, session=_Session(ready_after=1, fail_adoption=True)
    )
    bridge.step()
    with pytest.raises(RuntimeError, match="adoption_failed"):
        bridge.mark_adopted()
    # The caller still owns the ready bootstrap when adoption is declined.
    bridge.cancel()
    assert _Bootstrap.made[-1].closed
    assert session.cancelled


def test_cancel_before_adoption_cancels_cooperative_owner(monkeypatch):
    bridge, session = _bridge(monkeypatch)
    bridge.cancel()
    assert session.cancelled
    assert bridge.state is NativeMTPSpecPrefillBridgeState.TERMINAL
