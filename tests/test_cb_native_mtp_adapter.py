"""MLX-free contracts for standard-text native-MTP CB dispatch."""

from __future__ import annotations

from dataclasses import dataclass, replace
import asyncio
from collections import deque
from contextlib import contextmanager
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest


@dataclass(frozen=True)
class _Emission:
    uid: int
    token: int
    logprobs: object
    from_draft: bool
    finish_reason: str | None = None


class _Epoch:
    def __init__(self, active_uids):
        self.active_uids = tuple(active_uids)
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _Initial(_Epoch):
    def resume(self):
        return _Ready(self.active_uids)


class _RejectInitial(_Epoch):
    def resume(self):
        return _RejectReady(self.active_uids)


class _Ready(_Epoch):
    def decide(self):
        return _Decision(self.active_uids)


class _RejectReady(_Epoch):
    def decide(self):
        return _RejectDecision(self.active_uids)


class _Decision(_Epoch):
    def __init__(self, active_uids):
        super().__init__(active_uids)
        self.accepted_uids = (active_uids[0],)
        self.rejected_uids = tuple(active_uids[1:])

    def accept(self):
        return (
            tuple(
                _Emission(uid, uid + 10, object(), True, "length")
                for uid in self.active_uids
            ),
            _Accepted(self.active_uids),
        )

    def reject(self):
        return (
            tuple(
                _Emission(uid, uid + 10, object(), False, "length")
                for uid in self.active_uids
            ),
            _Rejected(self.active_uids),
        )

    def resolve(self):
        return (
            tuple(
                _Emission(uid, uid + 10, object(), uid in self.accepted_uids)
                for uid in self.active_uids
            ),
            _Mixed(self.accepted_uids, self.rejected_uids),
        )


class _RejectDecision(_Decision):
    def __init__(self, active_uids):
        _Epoch.__init__(self, active_uids)
        self.accepted_uids = ()
        self.rejected_uids = tuple(active_uids)


class _Accepted(_Epoch):
    def bonus(self):
        return (
            tuple(
                _Emission(uid, uid + 20, object(), False) for uid in self.active_uids
            ),
            _Bonus(self.active_uids),
        )


class _Bonus(_Epoch):
    def catch_up(self):
        return _Ready(self.active_uids)


class _Rejected(_Epoch):
    def redraft(self):
        return _Ready(self.active_uids)


class _Mixed(_Epoch):
    def __init__(self, accepted, rejected):
        super().__init__(accepted + rejected)
        self.accepted, self.rejected = accepted, rejected

    def resume_after_resolution(self):
        return (
            tuple(
                _Emission(uid, uid + 20, object(), False, "length")
                for uid in self.accepted
            ),
            _MixedBonus(self.rejected),
        )


class _MixedBonus(_Epoch):
    def resume_after_bonus(self):
        return _Ready(self.active_uids)


class _Generator:
    def __init__(self, admission):
        self.admission = admission

    def prefill(self, *, prefill_step_size):
        assert prefill_step_size == 4
        emissions = tuple(
            _Emission(row.uid, row.uid + 1, object(), False)
            for row in self.admission.rows
        )
        return emissions, _Initial(tuple(row.uid for row in self.admission.rows))

    def start_sparse(self):
        emissions = tuple(
            _Emission(row.uid, row.uid + 1, object(), False)
            for row in self.admission.rows
        )
        return emissions, _Initial(tuple(row.uid for row in self.admission.rows))


class _RejectGenerator(_Generator):
    def prefill(self, *, prefill_step_size):
        assert prefill_step_size == 4
        emissions = tuple(
            _Emission(row.uid, row.uid + 1, object(), False)
            for row in self.admission.rows
        )
        return emissions, _RejectInitial(tuple(row.uid for row in self.admission.rows))


class _PoisonedInitial(_Epoch):
    """A consumed public epoch whose model call has already closed its owner."""

    def __init__(self, active_uids, generator):
        super().__init__(active_uids)
        self._generator = generator

    def resume(self):
        self._generator.closed = True
        self._generator.cohort.poisoned = True
        raise RuntimeError("forced_mtp_failure")

    def cancel(self):
        raise AssertionError("stale poisoned epoch must not be cancelled")


class _PoisoningGenerator(_Generator):
    def __init__(self, admission):
        super().__init__(admission)
        self.closed = False
        self.cohort = SimpleNamespace(poisoned=False)

    def prefill(self, *, prefill_step_size):
        emissions = tuple(
            _Emission(row.uid, row.uid + 1, object(), False)
            for row in self.admission.rows
        )
        return emissions, _PoisonedInitial(
            tuple(row.uid for row in self.admission.rows), self
        )


@dataclass(frozen=True)
class _Row:
    uid: int
    prompt: tuple[int, ...]
    max_tokens: int
    seed: int | None
    eos_token_ids: frozenset[int]
    sampling_config: object


class _Admission:
    calls = []

    @classmethod
    def create(cls, model, rows, request_caches, **kwargs):
        cls.calls.append((model, tuple(rows), tuple(request_caches), kwargs))
        return SimpleNamespace(rows=tuple(rows))

    @classmethod
    def create_from_sparse_bootstraps(cls, model, rows, bootstraps):
        cls.calls.append((model, tuple(rows), tuple(bootstraps), "sparse"))
        return SimpleNamespace(rows=tuple(rows))


def _lifecycle(*, reject=False):
    return SimpleNamespace(
        NativeMTPRowSpec=_Row,
        NativeMTPAdmission=_Admission,
        NativeMTPBatchGenerator=_RejectGenerator if reject else _Generator,
        NativeMTPSamplingConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )


def _poisoning_lifecycle():
    lifecycle = _lifecycle()
    lifecycle.NativeMTPBatchGenerator = _PoisoningGenerator
    return lifecycle


class _Model:
    mtp_capability = SimpleNamespace(supported=True, reason="")

    def __init__(self):
        self.created = []

    def make_mtp_request_cache(self, *, prompt_cache=None):
        assert prompt_cache is None
        cache = object()
        self.created.append(cache)
        return cache


class _SparseBootstrap:
    def __init__(self, model, request):
        self.selected_logical_positions = (0, 2)
        self.selected_token_ids = (request.prompt_token_ids[0], request.prompt_token_ids[2])
        self.immediate_successor_token_ids = (request.prompt_token_ids[1],)
        self.target_cache = [object()]
        self.next_logical_position = len(request.prompt_token_ids)
        self.receipts = (
            SimpleNamespace(
                model_id=id(model),
                cache_container_id=id(self.target_cache),
                cache_entry_ids=tuple(id(item) for item in self.target_cache),
            ),
        )


class _FreshCacheFailureModel(_Model):
    def make_mtp_request_cache(self, *, prompt_cache=None):
        assert prompt_cache is None
        raise RuntimeError("fresh_cache_failed")


class _PinnedSparseModel:
    """Real cache-writing target/MTP double for pinned sparse admission."""

    mtp_capability = SimpleNamespace(supported=True, reason="")
    vocab_size = 11

    def __init__(self, *, moe=False):
        from mlx_lm.models.cache import KVCache

        self._cache_type = KVCache
        self.layers = [type("_Attention", (), {"is_linear": False})()]
        self.mtp = type("_MTP", (), {"layers": [object()]})()
        self.moe = moe
        self.target_calls = 0
        self.mtp_calls = 0
        self._forward = None

    def make_cache(self):
        return [self._cache_type()]

    def make_mtp_cache(self):
        return [self._cache_type()]

    @contextmanager
    def generation_forward_context(self, forward):
        prior, self._forward = self._forward, forward
        try:
            yield
        finally:
            self._forward = prior

    def _ack(self):
        if self._forward is not None:
            self._forward.logical_position_ack.acknowledge(
                self._forward.logical_positions
            )

    def _logits(self, inputs):
        delta = mx.where(inputs % 2 == 0, 1, self.vocab_size - 1) if self.moe else 1
        ids = (inputs.astype(mx.int32) + delta) % self.vocab_size
        logits = mx.full((*inputs.shape, self.vocab_size), -20.0)
        for row in range(inputs.shape[0]):
            for column in range(inputs.shape[1]):
                logits[row, column, ids[row, column]] = 20.0
        return logits

    def __call__(self, inputs, *, cache, return_hidden=False):
        self.target_calls += 1
        self._ack()
        values = mx.broadcast_to(
            inputs.astype(mx.float32)[:, None, :, None],
            (inputs.shape[0], 1, inputs.shape[1], 8),
        )
        cache[0].update_and_fetch(values, values)
        hidden = mx.stack((inputs.astype(mx.float32), inputs.astype(mx.float32)), axis=-1)
        logits = self._logits(inputs)
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden, next_tokens, cache):
        self.mtp_calls += 1
        self._ack()
        values = mx.broadcast_to(
            next_tokens.astype(mx.float32)[:, None, :, None],
            (next_tokens.shape[0], 1, next_tokens.shape[1], 8),
        )
        cache[0].update_and_fetch(values, values)
        return self._logits(next_tokens)


def _pinned_bootstrap(model, prompt, positions):
    from mlx_lm.generate import (
        GenerationForwardPhase,
        NativeMTPSparseBootstrap,
        attested_target_forward,
    )

    cache = model.make_cache()
    selected = tuple(prompt[position] for position in positions)
    successors = tuple(prompt[position + 1] for position in positions[:-1])
    receipts = []
    for index, token in enumerate(selected):
        successor = () if index == len(selected) - 1 else (successors[index],)
        (_, _), receipt = attested_target_forward(
            model,
            (token,),
            cache,
            phase=GenerationForwardPhase.PREFILL,
            logical_positions=(positions[index],),
            immediate_successor_token_ids=successor,
            model_forward_context=model.generation_forward_context,
        )
        receipts.append(receipt)
    return NativeMTPSparseBootstrap(
        receipts=tuple(receipts),
        selected_logical_positions=tuple(positions),
        selected_token_ids=selected,
        immediate_successor_token_ids=successors,
        target_cache=cache,
        next_logical_position=len(prompt),
    )


def _request(uid, *, seed=None):
    return SimpleNamespace(
        request_id=f"r{uid}",
        batch_uid=uid,
        prompt_token_ids=[1, 2, uid],
        sampling_params=SimpleNamespace(
            max_tokens=8,
            stop_token_ids=[],
            presence_penalty=0.0,
            repetition_penalty=1.0,
        ),
        native_mtp_eos_token_ids={99},
        native_mtp_config=SimpleNamespace(
            sampling=SimpleNamespace(
                temperature=0.7, top_p=0.9, top_k=7, min_p=0.1, seed=seed
            )
        ),
    )


def _real_request(uid, *, max_tokens=8):
    from vllm_mlx.request import Request, SamplingParams

    request = Request(
        request_id=f"real-{uid}",
        prompt=[1, 2, uid],
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )
    request.prompt_token_ids = [1, 2, uid]
    request.num_prompt_tokens = 3
    request.native_mtp_config = _request(uid).native_mtp_config
    return request


def _bare_scheduler(*requests):
    import vllm_mlx.scheduler as scheduler_mod

    scheduler = object.__new__(scheduler_mod.Scheduler)
    scheduler.waiting = deque(requests)
    scheduler.running = {}
    scheduler.requests = {request.request_id: request for request in requests}
    scheduler.finished_req_ids = set()
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.native_mtp_adapter = None
    scheduler.batch_generator = None
    scheduler._current_sampler_params = None
    scheduler.prefix_cache = None
    scheduler.memory_aware_cache = None
    scheduler.paged_cache_manager = None
    scheduler.block_aware_cache = None
    scheduler._ssd_tier = None
    scheduler._detokenizer_pool = {}
    scheduler._next_native_mtp_uid = 7
    scheduler._native_zero_token_ids = set()
    scheduler._native_cancelled_ids = set()
    scheduler._native_admission_errors = {}
    scheduler._pending_abort_ids = set()
    scheduler.config = SimpleNamespace(max_num_seqs=8, prefill_step_size=4)
    scheduler.model = _Model()
    scheduler.total_prompt_tokens = 0
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_processed = 0
    scheduler._step_count = 0
    scheduler._clear_cache_interval = 999
    scheduler._memory_log_interval = 999
    return scheduler


def _install_minimal_step_cleanup(monkeypatch, scheduler):
    def _cleanup(ids):
        for request_id in ids:
            uid = scheduler.request_id_to_uid.pop(request_id, None)
            if uid is not None:
                scheduler.uid_to_request_id.pop(uid, None)
            scheduler.running.pop(request_id, None)

    monkeypatch.setattr(scheduler, "_cleanup_finished", _cleanup)


def test_b1_uses_uniform_public_emission_result_and_fresh_request_cache():
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    model = _Model()
    adapter = NativeMTPContinuousBatchAdapter.create(
        model, [_request(1, seed=17)], lifecycle=_lifecycle(), prefill_step_size=4
    )
    assert model.created == []  # cancellation before prefill owns no cache bytes
    assert [item.uid for item in adapter.step()] == [1]
    adapter.step()
    adapter.step()
    emissions = adapter.step()
    assert [(item.uid, item.from_draft, item.finish_reason) for item in emissions] == [
        (1, True, "length")
    ]
    assert emissions[0].logprobs is not None
    _, rows, caches, kwargs = _Admission.calls[-1]
    assert rows[0].seed == 17
    assert caches == tuple(model.created)
    assert kwargs == {
        "prefix_cache": None,
        "media": None,
        "external_draft": None,
        "sparse_bootstrap": None,
        "logits_processors": None,
        "kv_bits": None,
        "max_kv_size": None,
    }


def test_uniform_reject_exposes_target_logprobs_and_finish_reason():
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    adapter = NativeMTPContinuousBatchAdapter.create(
        _Model(), [_request(1)], lifecycle=_lifecycle(reject=True), prefill_step_size=4
    )
    adapter.step()
    adapter.step()
    adapter.step()
    emissions = adapter.step()
    assert [(item.uid, item.from_draft, item.finish_reason) for item in emissions] == [
        (1, False, "length")
    ]
    assert emissions[0].logprobs is not None


def test_sparse_bootstraps_are_atomically_adopted_and_start_from_sparse_head():
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    model = _Model()
    requests = [_request(1), _request(2)]
    bootstraps = tuple(_SparseBootstrap(model, request) for request in requests)
    adapter = NativeMTPContinuousBatchAdapter.create(
        model,
        requests,
        lifecycle=_lifecycle(),
        sparse_bootstraps=bootstraps,
    )
    emissions = adapter.step()
    assert [item.uid for item in emissions] == [1, 2]
    assert model.created == []
    _, rows, received, kind = _Admission.calls[-1]
    assert kind == "sparse"
    assert [(row.uid, row.prompt) for row in rows] == [(1, (1, 1)), (2, (1, 2))]
    assert received == bootstraps


def test_sparse_adopted_callback_runs_only_after_public_admission():
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    model = _Model()
    request = _request(1)
    called = []
    adapter = NativeMTPContinuousBatchAdapter.create(
        model,
        [request],
        lifecycle=_lifecycle(),
        sparse_bootstraps=(_SparseBootstrap(model, request),),
        sparse_adopted=lambda: called.append("claimed"),
    )
    assert called == []
    adapter.step()
    assert called == ["claimed"]


def test_sparse_bootstrap_count_mismatch_fails_before_fresh_cache_creation():
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    model = _Model()
    with pytest.raises(RuntimeError, match="native_mtp_sparse_bootstrap_count_mismatch"):
        NativeMTPContinuousBatchAdapter.create(
            model,
            [_request(1), _request(2)],
            lifecycle=_lifecycle(),
            sparse_bootstraps=(_SparseBootstrap(model, _request(1)),),
        )
    assert model.created == []


def test_sparse_bootstrap_cannot_be_reordered_or_reused_before_admission():
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    model = _Model()
    first, second = _request(1), _request(2)
    with pytest.raises(RuntimeError, match="native_mtp_sparse_request_tokens_mismatch"):
        NativeMTPContinuousBatchAdapter.create(
            model,
            [first, second],
            lifecycle=_lifecycle(),
            sparse_bootstraps=(_SparseBootstrap(model, second), _SparseBootstrap(model, first)),
        )
    bootstrap = _SparseBootstrap(model, first)
    with pytest.raises(RuntimeError, match="native_mtp_sparse_bootstrap_reused"):
        NativeMTPContinuousBatchAdapter.create(
            model,
            [first, second],
            lifecycle=_lifecycle(),
            sparse_bootstraps=(bootstrap, bootstrap),
        )


@pytest.mark.parametrize("moe", [False, True])
def test_pinned_sparse_admission_uses_selected_prompt_without_target_replay(moe):
    """Exercise real receipt authority through the adapter's public path."""
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    model = _PinnedSparseModel(moe=moe)
    request = _request(7, seed=3)
    request.prompt_token_ids = [0, 1, 2, 3, 4]
    bootstrap = _pinned_bootstrap(model, request.prompt_token_ids, (0, 2, 4))
    target_calls = model.target_calls
    adapter = NativeMTPContinuousBatchAdapter.create(
        model, [request], sparse_bootstraps=(bootstrap,)
    )
    emissions = adapter.step()
    assert [item.uid for item in emissions] == [7]
    assert model.target_calls == target_calls
    assert model.mtp_calls > 0
    with pytest.raises(RuntimeError, match="native_mtp_sparse_bootstrap_already_claimed"):
        from mlx_lm.generate import NativeMTPAdmission, NativeMTPRowSpec

        NativeMTPAdmission.create_from_sparse_bootstraps(
            model, (NativeMTPRowSpec(8, (0, 2, 4), 2),), (bootstrap,)
        )


def test_pinned_bootstrap_forgery_is_rejected_before_claim_and_original_survives():
    from mlx_lm.generate import NativeMTPAdmission, NativeMTPRowSpec
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    model = _PinnedSparseModel()
    request = _request(1)
    request.prompt_token_ids = [0, 1, 2, 3, 4]
    bootstrap = _pinned_bootstrap(model, request.prompt_token_ids, (0, 2, 4))
    forged = replace(bootstrap, target_cache=[object()])
    with pytest.raises(RuntimeError, match="native_mtp_sparse_receipt_provenance_mismatch"):
        NativeMTPContinuousBatchAdapter.create(
            model, [request], sparse_bootstraps=(forged,)
        )
    admission = NativeMTPAdmission.create_from_sparse_bootstraps(
        model, (NativeMTPRowSpec(1, (0, 2, 4), 2),), (bootstrap,)
    )
    assert admission.rows[0].prompt == (0, 2, 4)


def test_pinned_bootstrap_close_abandons_authority_before_downstream_claim():
    from mlx_lm.generate import NativeMTPAdmission, NativeMTPRowSpec

    model = _PinnedSparseModel()
    bootstrap = _pinned_bootstrap(model, (0, 1, 2, 3, 4), (0, 2, 4))
    bootstrap.close()
    with pytest.raises(RuntimeError, match="native_mtp_sparse_bootstrap_already_claimed"):
        NativeMTPAdmission.create_from_sparse_bootstraps(
            model, (NativeMTPRowSpec(1, (0, 2, 4), 2),), (bootstrap,)
        )


def test_partial_sparse_admission_failure_consumes_every_bootstrap_authority():
    from mlx_lm.generate import NativeMTPAdmission, NativeMTPRowSpec

    model = _PinnedSparseModel()
    first = _pinned_bootstrap(model, (0, 1, 2, 3, 4), (0, 2, 4))
    second = _pinned_bootstrap(model, (5, 6, 7, 8, 9), (0, 2, 4))
    # The second bootstrap presents a receipt owned by the first target cache;
    # the first claim has already been made when this one fails validation.
    malformed = replace(second, receipts=(first.receipts[0],))
    with pytest.raises(RuntimeError):
        NativeMTPAdmission.create_from_sparse_bootstraps(
            model,
            (
                NativeMTPRowSpec(1, (0, 2, 4), 2),
                NativeMTPRowSpec(2, (5, 7, 9), 2),
            ),
            (first, malformed),
        )
    with pytest.raises(RuntimeError, match="native_mtp_sparse_bootstrap_already_claimed"):
        NativeMTPAdmission.create_from_sparse_bootstraps(
            model, (NativeMTPRowSpec(3, first.selected_token_ids, 2),), (first,)
        )
    # The untouched original second authority was never passed or reserved.
    admission = NativeMTPAdmission.create_from_sparse_bootstraps(
        model, (NativeMTPRowSpec(4, second.selected_token_ids, 2),), (second,)
    )
    assert admission.rows[0].uid == 4


def test_b_gt_1_mixed_resolution_filters_terminal_and_tracks_uid_local_telemetry():
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    adapter = NativeMTPContinuousBatchAdapter.create(
        _Model(),
        [_request(1, seed=3), _request(2, seed=5)],
        lifecycle=_lifecycle(),
        prefill_step_size=4,
    )
    adapter.step()
    adapter.step()
    adapter.step()
    decision_emissions = adapter.step()
    assert [(item.uid, item.from_draft) for item in decision_emissions] == [
        (1, True),
        (2, False),
    ]
    assert all(item.logprobs is not None for item in decision_emissions)
    assert [(item.uid, item.finish_reason) for item in adapter.step()] == [
        (1, "length")
    ]
    adapter.step()
    assert adapter.active_uids == (2,)
    assert adapter.telemetry_for(1) == {
        "target_tokens": 5,
        "draft_tokens": 1,
        "accepted_tokens": 1,
    }
    assert adapter.telemetry_for(2) == {
        "target_tokens": 4,
        "draft_tokens": 2,
        "accepted_tokens": 0,
    }


def test_cancellation_before_and_after_initial_emission_is_public_and_cohort_scoped():
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    before = NativeMTPContinuousBatchAdapter.create(
        _Model(), [_request(1)], lifecycle=_lifecycle()
    )
    assert before.cancel() == (1,)
    assert before.closed
    after = NativeMTPContinuousBatchAdapter.create(
        _Model(), [_request(1)], lifecycle=_lifecycle(), prefill_step_size=4
    )
    after.step()
    assert after.cancel() == (1,)
    assert after.closed


def test_fresh_cache_failure_is_deferred_until_prefill_and_remains_cancellable():
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    adapter = NativeMTPContinuousBatchAdapter.create(
        _FreshCacheFailureModel(),
        [_request(1)],
        lifecycle=_lifecycle(),
        prefill_step_size=4,
    )
    with pytest.raises(RuntimeError, match="^fresh_cache_failed$"):
        adapter.step()
    assert adapter.cancel() == (1,)
    assert adapter.closed


@pytest.mark.parametrize(
    ("name", "value", "reason"),
    [
        ("prefix_reused", True, "native_mtp_prefix_reuse_unsupported"),
        ("chunked_prefill", True, "native_mtp_chunked_prefill_unsupported"),
        ("quantized_kv", True, "native_mtp_quantized_cache_unsupported"),
        ("has_media", True, "native_mtp_media_unsupported"),
        ("external_draft", True, "native_mtp_external_draft_unsupported"),
        ("logits_processors", [object()], "native_mtp_logits_processors_unsupported"),
    ],
)
def test_adapter_fails_closed_for_every_unadmitted_composition(name, value, reason):
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    request = _request(1)
    setattr(request, name, value)
    with pytest.raises(RuntimeError, match=f"^{reason}$"):
        NativeMTPContinuousBatchAdapter.create(
            _Model(), [request], lifecycle=_lifecycle()
        )


def test_adapter_uses_no_private_lifecycle_or_legacy_batch_generator_surface():
    source = (
        Path(__file__).parents[1] / "vllm_mlx" / "native_mtp_cb_adapter.py"
    ).read_text()
    assert "_install_mtp" not in source
    assert "._draft" not in source
    assert "._replacement" not in source
    assert "._verify" not in source
    assert "from mlx_lm.generate import BatchGenerator" not in source
    assert "_install_mtp(" not in source


def test_pinned_public_lifecycle_import_surface_exposes_uniform_emissions():
    module = importlib.import_module("mlx_lm.generate")
    assert inspect.ismodule(module)
    assert hasattr(module, "NativeMTPRowSpec")
    assert hasattr(module, "NativeMTPAdmission")
    assert hasattr(module, "NativeMTPBatchGenerator")
    assert "Tuple" in str(
        inspect.signature(module.NativeMTPDecisionEpoch.accept).return_annotation
    )
    assert "Tuple" in str(
        inspect.signature(module.NativeMTPDecisionEpoch.reject).return_annotation
    )


def test_scheduler_native_admission_failure_is_one_terminal_error_per_row_and_queue_advances(
    monkeypatch,
):
    import vllm_mlx.native_mtp_cb_adapter as adapter_mod

    first, second, next_zero = (
        _real_request(1),
        _real_request(2),
        _real_request(3, max_tokens=0),
    )
    first.batch_uid, second.batch_uid = 41, 42
    scheduler = _bare_scheduler(first, second, next_zero)
    _install_minimal_step_cleanup(monkeypatch, scheduler)
    calls = []

    monkeypatch.setattr(
        adapter_mod.NativeMTPContinuousBatchAdapter,
        "create",
        classmethod(
            lambda cls, *args, **kwargs: (
                calls.append((args, kwargs)),
                (_ for _ in ()).throw(RuntimeError("create_failed")),
            )[1]
        ),
    )
    first_step = scheduler.step()
    assert [
        (output.request_id, output.finish_reason, output.native_mtp_error_reason)
        for output in first_step.outputs
    ] == [
        (first.request_id, "error", "create_failed"),
        (second.request_id, "error", "create_failed"),
    ]
    assert tuple(scheduler.waiting) == (next_zero,)
    assert (first.batch_uid, second.batch_uid, scheduler._next_native_mtp_uid) == (
        41,
        42,
        7,
    )
    assert (
        scheduler.running
        == scheduler.request_id_to_uid
        == scheduler.uid_to_request_id
        == {}
    )
    second_step = scheduler.step()
    assert [
        (output.request_id, output.finish_reason) for output in second_step.outputs
    ] == [(next_zero.request_id, "length")]
    assert len(calls) == 1


def test_scheduler_zero_budget_head_finishes_same_step_before_adapter_and_keeps_next_row(
    monkeypatch,
):
    import vllm_mlx.native_mtp_cb_adapter as adapter_mod

    zero, following = _real_request(1, max_tokens=0), _real_request(2)
    scheduler = _bare_scheduler(zero, following)
    _install_minimal_step_cleanup(monkeypatch, scheduler)
    monkeypatch.setattr(
        adapter_mod.NativeMTPContinuousBatchAdapter,
        "create",
        classmethod(lambda cls, *args, **kwargs: pytest.fail("adapter must not start")),
    )
    output = scheduler.step()
    assert [
        (item.request_id, item.finish_reason, item.completion_tokens)
        for item in output.outputs
    ] == [(zero.request_id, "length", 0)]
    assert tuple(scheduler.waiting) == (following,)
    assert scheduler._next_native_mtp_uid == 7


def test_native_emission_logprobs_survive_scheduler_collector_and_batched_conversion(
    monkeypatch,
):
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.output_collector import RequestOutputCollector
    from vllm_mlx.request import RequestOutput

    request = _real_request(1)
    scheduler = _bare_scheduler(request)
    scheduler.running[request.request_id] = request
    scheduler.uid_to_request_id[1] = request.request_id
    scheduler.native_mtp_adapter = SimpleNamespace(
        telemetry_for=lambda uid: {"draft_tokens": 1, "accepted_tokens": 1}
    )
    logprobs = object()
    detok = SimpleNamespace(
        add_token=lambda token: None, last_segment="x", finalize=lambda: None, text="x"
    )
    monkeypatch.setattr(scheduler, "_get_detokenizer", lambda request_id: detok)
    monkeypatch.setattr(scheduler, "_cleanup_detokenizer", lambda request_id: None)
    outputs, _ = scheduler._process_native_mtp_emissions(
        [SimpleNamespace(uid=1, token=5, logprobs=logprobs, finish_reason=None)]
    )
    collector = RequestOutputCollector()
    collector.put(outputs[0])
    assert collector.get_nowait().logprobs is logprobs

    class _Engine:
        async def generate(self, **kwargs):
            return RequestOutput(
                request_id="engine-request",
                output_text="ok",
                logprobs=logprobs,
                native_mtp_error_reason="native_error",
            )

    engine = object.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._engine = _Engine()
    native = asyncio.run(engine.generate("prompt", _native_mtp_request_config=object()))
    ordinary = asyncio.run(engine.generate("prompt"))
    assert native.logprobs is logprobs
    assert native.mtp_bypass_reason == "native_error"
    assert ordinary.logprobs is None
    assert ordinary.mtp_bypass_reason is None


def test_scheduler_native_abort_is_cohort_scoped_and_cleans_all_terminal_mappings(
    monkeypatch,
):
    first, second = _real_request(1), _real_request(2)
    scheduler = _bare_scheduler(first, second)
    scheduler.running = {first.request_id: first, second.request_id: second}
    scheduler.request_id_to_uid = {first.request_id: 1, second.request_id: 2}
    scheduler.uid_to_request_id = {1: first.request_id, 2: second.request_id}
    cancelled = []
    scheduler.native_mtp_adapter = SimpleNamespace(
        cancel=lambda: cancelled.append(True)
    )
    _install_minimal_step_cleanup(monkeypatch, scheduler)
    assert scheduler._do_abort_request(first.request_id)
    output = scheduler.step()
    assert cancelled == [True]
    assert {
        (item.request_id, item.finish_reason, item.native_mtp_error_reason)
        for item in output.outputs
    } == {
        (first.request_id, "abort", None),
        (second.request_id, "abort", None),
    }
    assert (
        scheduler.running
        == scheduler.request_id_to_uid
        == scheduler.uid_to_request_id
        == {}
    )


def test_two_pending_native_cohort_aborts_cancel_once_and_drain_exactly_once(
    monkeypatch,
):
    first, second = _real_request(1), _real_request(2)
    scheduler = _bare_scheduler(first, second)
    scheduler.waiting.clear()
    scheduler.running = {first.request_id: first, second.request_id: second}
    scheduler.request_id_to_uid = {first.request_id: 1, second.request_id: 2}
    scheduler.uid_to_request_id = {1: first.request_id, 2: second.request_id}
    scheduler._pending_abort_ids = {first.request_id, second.request_id}
    cancelled = []
    scheduler.native_mtp_adapter = SimpleNamespace(
        cancel=lambda: cancelled.append(True)
    )
    _install_minimal_step_cleanup(monkeypatch, scheduler)

    first_step = scheduler.step()
    assert cancelled == [True]
    assert {
        (item.request_id, item.finish_reason, item.native_mtp_error_reason)
        for item in first_step.outputs
    } == {
        (first.request_id, "abort", None),
        (second.request_id, "abort", None),
    }
    assert len(first_step.outputs) == 2
    assert (
        scheduler.running
        == scheduler.request_id_to_uid
        == scheduler.uid_to_request_id
        == {}
    )

    second_step = scheduler.step()
    assert second_step.outputs == []
    assert cancelled == [True]


def test_scheduler_native_prefill_failure_is_one_terminal_error_without_retry_hang(
    monkeypatch,
):
    request = _real_request(1)
    scheduler = _bare_scheduler(request)
    scheduler.waiting.clear()
    scheduler.running = {request.request_id: request}
    scheduler.request_id_to_uid = {request.request_id: 1}
    scheduler.uid_to_request_id = {1: request.request_id}
    scheduler.native_mtp_adapter = SimpleNamespace(
        step=lambda: (_ for _ in ()).throw(RuntimeError("fresh_cache_failed")),
        cancel=lambda: None,
    )
    _install_minimal_step_cleanup(monkeypatch, scheduler)
    first = scheduler.step()
    assert [
        (item.request_id, item.finish_reason, item.native_mtp_error_reason)
        for item in first.outputs
    ] == [(request.request_id, "error", "fresh_cache_failed")]
    assert scheduler.native_mtp_adapter is None
    assert (
        scheduler.running
        == scheduler.request_id_to_uid
        == scheduler.uid_to_request_id
        == {}
    )
    second = scheduler.step()
    assert second.outputs == []


def test_scheduler_preserves_primary_mtp_failure_when_public_epoch_is_already_poisoned(
    monkeypatch,
):
    """Terminal cleanup must not call a consumed public epoch a second time."""
    from vllm_mlx.native_mtp_cb_adapter import NativeMTPContinuousBatchAdapter

    request = _real_request(1)
    request.batch_uid = 1
    scheduler = _bare_scheduler(request)
    scheduler.waiting.clear()
    scheduler.running = {request.request_id: request}
    scheduler.request_id_to_uid = {request.request_id: request.batch_uid}
    scheduler.uid_to_request_id = {request.batch_uid: request.request_id}
    _install_minimal_step_cleanup(monkeypatch, scheduler)
    detokenizer = SimpleNamespace(
        last_segment="x",
        text="x",
        add_token=lambda _token_id: None,
        finalize=lambda: None,
    )
    monkeypatch.setattr(scheduler, "_get_detokenizer", lambda _request_id: detokenizer)
    adapter = NativeMTPContinuousBatchAdapter.create(
        scheduler.model,
        [request],
        lifecycle=_poisoning_lifecycle(),
        prefill_step_size=4,
    )
    scheduler.native_mtp_adapter = adapter

    first = scheduler.step()
    assert first.outputs[0].finish_reason is None
    generator = adapter._generator
    assert generator.closed is False
    assert generator.cohort.poisoned is False

    error = scheduler.step()
    assert [
        (item.request_id, item.finish_reason, item.native_mtp_error_reason)
        for item in error.outputs
    ] == [(request.request_id, "error", "forced_mtp_failure")]
    assert adapter.closed is True
    assert generator.closed is True
    assert generator.cohort.poisoned is True
    assert scheduler.native_mtp_adapter is None
    assert (
        scheduler.running
        == scheduler.request_id_to_uid
        == scheduler.uid_to_request_id
        == {}
    )

    retry = scheduler.step()
    assert retry.outputs == []


def test_inactive_scheduler_keeps_ordinary_constructor_and_insert_call_shapes():
    source = (Path(__file__).parents[1] / "vllm_mlx" / "scheduler.py").read_text()
    assert "BatchGenerator(\n            model=self.model," in source
    assert (
        "self.batch_generator.insert(\n                    [tokens_to_process],"
        in source
    )
    assert (
        "native_mtp_config"
        not in source.split("def _create_batch_generator", 1)[1].split(
            "def _make_prompt_cache_save_callback", 1
        )[0]
    )
