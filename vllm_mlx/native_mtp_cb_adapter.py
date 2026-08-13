# SPDX-License-Identifier: Apache-2.0
"""Public-lifecycle adapter for standard-text native-MTP continuous batching.

This module is deliberately independent from mlx-lm's ordinary
``BatchGenerator``.  It owns only the exported native-MTP epoch handles and
request-local cache admission; the Scheduler retains queueing and output I/O.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(slots=True)
class _Telemetry:
    target_tokens: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "target_tokens": self.target_tokens,
            "draft_tokens": self.draft_tokens,
            "accepted_tokens": self.accepted_tokens,
        }


class NativeMTPContinuousBatchAdapter:
    """Drive the public mlx-lm native-MTP lifecycle for one fresh cohort.

    A cohort is intentionally closed as a unit on cancellation or failure:
    the public lifecycle exposes no safe per-row cancellation handle.  The
    scheduler can subsequently admit surviving requests as a new fresh cohort.
    """

    def __init__(self, *, start: Any, requests: Iterable[Any], prefill_step_size: int):
        self._start = start
        self._generator = None
        self._sparse_start = False
        self._requests = {request.batch_uid: request for request in requests}
        self._telemetry = {uid: _Telemetry() for uid in self._requests}
        self._prefill_step_size = prefill_step_size
        self._epoch = None
        self._phase = "new"
        self._active_uids = tuple(self._requests)
        self._decision = None
        self._mixed_accepted = ()
        self._mixed_rejected = ()
        self._closed = False
        self._cleanup_error_reason = None

    @classmethod
    def create(
        cls,
        model: Any,
        requests: Iterable[Any],
        *,
        lifecycle: Any | None = None,
        prefill_step_size: int = 512,
        sparse_bootstraps: Iterable[Any] | None = None,
        sparse_adopted: Any | None = None,
    ) -> "NativeMTPContinuousBatchAdapter":
        requests = tuple(requests)
        if not requests:
            raise RuntimeError("native_mtp_batch_rows_and_caches_required")
        if lifecycle is None:
            # ``mlx_lm.generate`` is also re-exported as a callable package
            # attribute.  Import the module by its exact dotted name so the
            # public lifecycle classes cannot be shadowed by that function.
            lifecycle = importlib.import_module("mlx_lm.generate")

        capability = getattr(model, "mtp_capability", None)
        if capability is None or not getattr(capability, "supported", False):
            raise RuntimeError(
                "native_mtp_model_capability_missing"
                if capability is None
                else getattr(capability, "reason", "native_mtp_unsupported")
            )
        bootstraps = None if sparse_bootstraps is None else tuple(sparse_bootstraps)
        make_cache = getattr(model, "make_mtp_request_cache", None)
        if bootstraps is None and not callable(make_cache):
            raise RuntimeError("native_mtp_request_cache_factory_missing")

        if bootstraps is not None and len(bootstraps) != len(requests):
            raise RuntimeError("native_mtp_sparse_bootstrap_count_mismatch")
        if bootstraps is not None and len({id(item) for item in bootstraps}) != len(
            bootstraps
        ):
            raise RuntimeError("native_mtp_sparse_bootstrap_reused")
        if sparse_adopted is not None and not callable(sparse_adopted):
            raise TypeError("native_mtp_sparse_adopted_callback_invalid")

        rows = []
        for index, request in enumerate(requests):
            cls._validate_request(request)
            prompt = tuple(request.prompt_token_ids)
            if bootstraps is not None:
                prompt = cls._validate_sparse_bootstrap(
                    model, request, bootstraps[index]
                )
            sampling = request.native_mtp_config.sampling
            rows.append(
                lifecycle.NativeMTPRowSpec(
                    uid=request.batch_uid,
                    prompt=prompt,
                    max_tokens=request.sampling_params.max_tokens,
                    seed=sampling.seed,
                    eos_token_ids=frozenset(request.native_mtp_eos_token_ids),
                    sampling_config=lifecycle.NativeMTPSamplingConfig(
                        temperature=sampling.temperature,
                        top_p=sampling.top_p,
                        top_k=sampling.top_k,
                        min_p=sampling.min_p,
                        seed=sampling.seed,
                    ),
                )
            )

        def _start():
            if bootstraps is not None:
                admission = lifecycle.NativeMTPAdmission.create_from_sparse_bootstraps(
                    model, rows, bootstraps
                )
                if sparse_adopted is not None:
                    sparse_adopted()
                return lifecycle.NativeMTPBatchGenerator(admission), True
            # Delay fresh-cache ownership until the first prefill boundary so a
            # queued request can be cancelled before any model/cache mutation.
            request_caches = [make_cache(prompt_cache=None) for _ in rows]
            admission = lifecycle.NativeMTPAdmission.create(
                model,
                rows,
                request_caches,
                prefix_cache=None,
                media=None,
                external_draft=None,
                sparse_bootstrap=None,
                logits_processors=None,
                kv_bits=None,
                max_kv_size=None,
            )
            return lifecycle.NativeMTPBatchGenerator(admission), False

        return cls(
            start=_start,
            requests=requests,
            prefill_step_size=prefill_step_size,
        )

    @staticmethod
    def _validate_request(request: Any) -> None:
        config = getattr(request, "native_mtp_config", None)
        if config is None:
            raise RuntimeError("native_mtp_request_config_missing")
        if not getattr(request, "prompt_token_ids", None):
            raise RuntimeError("native_mtp_prompt_required")
        if getattr(request, "prefix_reused", False):
            raise RuntimeError("native_mtp_prefix_reuse_unsupported")
        if getattr(request, "chunked_prefill", False):
            raise RuntimeError("native_mtp_chunked_prefill_unsupported")
        if getattr(request, "quantized_kv", False):
            raise RuntimeError("native_mtp_quantized_cache_unsupported")
        if getattr(request, "has_media", False):
            raise RuntimeError("native_mtp_media_unsupported")
        if getattr(request, "external_draft", False):
            raise RuntimeError("native_mtp_external_draft_unsupported")
        if getattr(request, "logits_processors", None):
            raise RuntimeError("native_mtp_logits_processors_unsupported")
        if (
            getattr(request.sampling_params, "presence_penalty", 0.0) != 0.0
            or getattr(request.sampling_params, "repetition_penalty", 1.0) != 1.0
        ):
            raise RuntimeError("native_mtp_penalty_processors_unsupported")

    @staticmethod
    def _validate_sparse_bootstrap(
        model: Any, request: Any, bootstrap: Any
    ) -> tuple[int, ...]:
        """Bind one bootstrap to this exact request/model before claim.

        mlx-lm repeats these provenance checks while atomically claiming the
        receipts.  The adapter checks the scheduler-side association first so
        a misordered cohort cannot consume another request's authority.
        """
        prompt = tuple(request.prompt_token_ids)
        try:
            positions = tuple(bootstrap.selected_logical_positions)
            selected = tuple(bootstrap.selected_token_ids)
            successors = tuple(bootstrap.immediate_successor_token_ids)
            receipts = tuple(bootstrap.receipts)
            target_cache = bootstrap.target_cache
            cursor = bootstrap.next_logical_position
        except (AttributeError, TypeError) as exc:
            raise RuntimeError("native_mtp_sparse_bootstrap_invalid") from exc
        if not positions or len(selected) != len(positions):
            raise RuntimeError("native_mtp_sparse_bootstrap_invalid")
        if not isinstance(target_cache, list) or not target_cache:
            raise RuntimeError("native_mtp_sparse_target_cache_invalid")
        if (
            positions[-1] != len(prompt) - 1
            or cursor != len(prompt)
            or len(successors) != len(positions) - 1
        ):
            raise RuntimeError("native_mtp_sparse_request_cursor_mismatch")
        if (
            any(
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
                or position >= len(prompt)
                for position in positions
            )
            or tuple(sorted(set(positions))) != positions
        ):
            raise RuntimeError("native_mtp_sparse_positions_invalid")
        if selected != tuple(prompt[position] for position in positions):
            raise RuntimeError("native_mtp_sparse_request_tokens_mismatch")
        if successors != tuple(prompt[position + 1] for position in positions[:-1]):
            raise RuntimeError("native_mtp_sparse_request_successors_mismatch")
        if not receipts:
            raise RuntimeError("native_mtp_sparse_receipts_missing")
        cache_ids = tuple(id(entry) for entry in target_cache)
        for receipt in receipts:
            if (
                getattr(receipt, "model_id", None) != id(model)
                or getattr(receipt, "cache_container_id", None) != id(target_cache)
                or getattr(receipt, "cache_entry_ids", None) != cache_ids
            ):
                raise RuntimeError("native_mtp_sparse_receipt_provenance_mismatch")
        return selected

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active_uids(self) -> tuple[int, ...]:
        return () if self._closed else self._active_uids

    @property
    def cleanup_error_reason(self) -> str | None:
        """Return a non-primary terminal cleanup failure, if one occurred."""
        return self._cleanup_error_reason

    def telemetry_for(self, uid: int) -> dict[str, int]:
        return self._telemetry[uid].snapshot()

    def _add_target(self, uids: Iterable[int], count: int) -> None:
        for uid in uids:
            self._telemetry[uid].target_tokens += count

    def _add_drafts(self, uids: Iterable[int]) -> None:
        for uid in uids:
            self._telemetry[uid].draft_tokens += 1

    def _record_emissions(self, emissions: Iterable[Any]) -> tuple[Any, ...]:
        emissions = tuple(emissions)
        for emission in emissions:
            if emission.from_draft:
                self._telemetry[emission.uid].accepted_tokens += 1
            if emission.finish_reason is not None:
                self._active_uids = tuple(
                    uid for uid in self._active_uids if uid != emission.uid
                )
        return emissions

    def _set_epoch(self, epoch: Any) -> None:
        self._epoch = epoch
        if hasattr(epoch, "active_uids"):
            self._active_uids = tuple(epoch.active_uids)

    def step(self) -> tuple[Any, ...]:
        """Advance exactly one public lifecycle boundary and return emissions."""
        if self._closed:
            return ()
        if self._phase == "new":
            self._generator, self._sparse_start = self._start()
            if self._sparse_start:
                emissions, epoch = self._generator.start_sparse()
            else:
                emissions, epoch = self._generator.prefill(
                    prefill_step_size=self._prefill_step_size
                )
            self._set_epoch(epoch)
            self._add_target((item.uid for item in emissions), 1)
            self._phase = "initial"
            return self._record_emissions(emissions)
        if self._phase == "initial":
            active = self._epoch.active_uids
            self._set_epoch(self._epoch.resume())
            self._add_drafts(active)
            self._phase = "ready"
            return ()
        if self._phase == "ready":
            active = self._epoch.active_uids
            self._set_epoch(self._epoch.decide())
            self._decision = self._epoch
            self._add_target(active, 2)
            self._phase = "decision"
            return ()
        if self._phase == "decision":
            accepted = tuple(self._decision.accepted_uids)
            rejected = tuple(self._decision.rejected_uids)
            if accepted and rejected:
                emissions, epoch = self._epoch.resolve()
                self._set_epoch(epoch)
                # Accepted rows rerun the verified target call after rollback.
                self._add_target(accepted, 2)
                self._mixed_accepted, self._mixed_rejected = accepted, rejected
                self._phase = "mixed"
                return self._record_emissions(emissions)
            if accepted:
                emissions, epoch = self._epoch.accept()
                self._set_epoch(epoch)
                self._phase = "accepted"
                return self._record_emissions(emissions)
            emissions, epoch = self._epoch.reject()
            self._set_epoch(epoch)
            # Rejection restores then replays the target head in this public
            # boundary before the replacement token is observable.
            self._add_target(rejected, 1)
            self._phase = "rejected"
            return self._record_emissions(emissions)
        if self._phase == "accepted":
            emissions, epoch = self._epoch.bonus()
            self._set_epoch(epoch)
            self._phase = "bonus"
            return self._record_emissions(emissions)
        if self._phase == "bonus":
            active = self._epoch.active_uids
            self._set_epoch(self._epoch.catch_up())
            self._add_drafts(active)
            self._phase = "ready"
            return ()
        if self._phase == "rejected":
            active = self._epoch.active_uids
            self._set_epoch(self._epoch.redraft())
            self._add_drafts(active)
            self._phase = "ready"
            return ()
        if self._phase == "mixed":
            emissions, epoch = self._epoch.resume_after_resolution()
            self._set_epoch(epoch)
            # Rejected rows replay their head then produce a new draft.
            self._add_target(self._mixed_rejected, 1)
            self._add_drafts(self._mixed_rejected)
            self._phase = "mixed_bonus"
            return self._record_emissions(emissions)
        if self._phase == "mixed_bonus":
            before = set(self._mixed_accepted)
            self._set_epoch(self._epoch.resume_after_bonus())
            self._add_drafts(before.intersection(self._epoch.active_uids))
            self._phase = "ready"
            return ()
        raise RuntimeError("native_mtp_batch_lifecycle_phase_invalid")

    def cancel(self) -> tuple[int, ...]:
        """Cancel the entire public cohort and return affected UIDs."""
        if self._closed:
            return ()
        affected = self.active_uids
        # A lifecycle operation can consume its epoch and poison/close the
        # generator before its exception reaches the scheduler.  That stale
        # handle is not an actionable cleanup owner: calling ``cancel`` on it
        # raises ``native_mtp_epoch_moved`` and used to hide the model error.
        # Mark this adapter closed first so repeated terminal cleanup is also
        # harmless.  A live epoch is still cancelled normally.
        self._closed = True
        if self._epoch is None or getattr(self._generator, "closed", False):
            return affected
        try:
            self._epoch.cancel()
        except Exception as error:
            # Cancellation has no replacement primary failure of its own.
            # Keep it observable to scheduler-owned cleanup without allowing
            # it to replace a model/admission failure already in flight.
            self._cleanup_error_reason = str(error) or type(error).__name__
        return affected
