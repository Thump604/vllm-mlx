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

    @classmethod
    def create(
        cls,
        model: Any,
        requests: Iterable[Any],
        *,
        lifecycle: Any | None = None,
        prefill_step_size: int = 512,
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
        make_cache = getattr(model, "make_mtp_request_cache", None)
        if not callable(make_cache):
            raise RuntimeError("native_mtp_request_cache_factory_missing")

        rows = []
        for request in requests:
            cls._validate_request(request)
            sampling = request.native_mtp_config.sampling
            rows.append(
                lifecycle.NativeMTPRowSpec(
                    uid=request.batch_uid,
                    prompt=tuple(request.prompt_token_ids),
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
            return lifecycle.NativeMTPBatchGenerator(admission)

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

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active_uids(self) -> tuple[int, ...]:
        return () if self._closed else self._active_uids

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
            self._generator = self._start()
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
        if self._epoch is not None:
            self._epoch.cancel()
        self._closed = True
        return affected
