# SPDX-License-Identifier: Apache-2.0
"""Request-local contract for native Qwen MTP decoding.

This module deliberately contains no model or MLX imports.  Public request
validation and route admission can therefore finish before generation mutates
model, cache, sampler, or processor state.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class NativeMTPRequestError(ValueError):
    """An explicit native-MTP request cannot be honored safely."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def native_mtp_consumer_supported(consumer: Callable[..., Any]) -> bool:
    """Whether a generation callable exposes the exact native-MTP seam."""

    try:
        parameters = inspect.signature(consumer).parameters
    except (TypeError, ValueError):
        return False
    return "mtp" in parameters and "mtp_sampling_config" in parameters


def resolve_native_mtp_consumer() -> Callable[..., Any] | None:
    """Return the installed mlx-lm consumer only when its contract is exact."""

    try:
        from mlx_lm import stream_generate
        from mlx_lm.generate import NativeMTPSamplingConfig  # noqa: F401
    except (ImportError, AttributeError):
        return None
    return stream_generate if native_mtp_consumer_supported(stream_generate) else None


@dataclass(frozen=True, slots=True)
class NativeMTPSampling:
    """Immutable sampling inputs consumed only by a selected native-MTP path."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    seed: int | None


@dataclass(frozen=True, slots=True)
class NativeMTPRequestConfig:
    """The complete request-effective native-MTP decode configuration."""

    sampling: NativeMTPSampling
    num_draft_tokens: int

    def __post_init__(self) -> None:
        if self.num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be positive")

    def mlx_lm_call_kwargs(
        self, consumer: Callable[..., Any] | None = None
    ) -> dict[str, Any]:
        """Build the exact request-local kwargs required by current mlx-lm."""

        consumer = consumer or resolve_native_mtp_consumer()
        if consumer is None or not native_mtp_consumer_supported(consumer):
            raise NativeMTPRequestError("native_mtp_consumer_contract_missing")
        try:
            from mlx_lm.generate import NativeMTPSamplingConfig
        except ImportError as exc:
            raise NativeMTPRequestError(
                "native_mtp_consumer_contract_missing"
            ) from exc
        return {
            "mtp": True,
            "mtp_sampling_config": NativeMTPSamplingConfig(
                temperature=self.sampling.temperature,
                top_p=self.sampling.top_p,
                top_k=self.sampling.top_k,
                min_p=self.sampling.min_p,
                seed=self.sampling.seed,
            ),
        }


@dataclass(frozen=True, slots=True)
class NativeMTPServerState:
    """Read-only engine admission state used by the public request layer."""

    server_default: bool
    capable: bool
    num_draft_tokens: int = 1
    supports_penalty_processors: bool = True
    incompatibility: str | None = None


@dataclass(frozen=True, slots=True)
class NativeMTPDecision:
    """Result of resolving public intent against server admission."""

    requested: bool | None
    selected: bool
    config: NativeMTPRequestConfig | None = None
    bypass_reason: str | None = None

    def __post_init__(self) -> None:
        if self.selected != (self.config is not None):
            raise ValueError("selected native MTP requires exactly one request config")
        if self.selected and self.bypass_reason is not None:
            raise ValueError("selected native MTP cannot have a bypass reason")


def resolve_native_mtp_request(
    *,
    requested: bool | None,
    sampling: NativeMTPSampling,
    server_default: bool,
    capable: bool,
    num_draft_tokens: int,
    incompatibility: str | None = None,
) -> NativeMTPDecision:
    """Resolve native MTP without mutating engine or model state.

    ``None`` preserves the existing server default.  Explicit ``False`` is an
    unreported opt-out.  Explicit ``True`` never silently downgrades: a missing
    capability or any route incompatibility raises with a stable reason.
    """

    if requested is False:
        return NativeMTPDecision(requested=False, selected=False)

    wants_mtp = server_default if requested is None else True
    if not wants_mtp:
        return NativeMTPDecision(requested=None, selected=False)

    reason = incompatibility
    if reason is None and not capable:
        reason = "native_mtp_unsupported"
    if reason is not None:
        if requested is True:
            raise NativeMTPRequestError(reason)
        return NativeMTPDecision(
            requested=None,
            selected=False,
            bypass_reason=reason,
        )

    return NativeMTPDecision(
        requested=requested,
        selected=True,
        config=NativeMTPRequestConfig(
            sampling=sampling,
            num_draft_tokens=num_draft_tokens,
        ),
    )
