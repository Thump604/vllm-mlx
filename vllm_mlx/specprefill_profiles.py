# SPDX-License-Identifier: Apache-2.0
"""Calibrated, fail-closed eligibility profiles for SpecPrefill.

This registry is intentionally pure Python.  It does not load a model or
choose an architecture dynamically: the caller must supply the exact artifact,
adapter, engine, and speculation-composition cell.  A production request is
eligible only when that exact cell has an explicit calibrated profile marked
certified.  The seed registry contains no certified artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

_CONTEXT_QUALIFICATION_LADDER = (
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    96 * 1024,
    127 * 1024,
    192 * 1024,
    255 * 1024,
)
_CB_CONCURRENCY_LADDER = (1, 2, 4, 8)


class SpecPrefillEngine(str, Enum):
    """Execution engines whose calibration results cannot be shared."""

    SIMPLE = "simple"
    CONTINUOUS_BATCHING = "continuous_batching"


class SpecPrefillCell(str, Enum):
    """Independently qualified sparse-only and MTP-composed feature cells."""

    SPARSE_ONLY = "sparse_only"
    COMBINED_MTP = "combined_mtp"


class SpecPrefillProfileTier(str, Enum):
    """Production profiles and explicitly non-production diagnostic profiles."""

    PRODUCTION = "production"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class SpecPrefillTuning:
    """Selector controls calibrated for one exact feature cell."""

    keep_pct: float
    backbone_pct: float
    halo_chunks: int
    anchor_chunks: int
    chunk_size: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.keep_pct) or not 0.0 < self.keep_pct <= 1.0:
            raise ValueError("keep_pct must be finite and in (0, 1]")
        if not math.isfinite(self.backbone_pct) or not 0.0 <= self.backbone_pct <= 1.0:
            raise ValueError("backbone_pct must be finite and in [0, 1]")
        if self.halo_chunks < 0 or self.anchor_chunks < 0:
            raise ValueError("halo_chunks and anchor_chunks must be non-negative")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")


@dataclass(frozen=True)
class SpecPrefillProfileKey:
    """Immutable model/artifact/route identity for one eligibility cell."""

    target_artifact_id: str
    target_artifact_hash: str
    tokenizer_artifact_hash: str
    scorer_artifact_id: str
    scorer_artifact_hash: str
    adapter_id: str
    engine: SpecPrefillEngine
    cell: SpecPrefillCell

    def __post_init__(self) -> None:
        for name in (
            "target_artifact_id",
            "scorer_artifact_id",
            "adapter_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "target_artifact_hash",
            "tokenizer_artifact_hash",
            "scorer_artifact_hash",
        ):
            _validate_sha256(name, getattr(self, name))
        if not isinstance(self.engine, SpecPrefillEngine):
            raise ValueError("engine must be SpecPrefillEngine")
        if not isinstance(self.cell, SpecPrefillCell):
            raise ValueError("cell must be SpecPrefillCell")


@dataclass(frozen=True)
class SpecPrefillCalibration:
    """Measured eligibility bounds and value gates for one profile cell."""

    selector_version: str
    tuning: SpecPrefillTuning
    crossover_tokens: int
    max_context_tokens: int
    residency_limit_bytes: int
    min_ttft_improvement_pct: float
    max_total_latency_regression_pct: float
    max_decode_throughput_regression_pct: float
    required_context_tokens: tuple[int, ...]
    required_concurrency_levels: tuple[int, ...]
    max_p95_inter_token_latency_regression_pct: float
    min_prefill_heavy_throughput_improvement_pct: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.selector_version, str)
            or not self.selector_version.strip()
        ):
            raise ValueError(
                "selector_version must be a non-empty immutable identifier"
            )
        if not isinstance(self.tuning, SpecPrefillTuning):
            raise ValueError("tuning must be SpecPrefillTuning")
        if self.crossover_tokens <= 0:
            raise ValueError("crossover_tokens must be positive")
        if self.max_context_tokens < self.crossover_tokens:
            raise ValueError("max_context_tokens must be at least crossover_tokens")
        if self.residency_limit_bytes <= 0:
            raise ValueError("residency_limit_bytes must be positive")
        required_context = _normalize_ladder(
            "required_context_tokens", self.required_context_tokens
        )
        required_concurrency = _normalize_ladder(
            "required_concurrency_levels", self.required_concurrency_levels
        )
        object.__setattr__(self, "required_context_tokens", required_context)
        object.__setattr__(self, "required_concurrency_levels", required_concurrency)
        if any(token > self.max_context_tokens for token in required_context):
            raise ValueError("required_context_tokens cannot exceed max_context_tokens")
        required_canonical_context = {
            token
            for token in _CONTEXT_QUALIFICATION_LADDER
            if token <= self.max_context_tokens
        }
        if not required_canonical_context.issubset(required_context):
            raise ValueError(
                "required_context_tokens must include every applicable context ladder rung"
            )
        if self.required_concurrency_levels == ():
            raise ValueError("required_concurrency_levels must not be empty")
        if self.required_concurrency_levels[0] != 1:
            raise ValueError("required_concurrency_levels must include one request")
        if (
            not math.isfinite(self.min_ttft_improvement_pct)
            or self.min_ttft_improvement_pct <= 0.0
        ):
            raise ValueError("min_ttft_improvement_pct must be finite and positive")
        for name in (
            "max_total_latency_regression_pct",
            "max_decode_throughput_regression_pct",
            "max_p95_inter_token_latency_regression_pct",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not math.isfinite(self.min_prefill_heavy_throughput_improvement_pct)
            or self.min_prefill_heavy_throughput_improvement_pct <= 0.0
        ):
            raise ValueError(
                "min_prefill_heavy_throughput_improvement_pct must be finite and positive"
            )

        if self.required_concurrency_levels and any(
            level > 1 for level in self.required_concurrency_levels
        ):
            if not set(_CB_CONCURRENCY_LADDER).issubset(
                self.required_concurrency_levels
            ):
                raise ValueError("CB required_concurrency_levels must include 1/2/4/8")


@dataclass(frozen=True)
class SpecPrefillQualificationEvidence:
    """Immutable report-backed proof for one production profile cell."""

    report_id: str
    report_sha256: str
    key: SpecPrefillProfileKey
    selector_version: str
    tested_context_tokens: tuple[int, ...]
    tested_concurrency_levels: tuple[int, ...]
    deterministic_baseline_successes: int
    preserved_baseline_successes: int
    fabricated_or_source_corruption_count: int
    quality_noninferiority_ci_lower_points: float
    median_ttft_improvement_pct: float
    median_total_latency_regression_pct: float
    decode_throughput_regression_pct: float
    oom_count: int
    swap_escalation_count: int
    unbounded_retry_count: int
    peak_resident_bytes: int
    admission_safety_reserve_pct: float
    cb_p95_inter_token_latency_regression_pct: float | None
    cb_aggregate_throughput_regression_pct: float | None
    cb_prefill_heavy_throughput_improvement_pct: float | None
    mtp_evidence_id: str | None
    mtp_evidence_sha256: str | None
    mtp_drafts: int
    mtp_accepted: int

    def __post_init__(self) -> None:
        if not isinstance(self.report_id, str) or not self.report_id.strip():
            raise ValueError("report_id must be a non-empty artifact identifier")
        _validate_sha256("report_sha256", self.report_sha256)
        if not isinstance(self.key, SpecPrefillProfileKey):
            raise ValueError("evidence key must be SpecPrefillProfileKey")
        if (
            not isinstance(self.selector_version, str)
            or not self.selector_version.strip()
        ):
            raise ValueError("evidence selector_version must be non-empty")
        for name in (
            "deterministic_baseline_successes",
            "preserved_baseline_successes",
            "fabricated_or_source_corruption_count",
            "oom_count",
            "swap_escalation_count",
            "unbounded_retry_count",
            "mtp_drafts",
            "mtp_accepted",
            "peak_resident_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        tested_context = _normalize_ladder(
            "tested_context_tokens", self.tested_context_tokens
        )
        tested_concurrency = _normalize_ladder(
            "tested_concurrency_levels", self.tested_concurrency_levels
        )
        object.__setattr__(self, "tested_context_tokens", tested_context)
        object.__setattr__(self, "tested_concurrency_levels", tested_concurrency)
        for name in (
            "quality_noninferiority_ci_lower_points",
            "median_ttft_improvement_pct",
            "median_total_latency_regression_pct",
            "decode_throughput_regression_pct",
            "admission_safety_reserve_pct",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        cb_values = (
            self.cb_p95_inter_token_latency_regression_pct,
            self.cb_aggregate_throughput_regression_pct,
            self.cb_prefill_heavy_throughput_improvement_pct,
        )
        if any(value is not None and not math.isfinite(value) for value in cb_values):
            raise ValueError("CB evidence values must be finite when set")
        if (self.mtp_evidence_id is None) != (self.mtp_evidence_sha256 is None):
            raise ValueError(
                "MTP evidence identifier and hash must be provided together"
            )
        if self.mtp_evidence_id is not None:
            if not self.mtp_evidence_id.strip():
                raise ValueError("MTP evidence identifier must be non-empty")
            _validate_sha256("mtp_evidence_sha256", self.mtp_evidence_sha256)

    def assert_certifies(self, calibration: SpecPrefillCalibration) -> None:
        """Raise unless this evidence proves every gate advertised by a profile."""
        if self.selector_version != calibration.selector_version:
            raise ValueError(
                "qualification evidence selector version does not match calibration"
            )
        if not set(calibration.required_context_tokens).issubset(
            self.tested_context_tokens
        ):
            raise ValueError(
                "qualification evidence is missing required context ladder rungs"
            )
        if not set(calibration.required_concurrency_levels).issubset(
            self.tested_concurrency_levels
        ):
            raise ValueError(
                "qualification evidence is missing required concurrency ladder levels"
            )
        if self.deterministic_baseline_successes <= 0:
            raise ValueError("qualification evidence requires baseline successes")
        if self.preserved_baseline_successes != self.deterministic_baseline_successes:
            raise ValueError(
                "qualification evidence does not preserve baseline successes"
            )
        if self.fabricated_or_source_corruption_count != 0:
            raise ValueError(
                "qualification evidence records fabricated/source corruption"
            )
        if self.quality_noninferiority_ci_lower_points < -2.0:
            raise ValueError(
                "qualification evidence fails quality noninferiority CI gate"
            )
        if self.median_ttft_improvement_pct < calibration.min_ttft_improvement_pct:
            raise ValueError("qualification evidence fails TTFT improvement gate")
        if (
            self.median_total_latency_regression_pct
            > calibration.max_total_latency_regression_pct
        ):
            raise ValueError("qualification evidence fails total latency gate")
        if (
            self.decode_throughput_regression_pct
            > calibration.max_decode_throughput_regression_pct
        ):
            raise ValueError("qualification evidence fails decode throughput gate")
        if self.oom_count != 0:
            raise ValueError("qualification evidence records OOM")
        if self.swap_escalation_count != 0:
            raise ValueError("qualification evidence records swap escalation")
        if self.unbounded_retry_count != 0:
            raise ValueError("qualification evidence records unbounded retry")
        if self.peak_resident_bytes > calibration.residency_limit_bytes:
            raise ValueError(
                "qualification evidence resident peak exceeds residency limit"
            )
        if self.admission_safety_reserve_pct < 10.0:
            raise ValueError(
                "qualification evidence admission safety reserve is below 10 percent"
            )
        if self.key.engine is SpecPrefillEngine.CONTINUOUS_BATCHING:
            if any(
                value is None
                for value in (
                    self.cb_p95_inter_token_latency_regression_pct,
                    self.cb_aggregate_throughput_regression_pct,
                    self.cb_prefill_heavy_throughput_improvement_pct,
                )
            ):
                raise ValueError("qualification evidence is missing CB metrics")
            if (
                self.cb_p95_inter_token_latency_regression_pct
                > calibration.max_p95_inter_token_latency_regression_pct
            ):
                raise ValueError("qualification evidence fails CB p95 ITL gate")
            if self.cb_aggregate_throughput_regression_pct > 0.0:
                raise ValueError(
                    "qualification evidence fails CB aggregate throughput gate"
                )
            if (
                self.cb_prefill_heavy_throughput_improvement_pct
                < calibration.min_prefill_heavy_throughput_improvement_pct
            ):
                raise ValueError(
                    "qualification evidence fails CB prefill-heavy throughput gate"
                )
        if self.key.cell is SpecPrefillCell.COMBINED_MTP:
            if self.mtp_evidence_id is None or self.mtp_drafts <= 0:
                raise ValueError("qualification evidence is missing MTP evidence")


@dataclass(frozen=True)
class SpecPrefillProfile:
    """A calibrated profile, not a wildcard default for a model family."""

    key: SpecPrefillProfileKey
    tier: SpecPrefillProfileTier
    calibration: SpecPrefillCalibration
    qualification_evidence: SpecPrefillQualificationEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, SpecPrefillProfileKey):
            raise ValueError("key must be SpecPrefillProfileKey")
        if not isinstance(self.tier, SpecPrefillProfileTier):
            raise ValueError("tier must be SpecPrefillProfileTier")
        if not isinstance(self.calibration, SpecPrefillCalibration):
            raise ValueError("calibration must be SpecPrefillCalibration")
        if self.key.engine is SpecPrefillEngine.CONTINUOUS_BATCHING and not set(
            _CB_CONCURRENCY_LADDER
        ).issubset(self.calibration.required_concurrency_levels):
            raise ValueError("CB profiles must require the 1/2/4/8 concurrency ladder")
        if self.tier is SpecPrefillProfileTier.DIAGNOSTIC:
            if self.qualification_evidence is not None:
                raise ValueError("diagnostic profiles cannot carry production evidence")
            return
        if self.qualification_evidence is not None:
            if self.qualification_evidence.key != self.key:
                raise ValueError(
                    "qualification evidence key does not match profile key"
                )
            self.qualification_evidence.assert_certifies(self.calibration)

    @property
    def production_certified(self) -> bool:
        return (
            self.tier is SpecPrefillProfileTier.PRODUCTION
            and self.qualification_evidence is not None
        )


@dataclass(frozen=True)
class SpecPrefillProfileDecision:
    """An explicit eligibility outcome for sparse execution or dense fallback."""

    eligible: bool
    production_certified: bool
    fallback_reason: str | None
    profile: SpecPrefillProfile | None = None

    @property
    def tuning(self) -> SpecPrefillTuning | None:
        return self.profile.calibration.tuning if self.profile else None

    @property
    def selector_version(self) -> str | None:
        return self.profile.calibration.selector_version if self.profile else None


class SpecPrefillProfileRegistry:
    """Immutable profile lookup with no family-level or global tuning fallback."""

    def __init__(self, profiles: Iterable[SpecPrefillProfile] = ()) -> None:
        entries: dict[
            tuple[SpecPrefillProfileKey, SpecPrefillProfileTier], SpecPrefillProfile
        ] = {}
        for profile in profiles:
            if not isinstance(profile, SpecPrefillProfile):
                raise ValueError("profiles must contain SpecPrefillProfile values")
            profile_key = (profile.key, profile.tier)
            if profile_key in entries:
                raise ValueError("duplicate SpecPrefill profile key and tier")
            entries[profile_key] = profile
        self._profiles = entries

    @property
    def profiles(self) -> tuple[SpecPrefillProfile, ...]:
        return tuple(self._profiles.values())

    def resolve(
        self,
        key: SpecPrefillProfileKey,
        *,
        prompt_tokens: int,
        residency_bytes: int,
        diagnostic: bool = False,
    ) -> SpecPrefillProfileDecision:
        if not isinstance(key, SpecPrefillProfileKey):
            raise ValueError("key must be SpecPrefillProfileKey")
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if residency_bytes < 0:
            raise ValueError("residency_bytes must be non-negative")

        tier = (
            SpecPrefillProfileTier.DIAGNOSTIC
            if diagnostic
            else SpecPrefillProfileTier.PRODUCTION
        )
        profile = self._profiles.get((key, tier))
        if profile is None:
            return _dense_fallback("profile_not_registered")
        if not diagnostic and not profile.production_certified:
            return _dense_fallback("uncalibrated_profile", profile)

        calibration = profile.calibration
        if prompt_tokens < calibration.crossover_tokens:
            return _dense_fallback("below_calibrated_crossover", profile)
        if prompt_tokens > calibration.max_context_tokens:
            return _dense_fallback("above_calibrated_max_context", profile)
        if residency_bytes > calibration.residency_limit_bytes:
            return _dense_fallback("residency_limit_exceeded", profile)
        return SpecPrefillProfileDecision(
            eligible=True,
            production_certified=(
                profile.production_certified
                and profile.tier is SpecPrefillProfileTier.PRODUCTION
            ),
            fallback_reason=None,
            profile=profile,
        )


def _dense_fallback(
    reason: str, profile: SpecPrefillProfile | None = None
) -> SpecPrefillProfileDecision:
    return SpecPrefillProfileDecision(
        eligible=False,
        production_certified=False,
        fallback_reason=reason,
        profile=profile,
    )


def _validate_sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _normalize_ladder(name: str, values: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    for value in normalized:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must contain positive integer values")
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{name} must be sorted and unique")
    return normalized


# Qualification has not run for this registry program.  New production entries
# must be added only with artifact-backed calibration evidence and explicit
# certification; no Qwen or Gemma4 artifact is seeded as production-ready.
EMPTY_SPECPREFILL_PROFILE_REGISTRY = SpecPrefillProfileRegistry()
