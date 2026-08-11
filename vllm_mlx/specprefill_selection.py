# SPDX-License-Identifier: Apache-2.0
"""Pure-Python semantic selection contracts for SpecPrefill.

The scorer computes chunk importance on device, but selection identity must be
defined without device state.  This module keeps policy, mandatory retention
sources, and the final selection fingerprint immutable and independently
testable.  Callers provide control-token positions after tokenization; this
module deliberately never guesses tokenizer-specific control IDs.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence

SPECPREFILL_SELECTOR_VERSION = "hybrid-chunk-v2"


@dataclass(frozen=True)
class SelectionPolicy:
    """Every policy control that can alter a sparse target cache."""

    keep_pct: float
    backbone_pct: float
    halo_chunks: int
    anchor_chunks: int
    chunk_size: int
    selector_version: str = SPECPREFILL_SELECTOR_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.keep_pct, bool)
            or not isinstance(self.keep_pct, Real)
            or not math.isfinite(self.keep_pct)
            or not 0 < self.keep_pct <= 1
        ):
            raise ValueError("keep_pct must be finite and in (0, 1]")
        if (
            isinstance(self.backbone_pct, bool)
            or not isinstance(self.backbone_pct, Real)
            or not math.isfinite(self.backbone_pct)
            or not 0 <= self.backbone_pct <= 1
        ):
            raise ValueError("backbone_pct must be finite and in [0, 1]")
        if (
            isinstance(self.halo_chunks, bool)
            or not isinstance(self.halo_chunks, int)
            or self.halo_chunks < 0
        ):
            raise ValueError("halo_chunks must be non-negative")
        if (
            isinstance(self.anchor_chunks, bool)
            or not isinstance(self.anchor_chunks, int)
            or self.anchor_chunks <= 0
        ):
            raise ValueError("anchor_chunks must be positive")
        if (
            isinstance(self.chunk_size, bool)
            or not isinstance(self.chunk_size, int)
            or self.chunk_size <= 0
        ):
            raise ValueError("chunk_size must be positive")
        if not isinstance(self.selector_version, str) or not self.selector_version:
            raise ValueError("selector_version must be a non-empty string")


@dataclass(frozen=True)
class RotatingTailRequirement:
    """Logical tail which a rotating target cache must retain in full.

    Chunk selection can retain a few extra tokens immediately before the tail;
    it must never omit a token in the final ``window_tokens`` logical range.
    ``window_tokens`` is part of the selection semantic identity even when two
    nearby window sizes happen to expand to the same chunk boundary.
    """

    window_tokens: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_tokens, bool)
            or not isinstance(self.window_tokens, int)
            or self.window_tokens <= 0
        ):
            raise ValueError("window_tokens must be a positive integer")

    def tail_start(self, prompt_length: int) -> int:
        return max(0, prompt_length - self.window_tokens)

    def required_chunks(self, prompt_length: int, chunk_size: int) -> tuple[int, ...]:
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        start_chunk = self.tail_start(prompt_length) // chunk_size
        final_chunk = (prompt_length - 1) // chunk_size
        return tuple(range(start_chunk, final_chunk + 1))


@dataclass(frozen=True)
class SelectionProvenance:
    """Immutable explanation of every source that retained a chunk."""

    importance_chunks: tuple[int, ...] = ()
    backbone_chunks: tuple[int, ...] = ()
    anchor_chunks: tuple[int, ...] = ()
    control_anchor_indices: tuple[int, ...] = ()
    control_anchor_chunks: tuple[int, ...] = ()
    rotating_tail_chunks: tuple[int, ...] = ()
    halo_chunks: tuple[int, ...] = ()


@dataclass(frozen=True)
class SelectionPlan:
    """Deterministic sparse-token selection for one prompt.

    Positions and provenance are immutable host values, suitable for request
    telemetry and exact cache identity.  The MLX execution boundary converts
    ``selected_indices`` to a device array; it must not reinterpret this plan.
    """

    prompt_length: int
    policy: SelectionPolicy
    selected_chunks: tuple[int, ...]
    selected_indices: tuple[int, ...]
    provenance: SelectionProvenance = SelectionProvenance()
    rotating_tail_requirement: RotatingTailRequirement | None = None

    def __post_init__(self) -> None:
        if self.prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        if tuple(sorted(set(self.selected_chunks))) != self.selected_chunks:
            raise ValueError("selected_chunks must be sorted and unique")
        if tuple(sorted(set(self.selected_indices))) != self.selected_indices:
            raise ValueError("selected_indices must be sorted and unique")
        if not self.selected_indices:
            raise ValueError("SelectionPlan must retain at least one token")
        if (
            self.selected_indices[0] < 0
            or self.selected_indices[-1] >= self.prompt_length
        ):
            raise ValueError("selected_indices are outside the prompt")

        expected = tuple(
            token
            for chunk in self.selected_chunks
            for token in range(
                chunk * self.policy.chunk_size,
                min((chunk + 1) * self.policy.chunk_size, self.prompt_length),
            )
        )
        if expected != self.selected_indices:
            raise ValueError("selected_indices must contain complete selected chunks")

        n_chunks = math.ceil(self.prompt_length / self.policy.chunk_size)
        self._validate_chunks(
            "importance_chunks",
            self.provenance.importance_chunks,
            n_chunks,
            ordered=True,
        )
        for source_name, values in (
            ("backbone_chunks", self.provenance.backbone_chunks),
            ("anchor_chunks", self.provenance.anchor_chunks),
            ("control_anchor_chunks", self.provenance.control_anchor_chunks),
            ("rotating_tail_chunks", self.provenance.rotating_tail_chunks),
            ("halo_chunks", self.provenance.halo_chunks),
        ):
            self._validate_chunks(source_name, values, n_chunks)
        self._validate_indices(
            "control_anchor_indices", self.provenance.control_anchor_indices
        )
        selected = set(self.selected_chunks)
        for source_name, chunks in (
            ("importance_chunks", self.provenance.importance_chunks),
            ("backbone_chunks", self.provenance.backbone_chunks),
            ("anchor_chunks", self.provenance.anchor_chunks),
            ("control_anchor_chunks", self.provenance.control_anchor_chunks),
            ("rotating_tail_chunks", self.provenance.rotating_tail_chunks),
            ("halo_chunks", self.provenance.halo_chunks),
        ):
            if not set(chunks).issubset(selected):
                raise ValueError(f"{source_name} must be retained in selected_chunks")

        expected_anchors = _anchor_chunk_ids(n_chunks, self.policy.anchor_chunks)
        if self.provenance.anchor_chunks != expected_anchors:
            raise ValueError("anchor_chunks must match the policy-derived anchors")
        expected_backbone = _stratified_chunks(
            n_chunks, math.ceil(n_chunks * self.policy.backbone_pct)
        )
        if self.provenance.backbone_chunks != expected_backbone:
            raise ValueError("backbone_chunks must match the policy-derived backbone")
        expected_control_chunks = tuple(
            sorted(
                {
                    index // self.policy.chunk_size
                    for index in self.provenance.control_anchor_indices
                }
            )
        )
        if self.provenance.control_anchor_chunks != expected_control_chunks:
            raise ValueError("control_anchor_chunks must match control_anchor_indices")
        if self.rotating_tail_requirement is not None:
            required_tail = self.rotating_tail_requirement.required_chunks(
                self.prompt_length, self.policy.chunk_size
            )
            if self.provenance.rotating_tail_chunks != required_tail:
                raise ValueError(
                    "rotating_tail_chunks must exactly describe the tail requirement"
                )
            if not set(required_tail).issubset(self.selected_chunks):
                raise ValueError("selected_chunks must retain the rotating tail")
        elif self.provenance.rotating_tail_chunks:
            raise ValueError("rotating_tail_chunks require a rotating_tail_requirement")

    def _validate_chunks(
        self,
        source_name: str,
        chunks: tuple[int, ...],
        n_chunks: int,
        *,
        ordered: bool = False,
    ) -> None:
        if len(set(chunks)) != len(chunks):
            raise ValueError(f"{source_name} must be unique")
        if not ordered and tuple(sorted(chunks)) != chunks:
            raise ValueError(f"{source_name} must be sorted")
        if any(
            not isinstance(chunk, int) or chunk < 0 or chunk >= n_chunks
            for chunk in chunks
        ):
            raise ValueError(f"{source_name} contains an invalid chunk")

    def _validate_indices(self, source_name: str, indices: tuple[int, ...]) -> None:
        if tuple(sorted(set(indices))) != indices:
            raise ValueError(f"{source_name} must be sorted and unique")
        if any(
            not isinstance(index, int) or index < 0 or index >= self.prompt_length
            for index in indices
        ):
            raise ValueError(f"{source_name} contains an invalid token index")

    @property
    def chunk_size(self) -> int:
        """Compatibility view of :attr:`policy.chunk_size`."""
        return self.policy.chunk_size

    @property
    def keep_pct(self) -> float:
        """Compatibility view of :attr:`policy.keep_pct`."""
        return self.policy.keep_pct

    @property
    def selector_version(self) -> str:
        """Compatibility view of :attr:`policy.selector_version`."""
        return self.policy.selector_version

    @property
    def importance_chunks(self) -> tuple[int, ...]:
        return self.provenance.importance_chunks

    @property
    def backbone_chunks(self) -> tuple[int, ...]:
        return self.provenance.backbone_chunks

    @property
    def anchor_chunks(self) -> tuple[int, ...]:
        return self.provenance.anchor_chunks

    @property
    def control_anchor_indices(self) -> tuple[int, ...]:
        return self.provenance.control_anchor_indices

    @property
    def control_anchor_chunks(self) -> tuple[int, ...]:
        return self.provenance.control_anchor_chunks

    @property
    def rotating_tail_chunks(self) -> tuple[int, ...]:
        return self.provenance.rotating_tail_chunks

    @property
    def halo_chunks(self) -> tuple[int, ...]:
        return self.provenance.halo_chunks

    @property
    def selected_token_count(self) -> int:
        return len(self.selected_indices)

    def as_mx_indices(self):
        """Return executor-ready MLX indices at the explicit device boundary.

        The local import keeps selector construction and validation usable in
        non-MLX scheduling and cache-admission code.
        """
        import mlx.core as mx

        return mx.array(self.selected_indices, dtype=mx.int32)

    @property
    def fingerprint(self) -> str:
        """Stable semantic identity for exact sparse-cache reuse."""
        tail_window = (
            "none"
            if self.rotating_tail_requirement is None
            else str(self.rotating_tail_requirement.window_tokens)
        )
        fields = (
            self.selector_version,
            str(self.prompt_length),
            self.keep_pct.hex(),
            self.policy.backbone_pct.hex(),
            str(self.policy.halo_chunks),
            str(self.policy.anchor_chunks),
            str(self.chunk_size),
            ",".join(map(str, self.selected_chunks)),
            ",".join(map(str, self.selected_indices)),
            ",".join(map(str, self.importance_chunks)),
            ",".join(map(str, self.backbone_chunks)),
            ",".join(map(str, self.anchor_chunks)),
            ",".join(map(str, self.control_anchor_indices)),
            ",".join(map(str, self.control_anchor_chunks)),
            tail_window,
            ",".join(map(str, self.rotating_tail_chunks)),
            ",".join(map(str, self.halo_chunks)),
        )
        return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()


def _stratified_chunks(n_chunks: int, count: int) -> tuple[int, ...]:
    """Select stable center-of-stratum representatives across a prompt."""
    if count <= 0:
        return ()
    if count >= n_chunks:
        return tuple(range(n_chunks))
    selected = {
        min(n_chunks - 1, int((index + 0.5) * n_chunks / count))
        for index in range(count)
    }
    for candidate in range(n_chunks):
        if len(selected) >= count:
            break
        selected.add(candidate)
    return tuple(sorted(selected))


def _anchor_chunk_ids(n_chunks: int, count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    count = min(count, n_chunks)
    anchors = set(range(count))
    anchors.update(range(max(0, n_chunks - count), n_chunks))
    return tuple(sorted(anchors))


def _canonical_control_indices(
    control_token_indices: Iterable[int], prompt_length: int
) -> tuple[int, ...]:
    values = tuple(control_token_indices)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= prompt_length
        for index in values
    ):
        raise ValueError("control_token_indices must be prompt-local integer positions")
    return tuple(sorted(set(values)))


def build_selection_plan_from_chunk_scores(
    *,
    prompt_length: int,
    chunk_scores: Sequence[float],
    policy: SelectionPolicy,
    control_token_indices: Iterable[int] = (),
    rotating_tail_requirement: RotatingTailRequirement | None = None,
) -> SelectionPlan:
    """Build a plan from host-resident chunk scores without MLX dependencies."""
    if not isinstance(policy, SelectionPolicy):
        raise TypeError("policy must be a SelectionPolicy")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    n_chunks = math.ceil(prompt_length / policy.chunk_size)
    if len(chunk_scores) != n_chunks:
        raise ValueError("chunk_scores length must match the prompt chunk count")
    if not all(math.isfinite(score) for score in chunk_scores):
        raise ValueError("chunk_scores must contain only finite values")

    control_indices = _canonical_control_indices(control_token_indices, prompt_length)
    control_chunks = tuple(
        sorted({index // policy.chunk_size for index in control_indices})
    )
    tail_chunks = (
        ()
        if rotating_tail_requirement is None
        else rotating_tail_requirement.required_chunks(prompt_length, policy.chunk_size)
    )
    anchors = _anchor_chunk_ids(n_chunks, policy.anchor_chunks)
    backbone = _stratified_chunks(n_chunks, math.ceil(n_chunks * policy.backbone_pct))
    mandatory = set(anchors)
    mandatory.update(control_chunks)
    mandatory.update(tail_chunks)
    selected = set(mandatory)
    ranked = tuple(
        sorted(range(n_chunks), key=lambda chunk: (-chunk_scores[chunk], chunk))
    )
    desired_chunks = max(1, math.ceil(n_chunks * policy.keep_pct))
    desired_tokens = max(1, math.ceil(prompt_length * policy.keep_pct))
    selected.update(backbone)
    # Mandatory sources are a correctness floor.  They may exceed the score
    # budget, exactly like the old fixed first/last anchors.
    importance_budget = max(1, desired_chunks - len(backbone))
    chunk_budget = min(n_chunks, desired_chunks + len(mandatory))
    importance_seeds: list[int] = []
    halo_chunks: set[int] = set()

    def selected_token_count() -> int:
        return sum(
            min(policy.chunk_size, prompt_length - chunk * policy.chunk_size)
            for chunk in selected
        )

    for seed in ranked:
        if (
            len(importance_seeds) >= importance_budget
            and selected_token_count() >= desired_tokens
        ):
            break
        was_seed_selected = seed in selected
        candidates = [seed]
        for distance in range(1, policy.halo_chunks + 1):
            candidates.extend((seed - distance, seed + distance))
        for candidate in candidates:
            if not 0 <= candidate < n_chunks:
                continue
            was_selected = candidate in selected
            if len(selected) >= chunk_budget and candidate not in selected:
                continue
            selected.add(candidate)
            if candidate == seed and not was_seed_selected:
                importance_seeds.append(seed)
            elif candidate != seed and not was_selected:
                halo_chunks.add(candidate)

    # A partial last chunk can meet a chunk budget without meeting the token
    # retention target; fill deterministically until both are satisfied.
    for candidate in ranked:
        if len(selected) >= chunk_budget and selected_token_count() >= desired_tokens:
            break
        if (
            selected_token_count() < desired_tokens
            or len(selected) < chunk_budget
            or candidate in selected
        ):
            selected.add(candidate)

    selected_chunks = tuple(sorted(selected))
    selected_indices = tuple(
        token
        for chunk in selected_chunks
        for token in range(
            chunk * policy.chunk_size,
            min((chunk + 1) * policy.chunk_size, prompt_length),
        )
    )
    return SelectionPlan(
        prompt_length=prompt_length,
        policy=policy,
        selected_chunks=selected_chunks,
        selected_indices=selected_indices,
        provenance=SelectionProvenance(
            importance_chunks=tuple(importance_seeds),
            backbone_chunks=backbone,
            anchor_chunks=anchors,
            control_anchor_indices=control_indices,
            control_anchor_chunks=control_chunks,
            rotating_tail_chunks=tail_chunks,
            halo_chunks=tuple(sorted(halo_chunks)),
        ),
        rotating_tail_requirement=rotating_tail_requirement,
    )
