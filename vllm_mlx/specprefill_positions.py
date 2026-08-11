# SPDX-License-Identifier: Apache-2.0
"""Request-local logical-position contracts for target SpecPrefill execution.

Sparse prefill stores selected KV entries contiguously while their semantic
positions are not contiguous.  Consequently a cache's physical occupancy is
not an authority for RoPE/M-RoPE positions.  This module is deliberately
host-only: it builds immutable invocation plans from
:mod:`vllm_mlx.specprefill_cache` and never modifies model modules, cache
objects, or global RoPE state.

An executor may run a plan only when its target family has an explicit
transport for the required non-contiguous positions.  The conservative
capability matrix is intentional:

* normal Qwen attention and Qwen3.5/3.6 text/hybrid targets currently accept
  cache-derived scalar offsets only, so sparse position transport is blocked;
* Qwen3.5/3.6 VLM attention accepts explicit M-RoPE IDs shaped ``(3, B, L)``;
* Gemma4 forwards an explicit scalar offset through shared-KV layers, but its
  KV-owning attention replaces that value with the physical cache offset.  No
  current public call accepts a non-contiguous position sequence, so sparse
  prefill and sparse decode are both blocked pending a future sparse hook.

This prevents the former model-global ``.rope`` replacement from being used as
an implicit execution mechanism.  A later family-local hook can advertise a
new capability only after its own keep-ratio-one oracle and batch tests pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from .specprefill_cache import SparseCacheState, SparseCacheStateError


class TargetPositionError(SparseCacheStateError):
    """A target cannot faithfully execute the requested logical positions."""


class TargetPositionFamily(str, Enum):
    """Explicit target families with materially different position APIs."""

    QWEN_DENSE = "qwen_dense"
    QWEN35_TEXT_HYBRID = "qwen35_text_hybrid"
    QWEN35_VLM_HYBRID = "qwen35_vlm_hybrid"
    QWEN35_VLM_MOE = "qwen35_vlm_moe"
    GEMMA4_DENSE = "gemma4_dense"
    GEMMA4_A4B = "gemma4_a4b"


class PositionTransport(str, Enum):
    """How a target receives request-local positions without global mutation."""

    CACHE_DERIVED_OFFSET = "cache_derived_offset"
    QWEN35_MROPE_IDS = "qwen35_mrope_ids"
    GEMMA4_EXPLICIT_OFFSET = "gemma4_explicit_offset"


class PositionPhase(str, Enum):
    """One target-cache lifecycle operation represented by a position plan."""

    SPARSE_PREFILL = "sparse_prefill"
    DECODE = "decode"
    VERIFY = "verify"


@dataclass(frozen=True)
class TargetPositionAdapter:
    """Static family contract, including deliberately unsupported cases."""

    family: TargetPositionFamily
    model_types: tuple[str, ...]
    transport: PositionTransport
    supports_noncontiguous_prefill: bool
    supports_heterogeneous_batch_rows: bool
    supports_shared_kv: bool
    supports_partial_rope: bool
    uses_q_norm: bool
    cache_layout: str
    position_id_planes: int = 1

    def __post_init__(self) -> None:
        if not self.model_types:
            raise TargetPositionError("target adapter must name model types")
        if self.position_id_planes <= 0:
            raise TargetPositionError("position_id_planes must be positive")
        if self.transport is PositionTransport.QWEN35_MROPE_IDS:
            if self.position_id_planes != 3:
                raise TargetPositionError("Qwen3.5 M-RoPE requires exactly 3 planes")
        elif self.position_id_planes != 1:
            raise TargetPositionError(
                "only Qwen3.5 M-RoPE has more than one position-ID plane"
            )


QWEN_DENSE_TARGET = TargetPositionAdapter(
    family=TargetPositionFamily.QWEN_DENSE,
    model_types=("qwen3", "qwen3_moe"),
    transport=PositionTransport.CACHE_DERIVED_OFFSET,
    supports_noncontiguous_prefill=False,
    supports_heterogeneous_batch_rows=False,
    supports_shared_kv=False,
    supports_partial_rope=False,
    uses_q_norm=True,
    cache_layout="one_attention_cache_per_layer",
)

QWEN35_TEXT_HYBRID_TARGET = TargetPositionAdapter(
    family=TargetPositionFamily.QWEN35_TEXT_HYBRID,
    model_types=("qwen3_5", "qwen3_5_moe", "qwen3_next"),
    transport=PositionTransport.CACHE_DERIVED_OFFSET,
    supports_noncontiguous_prefill=False,
    supports_heterogeneous_batch_rows=False,
    supports_shared_kv=False,
    supports_partial_rope=True,
    uses_q_norm=True,
    cache_layout="one_cache_entry_per_hybrid_layer",
)

QWEN35_VLM_HYBRID_TARGET = TargetPositionAdapter(
    family=TargetPositionFamily.QWEN35_VLM_HYBRID,
    model_types=("qwen3_5",),
    transport=PositionTransport.QWEN35_MROPE_IDS,
    supports_noncontiguous_prefill=True,
    supports_heterogeneous_batch_rows=True,
    supports_shared_kv=False,
    supports_partial_rope=True,
    uses_q_norm=True,
    cache_layout="one_cache_entry_per_hybrid_layer",
    position_id_planes=3,
)

QWEN35_VLM_MOE_TARGET = TargetPositionAdapter(
    family=TargetPositionFamily.QWEN35_VLM_MOE,
    model_types=("qwen3_5_moe",),
    transport=PositionTransport.QWEN35_MROPE_IDS,
    supports_noncontiguous_prefill=True,
    supports_heterogeneous_batch_rows=True,
    supports_shared_kv=False,
    supports_partial_rope=True,
    uses_q_norm=True,
    cache_layout="one_cache_entry_per_hybrid_layer",
    position_id_planes=3,
)

GEMMA4_DENSE_TARGET = TargetPositionAdapter(
    family=TargetPositionFamily.GEMMA4_DENSE,
    model_types=("gemma4", "gemma4_text"),
    transport=PositionTransport.GEMMA4_EXPLICIT_OFFSET,
    supports_noncontiguous_prefill=False,
    supports_heterogeneous_batch_rows=False,
    supports_shared_kv=True,
    supports_partial_rope=True,
    uses_q_norm=True,
    cache_layout="one_cache_entry_per_layer_or_previous_kv_owner",
)

GEMMA4_A4B_TARGET = TargetPositionAdapter(
    family=TargetPositionFamily.GEMMA4_A4B,
    model_types=("gemma4",),
    transport=PositionTransport.GEMMA4_EXPLICIT_OFFSET,
    supports_noncontiguous_prefill=False,
    supports_heterogeneous_batch_rows=False,
    supports_shared_kv=True,
    supports_partial_rope=True,
    uses_q_norm=True,
    cache_layout="one_cache_entry_per_layer_or_previous_kv_owner",
)

TARGET_POSITION_ADAPTERS: dict[TargetPositionFamily, TargetPositionAdapter] = {
    adapter.family: adapter
    for adapter in (
        QWEN_DENSE_TARGET,
        QWEN35_TEXT_HYBRID_TARGET,
        QWEN35_VLM_HYBRID_TARGET,
        QWEN35_VLM_MOE_TARGET,
        GEMMA4_DENSE_TARGET,
        GEMMA4_A4B_TARGET,
    )
}


def target_position_adapter(
    family: TargetPositionFamily | str,
) -> TargetPositionAdapter:
    """Resolve one adapter without attribute-based model-family guessing."""
    try:
        return TARGET_POSITION_ADAPTERS[TargetPositionFamily(family)]
    except (KeyError, ValueError) as exc:
        supported = ", ".join(adapter.value for adapter in TargetPositionFamily)
        raise TargetPositionError(
            f"unsupported target position family {family!r}; supported: {supported}"
        ) from exc


@dataclass(frozen=True)
class PositionRow:
    """One row's immutable semantic and physical coordinates for one call."""

    logical_positions: tuple[int, ...]
    physical_cache_length: int

    def __post_init__(self) -> None:
        if not self.logical_positions:
            raise TargetPositionError("position rows must contain at least one token")
        if self.physical_cache_length < 0:
            raise TargetPositionError("physical cache length must be non-negative")
        previous = -1
        for position in self.logical_positions:
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
            ):
                raise TargetPositionError(
                    "logical positions must be non-negative integer coordinates"
                )
            if position <= previous:
                raise TargetPositionError(
                    "logical positions must be strictly increasing"
                )
            previous = position

    @property
    def token_count(self) -> int:
        return len(self.logical_positions)

    @property
    def is_contiguous(self) -> bool:
        start = self.logical_positions[0]
        return self.logical_positions == tuple(range(start, start + self.token_count))


@dataclass(frozen=True)
class TargetPositionPlan:
    """A no-side-effect target invocation plan derived from sparse cache state.

    ``physical_cache_lengths`` describes only the target cache's real KV
    occupancy.  ``logical_positions`` is the independent RoPE/M-RoPE contract.
    An executor must pass the latter through an adapter-specific argument and
    must never rewrite cache ``offset`` to make the two appear equal.
    """

    adapter: TargetPositionAdapter
    phase: PositionPhase
    rows: tuple[PositionRow, ...]
    source_state: SparseCacheState

    def __post_init__(self) -> None:
        if not self.rows:
            raise TargetPositionError("position plan must contain at least one row")
        if self.source_state.row_count != len(self.rows):
            raise TargetPositionError("position plan rows must match sparse cache rows")
        if self.phase is PositionPhase.DECODE and any(
            row.token_count != 1 for row in self.rows
        ):
            raise TargetPositionError("decode plan requires exactly one token per row")

    @property
    def logical_positions(self) -> tuple[tuple[int, ...], ...]:
        return tuple(row.logical_positions for row in self.rows)

    @property
    def physical_cache_lengths(self) -> tuple[int, ...]:
        return tuple(row.physical_cache_length for row in self.rows)

    @property
    def is_dense_equivalent(self) -> bool:
        """True iff every row has ordinary dense zero-origin positions."""
        return all(
            row.logical_positions == tuple(range(row.token_count))
            and row.physical_cache_length == 0
            for row in self.rows
        )

    def require_executable(self) -> None:
        """Fail closed before an engine calls a target model.

        This validates target API capability, not merely that a plan can be
        represented in host metadata.  It intentionally keeps Gemma and
        cache-offset-only Qwen sparse prefill disabled until a request-local
        position hook exists.
        """
        has_noncontiguous = any(not row.is_contiguous for row in self.rows)
        if self.phase is PositionPhase.SPARSE_PREFILL and has_noncontiguous:
            if not self.adapter.supports_noncontiguous_prefill:
                raise TargetPositionError(
                    f"{self.adapter.family.value} has no request-local "
                    "non-contiguous sparse-position transport"
                )
        if len(self.rows) > 1 and not self.adapter.supports_heterogeneous_batch_rows:
            if len(set(self.logical_positions)) > 1:
                raise TargetPositionError(
                    f"{self.adapter.family.value} does not expose verified "
                    "heterogeneous-row logical-position transport"
                )
        token_counts = {row.token_count for row in self.rows}
        if len(token_counts) > 1:
            raise TargetPositionError(
                "a batched target invocation requires equal token counts; "
                "scheduler must use padded, adapter-verified transport or separate lanes"
            )
        if self.adapter.transport in (
            PositionTransport.CACHE_DERIVED_OFFSET,
            PositionTransport.GEMMA4_EXPLICIT_OFFSET,
        ):
            for row in self.rows:
                expected = tuple(
                    range(
                        row.physical_cache_length,
                        row.physical_cache_length + row.token_count,
                    )
                )
                if row.logical_positions != expected:
                    raise TargetPositionError(
                        f"{self.adapter.family.value} derives owner-layer positions "
                        "from physical cache offset and cannot execute this "
                        "logical-position plan"
                    )

    def qwen35_mrope_position_ids(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Return exact Qwen3.5 VLM M-RoPE host shape ``(3, B, L)``.

        The current mlx-vlm Qwen3.5 attention accepts three M-RoPE planes. For
        text-only sparse prefill all three carry the same logical text indices;
        media M-RoPE is deliberately routed dense outside this module.
        """
        if self.adapter.transport is not PositionTransport.QWEN35_MROPE_IDS:
            raise TargetPositionError("target does not accept Qwen3.5 M-RoPE IDs")
        self.require_executable()
        row_ids = self.logical_positions
        return tuple(row_ids for _ in range(self.adapter.position_id_planes))

    def gemma4_offsets(self) -> tuple[int, ...]:
        """Return scalar starting offsets for Gemma's explicit-offset call path.

        This helper is valid for normal dense-contiguous decode only.  Gemma's
        owner attention overwrites its ``offset`` from the physical cache;
        shared-KV layers merely carry that owner value.  It therefore never
        claims that the current Gemma API can execute sparse decode, sparse
        prefill, or batch different row offsets.
        """
        if self.adapter.transport is not PositionTransport.GEMMA4_EXPLICIT_OFFSET:
            raise TargetPositionError("target does not accept Gemma4 explicit offsets")
        self.require_executable()
        return tuple(row.logical_positions[0] for row in self.rows)


def sparse_prefill_plan(
    adapter: TargetPositionAdapter | TargetPositionFamily | str,
    state: SparseCacheState,
) -> TargetPositionPlan:
    """Plan initial sparse target prefill from immutable selected positions.

    ``SparseCacheState.from_selection`` describes the cache *after* selected
    prompt entries are written, so each initial row starts with physical
    occupancy zero.  Sparse prefix reuse is intentionally not modeled here.
    """
    resolved = _resolve_adapter(adapter)
    rows = tuple(
        PositionRow(row.logical_positions, physical_cache_length=0)
        for row in state.rows
    )
    return TargetPositionPlan(resolved, PositionPhase.SPARSE_PREFILL, rows, state)


def decode_plan(
    adapter: TargetPositionAdapter | TargetPositionFamily | str,
    state: SparseCacheState,
) -> TargetPositionPlan:
    """Plan one decode token per row at its independent logical cursor."""
    resolved = _resolve_adapter(adapter)
    rows = tuple(
        PositionRow((row.next_logical_position,), row.physical_valid_length)
        for row in state.rows
    )
    return TargetPositionPlan(resolved, PositionPhase.DECODE, rows, state)


def verify_plan(
    adapter: TargetPositionAdapter | TargetPositionFamily | str,
    state: SparseCacheState,
    token_counts: int | Sequence[int],
) -> TargetPositionPlan:
    """Plan a target speculative-verification block without advancing state."""
    resolved = _resolve_adapter(adapter)
    counts = _normalize_counts(token_counts, state.row_count)
    rows = tuple(
        PositionRow(
            tuple(range(row.next_logical_position, row.next_logical_position + count)),
            row.physical_valid_length,
        )
        for row, count in zip(state.rows, counts, strict=True)
    )
    return TargetPositionPlan(resolved, PositionPhase.VERIFY, rows, state)


@dataclass(frozen=True)
class MTPPositionPlan:
    """Pair target and assistant cache positions without conflating occupancy."""

    target: TargetPositionPlan
    assistant: TargetPositionPlan

    def __post_init__(self) -> None:
        if self.target.phase is not self.assistant.phase:
            raise TargetPositionError("target and assistant must have the same phase")
        if self.target.logical_positions != self.assistant.logical_positions:
            raise TargetPositionError(
                "target and assistant logical positions must agree for MTP"
            )

    @property
    def physical_cache_lengths(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Independent target/assistant physical occupancy for executor wiring."""
        return self.target.physical_cache_lengths, self.assistant.physical_cache_lengths


def mtp_decode_plan(
    target_adapter: TargetPositionAdapter | TargetPositionFamily | str,
    target_state: SparseCacheState,
    assistant_adapter: TargetPositionAdapter | TargetPositionFamily | str,
    assistant_state: SparseCacheState,
) -> MTPPositionPlan:
    """Build matching logical decode plans for native or external MTP."""
    return MTPPositionPlan(
        target=decode_plan(target_adapter, target_state),
        assistant=decode_plan(assistant_adapter, assistant_state),
    )


def gemma_previous_kv_cache_map(previous_kvs: Iterable[int]) -> dict[int, int]:
    """Validate Gemma4 compact shared-KV ownership without touching caches.

    Gemma uses a cache slot per layer while late shared-KV layers read the
    tuple and scalar offset produced by an earlier owner.  The map makes the
    ownership relation explicit for request-local executor state.
    """
    owners = tuple(previous_kvs)
    if not owners:
        raise TargetPositionError("Gemma previous_kvs must not be empty")
    mapping: dict[int, int] = {}
    for index, owner in enumerate(owners):
        if isinstance(owner, bool) or not isinstance(owner, int):
            raise TargetPositionError("Gemma previous_kvs entries must be integers")
        if owner < 0 or owner > index:
            raise TargetPositionError(
                "Gemma previous_kvs owner must reference this or an earlier layer"
            )
        mapping[index] = owner
    return mapping


def _resolve_adapter(
    adapter: TargetPositionAdapter | TargetPositionFamily | str,
) -> TargetPositionAdapter:
    if isinstance(adapter, TargetPositionAdapter):
        return adapter
    return target_position_adapter(adapter)


def _normalize_counts(counts: int | Sequence[int], row_count: int) -> tuple[int, ...]:
    if isinstance(counts, int) and not isinstance(counts, bool):
        values = (counts,) * row_count
    elif isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)):
        values = tuple(counts)
    else:
        raise TargetPositionError(
            "verification token counts must be an integer sequence"
        )
    if len(values) != row_count:
        raise TargetPositionError("verification token counts need one value per row")
    for count in values:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise TargetPositionError(
                "verification token counts must be positive integers"
            )
    return values
