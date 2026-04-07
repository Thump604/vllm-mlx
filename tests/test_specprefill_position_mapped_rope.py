# SPDX-License-Identifier: Apache-2.0
"""Tests for _PositionMappedRoPE compatibility with ProportionalRoPE.

Session 84 root-cause: Gemma 4 26B's full-attention layers use
``ProportionalRoPE`` (mlx_vlm.models.gemma4.rope_utils) which exposes
``self.dims = full_head_dim`` while only rotating ``self.rotated_dims``
of those dims, via a SPLIT-and-STITCH pattern that
``manual_rope_with_freqs`` cannot reproduce by reading ``rope.dims``
alone. The result was a ``[broadcast_shapes]`` failure inside the first
sparse-prefill chunk on Gemma 4. These tests pin the math against the
canonical ``ProportionalRoPE`` so the position-mapped version stays
correct as the rope class evolves upstream.
"""

from __future__ import annotations

import math

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


# Minimal local copy of ProportionalRoPE so the test does not depend on
# mlx_vlm being importable in this venv. The math must match the
# upstream implementation; if it diverges the test fails loudly.
class _ProportionalRoPEReference:
    def __init__(
        self,
        dims: int,
        rotated_dims: int,
        base: float = 1_000_000.0,
        traditional: bool = False,
    ):
        self.dims = dims
        self.rotated_dims = rotated_dims
        self.traditional = traditional
        if rotated_dims > 0:
            exponents = mx.arange(0, rotated_dims, 2, dtype=mx.float32) / dims
            self._freqs = base**exponents
        else:
            self._freqs = None

    def __call__(self, x, offset: int = 0):
        if self.rotated_dims <= 0:
            return x

        head = x[..., : self.dims]
        tail = x[..., self.dims :]
        half = self.dims // 2

        left = head[..., :half]
        right = head[..., half:]
        rotated = mx.concatenate(
            [
                left[..., : self.rotated_dims // 2],
                right[..., : self.rotated_dims // 2],
            ],
            axis=-1,
        )
        rotated = mx.fast.rope(
            rotated,
            self.rotated_dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )

        left_out = mx.concatenate(
            [
                rotated[..., : self.rotated_dims // 2],
                left[..., self.rotated_dims // 2 :],
            ],
            axis=-1,
        )
        right_out = mx.concatenate(
            [
                rotated[..., self.rotated_dims // 2 :],
                right[..., self.rotated_dims // 2 :],
            ],
            axis=-1,
        )
        head = mx.concatenate([left_out, right_out], axis=-1)

        if tail.shape[-1] == 0:
            return head
        return mx.concatenate([head, tail], axis=-1)


def _rand(shape, seed=0):
    mx.random.seed(seed)
    return mx.random.normal(shape)


class TestPositionMappedProportionalRoPE:
    """Math regression for position-mapped ProportionalRoPE."""

    def test_position_mapped_matches_offset_zero_contiguous(self):
        """For contiguous positions [0..L-1], the position-mapped
        helper must produce bit-identical output to ProportionalRoPE
        called with offset=0."""
        from vllm_mlx.specprefill import (
            _PositionMappedRoPE,
            manual_rope_proportional,
        )

        full_dims = 512
        rotated_dims = 128
        L = 16
        x = _rand((1, 2, L, full_dims), seed=1)

        ref_rope = _ProportionalRoPEReference(
            dims=full_dims, rotated_dims=rotated_dims, base=1_000_000.0
        )
        expected = ref_rope(x, offset=0)

        positions = mx.arange(L, dtype=mx.int32)
        actual = manual_rope_proportional(
            x,
            positions=positions,
            full_dims=full_dims,
            rotated_dims=rotated_dims,
            freqs=ref_rope._freqs,
        )

        assert actual.shape == expected.shape == (1, 2, L, full_dims)
        # Bit-identical at offset 0; allow tiny float drift from
        # different intermediate ops.
        max_err = float(mx.max(mx.abs(actual - expected)).item())
        assert max_err < 1e-5, f"max_err={max_err}"

    def test_position_mapped_matches_offset_nonzero_contiguous(self):
        """For contiguous positions [offset..offset+L-1] the helper
        must match ProportionalRoPE(x, offset=offset)."""
        from vllm_mlx.specprefill import manual_rope_proportional

        full_dims = 256
        rotated_dims = 64
        L = 8
        offset = 7
        x = _rand((1, 4, L, full_dims), seed=2)

        ref_rope = _ProportionalRoPEReference(
            dims=full_dims, rotated_dims=rotated_dims, base=1_000_000.0
        )
        expected = ref_rope(x, offset=offset)

        positions = mx.arange(offset, offset + L, dtype=mx.int32)
        actual = manual_rope_proportional(
            x,
            positions=positions,
            full_dims=full_dims,
            rotated_dims=rotated_dims,
            freqs=ref_rope._freqs,
        )
        max_err = float(mx.max(mx.abs(actual - expected)).item())
        assert max_err < 1e-5, f"max_err={max_err}"

    def test_position_mapped_sparse_positions_per_row(self):
        """For arbitrary (non-contiguous) positions, each row must
        rotate independently at its OWN position. We construct the
        expected output by calling the reference rope row-by-row."""
        from vllm_mlx.specprefill import manual_rope_proportional

        full_dims = 256
        rotated_dims = 64
        positions_list = [3, 17, 42, 99]
        x = _rand((1, 2, len(positions_list), full_dims), seed=3)

        ref_rope = _ProportionalRoPEReference(
            dims=full_dims, rotated_dims=rotated_dims, base=1_000_000.0
        )

        # Build expected row-by-row.
        rows = []
        for i, pos in enumerate(positions_list):
            row = x[..., i : i + 1, :]
            rotated_row = ref_rope(row, offset=int(pos))
            rows.append(rotated_row)
        expected = mx.concatenate(rows, axis=-2)

        positions = mx.array(positions_list, dtype=mx.int32)
        actual = manual_rope_proportional(
            x,
            positions=positions,
            full_dims=full_dims,
            rotated_dims=rotated_dims,
            freqs=ref_rope._freqs,
        )
        # Tolerance matches the standard-RoPE test below: the
        # reference path goes through ``mx.fast.rope`` per row while
        # the position-mapped helper uses pure mlx ops, so float32
        # drift at ~1e-5 is expected and not a math bug.
        max_err = float(mx.max(mx.abs(actual - expected)).item())
        assert max_err < 1e-4, f"max_err={max_err}"

    def test_no_rotation_when_rotated_dims_zero(self):
        """If rotated_dims is 0, the helper must return the input
        unchanged (matching ProportionalRoPE's no-op branch)."""
        from vllm_mlx.specprefill import manual_rope_proportional

        x = _rand((1, 1, 4, 32), seed=4)
        positions = mx.array([0, 1, 2, 3], dtype=mx.int32)
        out = manual_rope_proportional(
            x,
            positions=positions,
            full_dims=32,
            rotated_dims=0,
            freqs=None,
        )
        assert mx.array_equal(out, x).item()

    def test_position_mapped_rope_detects_proportional_rope_class(self):
        """``_PositionMappedRoPE`` must detect ProportionalRoPE-style
        ropes via the ``rotated_dims`` attribute and route through the
        proportional helper instead of plain ``manual_rope_with_freqs``,
        which would broadcast-mismatch."""
        from vllm_mlx.specprefill import _PositionMappedRoPE

        full_dims = 512
        rotated_dims = 128
        L = 8
        ref_rope = _ProportionalRoPEReference(
            dims=full_dims, rotated_dims=rotated_dims, base=1_000_000.0
        )

        all_positions = mx.arange(L, dtype=mx.int32)
        wrapper = _PositionMappedRoPE(
            ref_rope, all_positions, cache_start=0
        )

        # Confirm the wrapper detected the proportional pattern.
        assert getattr(wrapper, "_proportional_mode", False) is True
        assert wrapper._rotated_dims == rotated_dims

        x = _rand((1, 2, L, full_dims), seed=5)
        actual = wrapper(x, offset=0)
        expected = ref_rope(x, offset=0)
        max_err = float(mx.max(mx.abs(actual - expected)).item())
        assert max_err < 1e-5, f"max_err={max_err}"

    def test_position_mapped_rope_standard_path_unaffected(self):
        """Plain ``nn.RoPE`` (no ``rotated_dims`` attribute) must still
        flow through the existing ``manual_rope`` path and produce
        results that match ``nn.RoPE`` for contiguous offsets."""
        from vllm_mlx.specprefill import _PositionMappedRoPE

        full_dims = 64
        L = 4
        rope = nn.RoPE(full_dims, traditional=False, base=10000.0)
        x = _rand((1, 2, L, full_dims), seed=6)
        expected = rope(x, offset=0)

        positions = mx.arange(L, dtype=mx.int32)
        wrapper = _PositionMappedRoPE(rope, positions, cache_start=0)
        assert getattr(wrapper, "_proportional_mode", False) is False
        actual = wrapper(x, offset=0)
        max_err = float(mx.max(mx.abs(actual - expected)).item())
        assert max_err < 1e-4, f"max_err={max_err}"
