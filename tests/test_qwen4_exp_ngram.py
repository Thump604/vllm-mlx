import numpy as np

from vllm_mlx.utils.qwen4_exp_ngram import Qwen4ExpNGramLayout


def _official_layout() -> Qwen4ExpNGramLayout:
    return Qwen4ExpNGramLayout(
        unigram_vocab_size=248_320,
        embedding_dim=2_560,
        eos_token_id=248_044,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        divisible_by=128,
        split_parts=128,
        ple_layer_index=0,
        seed=1234,
    )


def test_released_flash_next_ngram_layout_matches_weight_headers():
    layout = _official_layout()

    assert layout.ngram_heads == 16
    assert layout.head_dim == 160
    assert layout.padded_rows == 320_001_536
    assert layout.rows_per_split == 2_500_012
    assert layout.bf16_bytes == 102_400_491_520


def test_each_token_selects_one_row_per_ngram_head():
    layout = _official_layout()
    indices = layout.indices_for_tokens([248_044, 10, 20, 30])

    assert len(indices) == 4
    assert all(len(rows) == 16 for rows in indices)
    assert all(0 <= row < layout.padded_rows for rows in indices for row in rows)


def test_eos_resets_ngram_history():
    layout = _official_layout()

    after_history = layout.indices_for_tokens([10, 20, 248_044, 30])[-1]
    after_fresh_eos = layout.indices_for_tokens([248_044, 30])[-1]

    assert after_history == after_fresh_eos


def test_global_rows_map_to_released_split_shape():
    layout = _official_layout()

    assert layout.split_address(0) == (0, 0)
    assert layout.split_address(2_500_012) == (1, 0)
    assert layout.split_address(layout.padded_rows - 1) == (127, 2_500_011)


def _transformers_shift_reference(layout, token_ids):
    """Independent form of Transformers' vectorized shift/gather algorithm."""
    tokens = np.asarray(token_ids, dtype=np.int64)[None, :]
    positions = np.arange(tokens.shape[1], dtype=np.int64)
    shifted = []
    for shift in range(layout.ngram_size):
        if shift == 0:
            shifted.append(tokens)
            continue
        eos_positions = np.where(tokens == layout.eos_token_id, positions, -1)
        previous_inclusive = np.maximum.accumulate(eos_positions, axis=1)
        previous = np.concatenate(
            [np.full((1, 1), -1, dtype=np.int64), previous_inclusive[:, :-1]],
            axis=1,
        )
        segment_start = previous + 1
        source = positions - shift
        gathered = np.take_along_axis(tokens, np.maximum(source, 0)[None, :], axis=1)
        valid = ((positions[None, :] - segment_start) >= shift) & (source[None, :] >= 0)
        shifted.append(np.where(valid, gathered, layout.eos_token_id))
    return shifted


def test_randomized_indices_match_transformers_reference_formulation():
    layout = _official_layout()
    rng = np.random.default_rng(604)
    sizes = np.asarray(layout.head_vocab_sizes)
    offsets = np.asarray(layout.head_offsets)
    from vllm_mlx.utils.qwen4_exp_ngram import _build_layer_multipliers

    multipliers = _build_layer_multipliers(
        layout.unigram_vocab_size,
        layout.ngram_size,
        layout.ple_layer_index,
        layout.seed,
    )
    for _ in range(30):
        tokens = rng.integers(0, layout.unigram_vocab_size - 1, size=37).tolist()
        tokens[rng.integers(1, len(tokens) - 1)] = layout.eos_token_id
        shifted = _transformers_shift_reference(layout, tokens)
        blocks = []
        for ngram in range(2, layout.ngram_size + 1):
            start = (ngram - 2) * layout.heads_per_ngram
            end = start + layout.heads_per_ngram
            mixed = shifted[0] * multipliers[0]
            for position in range(1, ngram):
                mixed = np.bitwise_xor(mixed, shifted[position] * multipliers[position])
            blocks.append(
                np.remainder(mixed[..., None], sizes[start:end]) + offsets[start:end]
            )
        expected = np.concatenate(blocks, axis=-1)[0]
        np.testing.assert_array_equal(layout.indices_for_tokens(tokens), expected)
