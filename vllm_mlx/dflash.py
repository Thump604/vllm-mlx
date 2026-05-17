# SPDX-License-Identifier: Apache-2.0
"""Default-off DFlash speculative decoding support.

This module keeps the DFlash draft contract and target-layer capture boundary
separate from the serving engines. The first supported target is Qwen 3.6
35B-A3B with the published z-lab drafter config.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import (
    KVCache,
    RotatingKVCache,
    can_trim_prompt_cache,
    make_prompt_cache,
)
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

QWEN36_35B_A3B_DFLASH_REVISION = "42d3b34d588423cdae7ba8f53a8cf7789346a719"
QWEN36_35B_A3B_TARGET_LAYER_IDS = (1, 10, 19, 28, 37)
QWEN36_35B_A3B_TARGET_LAYER_COUNT = 40
QWEN36_35B_A3B_MASK_TOKEN_ID = 248070
QWEN36_35B_A3B_BLOCK_SIZE = 16


class DFlashCompatibilityError(ValueError):
    """Raised when a DFlash draft does not match the selected target."""


@dataclass(frozen=True)
class DFlashDraftConfig:
    """Validated DFlash draft config loaded from ``config.json``."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    mask_token_id: int
    rope_scaling: dict[str, Any] | None = None
    layer_types: tuple[str, ...] = ()
    sliding_window: int | None = None
    use_sliding_window: bool = False
    final_logit_softcapping: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DFlashDraftConfig":
        """Build a draft config from a Hugging Face DFlash ``config.json``."""
        dflash_config = raw.get("dflash_config") or {}
        layer_types = tuple(
            raw.get("layer_types") or ["full_attention"] * raw["num_hidden_layers"]
        )
        if len(layer_types) != raw["num_hidden_layers"]:
            raise DFlashCompatibilityError(
                "layer_types length must match num_hidden_layers"
            )
        unknown_layer_types = set(layer_types) - {"full_attention", "sliding_attention"}
        if unknown_layer_types:
            raise DFlashCompatibilityError(
                f"Unsupported DFlash layer_types: {sorted(unknown_layer_types)}"
            )
        if "sliding_attention" in layer_types and raw.get("sliding_window") is None:
            raise DFlashCompatibilityError(
                "sliding_attention drafts must define sliding_window"
            )
        return cls(
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            intermediate_size=int(raw["intermediate_size"]),
            vocab_size=int(raw["vocab_size"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(raw["rope_theta"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            block_size=int(raw["block_size"]),
            target_layer_ids=tuple(int(i) for i in dflash_config["target_layer_ids"]),
            num_target_layers=int(raw["num_target_layers"]),
            mask_token_id=int(dflash_config["mask_token_id"]),
            rope_scaling=raw.get("rope_scaling"),
            layer_types=layer_types,
            sliding_window=raw.get("sliding_window"),
            use_sliding_window=bool(raw.get("use_sliding_window", False)),
            final_logit_softcapping=raw.get("final_logit_softcapping"),
        )

    def validate_qwen35_a3b_contract(
        self, *, target_model: Any, target_tokenizer: Any
    ) -> None:
        """Validate the published Qwen 3.6 35B-A3B DFlash contract."""
        if self.target_layer_ids != QWEN36_35B_A3B_TARGET_LAYER_IDS:
            raise DFlashCompatibilityError(
                "target_layer_ids mismatch: "
                f"{self.target_layer_ids} != {QWEN36_35B_A3B_TARGET_LAYER_IDS}"
            )
        if self.num_target_layers != QWEN36_35B_A3B_TARGET_LAYER_COUNT:
            raise DFlashCompatibilityError(
                "num_target_layers mismatch: "
                f"{self.num_target_layers} != {QWEN36_35B_A3B_TARGET_LAYER_COUNT}"
            )
        target_layers = _get_layers(target_model)
        if len(target_layers) != self.num_target_layers:
            raise DFlashCompatibilityError(
                "target model layer count mismatch: "
                f"{len(target_layers)} != {self.num_target_layers}"
            )
        if self.mask_token_id != QWEN36_35B_A3B_MASK_TOKEN_ID:
            raise DFlashCompatibilityError(
                "mask_token_id mismatch: "
                f"{self.mask_token_id} != {QWEN36_35B_A3B_MASK_TOKEN_ID}"
            )
        tokenizer_size = None
        try:
            tokenizer_size = len(target_tokenizer)
        except TypeError:
            tokenizer_size = getattr(target_tokenizer, "vocab_size", None)

        if tokenizer_size is not None and self.mask_token_id >= int(tokenizer_size):
            raise DFlashCompatibilityError(
                "mask_token_id "
                f"{self.mask_token_id} outside target tokenizer size {tokenizer_size}"
            )
        if self.block_size != QWEN36_35B_A3B_BLOCK_SIZE:
            raise DFlashCompatibilityError(
                f"block_size mismatch: {self.block_size} != {QWEN36_35B_A3B_BLOCK_SIZE}"
            )
        if self.sliding_window is not None or self.use_sliding_window:
            raise DFlashCompatibilityError(
                "sliding_window/use_sliding_window is not supported for the first "
                "Qwen 35B DFlash milestone"
            )


def _get_layers(model: Any) -> list[Any]:
    """Find the decoder layer list on mlx-lm style model wrappers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        return model.language_model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise DFlashCompatibilityError(
        f"Cannot find target layers on {type(model).__name__}"
    )


class _LayerCaptureWrapper:
    def __init__(self, layer: Any, capture: "DFlashLayerCapture", storage_idx: int):
        self._layer = layer
        self._capture = capture
        self._storage_idx = storage_idx

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        self._capture.record(
            self._storage_idx, out[0] if isinstance(out, tuple) else out
        )
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)


class DFlashLayerCapture:
    """Request-scoped target hidden-state capture with deterministic teardown."""

    def __init__(self, target_model: Any, layer_ids: tuple[int, ...]):
        self._target_model = target_model
        self._layer_ids = tuple(layer_ids)
        self._layers: list[Any] | None = None
        self._original_layers: dict[int, Any] = {}
        self._hidden_states: list[Any | None] = [None] * len(self._layer_ids)
        self._closed = False

    def __enter__(self) -> "DFlashLayerCapture":
        self._layers = _get_layers(self._target_model)
        for storage_idx, layer_idx in enumerate(self._layer_ids):
            if layer_idx < 0 or layer_idx >= len(self._layers):
                raise DFlashCompatibilityError(
                    f"target_layer_id {layer_idx} outside target layers"
                )
            original = self._layers[layer_idx]
            self._original_layers[layer_idx] = original
            self._layers[layer_idx] = _LayerCaptureWrapper(original, self, storage_idx)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def record(self, storage_idx: int, hidden: Any) -> None:
        self._hidden_states[storage_idx] = hidden

    def clear(self) -> None:
        self._hidden_states = [None] * len(self._layer_ids)

    def concat_hidden_states(self) -> mx.array:
        if any(state is None for state in self._hidden_states):
            raise RuntimeError("missing captured hidden states for DFlash draft")
        return mx.concatenate(self._hidden_states, axis=-1)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._layers is not None:
                for layer_idx, original in self._original_layers.items():
                    self._layers[layer_idx] = original
        finally:
            self._closed = True
            self._layers = None
            self._original_layers.clear()
            self.clear()


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashDraftConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim**-0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.q_proj = nn.Linear(dim, self.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * config.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        batch, length, _ = x.shape
        context_len = x_ctx.shape[1]
        if self.is_sliding:
            keep_ctx = self.sliding_window - 1
            if context_len > keep_ctx:
                skip = context_len - keep_ctx
                x_ctx = x_ctx[:, skip:]
                context_len = x_ctx.shape[1]
                cache.offset += skip
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(
            queries.reshape(batch, length, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(
            ctx_keys.reshape(batch, context_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(
            batch, context_len, self.n_kv_heads, -1
        ).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(
            prop_keys.reshape(batch, length, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(batch, length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        queries = rope(queries, offset=cache.offset + context_len)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + context_len)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        cached_len = keys.shape[2]
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        mask = None
        if self.is_sliding:
            mask = (
                "causal"
                if cached_len + length <= self.sliding_window
                else create_causal_mask(
                    length, offset=cached_len, window_size=self.sliding_window
                )
            )
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashDraftConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashDraftConfig):
        super().__init__()
        self.config = config
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = initialize_rope(
            dims=config.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.embed_tokens = None
        self.lm_head = None
        self.embed_scale = 1.0

    def bind(self, target_model: Any) -> None:
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        else:
            raise DFlashCompatibilityError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = getattr(
            self.embed_tokens, "embed_scale", getattr(inner, "embed_scale", 1.0)
        )
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self.embed_tokens.as_linear
        )

    def make_cache(self) -> list[Any]:
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                if self.config.sliding_window is None:
                    raise DFlashCompatibilityError(
                        "sliding_attention draft missing sliding_window"
                    )
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    def __call__(self, inputs, target_hidden, cache, logits_start: int = 0):
        h = self.embed_tokens(inputs) * self.embed_scale
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        if logits_start:
            h = h[:, logits_start:]
        logits = self.lm_head(self.norm(h))
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits


@dataclass(frozen=True)
class DFlashGenerationResponse:
    text: str
    tokens: list[int]
    accepted: int
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    finish_reason: str | None = None


def _resolve_draft_path(model_id_or_path: str, revision: str | None = None) -> Path:
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        return path
    return Path(
        snapshot_download(
            model_id_or_path,
            revision=revision,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )


def load_dflash_draft(
    model_id_or_path: str,
    *,
    revision: str | None = None,
) -> DFlashDraftModel:
    path = _resolve_draft_path(model_id_or_path, revision)
    cfg = DFlashDraftConfig.from_dict(json.loads((path / "config.json").read_text()))
    weights = {
        key: value
        for weight_file in path.glob("*.safetensors")
        for key, value in mx.load(str(weight_file)).items()
    }
    model = DFlashDraftModel(cfg)
    model.load_weights(list(weights.items()))
    return model


def _trim_recent_cache(cache: list[Any], num_tokens: int) -> None:
    if num_tokens <= 0:
        return
    for c in cache:
        n = min(getattr(c, "offset", num_tokens), num_tokens)
        if n <= 0:
            continue
        if isinstance(c, RotatingKVCache) and c.keys is not None:
            c.keys = c._temporal_order(c.keys)
            c.values = c._temporal_order(c.values)
            c.keys = c.keys[..., :-n, :]
            c.values = c.values[..., :-n, :]
            c.offset -= n
            c._idx = c.keys.shape[2]
        elif hasattr(c, "trim"):
            c.trim(n)


def _copy_cache_state_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_copy_cache_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_cache_state_value(item) for item in value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return mx.array(value)
    return value


def _snapshot_cache_state(cache: list[Any]) -> list[tuple[Any, Any | None]]:
    snapshot = []
    for c in cache:
        state = _copy_cache_state_value(getattr(c, "state", None))
        meta_state = getattr(c, "meta_state", None)
        snapshot.append((state, meta_state))
    return snapshot


def _restore_cache_state(
    cache: list[Any], snapshot: list[tuple[Any, Any | None]]
) -> None:
    for c, (state, meta_state) in zip(cache, snapshot):
        if state is not None:
            c.state = _copy_cache_state_value(state)
        if meta_state is not None and hasattr(c, "meta_state"):
            c.meta_state = meta_state


def _can_trim_prompt_cache(cache: list[Any]) -> bool:
    try:
        return can_trim_prompt_cache(cache)
    except AttributeError:
        return False


class DFlashSpeculativeDecoder:
    """Qwen 35B DFlash draft runner with aggregate acceptance stats."""

    def __init__(
        self,
        draft: DFlashDraftModel,
        *,
        draft_model_name: str,
        block_size: int | None = None,
    ):
        self.draft = draft
        self.draft_model_name = draft_model_name
        self.block_size = block_size or int(draft.config.block_size)
        self._draft_tokens = 0
        self._accepted_tokens = 0
        self._rejected_tokens = 0
        self._errors = 0
        self._acceptance_by_block: list[dict[str, int | float | None]] = []

    @classmethod
    def load_qwen35(
        cls,
        draft_model: str,
        *,
        target_model: Any,
        target_tokenizer: Any,
        block_size: int | None = None,
        revision: str = QWEN36_35B_A3B_DFLASH_REVISION,
    ) -> "DFlashSpeculativeDecoder":
        draft = load_dflash_draft(draft_model, revision=revision)
        draft.config.validate_qwen35_a3b_contract(
            target_model=target_model,
            target_tokenizer=target_tokenizer,
        )
        if block_size is not None and block_size != draft.config.block_size:
            raise DFlashCompatibilityError(
                f"Qwen 35B DFlash block_size must be {draft.config.block_size}; "
                f"got {block_size}"
            )
        draft.bind(target_model)
        return cls(draft, draft_model_name=draft_model, block_size=block_size)

    def snapshot_stats(self) -> dict[str, Any]:
        total = self._draft_tokens
        blocks = len(self._acceptance_by_block)
        return {
            "draft_model": self.draft_model_name,
            "block_size": self.block_size,
            "draft_tokens": self._draft_tokens,
            "accepted_tokens": self._accepted_tokens,
            "rejected_tokens": self._rejected_tokens,
            "errors": self._errors,
            "acceptance_rate": (self._accepted_tokens / total if total > 0 else None),
            "blocks": blocks,
            "avg_accepted_per_block": (
                self._accepted_tokens / blocks if blocks > 0 else None
            ),
            "acceptance_by_block": list(self._acceptance_by_block),
        }

    def _record_block(self, *, draft_count: int, accepted_count: int) -> None:
        self._draft_tokens += draft_count
        self._accepted_tokens += accepted_count
        self._rejected_tokens += draft_count - accepted_count
        self._acceptance_by_block.append(
            {
                "draft_tokens": draft_count,
                "accepted_tokens": accepted_count,
                "acceptance_rate": (
                    accepted_count / draft_count if draft_count > 0 else None
                ),
            }
        )

    def stream_generate(
        self,
        *,
        target_model: Any,
        tokenizer: Any,
        prompt: str | list[int] | mx.array,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        cancel_check: Any | None = None,
        **_: Any,
    ) -> Iterator[DFlashGenerationResponse]:
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
        tok = (
            tokenizer
            if isinstance(tokenizer, TokenizerWrapper)
            else TokenizerWrapper(tokenizer)
        )

        if not isinstance(prompt, mx.array):
            if isinstance(prompt, str):
                add_special_tokens = tok.bos_token is None or not prompt.startswith(
                    tok.bos_token
                )
                prompt = tok.encode(prompt, add_special_tokens=add_special_tokens)
            prompt = mx.array(prompt)

        detokenizer = tok.detokenizer
        eos_token_ids = set(getattr(tok, "eos_token_ids", None) or [])
        eos_token_id = getattr(tok, "eos_token_id", None)
        if eos_token_id is not None:
            eos_token_ids.add(eos_token_id)
        tokens = prompt.tolist()
        target_cache = make_prompt_cache(target_model)
        target_cache_can_trim = _can_trim_prompt_cache(target_cache)
        draft_cache = self.draft.make_cache()

        try:
            with DFlashLayerCapture(
                target_model, self.draft.config.target_layer_ids
            ) as capture:
                tic = time.perf_counter()
                logits = target_model(prompt[None], cache=target_cache)
                hidden = capture.concat_hidden_states()
                mx.eval(logits, hidden)
                if cancel_check is not None:
                    cancel_check()
                prompt_tps = prompt.size / max(time.perf_counter() - tic, 1e-9)

                start = time.perf_counter()
                token = sampler(logits[:, -1:])[0, 0].item()
                tokens.append(token)
                generated = 1
                detokenizer.add_token(token)
                if token in eos_token_ids:
                    detokenizer.finalize()
                    yield self._response(
                        detokenizer.last_segment,
                        [token],
                        0,
                        prompt.size,
                        prompt_tps,
                        generated,
                        start,
                        "stop",
                    )
                    return
                yield self._response(
                    detokenizer.last_segment,
                    [token],
                    0,
                    prompt.size,
                    prompt_tps,
                    generated,
                    start,
                )

                while generated < max_tokens:
                    if cancel_check is not None:
                        cancel_check()
                    bs = min(self.block_size, max_tokens - generated + 1)
                    if bs <= 1:
                        break

                    block = mx.array(
                        [[tokens[-1]] + [self.draft.config.mask_token_id] * (bs - 1)]
                    )
                    draft_logits = self.draft(
                        block,
                        hidden,
                        draft_cache,
                        logits_start=1,
                    )
                    trim_n = draft_cache[0].offset - (prompt.size + generated - 1)
                    if trim_n > 0:
                        _trim_recent_cache(draft_cache, trim_n)
                    draft_tokens = sampler(draft_logits)
                    mx.eval(draft_tokens)

                    capture.clear()
                    verify_snapshot = (
                        None
                        if target_cache_can_trim
                        else _snapshot_cache_state(target_cache)
                    )
                    verify_input = mx.concatenate(
                        [mx.array([[tokens[-1]]]), draft_tokens], axis=1
                    )
                    logits = target_model(verify_input, cache=target_cache)
                    hidden = capture.concat_hidden_states()
                    target_tokens = sampler(logits)
                    mx.eval(target_tokens, hidden)
                    if cancel_check is not None:
                        cancel_check()

                    d_list = draft_tokens[0].tolist()
                    t_list = target_tokens[0].tolist()
                    accepted = next(
                        (
                            i
                            for i, draft_tok in enumerate(d_list)
                            if draft_tok != t_list[i]
                        ),
                        len(d_list),
                    )
                    self._record_block(
                        draft_count=len(d_list),
                        accepted_count=accepted,
                    )

                    new_tokens = d_list[:accepted] + [t_list[accepted]]
                    new_tokens = new_tokens[: max_tokens - generated]

                    eos_idx = next(
                        (
                            i
                            for i, token_id in enumerate(new_tokens)
                            if token_id in eos_token_ids
                        ),
                        None,
                    )
                    finish_reason = None
                    if eos_idx is not None:
                        new_tokens = new_tokens[: eos_idx + 1]
                        finish_reason = "stop"

                    commit_len = len(new_tokens)
                    trim = bs - commit_len
                    if trim > 0:
                        if target_cache_can_trim:
                            _trim_recent_cache(target_cache, trim)
                            hidden = hidden[:, :commit_len, :]
                        else:
                            if verify_snapshot is None:
                                raise DFlashCompatibilityError(
                                    "DFlash target cache snapshot missing for "
                                    "non-trimmable rollback"
                                )
                            _restore_cache_state(target_cache, verify_snapshot)
                            capture.clear()
                            logits = target_model(
                                verify_input[:, :commit_len],
                                cache=target_cache,
                            )
                            hidden = capture.concat_hidden_states()
                            mx.eval(logits, hidden)
                    else:
                        hidden = hidden[:, :commit_len, :]

                    for token_id in new_tokens:
                        detokenizer.add_token(token_id)
                    tokens.extend(new_tokens)
                    generated += len(new_tokens)

                    if generated % 256 == 0:
                        mx.clear_cache()

                    yield self._response(
                        detokenizer.last_segment,
                        new_tokens,
                        accepted,
                        prompt.size,
                        prompt_tps,
                        generated,
                        start,
                        finish_reason,
                    )

                    if finish_reason is not None:
                        detokenizer.finalize()
                        return

                detokenizer.finalize()
                yield self._response(
                    detokenizer.last_segment,
                    [],
                    0,
                    prompt.size,
                    prompt_tps,
                    generated,
                    start,
                    "length",
                )
        except Exception:
            self._errors += 1
            raise

    def _response(
        self,
        text: str,
        tokens: list[int],
        accepted: int,
        prompt_tokens: int,
        prompt_tps: float,
        generation_tokens: int,
        start: float,
        finish_reason: str | None = None,
    ) -> DFlashGenerationResponse:
        return DFlashGenerationResponse(
            text=text,
            tokens=tokens,
            accepted=accepted,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            prompt_tps=prompt_tps,
            generation_tps=generation_tokens / max(time.perf_counter() - start, 1e-9),
            peak_memory_gb=mx.get_peak_memory() / 1e9,
            finish_reason=finish_reason,
        )
