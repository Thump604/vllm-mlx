"""Offline Gemma 4 Thump session restart-recovery harness."""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from vllm_mlx.specprefill import cleanup_rope
from vllm_mlx.utils.tokenizer import _load_strict_false

from .capture import (
    CaptureCollector,
    LayerCapture,
    capture_into,
    install_gemma4_capture_patch,
)
from .session import SessionCheckpoint, SessionSubstrate, model_id_hash_for_path

FEATURE_FLAG_ENV = "VLLM_MLX_ENABLE_THUMP_SESSION_RECOVERY"


def session_recovery_enabled() -> bool:
    return os.environ.get(FEATURE_FLAG_ENV) == "1"


def _sampler(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def _current_rss_bytes() -> int:
    proc = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        check=True,
        text=True,
        capture_output=True,
    )
    return int(proc.stdout.strip()) * 1024


def _prefill_tokens(
    model: Any,
    tokens: list[int],
    cache: list[Any],
    *,
    prefill_step_size: int,
    collector: CaptureCollector | None = None,
) -> None:
    if not tokens:
        return
    prompt = mx.array(tokens, dtype=mx.int32)
    offset = 0
    while offset < len(tokens):
        chunk = min(prefill_step_size, len(tokens) - offset)
        context = capture_into(collector) if collector is not None else nullcontext()
        with context:
            model(prompt[offset : offset + chunk][None], cache=cache)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
        offset += chunk
        mx.clear_cache()


def _replay_tokens_exact(
    model: Any,
    tokens: list[int],
    cache: list[Any],
    *,
    collector: CaptureCollector | None = None,
) -> None:
    if not tokens:
        return
    for token_id in tokens:
        context = capture_into(collector) if collector is not None else nullcontext()
        with context:
            model(mx.array([int(token_id)], dtype=mx.int32)[None], cache=cache)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
        mx.clear_cache()


def _reconstruct_tokens(
    model: Any,
    tokens: list[int],
    cache: list[Any],
    *,
    prefill_step_size: int,
    exact_replay: bool = False,
    collector: CaptureCollector | None = None,
) -> None:
    if exact_replay:
        _replay_tokens_exact(model, tokens, cache, collector=collector)
        return
    _prefill_tokens(
        model,
        tokens,
        cache,
        prefill_step_size=prefill_step_size,
        collector=collector,
    )


def _merge_captures(
    left: dict[int, LayerCapture],
    right: dict[int, LayerCapture],
) -> dict[int, LayerCapture]:
    out: dict[int, LayerCapture] = {}
    for layer_idx in sorted(set(left) | set(right)):
        left_cap = left.get(layer_idx)
        right_cap = right.get(layer_idx)
        if left_cap is None:
            out[layer_idx] = right_cap
            continue
        if right_cap is None:
            out[layer_idx] = left_cap
            continue
        out[layer_idx] = LayerCapture(
            keys=np.concatenate([left_cap.keys, right_cap.keys], axis=0),
            values=np.concatenate([left_cap.values, right_cap.values], axis=0),
        )
    return out


def _decode_tokens(
    model: Any,
    tokenizer: Any,
    cache: list[Any],
    last_token: int,
    *,
    max_new_tokens: int,
    collector: CaptureCollector | None = None,
) -> tuple[float, str, list[int]]:
    output_tokens: list[int] = []
    continuation_ms: float | None = None
    decode_start = time.perf_counter()
    eos_ids = getattr(tokenizer, "eos_token_id", None)
    eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    input_token = int(last_token)
    for _ in range(max_new_tokens):
        context = capture_into(collector) if collector is not None else nullcontext()
        with context:
            logits = model(mx.array([input_token], dtype=mx.int32)[None], cache=cache)
        logits = logits[:, -1, :]
        token = _sampler(logits)
        mx.eval(token)
        mx.eval([c.state for c in cache if hasattr(c, "state")])
        token_id = int(token.item())
        output_tokens.append(token_id)
        if continuation_ms is None:
            continuation_ms = (time.perf_counter() - decode_start) * 1000.0
        if token_id in eos_set:
            break
        input_token = token_id
        mx.clear_cache()
    return continuation_ms or 0.0, tokenizer.decode(output_tokens), output_tokens


@dataclass
class SessionRecoveryTrace:
    name: str
    prompt_text: str
    seed_new_tokens: int = 16
    continue_new_tokens: int = 32
    description: str = ""
    prefill_step_size: int = 128
    capture_step_size: int | None = None
    exact_replay: bool = False
    exact_hot_restart: bool = False

    @classmethod
    def from_path(cls, path: str | Path) -> "SessionRecoveryTrace":
        payload = json.loads(Path(path).read_text())
        if "prompt_text" not in payload:
            payload = {
                "name": payload["name"],
                "description": payload.get("description", ""),
                "prompt_text": (
                    payload["prefix_text"]
                    + payload.get("insert_text", "")
                    + payload["suffix_text"]
                ),
                "prefill_step_size": payload.get("prefill_step_size", 128),
                "capture_step_size": payload.get("capture_step_size"),
            }
        return cls(**payload)


@dataclass
class CheckpointArtifact:
    trace_name: str
    model_path: str
    manifest_path: str
    root_dir: str
    prompt_tokens: list[int]
    seed_tokens: list[int]
    prompt_token_count: int
    generated_tokens: int
    context_tokens: int
    artifact_size_bytes: int
    model_id_hash: int
    session_id: int
    sequence_id: int
    block_size_tokens: int
    prefill_step_size: int
    capture_step_size: int
    exact_hot_restart: bool = False
    checkpoint_latency_ms: float = 0.0
    checkpoint_rss_telemetry: dict[str, int | float] | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "CheckpointArtifact":
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class RecoveryRunResult:
    variant: str
    validate_latency_ms: float
    validate_materialize_latency_ms: float
    cold_rebuild_latency_ms: float
    continuation_latency_ms: float
    output_text: str
    output_tokens: list[int]
    fallback_count: int
    fallback_reason: str | None
    session_total_ms: float


@dataclass
class RecoveryComparison:
    model_path: str
    trace_name: str
    feature_flag_env: str
    checkpoint: dict[str, Any]
    restore: dict[str, Any]
    cold_rebuild: dict[str, Any]
    exact_fidelity: bool
    restore_meaningfully_avoids_rebuild: bool
    fallback_rate: float
    go_no_go: str
    telemetry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SessionRecoveryRunner:
    def __init__(
        self,
        model_path: str | Path,
        *,
        thump_lib_path: str | Path | None = None,
        block_size_tokens: int = 1,
    ) -> None:
        install_gemma4_capture_patch()
        outer_model, tokenizer = _load_strict_false(
            str(model_path),
            {"trust_remote_code": True},
        )
        self.outer_model = outer_model
        self.model = outer_model.language_model
        self.tokenizer = tokenizer
        self.model_path = str(model_path)
        self.thump_lib_path = thump_lib_path
        self.block_size_tokens = block_size_tokens
        self.model_id_hash = model_id_hash_for_path(model_path)

    def build_prompt_tokens(self, trace: SessionRecoveryTrace) -> list[int]:
        tokens = list(self.tokenizer.encode(trace.prompt_text))
        if not tokens:
            raise ValueError("prompt_text must produce at least one token")
        return tokens

    def create_checkpoint(
        self,
        trace: SessionRecoveryTrace,
        *,
        bundle_dir: str | Path,
    ) -> CheckpointArtifact:
        prompt_tokens = self.build_prompt_tokens(trace)
        prefix_tokens = prompt_tokens[:-1]
        last_prompt_token = prompt_tokens[-1]
        capture_step_size = trace.capture_step_size or trace.prefill_step_size
        checkpoint_rss_telemetry: dict[str, int | float] = {
            "rss_before_prefill_bytes": _current_rss_bytes(),
        }

        prefix_cache = self.model.make_cache()
        prefix_collector = None if trace.exact_hot_restart else CaptureCollector()
        prefix_prefill_start = time.perf_counter()
        _reconstruct_tokens(
            self.model,
            prefix_tokens,
            prefix_cache,
            prefill_step_size=capture_step_size,
            exact_replay=trace.exact_replay,
            collector=prefix_collector,
        )
        checkpoint_rss_telemetry["prefix_prefill_latency_ms"] = (
            time.perf_counter() - prefix_prefill_start
        ) * 1000.0
        checkpoint_rss_telemetry["rss_after_prefix_prefill_bytes"] = _current_rss_bytes()
        seed_collector = None if trace.exact_hot_restart else CaptureCollector()
        seed_decode_start = time.perf_counter()
        _continuation_ms, _seed_text, seed_tokens = _decode_tokens(
            self.model,
            self.tokenizer,
            prefix_cache,
            last_prompt_token,
            max_new_tokens=trace.seed_new_tokens,
            collector=seed_collector,
        )
        checkpoint_rss_telemetry["seed_decode_latency_ms"] = (
            time.perf_counter() - seed_decode_start
        ) * 1000.0
        checkpoint_rss_telemetry["rss_after_seed_decode_bytes"] = _current_rss_bytes()
        if not seed_tokens:
            raise RuntimeError("seed decode produced no tokens")

        full_capture: dict[int, LayerCapture] | None = None
        capture_merge_start = time.perf_counter()
        if prefix_collector is not None and seed_collector is not None:
            full_capture = _merge_captures(
                prefix_collector.joined(), seed_collector.joined()
            )
        checkpoint_rss_telemetry["capture_merge_latency_ms"] = (
            time.perf_counter() - capture_merge_start
        ) * 1000.0
        checkpoint_rss_telemetry["rss_after_capture_merge_bytes"] = _current_rss_bytes()
        context_tokens = len(prompt_tokens) + len(seed_tokens) - 1
        total_blocks = max(
            8,
            int((context_tokens + self.block_size_tokens - 1) / self.block_size_tokens)
            + 8,
        )
        checkpoint_start = time.perf_counter()
        session_init_start = time.perf_counter()
        session = SessionSubstrate.from_gemma4_model(
            self.model,
            block_size_tokens=self.block_size_tokens,
            block_capacity=total_blocks,
            root_dir=bundle_dir,
            lib_path=self.thump_lib_path,
            exact_hot_restart=trace.exact_hot_restart,
        )
        checkpoint_rss_telemetry["session_init_latency_ms"] = (
            time.perf_counter() - session_init_start
        ) * 1000.0
        checkpoint_rss_telemetry["rss_after_session_init_bytes"] = _current_rss_bytes()
        capture_write_start = time.perf_counter()
        if trace.exact_hot_restart:
            session.initialize_from_live_cache(prefix_cache, total_tokens=context_tokens)
        else:
            session.initialize_from_capture(full_capture, total_tokens=context_tokens)
        checkpoint_rss_telemetry["capture_write_latency_ms"] = (
            time.perf_counter() - capture_write_start
        ) * 1000.0
        checkpoint_rss_telemetry["rss_after_capture_write_bytes"] = _current_rss_bytes()
        session_id = int(time.time_ns() & 0xFFFFFFFFFFFFFFFF)
        sequence_id = int((time.time_ns() ^ os.getpid()) & 0xFFFFFFFFFFFFFFFF)
        manifest_path = Path(bundle_dir) / "session.tsmf"
        checkpoint_write_start = time.perf_counter()
        checkpoint = session.checkpoint(
            manifest_path,
            model_id_hash=self.model_id_hash,
            session_id=session_id,
            sequence_id=sequence_id,
            prompt_tokens=len(prompt_tokens),
            generated_tokens=len(seed_tokens),
        )
        checkpoint_rss_telemetry["checkpoint_write_latency_ms"] = (
            time.perf_counter() - checkpoint_write_start
        ) * 1000.0
        checkpoint_rss_telemetry["rss_after_checkpoint_write_bytes"] = _current_rss_bytes()
        checkpoint_latency_ms = (time.perf_counter() - checkpoint_start) * 1000.0
        session.close()
        checkpoint_rss_telemetry["rss_after_session_close_bytes"] = _current_rss_bytes()
        checkpoint_rss_telemetry["checkpoint_peak_rss_bytes"] = max(
            int(value)
            for key, value in checkpoint_rss_telemetry.items()
            if key.startswith("rss_")
        )
        cleanup_rope(self.model)
        return CheckpointArtifact(
            trace_name=trace.name,
            model_path=self.model_path,
            manifest_path=str(checkpoint.manifest_path),
            root_dir=str(checkpoint.root_dir),
            prompt_tokens=prompt_tokens,
            seed_tokens=seed_tokens,
            prompt_token_count=len(prompt_tokens),
            generated_tokens=len(seed_tokens),
            context_tokens=checkpoint.context_tokens,
            artifact_size_bytes=checkpoint.artifact_bytes,
            model_id_hash=checkpoint.model_id_hash,
            session_id=checkpoint.session_id,
            sequence_id=checkpoint.sequence_id,
            block_size_tokens=self.block_size_tokens,
            prefill_step_size=trace.prefill_step_size,
            capture_step_size=capture_step_size,
            exact_hot_restart=trace.exact_hot_restart,
            checkpoint_latency_ms=checkpoint_latency_ms,
            checkpoint_rss_telemetry=checkpoint_rss_telemetry,
        )

    def restore_and_continue(
        self,
        trace: SessionRecoveryTrace,
        artifact: CheckpointArtifact,
    ) -> RecoveryRunResult:
        if not session_recovery_enabled():
            fallback = self.cold_rebuild_and_continue(trace, artifact)
            fallback.variant = "restore_fallback"
            fallback.fallback_count = 1
            fallback.fallback_reason = f"{FEATURE_FLAG_ENV} is disabled"
            return fallback

        session_start = time.perf_counter()
        try:
            validate_start = time.perf_counter()
            session, _checkpoint = SessionSubstrate.attach_gemma4_checkpoint(
                self.model,
                artifact.manifest_path,
                block_size_tokens=artifact.block_size_tokens,
                lib_path=self.thump_lib_path,
                expected_model_id_hash=artifact.model_id_hash,
                require_exact_hot_restart=artifact.exact_hot_restart,
            )
            validate_latency_ms = (time.perf_counter() - validate_start) * 1000.0
            cache = session.materialize_prompt_cache(
                self.model,
                upto_tokens=artifact.context_tokens,
            )
            validate_materialize_latency_ms = (
                time.perf_counter() - validate_start
            ) * 1000.0
            continuation_ms, output_text, output_tokens = _decode_tokens(
                self.model,
                self.tokenizer,
                cache,
                artifact.seed_tokens[-1],
                max_new_tokens=trace.continue_new_tokens,
            )
            session.close()
            cleanup_rope(self.model)
            return RecoveryRunResult(
                variant="restore",
                validate_latency_ms=validate_latency_ms,
                validate_materialize_latency_ms=validate_materialize_latency_ms,
                cold_rebuild_latency_ms=0.0,
                continuation_latency_ms=continuation_ms,
                output_text=output_text,
                output_tokens=output_tokens,
                fallback_count=0,
                fallback_reason=None,
                session_total_ms=(time.perf_counter() - session_start) * 1000.0,
            )
        except Exception as exc:
            fallback = self.cold_rebuild_and_continue(trace, artifact)
            fallback.variant = "restore_fallback"
            fallback.fallback_count = 1
            fallback.fallback_reason = str(exc)
            return fallback

    def cold_rebuild_and_continue(
        self,
        trace: SessionRecoveryTrace,
        artifact: CheckpointArtifact,
    ) -> RecoveryRunResult:
        session_start = time.perf_counter()
        cache = self.model.make_cache()
        rebuild_start = time.perf_counter()
        _reconstruct_tokens(
            self.model,
            artifact.prompt_tokens + artifact.seed_tokens[:-1],
            cache,
            prefill_step_size=trace.prefill_step_size,
            exact_replay=trace.exact_replay,
        )
        rebuild_latency_ms = (time.perf_counter() - rebuild_start) * 1000.0
        continuation_ms, output_text, output_tokens = _decode_tokens(
            self.model,
            self.tokenizer,
            cache,
            artifact.seed_tokens[-1],
            max_new_tokens=trace.continue_new_tokens,
        )
        cleanup_rope(self.model)
        return RecoveryRunResult(
            variant="cold_rebuild",
            validate_latency_ms=0.0,
            validate_materialize_latency_ms=0.0,
            cold_rebuild_latency_ms=rebuild_latency_ms,
            continuation_latency_ms=continuation_ms,
            output_text=output_text,
            output_tokens=output_tokens,
            fallback_count=0,
            fallback_reason=None,
            session_total_ms=(time.perf_counter() - session_start) * 1000.0,
        )


def build_recovery_comparison(
    trace: SessionRecoveryTrace,
    artifact: CheckpointArtifact,
    restore: RecoveryRunResult,
    cold_rebuild: RecoveryRunResult,
) -> RecoveryComparison:
    exact_fidelity = restore.output_tokens == cold_rebuild.output_tokens
    fallback_rate = restore.fallback_count / 1.0
    avoids_rebuild = (
        restore.fallback_count == 0
        and exact_fidelity
        and restore.validate_materialize_latency_ms
        < cold_rebuild.cold_rebuild_latency_ms
    )
    go_no_go = "GO" if avoids_rebuild else "NO_GO"
    telemetry = {
        "checkpoint_latency_ms": artifact.checkpoint_latency_ms,
        "checkpoint_rss_telemetry": artifact.checkpoint_rss_telemetry or {},
        "checkpoint_peak_rss_bytes": (
            (artifact.checkpoint_rss_telemetry or {}).get("checkpoint_peak_rss_bytes")
        ),
        "restore_mode": restore.variant,
        "restore_validate_latency_ms": restore.validate_latency_ms,
        "restore_validate_materialize_latency_ms": restore.validate_materialize_latency_ms,
        "cold_rebuild_latency_ms": cold_rebuild.cold_rebuild_latency_ms,
        "artifact_size_bytes": artifact.artifact_size_bytes,
        "exact_hot_restart": artifact.exact_hot_restart,
        "prompt_token_count": artifact.prompt_token_count,
        "context_tokens": artifact.context_tokens,
        "generated_tokens": artifact.generated_tokens,
        "exact_fidelity": exact_fidelity,
        "fallback_count": restore.fallback_count,
        "fallback_rate": fallback_rate,
        "fallback_reason": restore.fallback_reason,
    }
    return RecoveryComparison(
        model_path=artifact.model_path,
        trace_name=trace.name,
        feature_flag_env=FEATURE_FLAG_ENV,
        checkpoint=asdict(artifact),
        restore=asdict(restore),
        cold_rebuild=asdict(cold_rebuild),
        exact_fidelity=exact_fidelity,
        restore_meaningfully_avoids_rebuild=avoids_rebuild,
        fallback_rate=fallback_rate,
        go_no_go=go_no_go,
        telemetry=telemetry,
    )
