"""Narrow localhost-only Thump seam for the qwen_code coexistence pilot."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_vlm.generate import PromptCacheState

from .session import SessionSubstrate, model_id_hash_for_path

logger = logging.getLogger(__name__)

FEATURE_FLAG_ENV = "VLLM_MLX_ENABLE_THUMP_QWEN_PILOT"
ARTIFACT_ROOT_ENV = "VLLM_MLX_THUMP_QWEN_PILOT_ARTIFACT_ROOT"
DEFAULT_BLOCK_SIZE_TOKENS = 16
DEFAULT_ARTIFACT_ROOT = Path("/Volumes/Lexar/ai-runtime-run")
SESSION_MANIFEST_NAME = "session.tsmf"
PROMPT_STATE_NAME = "prompt-state.json"


def qwen_pilot_enabled() -> bool:
    """Return whether the narrow qwen_code pilot is enabled."""
    return os.environ.get(FEATURE_FLAG_ENV) == "1"


def qwen_pilot_supported(model: Any) -> bool:
    """Limit the pilot to the current Gemma lane only."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return model_type in {"gemma4", "gemma4_text"}


def _stable_u64(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode(), digest_size=8).digest(),
        "little",
    )


def _artifact_bytes(root_dir: Path) -> int:
    return sum(path.stat().st_size for path in root_dir.iterdir() if path.is_file())


def qwen_pilot_artifact_root() -> Path:
    return Path(
        os.environ.get(ARTIFACT_ROOT_ENV, str(DEFAULT_ARTIFACT_ROOT))
    ).expanduser().resolve(strict=False)


def resolve_qwen_pilot_artifact_dir(artifact_path: str | Path) -> Path:
    artifact_dir = Path(artifact_path).expanduser()
    if not artifact_dir.is_absolute():
        raise PermissionError("artifact_path must be an absolute path")

    resolved_root = qwen_pilot_artifact_root()
    resolved_artifact = artifact_dir.resolve(strict=False)
    if not resolved_artifact.is_relative_to(resolved_root):
        raise PermissionError(
            f"artifact_path must stay under {resolved_root}"
        )
    return resolved_artifact


@dataclass
class LatestFinishedPromptState:
    token_ids: list[int]
    prompt_cache: list[Any]
    prompt_tokens: int
    completion_tokens: int
    route: str
    captured_at_s: float


@dataclass
class ArmedRestoreState:
    prompt_cache_state: PromptCacheState
    artifact_path: str
    cache_token_count: int
    qwen_session_id: str | None
    workspace_path: str | None
    armed_at_s: float


class QwenPilotManager:
    """Owns the exact hot-restart artifact + one-shot restored prefix cache."""

    def __init__(
        self,
        *,
        language_model: Any,
        model_path: str,
        block_size_tokens: int = DEFAULT_BLOCK_SIZE_TOKENS,
        thump_lib_path: str | Path | None = None,
    ) -> None:
        self.language_model = language_model
        self.model_path = model_path
        self.block_size_tokens = block_size_tokens
        self.thump_lib_path = thump_lib_path
        self.model_id_hash = model_id_hash_for_path(model_path)
        self._lock = threading.Lock()
        self._latest_finished: LatestFinishedPromptState | None = None
        self._armed_restore: ArmedRestoreState | None = None
        self._last_consume_status: dict[str, Any] | None = None

    def status(self) -> dict[str, Any]:
        """Summarize current in-memory pilot state for local debugging."""
        with self._lock:
            latest = self._latest_finished
            armed = self._armed_restore
            consume = self._last_consume_status
            return {
                "enabled": True,
                "supported_model": qwen_pilot_supported(self.language_model),
                "model_path": self.model_path,
                "latest_finished": (
                    None
                    if latest is None
                    else {
                        "route": latest.route,
                        "prompt_tokens": latest.prompt_tokens,
                        "completion_tokens": latest.completion_tokens,
                        "cache_token_count": len(latest.token_ids),
                        "captured_at_s": latest.captured_at_s,
                    }
                ),
                "armed_restore": (
                    None
                    if armed is None
                    else {
                        "artifact_path": armed.artifact_path,
                        "cache_token_count": armed.cache_token_count,
                        "qwen_session_id": armed.qwen_session_id,
                        "workspace_path": armed.workspace_path,
                        "armed_at_s": armed.armed_at_s,
                    }
                ),
                "last_consume_status": consume,
            }

    def record_finished_prompt_state(
        self,
        prompt_cache_state: PromptCacheState | None,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        route: str,
    ) -> None:
        """Capture the latest completed text-only chat turn for checkpointing."""
        if prompt_cache_state is None:
            return

        token_ids = list(prompt_cache_state.token_ids or [])
        prompt_cache = prompt_cache_state.cache
        if not token_ids or prompt_cache is None:
            return

        with self._lock:
            self._latest_finished = LatestFinishedPromptState(
                token_ids=token_ids,
                prompt_cache=prompt_cache,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                route=route,
                captured_at_s=time.time(),
            )

        logger.info(
            "Thump qwen pilot captured %s state (%d tokens, prompt=%d, completion=%d)",
            route,
            len(token_ids),
            prompt_tokens,
            completion_tokens,
        )

    def build_request_prompt_cache_state(
        self,
        token_ids: list[int],
    ) -> PromptCacheState:
        """Return a prompt-cache state for the next request, consuming one-shot restore."""
        fresh_state = PromptCacheState()
        with self._lock:
            armed = self._armed_restore
            if armed is None:
                return fresh_state

            prefix_len = armed.prompt_cache_state.find_prefix_length(token_ids)
            if 0 < prefix_len < len(token_ids):
                self._armed_restore = None
                self._last_consume_status = {
                    "status": "restore_match",
                    "artifact_path": armed.artifact_path,
                    "matched_prefix_tokens": prefix_len,
                    "cache_token_count": armed.cache_token_count,
                    "qwen_session_id": armed.qwen_session_id,
                    "workspace_path": armed.workspace_path,
                    "consumed_at_s": time.time(),
                }
                logger.info(
                    "Thump qwen pilot matched %d-token prefix from %s",
                    prefix_len,
                    armed.artifact_path,
                )
                return armed.prompt_cache_state

            fallback_reason = (
                "no_new_suffix" if prefix_len == len(token_ids) else "prompt_prefix_mismatch"
            )
            self._armed_restore = None
            self._last_consume_status = {
                "status": "cold_fallback",
                "artifact_path": armed.artifact_path,
                "matched_prefix_tokens": prefix_len,
                "fallback_reason": fallback_reason,
                "qwen_session_id": armed.qwen_session_id,
                "workspace_path": armed.workspace_path,
                "consumed_at_s": time.time(),
            }

        logger.warning(
            "Thump qwen pilot restore fell back cold (%s, matched=%d)",
            fallback_reason,
            prefix_len,
        )
        return fresh_state

    def checkpoint_latest_finished(
        self,
        artifact_path: str | Path,
        *,
        qwen_session_id: str | None = None,
        workspace_path: str | None = None,
    ) -> dict[str, Any]:
        """Checkpoint the latest completed text-only prompt cache into a Thump artifact."""
        with self._lock:
            latest = self._latest_finished

        if latest is None:
            raise ValueError("no finished qwen pilot state is available to checkpoint")

        artifact_dir = resolve_qwen_pilot_artifact_dir(artifact_path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact_dir / SESSION_MANIFEST_NAME
        prompt_state_path = artifact_dir / PROMPT_STATE_NAME

        cache_token_count = len(latest.token_ids)
        if cache_token_count <= 0:
            raise ValueError("cannot checkpoint an empty prompt cache state")

        block_capacity = max(
            8,
            math.ceil(cache_token_count / self.block_size_tokens) + 8,
        )
        sequence_id = _stable_u64(
            f"{qwen_session_id or 'pilot'}:{workspace_path or ''}:{time.time_ns()}"
        )
        session_id = _stable_u64(qwen_session_id or "qwen-pilot")

        start = time.perf_counter()
        session = SessionSubstrate.from_gemma4_model(
            self.language_model,
            block_size_tokens=self.block_size_tokens,
            block_capacity=block_capacity,
            root_dir=artifact_dir,
            lib_path=self.thump_lib_path,
            exact_hot_restart=True,
            model_path=self.model_path,
        )
        try:
            session.initialize_from_live_cache(
                latest.prompt_cache,
                total_tokens=cache_token_count,
            )
            checkpoint = session.checkpoint(
                manifest_path,
                model_id_hash=self.model_id_hash,
                session_id=session_id,
                sequence_id=sequence_id,
                prompt_tokens=cache_token_count,
                generated_tokens=0,
            )
        finally:
            session.close()

        prompt_state_payload = {
            "format_version": 1,
            "model_path": self.model_path,
            "model_id_hash": self.model_id_hash,
            "route": latest.route,
            "prompt_tokens": latest.prompt_tokens,
            "completion_tokens": latest.completion_tokens,
            "cache_token_count": cache_token_count,
            "token_ids": latest.token_ids,
            "qwen_session_id": qwen_session_id,
            "workspace_path": workspace_path,
            "session_id": session_id,
            "sequence_id": sequence_id,
            "block_size_tokens": self.block_size_tokens,
            "captured_at_s": latest.captured_at_s,
            "checkpoint_manifest_path": str(manifest_path),
        }
        prompt_state_path.write_text(
            json.dumps(prompt_state_payload, separators=(",", ":")),
            encoding="utf-8",
        )

        artifact_size_bytes = _artifact_bytes(artifact_dir)
        checkpoint_latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "artifact_path": str(artifact_dir),
            "manifest_path": str(manifest_path),
            "prompt_state_path": str(prompt_state_path),
            "checkpoint_latency_ms": checkpoint_latency_ms,
            "artifact_size_bytes": artifact_size_bytes,
            "block_size_tokens": self.block_size_tokens,
            "block_capacity": block_capacity,
            "cache_token_count": cache_token_count,
            "prompt_tokens": latest.prompt_tokens,
            "completion_tokens": latest.completion_tokens,
            "qwen_session_id": qwen_session_id,
            "workspace_path": workspace_path,
            "session_id": checkpoint.session_id,
            "sequence_id": checkpoint.sequence_id,
        }

    def arm_restore(
        self,
        artifact_path: str | Path,
    ) -> dict[str, Any]:
        """Validate + materialize a checkpoint into a one-shot prompt-cache restore."""
        artifact_dir = resolve_qwen_pilot_artifact_dir(artifact_path)
        prompt_state_path = artifact_dir / PROMPT_STATE_NAME
        manifest_path = artifact_dir / SESSION_MANIFEST_NAME
        payload = json.loads(prompt_state_path.read_text(encoding="utf-8"))
        token_ids = [int(token_id) for token_id in payload["token_ids"]]
        cache_token_count = int(payload.get("cache_token_count", len(token_ids)))
        qwen_session_id = payload.get("qwen_session_id")
        workspace_path = payload.get("workspace_path")

        start = time.perf_counter()
        try:
            session, _checkpoint = SessionSubstrate.attach_gemma4_checkpoint(
                self.language_model,
                manifest_path,
                block_size_tokens=int(
                    payload.get("block_size_tokens", self.block_size_tokens)
                ),
                lib_path=self.thump_lib_path,
                expected_model_id_hash=self.model_id_hash,
                require_exact_hot_restart=True,
                model_path=self.model_path,
            )
            try:
                cache = session.materialize_prompt_cache(
                    self.language_model,
                    upto_tokens=cache_token_count,
                )
            finally:
                session.close()

            prompt_cache_state = PromptCacheState()
            prompt_cache_state.update(token_ids, cache)
            armed = ArmedRestoreState(
                prompt_cache_state=prompt_cache_state,
                artifact_path=str(artifact_dir),
                cache_token_count=cache_token_count,
                qwen_session_id=qwen_session_id,
                workspace_path=workspace_path,
                armed_at_s=time.time(),
            )
            with self._lock:
                self._armed_restore = armed
                self._last_consume_status = None
            restore_latency_ms = (time.perf_counter() - start) * 1000.0
            return {
                "artifact_path": str(artifact_dir),
                "restore_validate_materialize_latency_ms": restore_latency_ms,
                "fallback_count": 0,
                "fallback_rate": 0.0,
                "fallback_reason": None,
                "restore_mode": "thump_hot_restart",
                "qwen_session_id": qwen_session_id,
                "workspace_path": workspace_path,
            }
        except Exception as exc:
            with self._lock:
                self._armed_restore = None
                self._last_consume_status = {
                    "status": "restore_validation_failed",
                    "artifact_path": str(artifact_dir),
                    "fallback_reason": str(exc),
                    "qwen_session_id": qwen_session_id,
                    "workspace_path": workspace_path,
                    "consumed_at_s": time.time(),
                }
            restore_latency_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("Thump qwen pilot restore validation failed: %s", exc)
            return {
                "artifact_path": str(artifact_dir),
                "restore_validate_materialize_latency_ms": restore_latency_ms,
                "fallback_count": 1,
                "fallback_rate": 1.0,
                "fallback_reason": str(exc),
                "restore_mode": "cold_fallback",
                "qwen_session_id": qwen_session_id,
                "workspace_path": workspace_path,
            }
