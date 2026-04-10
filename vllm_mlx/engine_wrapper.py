# venv/lib/python3.12/site-packages/vllm_mlx/engine_wrapper.py
"""Thin abstraction over mlx-lm BatchGenerator.

Decouples the TextBatchScheduler from BatchGenerator internals so that
upstream API changes (e.g. class renames, signature changes) only affect
this wrapper, not the scheduler logic.
"""

from typing import Any, Callable, List, Optional


class EngineWrapper:
    """Wrap a BatchGenerator with a stable interface."""

    def __init__(self, batch_gen):
        self._bg = batch_gen

    def _instance_attr(self, name: str) -> Any:
        """Read instance-backed internals without triggering mock auto-creation."""
        try:
            attrs = vars(self._bg)
        except TypeError:
            attrs = {}
        if name in attrs:
            return attrs[name]
        if hasattr(type(self._bg), name):
            return getattr(self._bg, name)
        return None

    @staticmethod
    def _safe_len(value: Any) -> int:
        if value is None:
            return 0
        try:
            return len(value)
        except TypeError:
            pass

        uids = getattr(value, "uids", None)
        if uids is not None:
            try:
                return len(uids)
            except TypeError:
                pass

        return 0

    def has_work(self) -> bool:
        """True if generation or prompt processing is pending."""
        try:
            return len(self._bg) > 0
        except (AttributeError, TypeError):
            pass

        return any(
            self._safe_len(self._instance_attr(name)) > 0
            for name in (
                "_unprocessed_sequences",
                "_currently_processing",
                "_prompt_batch",
                "_generation_batch",
                "active_batch",
            )
        )

    def step(self) -> tuple:
        """Run one BatchGenerator step.

        Returns:
            (prompt_responses, generation_responses) tuple.
        """
        return self._bg.next()

    def insert(
        self,
        prompts: List[List[int]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable]] = None,
        logits_processors: Optional[List[List[Callable]]] = None,
        state_machines: Optional[List[Any]] = None,
    ) -> List[int]:
        """Insert sequences into the batch. Returns UIDs."""
        return self._bg.insert(
            prompts,
            max_tokens=max_tokens,
            caches=caches,
            all_tokens=all_tokens,
            samplers=samplers,
            logits_processors=logits_processors,
            state_machines=state_machines,
        )

    def insert_segments(
        self,
        segments: List[List[List[int]]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable]] = None,
        logits_processors: Optional[List[List[Callable]]] = None,
        state_machines: Optional[List[Any]] = None,
    ) -> List[int]:
        """Insert segmented prompt sequences into the batch. Returns UIDs."""
        return self._bg.insert_segments(
            segments,
            max_tokens=max_tokens,
            caches=caches,
            all_tokens=all_tokens,
            samplers=samplers,
            logits_processors=logits_processors,
            state_machines=state_machines,
        )

    def remove(self, uids: List[int]):
        """Remove sequences from the batch."""
        self._bg.remove(uids)

    def extract_cache(self, uids: List[int]) -> dict:
        """Extract caches for prefix cache storage."""
        return self._bg.extract_cache(uids)

    def close(self):
        """Close the underlying BatchGenerator."""
        self._bg.close()
