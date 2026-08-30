# SPDX-License-Identifier: Apache-2.0
"""Thread-safe admission reservations for scheduler-owned requests."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional


class AdmissionCapacityError(RuntimeError):
    """Raised when admitting a request would exceed a configured limit."""

    code = "scheduler_capacity_exceeded"

    def __init__(
        self,
        *,
        resource: str,
        limit: int,
        current: int,
        requested: int,
    ) -> None:
        self.resource = resource
        self.limit = limit
        self.current = current
        self.requested = requested
        super().__init__(
            f"Scheduler {resource} capacity exceeded: "
            f"limit={limit}, current={current}, requested={requested}"
        )


@dataclass(frozen=True)
class AdmissionSnapshot:
    num_requests: int
    num_prompt_tokens: int


class AdmissionController:
    """Own request and prompt-token reservations until terminal cleanup."""

    def __init__(
        self,
        *,
        max_requests: Optional[int] = None,
        max_prompt_tokens: Optional[int] = None,
    ) -> None:
        self.max_requests = self._validate_limit("max_requests", max_requests)
        self.max_prompt_tokens = self._validate_limit(
            "max_prompt_tokens", max_prompt_tokens
        )
        self._lock = threading.Lock()
        self._prompt_tokens_by_request: dict[str, int] = {}
        self._num_prompt_tokens = 0

    @staticmethod
    def _validate_limit(name: str, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1 or None")
        return value

    def reserve(self, request_id: str, prompt_tokens: int) -> None:
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")

        with self._lock:
            if request_id in self._prompt_tokens_by_request:
                raise ValueError(f"Request {request_id} already has a reservation")

            num_requests = len(self._prompt_tokens_by_request)
            if self.max_requests is not None and num_requests + 1 > self.max_requests:
                raise AdmissionCapacityError(
                    resource="request",
                    limit=self.max_requests,
                    current=num_requests,
                    requested=1,
                )

            if (
                self.max_prompt_tokens is not None
                and self._num_prompt_tokens + prompt_tokens > self.max_prompt_tokens
            ):
                raise AdmissionCapacityError(
                    resource="prompt_token",
                    limit=self.max_prompt_tokens,
                    current=self._num_prompt_tokens,
                    requested=prompt_tokens,
                )

            self._prompt_tokens_by_request[request_id] = prompt_tokens
            self._num_prompt_tokens += prompt_tokens

    def release(self, request_id: str) -> bool:
        with self._lock:
            prompt_tokens = self._prompt_tokens_by_request.pop(request_id, None)
            if prompt_tokens is None:
                return False
            self._num_prompt_tokens -= prompt_tokens
            return True

    def clear(self) -> None:
        with self._lock:
            self._prompt_tokens_by_request.clear()
            self._num_prompt_tokens = 0

    def snapshot(self) -> AdmissionSnapshot:
        with self._lock:
            return AdmissionSnapshot(
                num_requests=len(self._prompt_tokens_by_request),
                num_prompt_tokens=self._num_prompt_tokens,
            )
