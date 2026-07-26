# SPDX-License-Identifier: Apache-2.0
"""Types shared by the public compatibility facade and private import engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping

SourceKind = Literal[
    "acquisition",
    "conversion",
    "registration",
    "registry",
    "cli_server",
    "qualification",
]
ProvenanceKind = Literal[
    "provider_fact",
    "derived_recommendation",
    "measured_result",
    "maintainer_policy",
]

_SHA256_LENGTH = 64


@dataclass(frozen=True)
class LegacySourceInput:
    """An already-loaded legacy document and its auditable source identity."""

    kind: SourceKind
    location: str
    sha256: str
    payload: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls, kind: SourceKind, value: Mapping[str, Any]
    ) -> "LegacySourceInput":
        """Validate and snapshot a mapping-shaped legacy source input."""
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError(f"{kind} source must contain a mapping 'payload'")
        location = value.get("location")
        sha256 = value.get("sha256")
        if not isinstance(location, str) or not location:
            raise ValueError(f"{kind} source must contain a non-empty 'location'")
        if not isinstance(sha256, str) or not _is_sha256(sha256):
            raise ValueError(f"{kind} source must contain a SHA-256 'sha256'")
        return cls(
            kind=kind,
            location=location,
            sha256=sha256,
            payload=deepcopy(dict(payload)),
        )


@dataclass(frozen=True)
class CompatibilityIssue:
    """A deterministic reason an imported fragment remains incomplete."""

    code: str
    severity: Literal["error", "warning"]
    pointer: str
    sources: tuple[str, ...]
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "pointer": self.pointer,
            "sources": list(self.sources),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ModelProfileImportResult:
    """The incomplete v1 import envelope; it never changes runtime state."""

    complete: bool
    sources: tuple[LegacySourceInput, ...]
    profile: Mapping[str, Any] | None
    issues: tuple[CompatibilityIssue, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return an envelope without source payloads and with copied data."""
        return {
            "schema_version": 1,
            "complete": self.complete,
            "sources": [
                {
                    "kind": source.kind,
                    "location": source.location,
                    "sha256": source.sha256,
                }
                for source in self.sources
            ],
            "profile": deepcopy(self.profile),
            "issues": [issue.as_dict() for issue in self.issues],
        }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )
