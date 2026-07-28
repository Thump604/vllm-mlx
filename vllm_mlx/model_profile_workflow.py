# SPDX-License-Identifier: Apache-2.0
"""Bind finalized workflow manifests to the explicit ModelProfile boundary."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from vllm_mlx._model_profile_compat_types import (
    LegacySourceInput,
    ModelProfileImportResult,
    SourceKind,
)
from vllm_mlx.model_profile_compat import (
    finalize_legacy_model_profile,
    import_legacy_model_profile,
)

_MANIFEST_KINDS: dict[SourceKind, str] = {
    "acquisition": "vllm-mlx-model-artifact",
    "conversion": "vllm-mlx-conversion",
    "registration": "vllm-mlx-model-registration",
}


@dataclass(frozen=True)
class WorkflowProfileEvidence:
    """Auditable workflow sources and their intentionally incomplete import."""

    acquisition: LegacySourceInput
    conversion: LegacySourceInput
    registration: LegacySourceInput
    imported: ModelProfileImportResult

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-ready evidence without duplicating untrusted payload bytes."""
        conversion = self.conversion.payload
        validation = conversion.get("artifact_validation")
        return {
            "kind": "vllm-mlx-workflow-profile-evidence",
            "schema_version": 1,
            "sources": [
                _source_summary(self.acquisition),
                _source_summary(self.conversion),
                _source_summary(self.registration),
            ],
            "acquisition": {
                "operation_id": self.acquisition.payload.get("operation_id"),
                "model_id": self.acquisition.payload.get("model_id"),
                "requested_revision": self.acquisition.payload.get("revision"),
                "resolved_revision": self.acquisition.payload.get("resolved_revision"),
            },
            "conversion": {
                "operation_id": conversion.get("operation_id"),
                "recipe": deepcopy(conversion.get("recipe")),
                "environment": deepcopy(conversion.get("environment")),
                "artifact_sha256": (
                    validation.get("artifact_sha256")
                    if isinstance(validation, Mapping)
                    else None
                ),
            },
            "import": self.imported.as_dict(),
            "promotion_required": True,
            "production_ready": False,
        }


def load_workflow_profile_evidence(
    *,
    acquisition_manifest: str | Path,
    conversion_manifest: str | Path,
    registration_manifest: str | Path,
) -> WorkflowProfileEvidence:
    """Load one coherent workflow triplet without changing runtime state.

    The manifests are consumed as immutable source records. This function does
    not infer missing ModelProfile fields or upgrade qualification status.
    """
    acquisition = _load_source("acquisition", acquisition_manifest)
    conversion = _load_source("conversion", conversion_manifest)
    registration = _load_source("registration", registration_manifest)
    _validate_workflow_links(acquisition, conversion, registration)
    imported = import_legacy_model_profile(
        acquisition=acquisition,
        conversion=conversion,
        registration=registration,
    )
    return WorkflowProfileEvidence(
        acquisition=acquisition,
        conversion=conversion,
        registration=registration,
        imported=imported,
    )


def finalize_workflow_profile(
    evidence: WorkflowProfileEvidence,
    completed_profile: Mapping[str, Any],
    *,
    profile_schema: Mapping[str, Any],
    import_schema: Mapping[str, Any],
) -> ModelProfileImportResult:
    """Finalize an explicit candidate while preserving workflow-derived facts."""
    return finalize_legacy_model_profile(
        evidence.imported,
        completed_profile,
        profile_schema=profile_schema,
        import_schema=import_schema,
    )


def _load_source(kind: SourceKind, path_value: str | Path) -> LegacySourceInput:
    path = Path(path_value).expanduser().resolve()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{kind} manifest is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} manifest must be a JSON object: {path}")
    if payload.get("kind") != _MANIFEST_KINDS[kind]:
        raise ValueError(f"{kind} manifest has an unexpected kind: {path}")
    return LegacySourceInput(
        kind=kind,
        location=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=deepcopy(payload),
    )


def _validate_workflow_links(
    acquisition: LegacySourceInput,
    conversion: LegacySourceInput,
    registration: LegacySourceInput,
) -> None:
    identity = conversion.payload.get("identity")
    source = identity.get("source") if isinstance(identity, Mapping) else None
    acquired_sha256 = (
        source.get("acquisition_manifest_sha256")
        if isinstance(source, Mapping)
        else None
    )
    if acquired_sha256 != acquisition.sha256:
        raise ValueError("conversion manifest is not bound to acquisition manifest")

    validation = conversion.payload.get("artifact_validation")
    digest = (
        validation.get("artifact_sha256") if isinstance(validation, Mapping) else None
    )
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("conversion manifest is missing its artifact digest")

    manifests = registration.payload.get("source_manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError("registration manifest is missing workflow sources")
    for kind, source_input in (
        ("acquisition", acquisition),
        ("conversion", conversion),
    ):
        record = manifests.get(kind)
        recorded = record.get("payload") if isinstance(record, Mapping) else None
        if recorded != source_input.payload:
            raise ValueError(
                f"registration manifest does not match {kind} manifest bytes"
            )


def _source_summary(source: LegacySourceInput) -> dict[str, str]:
    return {
        "kind": source.kind,
        "location": source.location,
        "sha256": source.sha256,
    }
