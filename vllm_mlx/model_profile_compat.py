# SPDX-License-Identifier: Apache-2.0
"""Stable public facade for bounded legacy ModelProfile compatibility imports."""

from __future__ import annotations

from typing import Any, Mapping

from vllm_mlx._model_profile_compat import _import_legacy_sources
from vllm_mlx._model_profile_compat_types import (
    CompatibilityIssue,
    LegacySourceInput,
    ModelProfileFinalizationError,
    ModelProfileImportResult,
)
from vllm_mlx._model_profile_finalization import finalize_import

__all__ = (
    "LegacySourceInput",
    "CompatibilityIssue",
    "ModelProfileFinalizationError",
    "ModelProfileImportResult",
    "finalize_legacy_model_profile",
    "import_legacy_model_profile",
)


def import_legacy_model_profile(
    *,
    acquisition: LegacySourceInput | Mapping[str, Any] | None = None,
    conversion: LegacySourceInput | Mapping[str, Any] | None = None,
    registration: LegacySourceInput | Mapping[str, Any] | None = None,
    registry_entry: LegacySourceInput | Mapping[str, Any] | None = None,
    cli_server: LegacySourceInput | Mapping[str, Any] | None = None,
    qualification: LegacySourceInput | Mapping[str, Any] | None = None,
) -> ModelProfileImportResult:
    """Map legacy records into an incomplete fragment without runtime effects.

    Raises:
        ValueError: No source was supplied, a source kind does not match its
            keyword, or a mapping has invalid source identity.
        TypeError: A source has an unsupported type or a non-mapping payload.
    """
    result: ModelProfileImportResult = _import_legacy_sources(
        (
            ("acquisition", acquisition),
            ("conversion", conversion),
            ("registration", registration),
            ("registry", registry_entry),
            ("cli_server", cli_server),
            ("qualification", qualification),
        )
    )
    return result


def finalize_legacy_model_profile(
    imported: ModelProfileImportResult,
    completed_profile: Mapping[str, Any],
    *,
    profile_schema: Mapping[str, Any],
    import_schema: Mapping[str, Any],
) -> ModelProfileImportResult:
    """Finalize an explicit candidate after preservation and contract checks."""
    result: ModelProfileImportResult = finalize_import(
        imported,
        completed_profile,
        profile_schema=profile_schema,
        import_schema=import_schema,
    )
    return result
