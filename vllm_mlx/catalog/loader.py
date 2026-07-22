# SPDX-License-Identifier: Apache-2.0
"""Load immutable ModelProfile documents from an injected catalog root."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping
from urllib.parse import urlparse

from vllm_mlx.model_profile import ModelProfileValidationError, validate_model_profile

_PROFILE_SCHEMA_NAME = "model-profile-v1.schema.json"
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class CatalogError(ValueError):
    """Base error for invalid or unusable catalog contents."""


class CatalogValidationError(CatalogError):
    """Raised when one or more catalog profiles fail validation."""


class ModelProfileCatalog:
    """Validated, deterministic, read-only view of a profile catalog.

    JSON files are discovered recursively below ``root``. The root is an input
    boundary only: file paths are not added to profiles or returned as catalog
    identity. When ``revision`` is omitted, ``get`` returns the highest
    revision for the requested profile ID.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        profile_schema: Mapping[str, Any] | None = None,
        schema_path: str | Path | None = None,
    ) -> None:
        if profile_schema is not None and schema_path is not None:
            raise ValueError("provide profile_schema or schema_path, not both")

        self._root = Path(root)
        if not self._root.is_dir():
            raise CatalogError("catalog root must be an existing directory")
        self._profile_schema = deepcopy(
            dict(profile_schema)
            if profile_schema is not None
            else _read_schema(schema_path)
        )
        self._profiles = self._load_profiles()

    def list_profiles(self) -> list[dict[str, Any]]:
        """Return all profiles in deterministic ID/revision order."""
        return [deepcopy(profile) for profile in self._profiles]

    def get(self, profile_id: str, revision: int | None = None) -> dict[str, Any]:
        """Return one defensive copy by ID and optional exact revision.

        Without a revision, the highest validated revision is selected. A
        missing ID or revision raises ``KeyError`` rather than silently
        returning a different profile.
        """
        matches = [
            profile for profile in self._profiles if profile["profile_id"] == profile_id
        ]
        if revision is not None:
            matches = [
                profile
                for profile in matches
                if profile["profile_revision"] == revision
            ]
        if not matches:
            identity = profile_id if revision is None else f"{profile_id}@{revision}"
            raise KeyError(identity)
        return deepcopy(matches[-1])

    def get_profile(
        self, profile_id: str, revision: int | None = None
    ) -> dict[str, Any]:
        """Alias for ``get`` for callers that prefer an explicit name."""
        return self.get(profile_id, revision)

    def _load_profiles(self) -> tuple[dict[str, Any], ...]:
        files = sorted(
            (path for path in self._root.rglob("*.json") if path.is_file()),
            key=lambda path: path.relative_to(self._root).as_posix(),
        )
        profiles: list[dict[str, Any]] = []
        identities: set[tuple[str, int]] = set()
        for path in files:
            relative = path.relative_to(self._root).as_posix()
            profile = _read_profile(path, relative)
            try:
                validate_model_profile(profile, self._profile_schema)
            except ModelProfileValidationError as error:
                raise CatalogValidationError(
                    f"{relative}: invalid ModelProfile: {error}"
                ) from error
            _validate_catalog_identity(profile, relative)
            identity = (profile["profile_id"], profile["profile_revision"])
            if identity in identities:
                raise CatalogValidationError(
                    f"{relative}: duplicate profile identity "
                    f"{profile['profile_id']}@{profile['profile_revision']}"
                )
            identities.add(identity)
            profiles.append(deepcopy(profile))

        profiles.sort(key=lambda item: (item["profile_id"], item["profile_revision"]))
        return tuple(profiles)


def load_catalog(
    root: str | Path,
    *,
    profile_schema: Mapping[str, Any] | None = None,
    schema_path: str | Path | None = None,
) -> ModelProfileCatalog:
    """Load and validate a catalog rooted at ``root``."""
    return ModelProfileCatalog(
        root, profile_schema=profile_schema, schema_path=schema_path
    )


def load_model_profiles(
    root: str | Path,
    *,
    profile_schema: Mapping[str, Any] | None = None,
    schema_path: str | Path | None = None,
) -> ModelProfileCatalog:
    """Explicitly named alias for ``load_catalog``."""
    return load_catalog(root, profile_schema=profile_schema, schema_path=schema_path)


def _read_schema(schema_path: str | Path | None) -> dict[str, Any]:
    path = (
        Path(schema_path)
        if schema_path is not None
        else Path(__file__).resolve().parents[2] / "schemas" / _PROFILE_SCHEMA_NAME
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"unable to load profile schema: {path.name}") from error
    if not isinstance(value, dict):
        raise CatalogError("profile schema must be a JSON object")
    return value


def _read_profile(path: Path, relative: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogValidationError(f"{relative}: invalid JSON") from error
    if not isinstance(value, dict):
        raise CatalogValidationError(f"{relative}: profile must be a JSON object")
    return value


def _validate_catalog_identity(profile: Mapping[str, Any], relative: str) -> None:
    for pointer, value in (
        ("/profile_id", profile["profile_id"]),
        ("/identity/artifact_id", profile["identity"]["artifact_id"]),
        ("/artifact/source_uri", profile["artifact"]["source_uri"]),
    ):
        if _is_local_absolute(value):
            raise CatalogValidationError(
                f"{relative}: {pointer} must not be a local absolute path"
            )


def _is_local_absolute(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if value.startswith(("/", "\\")) or _WINDOWS_DRIVE_PATH.match(value):
        return True
    parsed = urlparse(value)
    return parsed.scheme.lower() == "file" and (
        parsed.path.startswith("/") or PureWindowsPath(parsed.path).is_absolute()
    )
