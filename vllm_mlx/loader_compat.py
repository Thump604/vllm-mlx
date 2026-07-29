# SPDX-License-Identifier: Apache-2.0
"""Immutable loader compatibility contracts for model profiles.

This module deliberately has no MLX, mlx-lm, or mlx-vlm imports.  It is the
stable boundary between a reviewed model profile and a backend-specific loader.
Backend implementations consume the resolved policy; they must not infer a
strict=False fallback from a model name or a loosely recognized config shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping


class LoaderCompatibilityError(ValueError):
    """Raised when a profile lacks a usable backend loader contract."""


@dataclass(frozen=True)
class LoaderPolicy:
    """Backend-neutral policy that controls model weight matching."""

    backend_id: str
    loader_route: str
    weight_policy_mode: str
    allowed_unmatched_weight_prefixes: tuple[str, ...]
    dependency_constraints: Mapping[str, str]

    @property
    def strict(self) -> bool:
        return self.weight_policy_mode == "strict"


def resolve_loader_policy(profile: Mapping[str, Any]) -> LoaderPolicy:
    """Resolve the profile-owned loader policy without inspecting a model name."""
    backend = profile.get("backend")
    if not isinstance(backend, Mapping):
        raise LoaderCompatibilityError(
            "model profile does not declare a backend loader compatibility contract"
        )
    try:
        policy = backend["weight_policy"]
        prefixes = tuple(policy["allowed_unmatched_weight_prefixes"])
        constraints = dict(backend["dependency_constraints"])
        resolved = LoaderPolicy(
            backend_id=str(backend["backend_id"]),
            loader_route=str(backend["loader_route"]),
            weight_policy_mode=str(policy["mode"]),
            allowed_unmatched_weight_prefixes=prefixes,
            dependency_constraints=constraints,
        )
    except (KeyError, TypeError) as exc:
        raise LoaderCompatibilityError(
            "model profile backend loader compatibility contract is incomplete"
        ) from exc
    if resolved.strict and prefixes:
        raise LoaderCompatibilityError(
            "strict loader policy cannot allow unmatched weight prefixes"
        )
    return resolved


def build_load_receipt(
    profile: Mapping[str, Any],
    artifact_path: str | Path,
    *,
    installed_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a serializable receipt for one profile-directed load attempt.

    The receipt records the requested route and policy.  Backend adapters append
    observed unmatched keys only when their loader API can report them; this
    constructor never invents a successful load or silently dropped weights.
    """
    policy = resolve_loader_policy(profile)
    package_versions = (
        dict(installed_versions)
        if installed_versions is not None
        else _installed_backend_versions(policy.dependency_constraints)
    )
    return {
        "profile_id": profile.get("profile_id"),
        "profile_revision": profile.get("profile_revision"),
        "subject_digest": profile.get("subject_digest"),
        "artifact_path": str(Path(artifact_path).expanduser().resolve()),
        "loader_policy": asdict(policy),
        "installed_dependency_versions": package_versions,
        "observed_unmatched_weight_keys": None,
        "load_result": "not_started",
    }


def _installed_backend_versions(
    constraints: Mapping[str, str],
) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in sorted(constraints):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions
