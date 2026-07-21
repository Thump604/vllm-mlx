# SPDX-License-Identifier: Apache-2.0
"""Pure ModelProfile v1 serving-precedence resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from vllm_mlx.model_profile import validate_model_profile

_SAMPLING_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "seed",
    }
)


class ModelProfileResolutionError(ValueError):
    """Raised when an activation override violates the profile contract."""


@dataclass(frozen=True)
class EffectiveServingConfiguration:
    """Resolved serving values plus the source selected for each mutable value."""

    values: Mapping[str, Any]
    sources: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "values": deepcopy(dict(self.values)),
            "sources": dict(self.sources),
        }


def resolve_effective_serving_configuration(
    profile: Mapping[str, Any],
    *,
    profile_schema: Mapping[str, Any],
    activation_overrides: Mapping[str, Any] | None = None,
    runtime_fallbacks: Mapping[str, Any] | None = None,
) -> EffectiveServingConfiguration:
    """Resolve fallback < provider < profile < allowed activation values.

    The function is intentionally independent of server arguments and process
    state. It makes the v1 precedence contract executable without changing any
    runtime route.
    """
    validate_model_profile(profile, profile_schema)
    serving = profile["serving"]
    overrides = dict(activation_overrides or {})
    fallbacks = dict(runtime_fallbacks or {})
    allowed_overrides = set(serving["activation_policy"]["owner_override_fields"])
    unknown_overrides = sorted(set(overrides) - allowed_overrides)
    if unknown_overrides:
        raise ModelProfileResolutionError(
            f"activation overrides are not allowed: {unknown_overrides}"
        )
    unknown_fallbacks = sorted(set(fallbacks) - _SAMPLING_FIELDS)
    if unknown_fallbacks:
        raise ModelProfileResolutionError(
            f"runtime fallbacks are not schema sampling fields: {unknown_fallbacks}"
        )

    values = {
        "engine": serving["engine"],
        "route": serving["route"],
        "template": deepcopy(serving["template"]),
        "parsers": deepcopy(serving["parsers"]),
        "sampling": {},
        "limits": deepcopy(serving["limits"]),
        "features": {},
        "request_policy": deepcopy(serving["request_policy"]),
    }
    sources: dict[str, str] = {"/engine": "profile", "/route": "profile"}
    _record_leaf_sources(sources, values["template"], "/template", "profile")
    _record_leaf_sources(sources, values["parsers"], "/parsers", "profile")
    _record_leaf_sources(sources, values["limits"], "/limits", "profile")
    _record_leaf_sources(
        sources, values["request_policy"], "/request_policy", "profile"
    )

    sampling = values["sampling"]
    for source_name, source_values in (
        ("runtime_fallback", fallbacks),
        ("provider_default", serving["sampling"]["provider_defaults"]),
        ("profile_default", serving["sampling"]["profile_defaults"]),
    ):
        for key, value in source_values.items():
            sampling[key] = deepcopy(value)
            sources[f"/sampling/{key}"] = source_name

    for name, feature in serving["features"].items():
        resolved = deepcopy(feature)
        resolved["enabled"] = feature["mode"] == "enabled_by_default"
        values["features"][name] = resolved
        _record_leaf_sources(sources, resolved, f"/features/{name}", "profile_default")

    for field, value in overrides.items():
        section, name = field.split(".", 1)
        if section == "limits":
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ModelProfileResolutionError(
                    f"limit override {field!r} must be a positive integer"
                )
            values["limits"][name] = value
            sources[f"/limits/{name}"] = "activation_override"
        elif section == "features":
            if not isinstance(value, bool):
                raise ModelProfileResolutionError(
                    f"feature override {field!r} must be boolean"
                )
            if values["features"][name]["mode"] != "available_on_activation":
                raise ModelProfileResolutionError(
                    f"feature {name!r} is not available on activation"
                )
            values["features"][name]["enabled"] = value
            sources[f"/features/{name}/enabled"] = "activation_override"
        else:
            raise ModelProfileResolutionError(
                f"unsupported activation override section: {section}"
            )

    _validate_effective_limits(values["limits"])
    return EffectiveServingConfiguration(values=values, sources=sources)


def _record_leaf_sources(
    sources: dict[str, str], value: Any, pointer: str, source: str
) -> None:
    if isinstance(value, Mapping):
        if not value:
            sources[pointer] = source
            return
        for key, item in value.items():
            token = str(key).replace("~", "~0").replace("/", "~1")
            _record_leaf_sources(sources, item, f"{pointer}/{token}", source)
        return
    if isinstance(value, list):
        sources[pointer] = source
        return
    sources[pointer] = source


def _validate_effective_limits(limits: Mapping[str, Any]) -> None:
    advertised = limits["advertised_context"]
    serving = limits["serving_context"]
    default_output = limits["max_output_tokens"]
    request_output = limits["max_request_output_tokens"]
    max_kv = limits.get("max_kv_size")
    if serving > advertised:
        raise ModelProfileResolutionError("serving context exceeds advertised context")
    if default_output > serving or request_output > serving:
        raise ModelProfileResolutionError("output limit exceeds serving context")
    if default_output > request_output:
        raise ModelProfileResolutionError("default output exceeds request output cap")
    if max_kv is not None and max_kv < serving:
        raise ModelProfileResolutionError("max KV size is smaller than serving context")
