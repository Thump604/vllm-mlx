# SPDX-License-Identifier: Apache-2.0
"""Dependency-free SpecPrefill identity and outcome contracts.

This module is intentionally independent of MLX, mlx-lm, and runtime
integrations. It is the source/static contract consumed by later WS1, WS2.5,
and WS3 adapters.

The wire identity uses RFC 8785-style canonical JSON with an explicit
restricted number profile. Python integers must be safe IEEE-754 integers;
finite Python floats are serialized with ECMAScript NumberToString-compatible
formatting. The identity schema itself only emits strings, booleans, null,
and safe integers, so model metadata cannot silently introduce a
non-interoperable number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any

IDENTITY_SCHEMA_VERSION = "specprefill.identity.v1"
IDENTITY_CANONICALIZATION = "rfc8785"
IDENTITY_DIGEST_ALGORITHM = "sha256"
IDENTITY_NUMBER_PROFILE = "rfc8785-ieee754-safe-integer-v1"
MAX_SAFE_INTEGER = (1 << 53) - 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_UID_RE = re.compile(r"[!-~]{1,128}\Z")

_MODE_FIELDS = (
    "continuous_batching",
    "serialized",
    "mtp",
    "chunked_prefill",
    "rotating_cache",
    "ple",
    "qsa",
    "audio",
)

_MODEL_CACHE_FIELDS = (
    "model_id",
    "model_revision",
    "artifact_id",
    "artifact_digest",
    "weight_index_digest",
    "family",
    "architecture",
    "model_module",
    "language_module",
    "model_type",
    "config_digest",
    "tokenizer",
    "chat_template",
    "parser",
    "vision",
    "cache_schema",
    "mode",
)

_TOKENIZER_FIELDS = (
    "files",
    "config_digest",
    "implementation",
    "implementation_version",
    "added_tokens",
    "special_tokens",
    "encode_probes",
)

_COMPATIBILITY_SUMMARY_FIELDS = (
    "model_id",
    "model_revision",
    "artifact_id",
    "artifact_digest",
    "weight_index_digest",
    "family",
    "architecture",
    "model_type",
    "config_digest",
    "tokenizer_digest",
    "vision_config_digest",
    "media_mapping_digest",
    "media_token_ids",
)

_DISPOSITIONS = frozenset(
    {"not_attempted", "engaged", "fallback_dense", "cancelled", "failed"}
)
_DENSE_RESULTS = frozenset({"not_run", "succeeded", "failed", "cancelled"})
_BASE_REASONS = frozenset(
    {
        "none",
        "not_requested",
        "disabled",
        "unsupported_model",
        "tokenizer_mismatch",
        "template_parser_mismatch",
        "draft_mismatch",
        "media_audio_ineligible",
        "length_cap",
        "cache_unsafe",
        "scoring_failed",
        "cancellation",
        "runtime_error",
        "cb_required_but_unavailable",
        "cb_incompatible",
    }
)
_BYPASS_REASONS = frozenset(
    {
        "not_requested",
        "disabled",
        "unsupported_model",
        "tokenizer_mismatch",
        "template_parser_mismatch",
        "draft_mismatch",
        "media_audio_ineligible",
        "length_cap",
        "cache_unsafe",
        "cb_required_but_unavailable",
        "cb_incompatible",
    }
)
_FAILURE_REASONS = frozenset({"scoring_failed", "runtime_error"})


def _is_mode_reason(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("mode_incompatible_")
        and (
            len(value) > len("mode_incompatible_")
            and all(
                character in "abcdefghijklmnopqrstuvwxyz0123456789_"
                for character in value
            )
        )
    )


def _is_result_reason(value: Any) -> bool:
    return value in _BASE_REASONS or _is_mode_reason(value)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _reject_surrogates(value: str, path: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{path} contains a lone surrogate")


def _validate_json_value(value: Any, path: str = "$") -> None:
    """Validate the interoperable JSON subset used by the canonicalizer."""

    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        _reject_surrogates(value, path)
        return
    if isinstance(value, int):
        if isinstance(value, bool) or abs(value) > MAX_SAFE_INTEGER:
            raise ValueError(f"{path} must be a safe IEEE-754 integer (at most 2^53-1)")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        seen: set[str] = set()
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} has a non-string object key")
            _reject_surrogates(key, f"{path} key")
            if key in seen:
                raise ValueError(f"{path} has duplicate object key: {key!r}")
            seen.add(key)
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} has unsupported JSON value type: {type(value).__name__}")


def parse_identity_json(text: str) -> dict[str, Any]:
    """Parse JSON while rejecting duplicate keys and non-finite numbers."""

    if not isinstance(text, str):
        raise ValueError("identity JSON must be text")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            ("duplicate JSON object key", "non-finite")
        ):
            raise
        raise ValueError(f"invalid identity JSON: {exc}") from exc
    _validate_json_value(value)
    if not isinstance(value, dict):
        raise ValueError("identity JSON root must be an object")
    return value


def _ecmascript_number(value: int | float) -> str:
    """Serialize a Python number using the JSON/JCS ECMAScript spelling."""

    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > MAX_SAFE_INTEGER:
            raise ValueError("integer exceeds safe IEEE-754 range")
        return str(value)
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError("canonical numbers must be finite integers or floats")
    if value == 0.0:
        return "0"

    text = repr(value).lower()
    sign = ""
    if text.startswith("-"):
        sign, text = "-", text[1:]

    if "e" not in text:
        if text.endswith(".0"):
            text = text[:-2]
        return sign + text

    mantissa, exponent_text = text.split("e", 1)
    exponent = int(exponent_text)
    absolute = abs(value)
    if 1e-6 <= absolute < 1e21:
        digits = mantissa.replace(".", "")
        decimal_position = mantissa.find(".") if "." in mantissa else len(mantissa)
        decimal_position += exponent
        if decimal_position <= 0:
            return sign + "0." + ("0" * -decimal_position) + digits
        if decimal_position >= len(digits):
            return sign + digits + ("0" * (decimal_position - len(digits)))
        return sign + digits[:decimal_position] + "." + digits[decimal_position:]

    if mantissa.endswith(".0"):
        mantissa = mantissa[:-2]
    exponent_sign = "+" if exponent >= 0 else "-"
    return f"{sign}{mantissa}e{exponent_sign}{abs(exponent)}"


def _jcs(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return _ecmascript_number(value)
    if isinstance(value, float):
        return _ecmascript_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_jcs(item) for item in value) + "]"
    if isinstance(value, Mapping):
        ordered = sorted(value.items(), key=lambda item: item[0].encode("utf-16-be"))
        return (
            "{"
            + ",".join(_jcs(str(key)) + ":" + _jcs(item) for key, item in ordered)
            + "}"
        )
    raise ValueError(f"unsupported value for canonicalization: {type(value).__name__}")


def _clone(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return [_clone(item) for item in value]
    return value


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _require_keys(
    mapping: Mapping[str, Any],
    required: tuple[str, ...],
    allowed: tuple[str, ...],
    path: str,
) -> None:
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")
    missing = [key for key in required if key not in mapping or mapping[key] is None]
    if missing:
        raise ValueError(f"{path}.{missing[0]} is required and cannot be null")


def _require_string(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty string")
    _reject_surrogates(value, f"{path}.{key}")
    return value


def _require_bool(mapping: Mapping[str, Any], key: str, path: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a boolean")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, path: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path}.{key} must be a non-negative safe integer")
    if value > MAX_SAFE_INTEGER:
        raise ValueError(f"{path}.{key} exceeds safe IEEE-754 range")
    return value


def _require_sha256(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = _require_string(mapping, key, path)
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{path}.{key} must be lowercase SHA-256 hex")
    return value


def _validate_relative_path(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty relative POSIX path")
    _reject_surrogates(value, path)
    if "\x00" in value or "\\" in value or value.startswith("/"):
        raise ValueError(f"{path} must be a normalized relative POSIX path")
    if value.startswith("//") or value != str(PurePosixPath(value)):
        raise ValueError(f"{path} must be a normalized relative POSIX path")
    if any(part in {"", ".", ".."} for part in PurePosixPath(value).parts):
        raise ValueError(f"{path} must be a normalized relative POSIX path")
    return value


def _validate_file_records(value: Any, path: str, *, allow_empty: bool = True) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list")
    if not allow_empty and not value:
        raise ValueError(f"{path} must not be empty")
    paths: set[str] = set()
    for index, item in enumerate(value):
        entry_path = f"{path}[{index}]"
        entry = _require_mapping(item, entry_path)
        _require_keys(entry, ("path", "sha256"), ("path", "sha256"), entry_path)
        file_path = _validate_relative_path(entry["path"], f"{entry_path}.path")
        if file_path in paths:
            raise ValueError(f"{path} contains duplicate path: {file_path}")
        paths.add(file_path)
        _require_sha256(entry, "sha256", entry_path)


def _validate_token_records(value: Any, path: str) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list")
    ids: set[int] = set()
    tokens: set[str] = set()
    for index, item in enumerate(value):
        entry_path = f"{path}[{index}]"
        entry = _require_mapping(item, entry_path)
        _require_keys(entry, ("id", "token"), ("id", "token"), entry_path)
        token_id = _require_int(entry, "id", entry_path)
        token = _require_string(entry, "token", entry_path)
        if token_id in ids:
            raise ValueError(f"{path} contains duplicate token id: {token_id}")
        if token in tokens:
            raise ValueError(f"{path} contains duplicate token: {token!r}")
        ids.add(token_id)
        tokens.add(token)


def _validate_int_list(value: Any, path: str) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list")
    values: set[int] = set()
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"{path}[{index}] must be a non-negative integer")
        if item > MAX_SAFE_INTEGER:
            raise ValueError(f"{path}[{index}] exceeds safe IEEE-754 range")
        if item in values:
            raise ValueError(f"{path} contains duplicate id: {item}")
        values.add(item)


def _validate_encode_probes(value: Any, path: str) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list")
    texts: set[str] = set()
    for index, item in enumerate(value):
        entry_path = f"{path}[{index}]"
        entry = _require_mapping(item, entry_path)
        _require_keys(entry, ("text", "ids"), ("text", "ids"), entry_path)
        text = _require_string(entry, "text", entry_path)
        if text in texts:
            raise ValueError(f"{path} contains duplicate probe text")
        texts.add(text)
        _validate_int_list(entry["ids"], f"{entry_path}.ids")


def _validate_model_cache(cache: Any, path: str) -> None:
    cache = _require_mapping(cache, path)
    _require_keys(cache, _MODEL_CACHE_FIELDS, _MODEL_CACHE_FIELDS, path)
    for key in (
        "model_id",
        "model_revision",
        "artifact_id",
        "family",
        "architecture",
        "model_module",
        "language_module",
        "model_type",
    ):
        _require_string(cache, key, path)
    for key in ("artifact_digest", "weight_index_digest", "config_digest"):
        _require_sha256(cache, key, path)

    tokenizer_path = f"{path}.tokenizer"
    tokenizer = _require_mapping(cache["tokenizer"], tokenizer_path)
    _require_keys(tokenizer, _TOKENIZER_FIELDS, _TOKENIZER_FIELDS, tokenizer_path)
    _validate_file_records(
        tokenizer["files"], f"{tokenizer_path}.files", allow_empty=False
    )
    _require_sha256(tokenizer, "config_digest", tokenizer_path)
    _require_string(tokenizer, "implementation", tokenizer_path)
    _require_string(tokenizer, "implementation_version", tokenizer_path)
    _validate_token_records(tokenizer["added_tokens"], f"{tokenizer_path}.added_tokens")
    _validate_token_records(
        tokenizer["special_tokens"], f"{tokenizer_path}.special_tokens"
    )
    _validate_encode_probes(
        tokenizer["encode_probes"], f"{tokenizer_path}.encode_probes"
    )

    for key in ("chat_template", "parser"):
        section_path = f"{path}.{key}"
        section = _require_mapping(cache[key], section_path)
        _require_keys(
            section,
            ("name", "version", "sha256"),
            ("name", "version", "sha256"),
            section_path,
        )
        _require_string(section, "name", section_path)
        _require_string(section, "version", section_path)
        _require_sha256(section, "sha256", section_path)

    vision_path = f"{path}.vision"
    vision = _require_mapping(cache["vision"], vision_path)
    _require_keys(
        vision,
        (
            "enabled",
            "config_digest",
            "processor_files",
            "media_token_ids",
            "media_mapping_digest",
        ),
        (
            "enabled",
            "config_digest",
            "processor_files",
            "media_token_ids",
            "media_mapping_digest",
        ),
        vision_path,
    )
    _require_bool(vision, "enabled", vision_path)
    _require_sha256(vision, "config_digest", vision_path)
    _validate_file_records(vision["processor_files"], f"{vision_path}.processor_files")
    _validate_int_list(vision["media_token_ids"], f"{vision_path}.media_token_ids")
    _require_sha256(vision, "media_mapping_digest", vision_path)

    cache_schema_path = f"{path}.cache_schema"
    cache_schema = _require_mapping(cache["cache_schema"], cache_schema_path)
    _require_keys(
        cache_schema,
        ("name", "version", "sha256"),
        ("name", "version", "sha256"),
        cache_schema_path,
    )
    _require_string(cache_schema, "name", cache_schema_path)
    _require_string(cache_schema, "version", cache_schema_path)
    _require_sha256(cache_schema, "sha256", cache_schema_path)

    mode_path = f"{path}.mode"
    mode = _require_mapping(cache["mode"], mode_path)
    _require_keys(mode, _MODE_FIELDS, _MODE_FIELDS + ("capability_modes",), mode_path)
    for key in _MODE_FIELDS:
        _require_bool(mode, key, mode_path)
    capability_modes = mode.get("capability_modes")
    if not isinstance(capability_modes, (list, tuple)):
        raise ValueError(f"{mode_path}.capability_modes must be a list")
    seen_modes: set[str] = set()
    for index, item in enumerate(capability_modes):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{mode_path}.capability_modes[{index}] must be a string")
        if item in seen_modes:
            raise ValueError(f"{mode_path}.capability_modes contains duplicate: {item}")
        seen_modes.add(item)


def _validate_protocol_identity(protocol: Any, path: str) -> None:
    protocol = _require_mapping(protocol, path)
    required = (
        "template_digest",
        "parser_digest",
        "tool_schema_digest",
        "renderer",
        "version",
    )
    _require_keys(protocol, required, required, path)
    for key in ("template_digest", "parser_digest", "tool_schema_digest"):
        _require_sha256(protocol, key, path)
    _require_string(protocol, "renderer", path)
    _require_string(protocol, "version", path)


def _summary_from_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer_digest = hashlib.sha256(
        _jcs(cache["tokenizer"]).encode("utf-8")
    ).hexdigest()
    vision = cache["vision"]
    return {
        "model_id": cache["model_id"],
        "model_revision": cache["model_revision"],
        "artifact_id": cache["artifact_id"],
        "artifact_digest": cache["artifact_digest"],
        "weight_index_digest": cache["weight_index_digest"],
        "family": cache["family"],
        "architecture": cache["architecture"],
        "model_type": cache["model_type"],
        "config_digest": cache["config_digest"],
        "tokenizer_digest": tokenizer_digest,
        "vision_config_digest": vision["config_digest"],
        "media_mapping_digest": vision["media_mapping_digest"],
        "media_token_ids": list(vision["media_token_ids"]),
    }


def _validate_summary(summary: Any, path: str) -> None:
    summary = _require_mapping(summary, path)
    _require_keys(
        summary,
        _COMPATIBILITY_SUMMARY_FIELDS,
        _COMPATIBILITY_SUMMARY_FIELDS,
        path,
    )
    for key in (
        "model_id",
        "model_revision",
        "artifact_id",
        "family",
        "architecture",
        "model_type",
    ):
        _require_string(summary, key, path)
    for key in (
        "artifact_digest",
        "weight_index_digest",
        "config_digest",
        "tokenizer_digest",
        "vision_config_digest",
        "media_mapping_digest",
    ):
        _require_sha256(summary, key, path)
    _validate_int_list(summary["media_token_ids"], f"{path}.media_token_ids")


def build_draft_compatibility(
    target_cache: Mapping[str, Any],
    draft_cache: Mapping[str, Any],
    *,
    relation: str,
) -> dict[str, Any]:
    """Build the explicit target/draft relation stored in both manifests."""

    _validate_model_cache(target_cache, "target_cache")
    _validate_model_cache(draft_cache, "draft_cache")
    if not isinstance(relation, str) or not relation:
        raise ValueError("draft compatibility relation must be a non-empty string")
    target = _summary_from_cache(_normalize_model_cache(target_cache))
    draft = _summary_from_cache(_normalize_model_cache(draft_cache))
    return {"relation": relation, "target": target, "draft": draft}


def _validate_draft_compatibility(value: Any, path: str) -> None:
    relation = _require_mapping(value, path)
    _require_keys(
        relation,
        ("relation", "target", "draft"),
        ("relation", "target", "draft"),
        path,
    )
    _require_string(relation, "relation", path)
    _validate_summary(relation["target"], f"{path}.target")
    _validate_summary(relation["draft"], f"{path}.draft")


def _validate_identity_shape(
    manifest: Mapping[str, Any], *, require_digest: bool
) -> None:
    allowed = (
        "schema_version",
        "canonicalization",
        "digest_algorithm",
        "number_profile",
        "digest",
        "role",
        "model_cache_identity",
        "request_protocol_identity",
        "draft_compatibility",
    )
    required = (
        "schema_version",
        "canonicalization",
        "digest_algorithm",
        "number_profile",
        "role",
        "model_cache_identity",
        "request_protocol_identity",
        "draft_compatibility",
    )
    if require_digest:
        required += ("digest",)
    _require_keys(manifest, required, allowed, "manifest")
    if manifest["schema_version"] != IDENTITY_SCHEMA_VERSION:
        raise ValueError("manifest.schema_version is unsupported")
    if manifest["canonicalization"] != IDENTITY_CANONICALIZATION:
        raise ValueError("manifest.canonicalization is unsupported")
    if manifest["digest_algorithm"] != IDENTITY_DIGEST_ALGORITHM:
        raise ValueError("manifest.digest_algorithm is unsupported")
    if manifest["number_profile"] != IDENTITY_NUMBER_PROFILE:
        raise ValueError("manifest.number_profile is unsupported")
    if manifest["role"] not in {"target", "draft"}:
        raise ValueError("manifest.role must be target or draft")
    if require_digest:
        _require_sha256(manifest, "digest", "manifest")
    _validate_model_cache(
        manifest["model_cache_identity"], "manifest.model_cache_identity"
    )
    _validate_protocol_identity(
        manifest["request_protocol_identity"], "manifest.request_protocol_identity"
    )
    _validate_draft_compatibility(
        manifest["draft_compatibility"], "manifest.draft_compatibility"
    )


def _normalize_model_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    _validate_model_cache(cache, "model_cache_identity")
    result = _clone(cache)
    tokenizer = result["tokenizer"]
    tokenizer["files"].sort(key=lambda item: item["path"])
    tokenizer["added_tokens"].sort(key=lambda item: (item["id"], item["token"]))
    tokenizer["special_tokens"].sort(key=lambda item: (item["id"], item["token"]))
    # Encode probes and every ids array are sequence-semantic and retain order.
    result["vision"]["media_token_ids"].sort()
    result["mode"]["capability_modes"].sort()
    result["vision"]["processor_files"].sort(key=lambda item: item["path"])
    return result


def _normalize_identity_collections(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_identity_shape(manifest, require_digest="digest" in manifest)
    result = _clone(manifest)
    result["model_cache_identity"] = _normalize_model_cache(
        result["model_cache_identity"]
    )
    return result


def canonical_identity_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 canonical JSON bytes.

    Set-like manifest arrays are normalized only after strict record
    validation. Generic arrays preserve order.
    """

    _validate_json_value(value)
    if isinstance(value, Mapping) and "model_cache_identity" in value:
        value = _normalize_identity_collections(value)
    return _jcs(value).encode("utf-8")


def identity_digest(value: Mapping[str, Any]) -> str:
    """Digest an identity envelope, excluding its self-referential digest."""

    if not isinstance(value, Mapping):
        raise ValueError("identity digest input must be an object")
    payload: Mapping[str, Any] = value
    if "model_cache_identity" in value and "digest" in value:
        payload = {key: item for key, item in value.items() if key != "digest"}
    return hashlib.sha256(canonical_identity_bytes(payload)).hexdigest()


def validate_identity_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate schema, strict records, canonical set ordering, and digest."""

    if not isinstance(manifest, Mapping):
        raise ValueError("identity manifest must be an object")
    _validate_json_value(manifest)
    normalized = _normalize_identity_collections(manifest)
    expected = identity_digest(normalized)
    if manifest["digest"] != expected:
        raise ValueError("manifest.digest does not match canonical contents")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def freeze_identity_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and freeze an identity for the loaded-process lifetime."""

    validate_identity_manifest(manifest)
    return _freeze(_normalize_identity_collections(manifest))


def build_identity_manifest(
    model_cache_identity: Mapping[str, Any],
    request_protocol_identity: Mapping[str, Any],
    *,
    role: str,
    draft_compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a canonical immutable-process identity envelope."""

    envelope: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "canonicalization": IDENTITY_CANONICALIZATION,
        "digest_algorithm": IDENTITY_DIGEST_ALGORITHM,
        "number_profile": IDENTITY_NUMBER_PROFILE,
        "role": role,
        "model_cache_identity": model_cache_identity,
        "request_protocol_identity": request_protocol_identity,
        "draft_compatibility": draft_compatibility,
    }
    normalized = _normalize_identity_collections(envelope)
    normalized["digest"] = identity_digest(normalized)
    validate_identity_manifest(normalized)
    return normalized


def identity_manifest_for(model_or_manifest: Any) -> Mapping[str, Any] | None:
    """Resolve one unambiguous validated manifest from an owner object."""

    if isinstance(model_or_manifest, Mapping) and "schema_version" in model_or_manifest:
        candidates = [model_or_manifest]
    elif isinstance(model_or_manifest, Mapping):
        candidates = [
            model_or_manifest.get(name)
            for name in (
                "specprefill_identity_manifest",
                "runtime_identity_manifest",
                "identity_manifest",
            )
        ]
    else:
        candidates = [
            getattr(model_or_manifest, name, None)
            for name in (
                "specprefill_identity_manifest",
                "runtime_identity_manifest",
                "identity_manifest",
            )
        ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        return None

    frozen: list[Mapping[str, Any]] = []
    try:
        for candidate in candidates:
            frozen.append(freeze_identity_manifest(candidate))
    except (TypeError, ValueError, KeyError):
        return None
    digests = {manifest["digest"] for manifest in frozen}
    if len(digests) != 1:
        return None
    return frozen[0]


def _same_canonical(left: Any, right: Any) -> bool:
    return canonical_identity_bytes(left) == canonical_identity_bytes(right)


def identity_compatibility_reason(model: Any, draft_model: Any) -> str | None:
    """Return a distinct tokenizer/template/draft incompatibility reason."""

    target = identity_manifest_for(model)
    if target is None or target["role"] != "target":
        return "unsupported_model"
    draft = identity_manifest_for(draft_model)
    if draft is None or draft["role"] != "draft":
        return "draft_mismatch"

    target_cache = target["model_cache_identity"]
    draft_cache = draft["model_cache_identity"]
    if not _same_canonical(target_cache["tokenizer"], draft_cache["tokenizer"]):
        return "tokenizer_mismatch"
    for key in ("chat_template", "parser"):
        if not _same_canonical(target_cache[key], draft_cache[key]):
            return "template_parser_mismatch"

    target_relation = target["draft_compatibility"]
    draft_relation = draft["draft_compatibility"]
    if target_relation["relation"] != draft_relation["relation"]:
        return "draft_mismatch"
    target_summary = _summary_from_cache(target_cache)
    draft_summary = _summary_from_cache(draft_cache)
    if not _same_canonical(target_relation["target"], target_summary):
        return "draft_mismatch"
    if not _same_canonical(target_relation["draft"], draft_summary):
        return "draft_mismatch"
    if not _same_canonical(draft_relation["target"], target_summary):
        return "draft_mismatch"
    if not _same_canonical(draft_relation["draft"], draft_summary):
        return "draft_mismatch"
    return None


@dataclass(frozen=True)
class SpecPrefillOutcome:
    """Validated request result; dense_result is only the dense branch."""

    stable_request_uid: str = ""
    sequence_revision: int = -1
    requested: bool | None = None
    engaged: bool = False
    disposition: str = "not_attempted"
    reason: str = "not_requested"
    route: str = "mllm_media"
    model_module: str | None = None
    language_module: str | None = None
    model_type: str | None = None
    stage: str = "admission"
    attempted_tokens: int = 0
    scored_tokens: int = 0
    original_tokens: int = 0
    selected_tokens: int = 0
    cached_tokens: int = 0
    dense_result: str = "not_run"
    sparse_mutation_started: bool = False
    cache_mutation_started: bool = False
    manifest_digest: str | None = None
    terminal_ack_count: int = 0
    terminal_acknowledged: bool = False

    def validate(self) -> None:
        validate_specprefill_outcome(self)


def _validate_outcome_count(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if value > MAX_SAFE_INTEGER:
        raise ValueError(f"{name} exceeds safe IEEE-754 range")


def validate_specprefill_outcome(outcome: SpecPrefillOutcome) -> None:
    """Enforce truthful disposition/reason/dense-branch state combinations."""

    if not isinstance(outcome, SpecPrefillOutcome):
        raise ValueError("SpecPrefill outcome has an invalid type")
    if not isinstance(outcome.stable_request_uid, str) or not _REQUEST_UID_RE.fullmatch(
        outcome.stable_request_uid
    ):
        raise ValueError("stable_request_uid must be a non-empty stable token")
    if (
        not isinstance(outcome.sequence_revision, int)
        or isinstance(outcome.sequence_revision, bool)
        or outcome.sequence_revision < 0
        or outcome.sequence_revision > MAX_SAFE_INTEGER
    ):
        raise ValueError("sequence_revision must be a non-negative safe integer")
    if outcome.disposition not in _DISPOSITIONS:
        raise ValueError(f"unknown SpecPrefill disposition: {outcome.disposition}")
    if not _is_result_reason(outcome.reason):
        raise ValueError(f"unknown SpecPrefill reason: {outcome.reason}")
    if outcome.dense_result not in _DENSE_RESULTS:
        raise ValueError(f"unknown dense_result: {outcome.dense_result}")
    if outcome.stage not in {"admission", "terminal"}:
        raise ValueError("stage must be admission or terminal")
    if not isinstance(outcome.engaged, bool):
        raise ValueError("engaged must be a boolean")
    if outcome.engaged != (outcome.disposition == "engaged"):
        raise ValueError("engaged must agree with disposition")
    if outcome.requested is not None and not isinstance(outcome.requested, bool):
        raise ValueError("requested must be a boolean or null")
    if not isinstance(outcome.terminal_acknowledged, bool):
        raise ValueError("terminal_acknowledged must be a boolean")
    if (
        not isinstance(outcome.terminal_ack_count, int)
        or isinstance(outcome.terminal_ack_count, bool)
        or outcome.terminal_ack_count not in {0, 1}
    ):
        raise ValueError(
            "terminal_acknowledged must represent zero or exactly one acknowledgement"
        )
    if outcome.terminal_acknowledged != (outcome.terminal_ack_count == 1):
        raise ValueError("terminal_acknowledged must agree with terminal_ack_count")
    if (
        outcome.manifest_digest is not None
        and _SHA256_RE.fullmatch(outcome.manifest_digest) is None
    ):
        raise ValueError("manifest_digest must be lowercase SHA-256 hex")
    for name in (
        "attempted_tokens",
        "scored_tokens",
        "original_tokens",
        "selected_tokens",
        "cached_tokens",
    ):
        _validate_outcome_count(getattr(outcome, name), name)
    for name in ("sparse_mutation_started", "cache_mutation_started"):
        if not isinstance(getattr(outcome, name), bool):
            raise ValueError(f"{name} must be a boolean")

    if outcome.disposition == "engaged":
        if outcome.stage != "terminal":
            raise ValueError("engaged outcomes require terminal stage")
        if outcome.reason != "none":
            raise ValueError("engaged outcomes require reason=none")
        if outcome.selected_tokens <= 0:
            raise ValueError("engaged outcomes require positive sparse work")
        if outcome.dense_result != "not_run":
            raise ValueError(
                "engaged outcomes must keep dense_result=not_run; "
                "it only describes the dense bypass/fallback branch"
            )
    elif outcome.disposition == "fallback_dense":
        if outcome.stage != "terminal":
            raise ValueError("dense fallback outcomes require terminal stage")
        if outcome.reason == "none" or outcome.reason == "cancellation":
            raise ValueError("dense fallback requires a triggering reason")
        if outcome.dense_result == "not_run":
            raise ValueError("dense fallback requires a dense branch result")
    elif outcome.disposition == "cancelled":
        if outcome.stage != "terminal":
            raise ValueError("cancelled outcomes require terminal stage")
        if outcome.reason != "cancellation":
            raise ValueError("cancelled outcomes require reason=cancellation")
        if outcome.terminal_ack_count != 1:
            raise ValueError(
                "cancelled outcomes require exactly one terminal acknowledgement"
            )
        if outcome.dense_result not in {"not_run", "cancelled"}:
            raise ValueError("cancelled outcomes cannot report dense success")
    elif outcome.disposition == "failed":
        if outcome.stage != "terminal":
            raise ValueError("failed outcomes require terminal stage")
        if outcome.reason not in _FAILURE_REASONS:
            raise ValueError("failed outcomes require an execution failure reason")
        if outcome.dense_result == "succeeded":
            raise ValueError(
                "failed cannot describe a request completed by dense fallback"
            )
    elif outcome.disposition == "not_attempted":
        if outcome.reason not in _BYPASS_REASONS and not _is_mode_reason(
            outcome.reason
        ):
            raise ValueError(
                "not_attempted outcomes require a policy or capability reason"
            )
        if outcome.sparse_mutation_started or outcome.cache_mutation_started:
            raise ValueError("not_attempted outcomes cannot have sparse mutation")
        if outcome.stage == "admission" and outcome.dense_result != "not_run":
            raise ValueError(
                "admission not_attempted outcomes cannot report dense work"
            )
        if outcome.stage == "terminal" and outcome.dense_result == "not_run":
            raise ValueError(
                "terminal not_attempted outcomes must record the dense branch result"
            )


__all__ = [
    "IDENTITY_CANONICALIZATION",
    "IDENTITY_DIGEST_ALGORITHM",
    "IDENTITY_NUMBER_PROFILE",
    "IDENTITY_SCHEMA_VERSION",
    "MAX_SAFE_INTEGER",
    "SpecPrefillOutcome",
    "build_draft_compatibility",
    "build_identity_manifest",
    "canonical_identity_bytes",
    "freeze_identity_manifest",
    "identity_compatibility_reason",
    "identity_digest",
    "identity_manifest_for",
    "parse_identity_json",
    "validate_identity_manifest",
    "validate_specprefill_outcome",
]
