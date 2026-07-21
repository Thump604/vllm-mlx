# SPDX-License-Identifier: Apache-2.0
"""Pure import of legacy model-workflow records into ModelProfile v1 fragments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from vllm_mlx.model_profile import validate_import_result, validate_model_profile

SourceKind = Literal[
    "acquisition",
    "conversion",
    "registration",
    "registry",
    "cli_server",
    "qualification",
]

_SOURCE_ORDER: tuple[SourceKind, ...] = (
    "acquisition",
    "conversion",
    "registration",
    "registry",
    "cli_server",
    "qualification",
)
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class LegacySourceInput:
    """An already-loaded input document plus source identity and payload."""

    kind: SourceKind
    location: str
    sha256: str
    payload: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls, kind: SourceKind, value: Mapping[str, Any]
    ) -> "LegacySourceInput":
        """Create an input wrapper; output source descriptors omit payloads."""
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise TypeError(f"{kind} source must contain a mapping 'payload'")
        location = value.get("location")
        sha256 = value.get("sha256")
        if not isinstance(location, str) or not location:
            raise ValueError(f"{kind} source must contain a non-empty 'location'")
        if not isinstance(sha256, str) or not _is_sha256(sha256):
            raise ValueError(f"{kind} source must contain a SHA-256 'sha256'")
        return cls(kind=kind, location=location, sha256=sha256, payload=payload)


@dataclass(frozen=True)
class CompatibilityIssue:
    """A deterministic reason an imported fragment cannot be treated as complete."""

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
    """The bounded v1 compatibility envelope; it never changes runtime state."""

    complete: bool
    sources: tuple[LegacySourceInput, ...]
    profile: Mapping[str, Any] | None
    issues: tuple[CompatibilityIssue, ...]

    def as_dict(self) -> dict[str, Any]:
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


def finalize_legacy_model_profile(
    imported: ModelProfileImportResult,
    completed_profile: Mapping[str, Any],
    *,
    profile_schema: Mapping[str, Any],
    import_schema: Mapping[str, Any],
) -> ModelProfileImportResult:
    """Validate an explicit completion without discarding imported facts.

    Legacy inputs are intentionally insufficient on their own. The caller must
    supply a complete profile, and every fact imported from legacy sources must
    remain unchanged. Only missing-fact errors are resolved by that explicit
    completion; conflicts and malformed source evidence remain blocking.
    """
    if imported.complete:
        raise ValueError("legacy import is already complete")
    if imported.profile is None:
        raise ValueError("legacy import does not contain a profile fragment")

    blocking = tuple(
        issue
        for issue in imported.issues
        if issue.severity == "error" and issue.code != "missing_required_fact"
    )
    if blocking:
        codes = ", ".join(sorted({issue.code for issue in blocking}))
        raise ValueError(f"legacy import has unresolved errors: {codes}")

    conflicts = _fragment_conflicts(imported.profile, completed_profile)
    if conflicts:
        raise ValueError(
            "completed profile changes imported facts: " + ", ".join(conflicts)
        )

    candidate = deepcopy(dict(completed_profile))
    validate_model_profile(candidate, profile_schema)
    result = ModelProfileImportResult(
        complete=True,
        sources=imported.sources,
        profile=candidate,
        issues=tuple(issue for issue in imported.issues if issue.severity == "warning"),
    )
    validate_import_result(result.as_dict(), import_schema, profile_schema)
    return result


def _fragment_conflicts(
    fragment: Mapping[str, Any], candidate: Mapping[str, Any], pointer: str = ""
) -> tuple[str, ...]:
    conflicts: list[str] = []
    for key, value in fragment.items():
        child_pointer = f"{pointer}/{key}"
        if key not in candidate:
            conflicts.append(child_pointer)
            continue
        candidate_value = candidate[key]
        if isinstance(value, Mapping):
            if not isinstance(candidate_value, Mapping):
                conflicts.append(child_pointer)
            else:
                conflicts.extend(
                    _fragment_conflicts(value, candidate_value, child_pointer)
                )
        elif child_pointer == "/provenance/records" and isinstance(value, list):
            if not isinstance(candidate_value, list) or any(
                record not in candidate_value for record in value
            ):
                conflicts.append(child_pointer)
        elif candidate_value != value:
            conflicts.append(child_pointer)
    return tuple(conflicts)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _source(
    kind: SourceKind, value: LegacySourceInput | Mapping[str, Any] | None
) -> LegacySourceInput | None:
    if value is None:
        return None
    if isinstance(value, LegacySourceInput):
        if value.kind != kind:
            raise ValueError(f"expected {kind} source, received {value.kind}")
        if not value.location or not _is_sha256(value.sha256):
            raise ValueError(f"{kind} source has invalid location or SHA-256")
        return value
    if isinstance(value, Mapping):
        return LegacySourceInput.from_mapping(kind, value)
    raise TypeError(f"{kind} source must be a LegacySourceInput or mapping")


class _Importer:
    def __init__(self, sources: tuple[LegacySourceInput, ...]) -> None:
        self.sources = sources
        self.profile: dict[str, Any] = {"schema_version": 1}
        self.issues: list[CompatibilityIssue] = []
        self._field_sources: dict[str, LegacySourceInput] = {}
        self._provenance: dict[tuple[SourceKind, str], list[str]] = {}

    def assign(
        self,
        pointer: str,
        value: Any,
        source: LegacySourceInput,
        provenance_kind: str,
    ) -> None:
        if value is None:
            return
        parts = pointer.lstrip("/").split("/")
        target = self.profile
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        key = parts[-1]
        if key in target:
            if target[key] != value:
                first = self._field_sources[pointer]
                self.issues.append(
                    CompatibilityIssue(
                        code="conflicting_value",
                        severity="error",
                        pointer=pointer,
                        sources=(first.location, source.location),
                        detail=(
                            f"{first.kind} supplies {target[key]!r}; "
                            f"{source.kind} supplies {value!r}."
                        ),
                    )
                )
            return
        target[key] = deepcopy(value)
        self._field_sources[pointer] = source
        self._provenance.setdefault((source.kind, provenance_kind), []).append(pointer)

    def import_acquisition(self, source: LegacySourceInput) -> None:
        payload = source.payload
        inspection = _mapping(payload.get("inspection"))
        family = _mapping(inspection.get("model_family"))
        self.assign("/identity/provider", "huggingface", source, "provider_fact")
        self.assign(
            "/identity/repository_id", payload.get("model_id"), source, "provider_fact"
        )
        self.assign(
            "/identity/requested_revision",
            payload.get("revision"),
            source,
            "provider_fact",
        )
        resolved_revision = payload.get("resolved_revision")
        if _is_immutable_revision(resolved_revision):
            self.assign(
                "/identity/resolved_revision",
                resolved_revision,
                source,
                "provider_fact",
            )
        self.assign(
            "/artifact/source_uri", payload.get("model_id"), source, "provider_fact"
        )
        self._import_inspection(family, inspection, source, "provider_fact")

    def import_conversion(self, source: LegacySourceInput) -> None:
        payload = source.payload
        if payload.get("status") != "succeeded" or not isinstance(
            payload.get("output_inspection"), Mapping
        ):
            self.issues.append(
                CompatibilityIssue(
                    code="conversion_output_unverified",
                    severity="error",
                    pointer="/artifact",
                    sources=(source.location,),
                    detail=(
                        "only a succeeded conversion with output inspection can "
                        "contribute artifact facts"
                    ),
                )
            )
            return
        inspection = _mapping(payload.get("output_inspection"))
        family = _mapping(inspection.get("model_family"))
        self.assign("/artifact/format", "mlx", source, "derived_recommendation")
        self._import_inspection(family, inspection, source, "derived_recommendation")
        recipe = _mapping(payload.get("recipe"))
        if recipe:
            self.assign(
                "/artifact/quantization/bits",
                recipe.get("q_bits"),
                source,
                "derived_recommendation",
            )
            self.assign(
                "/artifact/quantization/group_size",
                recipe.get("q_group_size"),
                source,
                "derived_recommendation",
            )
            self.assign(
                "/artifact/quantization/mode",
                recipe.get("q_mode"),
                source,
                "derived_recommendation",
            )
            self.assign(
                "/artifact/quantization/source",
                "conversion_recipe",
                source,
                "derived_recommendation",
            )

    def _import_inspection(
        self,
        family: Mapping[str, Any],
        inspection: Mapping[str, Any],
        source: LegacySourceInput,
        provenance_kind: str,
    ) -> None:
        self.assign(
            "/artifact/model_type", family.get("model_type"), source, provenance_kind
        )
        self.assign(
            "/artifact/architectures",
            family.get("architectures"),
            source,
            provenance_kind,
        )
        self.assign(
            "/artifact/dtype", family.get("torch_dtype"), source, provenance_kind
        )
        self.assign(
            "/artifact/size_bytes",
            inspection.get("total_size_bytes"),
            source,
            provenance_kind,
        )
        quantization = _mapping(family.get("quantization"))
        self.assign(
            "/artifact/quantization/method",
            quantization.get("method", quantization.get("quant_method")),
            source,
            provenance_kind,
        )
        self.assign(
            "/artifact/quantization/bits",
            quantization.get("bits"),
            source,
            provenance_kind,
        )
        self.assign(
            "/artifact/quantization/group_size",
            quantization.get("group_size"),
            source,
            provenance_kind,
        )

    def import_registration(self, source: LegacySourceInput) -> None:
        payload = source.payload
        self.assign(
            "/identity/artifact_id",
            payload.get("artifact_id"),
            source,
            "maintainer_policy",
        )
        if payload.get("model_id") is not None and payload.get("artifact_id") is None:
            self.issues.append(
                CompatibilityIssue(
                    code="ambiguous_registration_model_id",
                    severity="error",
                    pointer="/identity/artifact_id",
                    sources=(source.location,),
                    detail=(
                        "registration model_id is a generic serving override and "
                        "does not establish immutable artifact identity"
                    ),
                )
            )
        self.assign(
            "/identity/served_model_name",
            payload.get("served_model_name"),
            source,
            "maintainer_policy",
        )
        alias = payload.get("preset_alias")
        if alias is not None:
            self.assign("/identity/aliases", [alias], source, "maintainer_policy")
        defaults = _mapping(payload.get("serving_defaults"))
        sampling = {
            key: defaults[key]
            for key in (
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "repetition_penalty",
            )
            if key in defaults
        }
        if sampling:
            self.assign(
                "/serving/sampling/profile_defaults",
                sampling,
                source,
                "maintainer_policy",
            )
        if "chat_template_kwargs" in defaults:
            self.assign(
                "/serving/template/default_kwargs",
                defaults["chat_template_kwargs"],
                source,
                "maintainer_policy",
            )
        parsers = _mapping(payload.get("parser_policy"))
        self.assign(
            "/serving/parsers/tool",
            parsers.get("tool_call_parser"),
            source,
            "maintainer_policy",
        )
        self.assign(
            "/serving/parsers/reasoning",
            parsers.get("reasoning_parser"),
            source,
            "maintainer_policy",
        )
        for feature in payload.get("feature_flags") or []:
            self.issues.append(
                CompatibilityIssue(
                    code="untyped_feature_declaration",
                    severity="warning",
                    pointer=f"/serving/features/{feature}",
                    sources=(source.location,),
                    detail=(
                        "registration feature_flags do not establish a v1 "
                        "feature mode or control."
                    ),
                )
            )
        if payload.get("production_ready") is not None:
            self.issues.append(
                CompatibilityIssue(
                    code="qualification_boolean_ignored",
                    severity="warning",
                    pointer="/qualification",
                    sources=(source.location,),
                    detail=(
                        "registration production_ready is not qualification "
                        "evidence."
                    ),
                )
            )

    def import_registry(self, source: LegacySourceInput) -> None:
        payload = source.payload
        # Registry names and sources are not identity aliases or artifact IDs.
        self._import_serving(payload, source, "maintainer_policy")
        for field in (
            "estimated_memory_bytes",
            "estimated_memory_gb",
            "force_mllm",
            "gpu_memory_utilization",
            "mllm",
            "model",
            "name",
            "path",
            "prefill_step_size",
            "preload",
            "source",
            "stream_interval",
        ):
            if field in payload:
                self.issues.append(
                    CompatibilityIssue(
                        code="registry_field_requires_target_contract",
                        severity="warning",
                        pointer="/serving",
                        sources=(source.location,),
                        detail=(
                            f"registry field {field!r} has no lossless ModelProfile "
                            "v1 destination"
                        ),
                    )
                )

    def import_cli_server(self, source: LegacySourceInput) -> None:
        self._import_serving(source.payload, source, "maintainer_policy")

    def _import_serving(
        self,
        payload: Mapping[str, Any],
        source: LegacySourceInput,
        provenance_kind: str,
    ) -> None:
        serving = _mapping(payload.get("serving"))
        values = dict(payload)
        for key, value in serving.items():
            if key in values and values[key] != value:
                self.issues.append(
                    CompatibilityIssue(
                        code="same_source_conflict",
                        severity="error",
                        pointer=f"/serving/{key}",
                        sources=(source.location,),
                        detail=(
                            f"top-level value {values[key]!r} conflicts with "
                            f"nested serving value {value!r}; nested value retained"
                        ),
                    )
                )
            values[key] = value
        self._report_unknown_serving_fields(values, source)
        for name in ("engine", "route"):
            self.assign(f"/serving/{name}", values.get(name), source, provenance_kind)
        for name in (
            "template",
            "parsers",
            "sampling",
            "limits",
            "features",
            "activation_policy",
            "request_policy",
        ):
            if name == "features":
                self._report_unknown_features(values.get(name), source)
            else:
                self._report_unknown_closed_fields(name, values.get(name), source)
                self._report_invalid_policy_values(name, values.get(name), source)
            value = _closed_serving_value(name, values.get(name))
            for key, item in value.items():
                self.assign(f"/serving/{name}/{key}", item, source, provenance_kind)
        # These are the current CLI field names; map only their direct semantics.
        if "continuous_batching" in values:
            batching = values["continuous_batching"]
            if self._assign_boolean_feature(
                "continuous_batching", batching, source, provenance_kind
            ):
                self.assign(
                    "/serving/engine",
                    "batched" if batching else "simple",
                    source,
                    provenance_kind,
                )
        if "enable_mtp" in values:
            self._assign_boolean_feature(
                "mtp", values["enable_mtp"], source, provenance_kind
            )
        if "specprefill_enabled" in values or "specprefill" in values:
            enabled = values.get("specprefill_enabled", values.get("specprefill"))
            settings = {
                "draft_model": values[key]
                for key in ("specprefill_draft_model",)
                if key in values
            }
            for key in (
                "specprefill_threshold",
                "specprefill_keep_pct",
                "specprefill_backbone_pct",
            ):
                if key in values:
                    self.issues.append(
                        CompatibilityIssue(
                            code="unsupported_feature_setting",
                            severity="warning",
                            pointer="/serving/features/specprefill/settings",
                            sources=(source.location,),
                            detail=f"{key} has no v1 feature-settings field.",
                        )
                    )
            self._assign_boolean_feature(
                "specprefill", enabled, source, provenance_kind, settings
            )
        for legacy, pointer in (
            ("max_tokens", "/serving/limits/max_output_tokens"),
            ("max_request_tokens", "/serving/limits/max_request_output_tokens"),
            ("max_kv_size", "/serving/limits/max_kv_size"),
        ):
            self.assign(pointer, values.get(legacy), source, provenance_kind)

    def _assign_boolean_feature(
        self,
        name: str,
        value: Any,
        source: LegacySourceInput,
        provenance_kind: str,
        settings: Mapping[str, Any] | None = None,
    ) -> bool:
        if not isinstance(value, bool):
            self.issues.append(
                CompatibilityIssue(
                    code="invalid_feature_declaration",
                    severity="error",
                    pointer=f"/serving/features/{name}",
                    sources=(source.location,),
                    detail=(
                        "legacy feature declarations must be boolean to map " "into v1."
                    ),
                )
            )
            return False
        feature: dict[str, Any] = {
            "mode": "enabled_by_default" if value else "guarded_off",
            "control": "none",
            "reason": (
                "enabled in the imported legacy configuration; override support "
                "is not established"
                if value
                else "disabled in the imported legacy configuration; feature "
                "support is not established"
            ),
        }
        if settings:
            feature["settings"] = deepcopy(dict(settings))
        self.assign(f"/serving/features/{name}", feature, source, provenance_kind)
        return True

    def _report_unknown_features(self, value: Any, source: LegacySourceInput) -> None:
        features = _mapping(value)
        for name in sorted(set(features) - _FEATURE_NAMES):
            self.issues.append(
                CompatibilityIssue(
                    code="unknown_feature_declaration",
                    severity="error",
                    pointer=f"/serving/features/{name}",
                    sources=(source.location,),
                    detail=(
                        "the legacy feature is outside the closed v1 feature "
                        "vocabulary."
                    ),
                )
            )
        for name in sorted(set(features) & _FEATURE_NAMES):
            feature = _mapping(features[name])
            for field in sorted(set(feature) - _FEATURE_FIELDS):
                self.issues.append(
                    CompatibilityIssue(
                        code="unknown_serving_field",
                        severity="warning",
                        pointer=f"/serving/features/{name}/{field}",
                        sources=(source.location,),
                        detail="field is outside the closed v1 feature vocabulary",
                    )
                )
            settings = _mapping(feature.get("settings"))
            for setting in sorted(set(settings) - _FEATURE_SETTING_NAMES):
                self.issues.append(
                    CompatibilityIssue(
                        code="unsupported_feature_setting",
                        severity="warning",
                        pointer=f"/serving/features/{name}/settings/{setting}",
                        sources=(source.location,),
                        detail=(
                            "the legacy setting is outside the closed v1 feature "
                            "vocabulary."
                        ),
                    )
                )

    def _report_unknown_serving_fields(
        self, values: Mapping[str, Any], source: LegacySourceInput
    ) -> None:
        for field in sorted(set(values) - _SERVING_TOP_LEVEL_FIELDS):
            if source.kind == "registry" and field in _REGISTRY_NON_PROFILE_FIELDS:
                continue
            self.issues.append(
                CompatibilityIssue(
                    code="unknown_serving_field",
                    severity="warning",
                    pointer=f"/serving/{field}",
                    sources=(source.location,),
                    detail=(
                        "legacy serving field has no declared ModelProfile v1 "
                        "mapping"
                    ),
                )
            )

    def _report_unknown_closed_fields(
        self, name: str, value: Any, source: LegacySourceInput
    ) -> None:
        raw = _mapping(value)
        for field in sorted(set(raw) - _CLOSED_SERVING_FIELDS[name]):
            self.issues.append(
                CompatibilityIssue(
                    code="unknown_serving_field",
                    severity="warning",
                    pointer=f"/serving/{name}/{field}",
                    sources=(source.location,),
                    detail=(
                        "field is outside the closed ModelProfile v1 serving "
                        "vocabulary"
                    ),
                )
            )
        if name == "sampling":
            for defaults_name in ("provider_defaults", "profile_defaults"):
                defaults = _mapping(raw.get(defaults_name))
                for field in sorted(set(defaults) - _SAMPLING_FIELDS):
                    self.issues.append(
                        CompatibilityIssue(
                            code="unknown_serving_field",
                            severity="warning",
                            pointer=f"/serving/sampling/{defaults_name}/{field}",
                            sources=(source.location,),
                            detail="field is outside the closed v1 sampling vocabulary",
                        )
                    )

    def _report_invalid_policy_values(
        self, name: str, value: Any, source: LegacySourceInput
    ) -> None:
        if value is not None and not isinstance(value, Mapping):
            self.issues.append(
                CompatibilityIssue(
                    code="invalid_policy_shape",
                    severity="error",
                    pointer=f"/serving/{name}",
                    sources=(source.location,),
                    detail="policy must be an object",
                )
            )
            return
        raw = _mapping(value)
        invalid: list[tuple[str, Any]] = []
        if name == "activation_policy":
            fields = raw.get("owner_override_fields")
            if fields is not None and not isinstance(fields, list):
                invalid.append(("owner_override_fields", fields))
                fields = []
            for field in fields or []:
                if not isinstance(field, str) or field not in _ACTIVATION_FIELDS:
                    invalid.append(("owner_override_fields", field))
        elif name == "request_policy":
            required = raw.get("required_fields")
            if required is not None and not isinstance(required, Mapping):
                invalid.append(("required_fields", required))
                required = {}
            for field in _mapping(required):
                if not isinstance(field, str) or field not in _REQUEST_FIELDS:
                    invalid.append(("required_fields", field))
            for collection in ("allowed_fields", "forbidden_fields"):
                fields = raw.get(collection)
                if fields is not None and not isinstance(fields, list):
                    invalid.append((collection, fields))
                    fields = []
                for field in fields or []:
                    if not isinstance(field, str) or field not in _REQUEST_FIELDS:
                        invalid.append((collection, field))
        for collection, field in invalid:
            self.issues.append(
                CompatibilityIssue(
                    code="invalid_policy_field",
                    severity="error",
                    pointer=f"/serving/{name}/{collection}",
                    sources=(source.location,),
                    detail=f"field {field!r} is outside the closed v1 policy vocabulary",
                )
            )

    def import_qualification(self, source: LegacySourceInput) -> None:
        payload = source.payload
        valid_evidence = _qualification_evidence(payload.get("evidence"))
        if valid_evidence is not None:
            self.assign(
                "/qualification/evidence", valid_evidence, source, "measured_result"
            )
        status = payload.get("qualification_status")
        if status in {"not_qualified", "failed", "blocked"}:
            self.assign("/qualification/status", status, source, "measured_result")
        if status == "qualified" and valid_evidence:
            self.assign("/qualification/status", status, source, "measured_result")
        if valid_evidence is None and (
            payload.get("production_ready") is not None
            or payload.get("status") is not None
            or status == "qualified"
        ):
            self.issues.append(
                CompatibilityIssue(
                    code="qualification_evidence_missing",
                    severity="error",
                    pointer="/qualification/evidence",
                    sources=(source.location,),
                    detail=(
                        "command status and production_ready do not establish "
                        "evidence bound to a subject digest."
                    ),
                )
            )

    def finish(self) -> ModelProfileImportResult:
        if self._provenance:
            records = []
            for (kind, provenance_kind), paths in self._provenance.items():
                source = next(item for item in self.sources if item.kind == kind)
                revision, rule_id, observed_at = _provenance_metadata(
                    source, provenance_kind
                )
                records.append(
                    {
                        "field_paths": paths,
                        "kind": provenance_kind,
                        "source": source.location,
                        "revision": revision,
                        "sha256": source.sha256,
                        "rule_id": rule_id,
                        "observed_at": observed_at,
                    }
                )
            self.profile["provenance"] = {"records": records}
        present = set(self._field_sources)
        for pointer in _REQUIRED_FACT_POINTERS:
            if pointer not in present:
                self.issues.append(
                    CompatibilityIssue(
                        code="missing_required_fact",
                        severity="error",
                        pointer=pointer,
                        sources=tuple(source.location for source in self.sources),
                        detail=(
                            "current legacy sources do not establish this "
                            "required ModelProfile fact."
                        ),
                    )
                )
        return ModelProfileImportResult(
            complete=False,
            sources=self.sources,
            profile=deepcopy(self.profile),
            issues=tuple(self.issues),
        )


_FEATURE_NAMES = frozenset(
    {
        "continuous_batching",
        "constrained_json",
        "kvq4",
        "kvq8",
        "mtp",
        "prefix_cache",
        "specprefill",
        "streaming",
    }
)
_FEATURE_SETTING_NAMES = frozenset(
    {
        "bits",
        "cache_memory_mb",
        "cache_type",
        "draft_model",
        "group_size",
        "max_concurrency",
        "num_draft_tokens",
    }
)
_FEATURE_FIELDS = frozenset({"mode", "control", "control_field", "settings", "reason"})
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
_ACTIVATION_FIELDS = frozenset(
    {
        "features.continuous_batching",
        "features.kvq4",
        "features.kvq8",
        "features.mtp",
        "features.prefix_cache",
        "features.specprefill",
        "limits.max_kv_size",
        "limits.max_output_tokens",
        "limits.max_request_output_tokens",
        "limits.serving_context",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "chat_template_kwargs.enable_thinking",
        "chat_template_kwargs.preserve_thinking",
        "max_tokens",
        "min_p",
        "presence_penalty",
        "reasoning_effort",
        "repetition_penalty",
        "response_format",
        "seed",
        "stream",
        "temperature",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    }
)
_CLOSED_SERVING_FIELDS = {
    "template": frozenset({"source", "sha256", "default_kwargs"}),
    "parsers": frozenset({"tool", "reasoning"}),
    "sampling": frozenset({"provider_defaults", "profile_defaults"}),
    "limits": frozenset(
        {
            "advertised_context",
            "serving_context",
            "max_output_tokens",
            "max_request_output_tokens",
            "max_kv_size",
        }
    ),
    "activation_policy": frozenset({"owner_override_fields"}),
    "request_policy": frozenset(
        {"required_fields", "allowed_fields", "forbidden_fields"}
    ),
}
_REGISTRY_NON_PROFILE_FIELDS = frozenset(
    {
        "estimated_memory_bytes",
        "estimated_memory_gb",
        "force_mllm",
        "gpu_memory_utilization",
        "mllm",
        "model",
        "name",
        "path",
        "prefill_step_size",
        "preload",
        "source",
        "stream_interval",
    }
)
_SERVING_TOP_LEVEL_FIELDS = (
    frozenset(
        {
            "activation_policy",
            "continuous_batching",
            "enable_mtp",
            "engine",
            "features",
            "limits",
            "max_kv_size",
            "max_request_tokens",
            "max_tokens",
            "parsers",
            "request_policy",
            "route",
            "sampling",
            "serving",
            "specprefill",
            "specprefill_backbone_pct",
            "specprefill_draft_model",
            "specprefill_enabled",
            "specprefill_keep_pct",
            "specprefill_threshold",
            "template",
        }
    )
    | _REGISTRY_NON_PROFILE_FIELDS
)


def _is_immutable_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 40 <= len(value) <= 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


_REQUIRED_FACT_POINTERS = (
    "/profile_id",
    "/profile_revision",
    "/subject_digest",
    "/identity/provider",
    "/identity/resolved_revision",
    "/identity/artifact_id",
    "/identity/served_model_name",
    "/artifact/source_uri",
    "/artifact/format",
    "/artifact/model_type",
    "/artifact/architectures",
    "/artifact/quantization/method",
    "/artifact/hashes/config_sha256",
    "/artifact/hashes/tokenizer_sha256",
    "/artifact/hashes/chat_template_sha256",
    "/artifact/hashes/weights_manifest_sha256",
    "/capabilities/modalities",
    "/capabilities/streaming",
    "/capabilities/tools",
    "/capabilities/reasoning",
    "/capabilities/structured_output",
    "/capabilities/api_surfaces",
    "/serving/engine",
    "/serving/route",
    "/serving/template/source",
    "/serving/template/sha256",
    "/serving/template/default_kwargs",
    "/serving/parsers/tool",
    "/serving/parsers/reasoning",
    "/serving/sampling/provider_defaults",
    "/serving/sampling/profile_defaults",
    "/serving/limits/advertised_context",
    "/serving/limits/serving_context",
    "/serving/limits/max_output_tokens",
    "/serving/limits/max_request_output_tokens",
    "/serving/features/continuous_batching",
    "/serving/features/constrained_json",
    "/serving/features/kvq4",
    "/serving/features/kvq8",
    "/serving/features/mtp",
    "/serving/features/prefix_cache",
    "/serving/features/specprefill",
    "/serving/features/streaming",
    "/serving/activation_policy/owner_override_fields",
    "/serving/request_policy/required_fields",
    "/serving/request_policy/allowed_fields",
    "/serving/request_policy/forbidden_fields",
    "/qualification/status",
    "/qualification/evidence",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _provenance_metadata(
    source: LegacySourceInput, provenance_kind: str
) -> tuple[str | None, str | None, str | None]:
    payload = source.payload
    revision = payload.get("resolved_revision")
    if not _is_immutable_revision(revision):
        revision = None
    rule_id = (
        None
        if provenance_kind == "provider_fact"
        else f"model-profile-compat-v1:{source.kind}"
    )
    inspection = _mapping(payload.get("inspection"))
    observed_at = next(
        (
            value
            for value in (
                payload.get("completed_at"),
                payload.get("created_at"),
                inspection.get("inspected_at"),
            )
            if isinstance(value, str) and value
        ),
        None,
    )
    return revision, rule_id, observed_at


def _closed_serving_value(name: str, value: Any) -> dict[str, Any]:
    """Copy only vocabulary owned by the closed v1 serving schema."""
    raw = _mapping(value)
    allowed = {
        "template": {"source", "sha256", "default_kwargs"},
        "parsers": {"tool", "reasoning"},
        "sampling": {"provider_defaults", "profile_defaults"},
        "limits": {
            "advertised_context",
            "serving_context",
            "max_output_tokens",
            "max_request_output_tokens",
            "max_kv_size",
        },
        "features": _FEATURE_NAMES,
        "activation_policy": {"owner_override_fields"},
        "request_policy": {"required_fields", "allowed_fields", "forbidden_fields"},
    }[name]
    result = {key: deepcopy(raw[key]) for key in allowed if key in raw}
    if name == "sampling":
        for key, item in list(result.items()):
            result[key] = {
                field: deepcopy(field_value)
                for field, field_value in _mapping(item).items()
                if field in _SAMPLING_FIELDS
            }
    if name == "features":
        for key, item in list(result.items()):
            result[key] = {
                field: deepcopy(field_value)
                for field, field_value in _mapping(item).items()
                if field in _FEATURE_FIELDS
            }
            if "settings" in result[key]:
                result[key]["settings"] = {
                    field: deepcopy(field_value)
                    for field, field_value in _mapping(result[key]["settings"]).items()
                    if field in _FEATURE_SETTING_NAMES
                }
    if name == "activation_policy" and "owner_override_fields" in result:
        fields = result["owner_override_fields"]
        result["owner_override_fields"] = (
            [
                field
                for field in fields
                if isinstance(field, str) and field in _ACTIVATION_FIELDS
            ]
            if isinstance(fields, list)
            else []
        )
    if name == "request_policy":
        required = _mapping(result.get("required_fields"))
        if "required_fields" in result:
            result["required_fields"] = {
                field: deepcopy(field_value)
                for field, field_value in required.items()
                if isinstance(field, str) and field in _REQUEST_FIELDS
            }
        for collection in ("allowed_fields", "forbidden_fields"):
            if collection in result:
                fields = result[collection]
                result[collection] = (
                    [
                        field
                        for field in fields
                        if isinstance(field, str) and field in _REQUEST_FIELDS
                    ]
                    if isinstance(fields, list)
                    else []
                )
    return result


def _qualification_evidence(value: Any) -> list[dict[str, Any]] | None:
    """Accept only passing evidence records that bind all v1 qualification facts."""
    if not isinstance(value, list) or not value:
        return None
    required = (
        "evidence_id",
        "kind",
        "location",
        "artifact_sha256",
        "result",
        "hardware_fingerprint",
        "workload_id",
        "subject_digest",
        "created_at",
    )
    required_keys = set(required)
    records: list[dict[str, Any]] = []
    for item in value:
        if (
            not isinstance(item, Mapping)
            or not required_keys <= item.keys()
            or not set(item).issubset(required_keys)
        ):
            return None
        if item["result"] != "pass":
            return None
        if not _is_sha256(item["artifact_sha256"]) or not _is_sha256(
            item["subject_digest"]
        ):
            return None
        records.append({key: deepcopy(item[key]) for key in required})
    return records


def import_legacy_model_profile(
    *,
    acquisition: LegacySourceInput | Mapping[str, Any] | None = None,
    conversion: LegacySourceInput | Mapping[str, Any] | None = None,
    registration: LegacySourceInput | Mapping[str, Any] | None = None,
    registry_entry: LegacySourceInput | Mapping[str, Any] | None = None,
    cli_server: LegacySourceInput | Mapping[str, Any] | None = None,
    qualification: LegacySourceInput | Mapping[str, Any] | None = None,
) -> ModelProfileImportResult:
    """Map legacy source records without reading files or changing runtime state.

    Mapping arguments use ``{"location", "sha256", "payload"}`` wrappers so the
    result can retain the exact source identity required by the v1 envelope.
    """
    source_values = (
        _source("acquisition", acquisition),
        _source("conversion", conversion),
        _source("registration", registration),
        _source("registry", registry_entry),
        _source("cli_server", cli_server),
        _source("qualification", qualification),
    )
    sources = tuple(source for source in source_values if source is not None)
    if not sources:
        raise ValueError("at least one legacy source is required")
    importer = _Importer(sources)
    for source in sources:
        getattr(importer, f"import_{source.kind}")(source)
    return importer.finish()
