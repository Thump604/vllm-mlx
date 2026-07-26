# SPDX-License-Identifier: Apache-2.0
"""Private engine for bounded legacy ModelProfile compatibility imports."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from vllm_mlx._model_profile_compat_types import (
    CompatibilityIssue,
    LegacySourceInput,
    ModelProfileImportResult,
    ProvenanceKind,
    SourceKind,
)
from vllm_mlx._model_profile_serving_compat import (
    import_registry as _import_registry,
    import_serving as _import_serving,
)


def _is_immutable_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 40 <= len(value) <= 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_source_input(
    kind: SourceKind, value: LegacySourceInput | Mapping[str, Any] | None
) -> LegacySourceInput | None:
    if value is None:
        return None
    if isinstance(value, LegacySourceInput):
        if value.kind != kind:
            raise ValueError(f"expected {kind} source, received {value.kind}")
        return LegacySourceInput.from_mapping(
            kind,
            {
                "location": value.location,
                "sha256": value.sha256,
                "payload": value.payload,
            },
        )
    if isinstance(value, Mapping):
        return LegacySourceInput.from_mapping(kind, value)
    raise TypeError(f"{kind} source must be a LegacySourceInput or mapping")


class _Importer:
    def __init__(self, sources: tuple[LegacySourceInput, ...]) -> None:
        self.sources = sources
        self.profile: dict[str, Any] = {"schema_version": 1}
        self.issues: list[CompatibilityIssue] = []
        self._field_sources: dict[str, LegacySourceInput] = {}
        self._provenance: dict[tuple[SourceKind, ProvenanceKind], list[str]] = {}

    def assign(
        self,
        pointer: str,
        value: Any,
        source: LegacySourceInput,
        provenance_kind: ProvenanceKind,
    ) -> None:
        """Assign a copied fact, retaining the first value and reporting conflict."""
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
        inspection = self._mapping_field(
            payload, "inspection", source, "/acquisition/inspection"
        )
        family = self._mapping_field(
            inspection,
            "model_family",
            source,
            "/acquisition/inspection/model_family",
        )
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
        output_inspection = payload.get("output_inspection")
        if payload.get("status") != "succeeded" or not isinstance(
            output_inspection, Mapping
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
        inspection = output_inspection
        family = self._mapping_field(
            inspection,
            "model_family",
            source,
            "/conversion/output_inspection/model_family",
        )
        self.assign("/artifact/format", "mlx", source, "derived_recommendation")
        self._import_inspection(family, inspection, source, "derived_recommendation")
        recipe = self._mapping_field(payload, "recipe", source, "/conversion/recipe")
        for pointer, key in (
            ("/artifact/quantization/bits", "q_bits"),
            ("/artifact/quantization/group_size", "q_group_size"),
            ("/artifact/quantization/mode", "q_mode"),
        ):
            self.assign(pointer, recipe.get(key), source, "derived_recommendation")
        if recipe:
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
        provenance_kind: ProvenanceKind,
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
        quantization = self._mapping_field(
            family,
            "quantization",
            source,
            "/inspection/model_family/quantization",
        )
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
        defaults = self._mapping_field(
            payload,
            "serving_defaults",
            source,
            "/registration/serving_defaults",
        )
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
        parsers = self._mapping_field(
            payload,
            "parser_policy",
            source,
            "/registration/parser_policy",
        )
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
        feature_flags = payload.get("feature_flags")
        if isinstance(feature_flags, list):
            for feature in feature_flags:
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
        _import_registry(source, self.assign, self.issues)

    def import_cli_server(self, source: LegacySourceInput) -> None:
        _import_serving(source, self.assign, self.issues)

    def _mapping_field(
        self,
        payload: Mapping[str, Any],
        field: str,
        source: LegacySourceInput,
        pointer: str,
    ) -> Mapping[str, Any]:
        value = payload.get(field)
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return value
        self.issues.append(
            CompatibilityIssue(
                code="invalid_source_field_shape",
                severity="error",
                pointer=pointer,
                sources=(source.location,),
                detail=f"legacy field {field!r} must be an object",
            )
        )
        return {}

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


def _provenance_metadata(
    source: LegacySourceInput, provenance_kind: ProvenanceKind
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
    inspection = _mapping_or_empty(payload.get("inspection"))
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


def _import_legacy_sources(
    source_inputs: tuple[
        tuple[SourceKind, LegacySourceInput | Mapping[str, Any] | None], ...
    ],
) -> ModelProfileImportResult:
    source_values = tuple(
        _normalize_source_input(kind, value) for kind, value in source_inputs
    )
    sources = tuple(source for source in source_values if source is not None)
    if not sources:
        raise ValueError("at least one legacy source is required")
    importer = _Importer(sources)
    dispatch: dict[SourceKind, Callable[[LegacySourceInput], None]] = {
        "acquisition": importer.import_acquisition,
        "conversion": importer.import_conversion,
        "registration": importer.import_registration,
        "registry": importer.import_registry,
        "cli_server": importer.import_cli_server,
    }
    for source in sources:
        dispatch[source.kind](source)
    return importer.finish()
