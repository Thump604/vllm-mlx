# SPDX-License-Identifier: Apache-2.0
"""Pure-Python contract tests for SpecPrefill identity and outcomes."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import hashlib

import pytest

from vllm_mlx.specprefill_contract import (
    SpecPrefillOutcome,
    build_draft_compatibility,
    build_identity_manifest,
    canonical_identity_bytes,
    freeze_identity_manifest,
    governed_target_identity_reason,
    identity_compatibility_reason,
    identity_digest,
    identity_manifest_for,
    parse_identity_json,
    validate_identity_manifest,
    validate_specprefill_outcome,
)


def _cache(
    *,
    model_id: str = "Qwen/Qwen3.8-27B",
    revision: str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    artifact_id: str = "Qwen3.8-27B-8bit-MLX",
    artifact_digest: str = "1" * 64,
    config_digest: str = "2" * 64,
    tokenizer_file_digest: str = "3" * 64,
    template_digest: str = "4" * 64,
    parser_digest: str = "5" * 64,
    family: str = "qwen3.5",
    architecture: str = "Qwen3_5ForConditionalGeneration",
    model_type: str = "qwen3_5",
    media_token_ids: list[int] | None = None,
):
    return {
        "model_id": model_id,
        "model_revision": revision,
        "artifact_id": artifact_id,
        "artifact_digest": artifact_digest,
        "weight_index_digest": "6" * 64,
        "family": family,
        "architecture": architecture,
        "model_module": "mlx_vlm.models.qwen3_5.qwen3_5",
        "language_module": "mlx_vlm.models.qwen3_5.language",
        "model_type": model_type,
        "config_digest": config_digest,
        "tokenizer": {
            "files": [
                {"path": "tokenizer.json", "sha256": tokenizer_file_digest},
                {"path": "tokenizer_config.json", "sha256": "7" * 64},
            ],
            "config_digest": "8" * 64,
            "implementation": "transformers",
            "implementation_version": "5.0.0",
            "added_tokens": [
                {"id": 248044, "token": "<|endoftext|>"},
                {"id": 248056, "token": "<|image_pad|>"},
            ],
            "special_tokens": [
                {"id": 248044, "token": "<|endoftext|>"},
                {"id": 248046, "token": "<|im_end|>"},
            ],
            "encode_probes": [
                {"text": "hello", "ids": [1, 2, 3]},
                {"text": "你好", "ids": [4, 5]},
            ],
        },
        "chat_template": {
            "name": "qwen3.8-installed",
            "version": "1",
            "sha256": template_digest,
        },
        "parser": {
            "name": "qwen3_xml",
            "version": "1",
            "sha256": parser_digest,
        },
        "vision": {
            "enabled": True,
            "config_digest": "9" * 64,
            "processor_files": [{"path": "processor_config.json", "sha256": "a" * 64}],
            "media_token_ids": media_token_ids or [248056, 248057],
            "media_mapping_digest": "b" * 64,
        },
        "cache_schema": {
            "name": "ws1-prefix-cache",
            "version": "promoted-api-v1",
            "sha256": "c" * 64,
        },
        "mode": {
            "continuous_batching": False,
            "serialized": True,
            "mtp": False,
            "chunked_prefill": False,
            "rotating_cache": False,
            "ple": False,
            "qsa": False,
            "audio": False,
            "capability_modes": ["media", "text"],
        },
    }


def _protocol(*, template_digest: str = "4" * 64, parser_digest: str = "5" * 64):
    return {
        "template_digest": template_digest,
        "parser_digest": parser_digest,
        "tool_schema_digest": "d" * 64,
        "renderer": "qwen3.8-runtime-renderer",
        "version": "1",
    }


def _pair(*, draft_cache=None, target_cache=None):
    target_cache = target_cache or _cache()
    draft_cache = draft_cache or _cache(
        model_id="Qwen/Qwen3.5-Draft",
        revision="draft-revision",
        artifact_id="Qwen3.5-Draft-MLX",
        artifact_digest="e" * 64,
        config_digest="f" * 64,
        family="qwen3.5",
        architecture="Qwen3_5ForCausalLM",
        model_type="qwen3_5",
    )
    relation = build_draft_compatibility(
        target_cache, draft_cache, relation="qwen3.8-27b-draft-v1"
    )
    target = build_identity_manifest(
        target_cache,
        _protocol(),
        role="target",
        draft_compatibility=relation,
    )
    draft = build_identity_manifest(
        draft_cache,
        _protocol(),
        role="draft",
        draft_compatibility=relation,
    )
    return target, draft


def _governed_target_fields(target):
    return (
        ("role", target["role"]),
        ("request_protocol_identity", deepcopy(target["request_protocol_identity"])),
        ("draft_compatibility", deepcopy(target["draft_compatibility"])),
    )


def _governed_cache_digest(target):
    return hashlib.sha256(
        canonical_identity_bytes(target["model_cache_identity"])
    ).hexdigest()


def test_wire_format_and_round_trip_are_explicit():
    target, _ = _pair()

    assert target["schema_version"] == "specprefill.identity.v1"
    assert target["canonicalization"] == "rfc8785"
    assert target["digest_algorithm"] == "sha256"
    assert len(target["digest"]) == 64
    validate_identity_manifest(target)

    parsed = parse_identity_json(canonical_identity_bytes(target).decode("utf-8"))
    assert parsed == target


def test_missing_or_null_top_level_digest_uses_public_value_error_surface():
    target, _ = _pair()
    target.pop("digest")

    with pytest.raises(ValueError, match="manifest.digest"):
        validate_identity_manifest(target)

    target, _ = _pair()
    target["digest"] = None
    with pytest.raises(ValueError, match="manifest.digest"):
        validate_identity_manifest(target)


def test_jcs_official_string_vector_and_utf16_property_order():
    # RFC 8785 Appendix B's string/literal shape, without unsupported floats.
    assert (
        canonical_identity_bytes(
            {"strings": ["€", "𝄞"], "literals": [None, True, False]}
        ).decode("utf-8")
        == '{"literals":[null,true,false],"strings":["€","𝄞"]}'
    )
    # JCS sorts object properties by UTF-16 code units, not Unicode code points.
    assert canonical_identity_bytes({"𐀀": 1, "": 2}).decode("utf-8") == (
        '{"𐀀":1,"":2}'
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "1"),
        (1.0, "1"),
        (1e-6, "0.000001"),
        (1e-7, "1e-7"),
        (1e20, "100000000000000000000"),
        (1e21, "1e+21"),
        (-0.0, "0"),
    ],
)
def test_ieee754_number_boundaries_use_ecmascript_number_format(value, expected):
    assert canonical_identity_bytes({"n": value}).decode("utf-8") == (
        '{"n":' + expected + "}"
    )


def test_unsafe_python_integers_and_nonfinite_numbers_are_rejected():
    canonical_identity_bytes({"n": (1 << 53) - 1})
    with pytest.raises(ValueError, match="safe IEEE-754 integer"):
        canonical_identity_bytes({"n": 1 << 53})
    with pytest.raises(ValueError, match="finite"):
        canonical_identity_bytes({"n": float("nan")})
    with pytest.raises(ValueError, match="finite"):
        canonical_identity_bytes({"n": float("inf")})


def test_control_unicode_and_lone_surrogates_are_handled_explicitly():
    assert canonical_identity_bytes({"x": "\b\n\t"}).decode("utf-8") == (
        '{"x":"\\b\\n\\t"}'
    )
    with pytest.raises(ValueError, match="surrogate"):
        canonical_identity_bytes({"x": "\ud800"})
    with pytest.raises(ValueError, match="surrogate"):
        canonical_identity_bytes({"\ud800": 1})


def test_duplicate_keys_and_nonfinite_json_are_rejected():
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        parse_identity_json('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="finite"):
        parse_identity_json('{"a": NaN}')


def test_set_like_identity_collections_sort_but_probe_sequence_does_not():
    target, _ = _pair()
    reordered = deepcopy(target)
    tokenizer = reordered["model_cache_identity"]["tokenizer"]
    tokenizer["files"].reverse()
    tokenizer["added_tokens"].reverse()
    tokenizer["special_tokens"].reverse()
    reordered["model_cache_identity"]["mode"]["capability_modes"].reverse()
    reordered["model_cache_identity"]["vision"]["media_token_ids"].reverse()

    assert identity_digest(target) == identity_digest(reordered)
    reordered["model_cache_identity"]["tokenizer"]["encode_probes"].reverse()
    assert identity_digest(target) != identity_digest(reordered)


@pytest.mark.parametrize("missing_or_null", ["missing", "null"])
def test_absent_and_null_required_identity_fields_fail_closed(missing_or_null):
    target, _ = _pair()
    if missing_or_null == "missing":
        del target["model_cache_identity"]["mode"]["audio"]
    else:
        target["model_cache_identity"]["mode"]["audio"] = None
    target["digest"] = "0" * 64

    with pytest.raises(ValueError, match="mode.audio"):
        validate_identity_manifest(target)
    assert (
        identity_manifest_for(SimpleNamespace(specprefill_identity_manifest=target))
        is None
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda cache: cache["tokenizer"]["added_tokens"].append(
            {"id": 248044, "token": "duplicate-id"}
        ),
        lambda cache: cache["tokenizer"]["special_tokens"].append(
            {"id": 248047, "token": "<|endoftext|>"}
        ),
        lambda cache: cache["tokenizer"]["files"].append(
            {"path": "../escape", "sha256": "1" * 64}
        ),
        lambda cache: cache["vision"]["media_token_ids"].append(248056),
    ],
)
def test_set_like_records_are_strictly_validated_before_digest(mutator):
    cache = _cache()
    mutator(cache)
    with pytest.raises(ValueError):
        build_identity_manifest(
            cache,
            _protocol(),
            role="target",
            draft_compatibility=build_draft_compatibility(
                _cache(), _cache(model_id="draft"), relation="r"
            ),
        )


def test_multiple_identity_sources_must_agree():
    target, _ = _pair()
    conflict = deepcopy(target)
    conflict["model_cache_identity"]["config_digest"] = "9" * 64
    conflict["digest"] = identity_digest(conflict)
    owner = SimpleNamespace(
        specprefill_identity_manifest=target,
        runtime_identity_manifest=conflict,
    )
    assert identity_manifest_for(owner) is None

    same = deepcopy(target)
    same["model_cache_identity"]["tokenizer"]["files"].reverse()
    same["digest"] = identity_digest(same)
    resolved = identity_manifest_for(
        SimpleNamespace(
            specprefill_identity_manifest=target,
            runtime_identity_manifest=same,
        )
    )
    assert resolved is not None
    assert resolved["digest"] == target["digest"]


def test_frozen_manifest_can_be_resolved_again_without_mutable_list_assumptions():
    target, _ = _pair()
    frozen = freeze_identity_manifest(target)
    resolved = identity_manifest_for(frozen)
    assert resolved is not None
    assert resolved["digest"] == target["digest"]


def test_model_cache_and_protocol_identity_are_separate():
    target, draft = _pair()
    target_owner = SimpleNamespace(specprefill_identity_manifest=target)
    draft_owner = SimpleNamespace(specprefill_identity_manifest=draft)
    assert identity_compatibility_reason(target_owner, draft_owner) is None

    changed_protocol = deepcopy(draft)
    changed_protocol["request_protocol_identity"]["tool_schema_digest"] = "e" * 64
    changed_protocol["digest"] = identity_digest(changed_protocol)
    assert (
        identity_compatibility_reason(
            target_owner,
            SimpleNamespace(specprefill_identity_manifest=changed_protocol),
        )
        is None
    )

    changed_template = deepcopy(draft)
    changed_template["model_cache_identity"]["chat_template"]["sha256"] = "f" * 64
    changed_template["digest"] = identity_digest(changed_template)
    assert (
        identity_compatibility_reason(
            target_owner,
            SimpleNamespace(specprefill_identity_manifest=changed_template),
        )
        == "template_parser_mismatch"
    )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("template_digest", "e" * 64, "template_parser_mismatch"),
        ("parser_digest", "f" * 64, "template_parser_mismatch"),
        ("renderer", "other-renderer", "draft_mismatch"),
        ("version", "2", "draft_mismatch"),
    ],
)
def test_draft_protocol_relation_rejects_wrong_protocol(field, value, reason):
    target, draft = _pair()
    changed = deepcopy(draft)
    changed["request_protocol_identity"][field] = value
    changed["digest"] = identity_digest(changed)

    assert (
        identity_compatibility_reason(
            SimpleNamespace(specprefill_identity_manifest=target),
            SimpleNamespace(specprefill_identity_manifest=changed),
        )
        == reason
    )


def test_governed_target_identity_requires_complete_external_anchors():
    target, _ = _pair()
    fields = _governed_target_fields(target)
    digest = _governed_cache_digest(target)

    assert (
        governed_target_identity_reason(
            target,
            expected_model_cache_digest=digest,
            expected_identity_fields=fields,
            registry_complete=True,
        )
        is None
    )

    assert (
        governed_target_identity_reason(
            target,
            expected_model_cache_digest=digest,
            expected_identity_fields=(("role", "target"),),
            registry_complete=True,
        )
        == "cache_unsafe"
    )
    assert (
        governed_target_identity_reason(
            target,
            expected_model_cache_digest=None,
            expected_identity_fields=fields,
            registry_complete=True,
        )
        == "cache_unsafe"
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda manifest: manifest.update({"role": "draft"}),
        lambda manifest: manifest["model_cache_identity"].update(
            {"model_revision": "wrong-revision"}
        ),
        lambda manifest: manifest["model_cache_identity"].update(
            {"config_digest": "d" * 64}
        ),
        lambda manifest: manifest["model_cache_identity"]["chat_template"].update(
            {"sha256": "e" * 64}
        ),
        lambda manifest: manifest["model_cache_identity"]["parser"].update(
            {"sha256": "f" * 64}
        ),
        lambda manifest: manifest["model_cache_identity"]["vision"].update(
            {"media_mapping_digest": "a" * 64}
        ),
        lambda manifest: manifest["draft_compatibility"].update(
            {"relation": "wrong-relation"}
        ),
        lambda manifest: manifest["request_protocol_identity"].update(
            {"renderer": "wrong-renderer"}
        ),
    ],
)
def test_governed_target_identity_rejects_mutated_or_rebound_manifest(mutator):
    target, _ = _pair()
    expected_fields = _governed_target_fields(target)
    expected_digest = _governed_cache_digest(target)
    mutated = deepcopy(target)
    mutator(mutated)
    # Rebinding the top-level digest must not turn an unauthorized identity
    # mutation into an eligible target.
    mutated["digest"] = identity_digest(mutated)

    assert (
        governed_target_identity_reason(
            mutated,
            expected_model_cache_digest=expected_digest,
            expected_identity_fields=expected_fields,
            registry_complete=True,
        )
        == "cache_unsafe"
    )


def test_copied_manifest_digest_is_rejected_before_governed_admission():
    target, _ = _pair()
    copied = deepcopy(target)
    copied["model_cache_identity"]["config_digest"] = "d" * 64

    assert (
        identity_manifest_for(SimpleNamespace(specprefill_identity_manifest=copied))
        is None
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda cache: cache.update({"model_revision": "wrong-revision"}),
        lambda cache: cache.update({"family": "unrelated-family"}),
        lambda cache: cache.update({"architecture": "OtherArchitecture"}),
        lambda cache: cache.update({"config_digest": "8" * 64}),
        lambda cache: cache["vision"].update({"media_mapping_digest": "9" * 64}),
    ],
)
def test_declared_draft_relation_rejects_structurally_copied_wrong_draft(mutator):
    target, _ = _pair()
    wrong_cache = _cache(
        model_id="Qwen3.5-Draft",
        revision="draft-revision",
        artifact_id="Qwen3.5-Draft-MLX",
        artifact_digest="e" * 64,
        config_digest="f" * 64,
        family="qwen3.5",
        architecture="Qwen3_5ForCausalLM",
        model_type="qwen3_5",
    )
    mutator(wrong_cache)
    relation = build_draft_compatibility(
        _cache(), wrong_cache, relation="qwen3.8-27b-draft-v1"
    )
    wrong = build_identity_manifest(
        wrong_cache,
        _protocol(),
        role="draft",
        draft_compatibility=relation,
    )
    # Copying target tokenizer/template fields does not turn an unrelated
    # family/config/media relation into an eligible draft.
    assert (
        identity_compatibility_reason(
            SimpleNamespace(specprefill_identity_manifest=target),
            SimpleNamespace(specprefill_identity_manifest=wrong),
        )
        == "draft_mismatch"
    )

    missing_relation = deepcopy(wrong)
    missing_relation.pop("draft_compatibility")
    missing_relation["digest"] = "0" * 64
    assert (
        identity_compatibility_reason(
            SimpleNamespace(specprefill_identity_manifest=target),
            SimpleNamespace(specprefill_identity_manifest=missing_relation),
        )
        == "draft_mismatch"
    )


def _outcome(**overrides):
    values = {
        "stable_request_uid": "req-123",
        "sequence_revision": 4,
        "requested": True,
        "stage": "terminal",
        "disposition": "engaged",
        "engaged": True,
        "reason": "none",
        "selected_tokens": 32,
        "scored_tokens": 64,
        "dense_result": "not_run",
        "terminal_ack_count": 0,
        "terminal_acknowledged": False,
    }
    values.update(overrides)
    return SpecPrefillOutcome(**values)


def test_result_state_invariants_and_dense_branch_semantics():
    validate_specprefill_outcome(_outcome())
    validate_specprefill_outcome(
        _outcome(
            disposition="fallback_dense",
            engaged=False,
            reason="scoring_failed",
            sparse_mutation_started=True,
            cache_mutation_started=True,
            dense_result="succeeded",
        )
    )
    validate_specprefill_outcome(
        _outcome(
            stage="admission",
            disposition="not_attempted",
            engaged=False,
            reason="length_cap",
            requested=True,
            selected_tokens=0,
        )
    )
    validate_specprefill_outcome(
        _outcome(
            disposition="not_attempted",
            engaged=False,
            reason="length_cap",
            selected_tokens=0,
            dense_result="succeeded",
        )
    )


def test_mode_reason_vocabulary_is_closed():
    validate_specprefill_outcome(
        _outcome(
            stage="admission",
            disposition="not_attempted",
            engaged=False,
            reason="mode_incompatible_mtp",
            selected_tokens=0,
        )
    )
    with pytest.raises(ValueError, match="unknown SpecPrefill reason"):
        validate_specprefill_outcome(
            _outcome(
                stage="admission",
                disposition="not_attempted",
                engaged=False,
                reason="mode_incompatible_untrusted_flag",
                selected_tokens=0,
            )
        )

    with pytest.raises(ValueError, match="dense_result"):
        validate_specprefill_outcome(_outcome(dense_result="succeeded"))
    with pytest.raises(ValueError, match="terminal acknowledgement"):
        validate_specprefill_outcome(
            _outcome(
                disposition="cancelled",
                engaged=False,
                reason="cancellation",
                dense_result="cancelled",
            )
        )
    with pytest.raises(ValueError, match="stable_request_uid"):
        validate_specprefill_outcome(_outcome(stable_request_uid=""))
    with pytest.raises(ValueError, match="sequence_revision"):
        validate_specprefill_outcome(_outcome(sequence_revision=-1))
    with pytest.raises(ValueError, match="dense fallback"):
        validate_specprefill_outcome(
            _outcome(
                disposition="fallback_dense",
                engaged=False,
                reason="scoring_failed",
                dense_result="not_run",
            )
        )


def test_cancelled_result_requires_exactly_one_ack_bound_to_uid_and_revision():
    validate_specprefill_outcome(
        _outcome(
            disposition="cancelled",
            engaged=False,
            reason="cancellation",
            dense_result="cancelled",
            terminal_ack_count=1,
            terminal_acknowledged=True,
        )
    )
    with pytest.raises(ValueError, match="acknowledged"):
        validate_specprefill_outcome(
            _outcome(
                disposition="cancelled",
                engaged=False,
                reason="cancellation",
                dense_result="cancelled",
                terminal_ack_count=2,
                terminal_acknowledged=True,
            )
        )
