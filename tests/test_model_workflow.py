# SPDX-License-Identifier: Apache-2.0
"""Tests for model artifact inspection, acquisition, and conversion helpers."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_mlx.model_workflow import (
    ACQUISITION_MARKER_NAME,
    CONVERSION_MANIFEST_NAME,
    MODEL_MANIFEST_NAME,
    QUALIFICATION_REQUEST_NAME,
    REGISTRATION_MANIFEST_NAME,
    AcquisitionOptions,
    ConversionOptions,
    QualificationOptions,
    RegistrationOptions,
    acquire_model,
    convert_model,
    inspect_model,
    qualify_model,
    register_model,
)


def test_inspect_local_model_reports_size_and_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "quantization": {"bits": 4, "group_size": 64},
                "max_position_embeddings": 32768,
            }
        )
    )
    (tmp_path / "model.safetensors").write_bytes(b"x" * 1024)

    payload = inspect_model(str(tmp_path))

    assert payload["source"] == "local"
    assert payload["file_count"] == 2
    assert payload["model_family"]["model_type"] == "qwen3"
    assert payload["mlx"]["looks_like_mlx_artifact"] is True
    assert payload["mlx"]["needs_conversion"] is False


def test_inspect_local_model_collects_metadata_evidence_without_loading_weights(
    tmp_path,
):
    template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    generation_config = {"eos_token_id": 42, "temperature": 0.7}
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fixture",
                "architectures": ["FixtureForCausalLM"],
                "license": "apache-2.0",
                "vision_config": {"model_type": "fixture_vision"},
            }
        )
    )
    (tmp_path / "tokenizer.json").write_text('{"version": "1.0"}')
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 4096, "chat_template": template})
    )
    (tmp_path / "special_tokens_map.json").write_text('{"eos_token": "</s>"}')
    (tmp_path / "generation_config.json").write_text(json.dumps(generation_config))
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    payload = inspect_model(str(tmp_path))
    evidence = payload["metadata_evidence"]

    assert evidence["revision"] == {
        "requested": None,
        "resolved": None,
        "source_kind": "local_directory",
        "resolved_is_immutable": None,
    }
    assert evidence["files"]["config.json"]["source_kind"] == "local_file"
    assert evidence["files"]["config.json"]["source_path"] == str(
        (tmp_path / "config.json").resolve()
    )
    assert (
        evidence["files"]["tokenizer.json"]["sha256"]
        == sha256(b'{"version": "1.0"}').hexdigest()
    )
    assert set(evidence["tokenizer_assets"]) == {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    assert evidence["chat_template"] == {
        "value": template,
        "sha256": sha256(template.encode()).hexdigest(),
        "source_kind": "embedded_json_field",
        "source_path": str((tmp_path / "tokenizer_config.json").resolve()),
        "source_field": "chat_template",
    }
    assert evidence["generation_config"]["data"] == generation_config
    assert evidence["license"] == {
        "identifier": "apache-2.0",
        "source_kind": "embedded_json_field",
        "source_path": str((tmp_path / "config.json").resolve()),
        "source_field": "license",
    }
    assert evidence["capabilities"]["declared_signals"]["vision_config_present"] is True
    assert evidence["capabilities"]["unknown"] == {
        "tool_calling": None,
        "reasoning": None,
        "structured_output": None,
        "parser_support": None,
        "runtime_support": None,
        "local_serving_context": None,
        "qualification": None,
    }


def test_inspect_local_model_uses_file_chat_template_and_preserves_unknowns(tmp_path):
    template = "{{ bos_token }}{% for message in messages %}{{ message }}{% endfor %}"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    (tmp_path / "chat_template.jinja").write_text(template)
    (tmp_path / "LICENSE").write_text("fixture license text")

    payload = inspect_model(str(tmp_path))
    evidence = payload["metadata_evidence"]

    assert evidence["chat_template"] == {
        "value": template,
        "sha256": sha256(template.encode()).hexdigest(),
        "source_kind": "local_file",
        "source_path": str((tmp_path / "chat_template.jinja").resolve()),
        "source_field": None,
    }
    assert evidence["license"] == {
        "identifier": None,
        "source_kind": "local_file",
        "source_path": str((tmp_path / "LICENSE").resolve()),
        "source_field": None,
    }
    assert evidence["generation_config"] == {
        "data": None,
        "sha256": None,
        "source_kind": None,
        "source_path": None,
    }
    assert evidence["capabilities"]["unknown"]["runtime_support"] is None


def test_file_chat_template_overrides_tokenizer_config_template(tmp_path):
    file_template = "{{ messages }}"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ stale_template }}"})
    )
    (tmp_path / "chat_template.jinja").write_text(file_template)

    evidence = inspect_model(str(tmp_path))["metadata_evidence"]

    assert evidence["chat_template"]["value"] == file_template
    assert evidence["chat_template"]["source_kind"] == "local_file"


def test_unreadable_file_template_does_not_fall_back(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ stale_template }}"})
    )
    (tmp_path / "chat_template.jinja").write_bytes(b"\xff\xfe")

    payload = inspect_model(str(tmp_path))

    assert payload["metadata_evidence"]["chat_template"]["value"] is None
    assert "not valid UTF-8" in payload["metadata_evidence"]["chat_template"]["error"]


def test_malformed_json_is_reported_and_empty_generation_config_is_preserved(
    tmp_path,
):
    (tmp_path / "config.json").write_text("{not-json")
    (tmp_path / "generation_config.json").write_text("{}")

    payload = inspect_model(str(tmp_path))
    evidence = payload["metadata_evidence"]

    assert any(
        "could not parse config.json" in warning for warning in payload["warnings"]
    )
    assert "parse_error" in evidence["files"]["config.json"]
    assert evidence["generation_config"]["data"] == {}


def test_inspect_local_metadata_hashes_are_stable_and_optional_files_are_absent(
    tmp_path,
):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fixture"}))

    first = inspect_model(str(tmp_path))["metadata_evidence"]
    second = inspect_model(str(tmp_path))["metadata_evidence"]

    assert first["files"] == second["files"]
    assert (
        first["files"]["config.json"]["sha256"]
        == sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    )
    assert first["tokenizer_assets"] == {}
    assert first["chat_template"]["value"] is None
    assert first["license"]["identifier"] is None


def test_inspect_hf_model_uses_metadata_without_weight_download(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "text_config": {
                    "model_type": "qwen3_5_moe",
                    "mtp_num_hidden_layers": 1,
                    "max_position_embeddings": 1_000_000,
                }
            }
        )
    )
    siblings = [
        SimpleNamespace(rfilename="config.json", size=100),
        SimpleNamespace(rfilename="model-00001-of-00002.safetensors", size=1000),
        SimpleNamespace(rfilename="tokenizer.json", size=200),
    ]
    info = SimpleNamespace(
        sha="a" * 40,
        siblings=siblings,
        card_data={"license": "apache-2.0"},
        library_name="transformers",
        pipeline_tag="text-generation",
        tags=["text-generation", "custom_code"],
    )

    with (
        patch("vllm_mlx.model_workflow.HfApi") as mock_api,
        patch("vllm_mlx.model_workflow.hf_hub_download") as mock_download,
    ):
        mock_api.return_value.model_info.return_value = info
        mock_download.return_value = str(config_path)
        payload = inspect_model("org/model", revision="main")

    assert payload["revision"] == "a" * 40
    assert payload["total_size_bytes"] == 1300
    assert payload["model_files_size_gb"] == 0.0
    assert payload["model_family"]["model_type"] == "qwen3_5_moe"
    assert payload["mlx"]["needs_conversion"] is True
    assert payload["warnings"] == [
        "very large advertised context; choose an explicit serving context before loading"
    ]
    assert payload["metadata_evidence"]["revision"] == {
        "requested": "main",
        "resolved": "a" * 40,
        "source_kind": "huggingface_repository",
        "resolved_is_immutable": True,
    }
    assert all(
        call.kwargs["revision"] == "a" * 40 for call in mock_download.call_args_list
    )
    assert payload["metadata_evidence"]["files"]["config.json"]["source_path"] == (
        "hf://org/model@" + "a" * 40 + "/config.json"
    )
    assert payload["metadata_evidence"]["license"] == {
        "identifier": "apache-2.0",
        "source_kind": "huggingface_model_card",
        "source_path": "hf://org/model@" + "a" * 40 + "/README.md",
        "source_field": "license",
    }
    declared = payload["metadata_evidence"]["capabilities"]["declared_signals"]
    assert declared["pipeline_tag"] == "text-generation"
    assert declared["library_name"] == "transformers"


def test_acquire_model_finalizes_target_and_writes_manifest(tmp_path):
    target = tmp_path / "model-final"
    staging_root = tmp_path / "stage"
    seen_env = {}

    def fake_snapshot_download(*args, **kwargs):
        staging = Path(kwargs["local_dir"])
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "config.json").write_text(
            json.dumps({"model_type": "llama", "quantization": {"bits": 4}})
        )
        (staging / "model.safetensors").write_bytes(b"weights")
        seen_env["fast_transfer"] = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        return str(staging)

    with (
        patch("vllm_mlx.model_workflow.find_spec", return_value=object()),
        patch("vllm_mlx.model_workflow.snapshot_download") as mock_download,
    ):
        mock_download.side_effect = fake_snapshot_download
        manifest = acquire_model(
            "org/model",
            options=AcquisitionOptions(
                revision="a" * 40,
                target_dir=str(target),
                staging_dir=str(staging_root),
                fast_transfer=True,
            ),
        )

    assert seen_env["fast_transfer"] == "1"
    assert target.exists()
    assert manifest["model_id"] == "org/model"
    assert manifest["path"] == str(target)
    assert manifest["fast_transfer"]["enabled"] is True
    assert len(manifest["operation_id"]) == 64
    manifest_path = target / MODEL_MANIFEST_NAME
    assert manifest_path.exists()
    saved = json.loads(manifest_path.read_text())
    assert saved["inspection"]["file_count"] == 2
    journal_path = Path(manifest["operation_journal_path"])
    journal = json.loads(journal_path.read_text())
    assert journal["status"] == "succeeded"
    assert journal["attempt"] == 1


def test_acquire_model_disables_fast_transfer_when_package_missing(tmp_path):
    target = tmp_path / "model-final"

    def fake_snapshot_download(*args, **kwargs):
        staging = Path(kwargs["local_dir"])
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "config.json").write_text(json.dumps({"model_type": "llama"}))
        return str(staging)

    with (
        patch("vllm_mlx.model_workflow.find_spec", return_value=None),
        patch("vllm_mlx.model_workflow.snapshot_download") as mock_download,
    ):
        mock_download.side_effect = fake_snapshot_download
        manifest = acquire_model(
            "org/model",
            options=AcquisitionOptions(
                revision="a" * 40,
                target_dir=str(target),
                fast_transfer=True,
            ),
        )

    assert manifest["fast_transfer"]["requested"] is True
    assert manifest["fast_transfer"]["enabled"] is False
    assert "not installed" in manifest["fast_transfer"]["reason"]


def test_acquire_model_refuses_existing_target(tmp_path):
    target = tmp_path / "model-final"
    target.mkdir()

    with patch("vllm_mlx.model_workflow.snapshot_download") as mock_download:
        with pytest.raises(FileExistsError):
            acquire_model(
                "org/model",
                options=AcquisitionOptions(revision="a" * 40, target_dir=str(target)),
            )

    mock_download.assert_not_called()


def test_targeted_acquisition_requires_immutable_revision(tmp_path):
    with patch("vllm_mlx.model_workflow.snapshot_download") as mock_download:
        with pytest.raises(ValueError, match="immutable"):
            acquire_model(
                "org/model",
                options=AcquisitionOptions(
                    revision="main", target_dir=str(tmp_path / "model")
                ),
            )

    mock_download.assert_not_called()


def test_acquire_model_retries_matching_failed_operation_in_same_staging_path(
    tmp_path,
):
    target = tmp_path / "model-final"
    staging_root = tmp_path / "stage"
    seen_staging = []

    def fake_snapshot_download(*args, **kwargs):
        staging = Path(kwargs["local_dir"])
        seen_staging.append(staging)
        (staging / "partial.bin").write_bytes(b"partial")
        if len(seen_staging) == 1:
            raise RuntimeError("interrupted transfer")
        (staging / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (staging / "model.safetensors").write_bytes(b"weights")
        return str(staging)

    options = AcquisitionOptions(
        revision="a" * 40,
        target_dir=str(target),
        staging_dir=str(staging_root),
        fast_transfer=False,
    )
    with patch(
        "vllm_mlx.model_workflow.snapshot_download",
        side_effect=fake_snapshot_download,
    ):
        with pytest.raises(RuntimeError, match="interrupted transfer"):
            acquire_model("org/model", options=options)
        manifest = acquire_model("org/model", options=options)

    assert seen_staging[0] == seen_staging[1]
    journal = json.loads(Path(manifest["operation_journal_path"]).read_text())
    assert journal["attempt"] == 2
    assert journal["status"] == "succeeded"


def test_acquire_model_rejects_conflicting_identity_for_existing_journal(tmp_path):
    target = tmp_path / "model-final"
    staging_root = tmp_path / "stage"

    with patch(
        "vllm_mlx.model_workflow.snapshot_download",
        side_effect=RuntimeError("failed"),
    ) as mock_download:
        with pytest.raises(RuntimeError, match="failed"):
            acquire_model(
                "org/model",
                options=AcquisitionOptions(
                    revision="a" * 40,
                    target_dir=str(target),
                    staging_dir=str(staging_root),
                ),
            )
        with pytest.raises(ValueError, match="identity conflict"):
            acquire_model(
                "org/model",
                options=AcquisitionOptions(
                    revision="b" * 40,
                    target_dir=str(target),
                    staging_dir=str(staging_root),
                ),
            )

    assert mock_download.call_count == 1


def test_acquire_model_rejects_staging_path_escape_in_matching_journal(tmp_path):
    target = tmp_path / "model-final"
    staging_root = tmp_path / "stage"
    options = AcquisitionOptions(
        revision="a" * 40,
        target_dir=str(target),
        staging_dir=str(staging_root),
        fast_transfer=False,
    )

    with patch(
        "vllm_mlx.model_workflow.snapshot_download",
        side_effect=RuntimeError("failed"),
    ) as mock_download:
        with pytest.raises(RuntimeError, match="failed"):
            acquire_model("org/model", options=options)
        journal_path = next(staging_root.glob("*.acquisition.json"))
        journal = json.loads(journal_path.read_text())
        journal["staging_path"] = str(tmp_path.parent / "escaped")
        journal_path.write_text(json.dumps(journal))

        with pytest.raises(ValueError, match="staging path conflict"):
            acquire_model("org/model", options=options)

    assert mock_download.call_count == 1


def test_acquire_model_rejects_malformed_existing_journal(tmp_path):
    target = tmp_path / "model-final"
    staging_root = tmp_path / "stage"
    options = AcquisitionOptions(
        revision="a" * 40,
        target_dir=str(target),
        staging_dir=str(staging_root),
        fast_transfer=False,
    )

    with patch(
        "vllm_mlx.model_workflow.snapshot_download",
        side_effect=RuntimeError("failed"),
    ) as mock_download:
        with pytest.raises(RuntimeError, match="failed"):
            acquire_model("org/model", options=options)
        journal_path = next(staging_root.glob("*.acquisition.json"))
        journal_path.write_text("{not-json")

        with pytest.raises(ValueError, match="invalid acquisition journal"):
            acquire_model("org/model", options=options)

    assert mock_download.call_count == 1


def test_matching_concurrent_acquisitions_are_serialized(tmp_path):
    target = tmp_path / "model-final"
    staging_root = tmp_path / "stage"
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def fake_snapshot_download(*args, **kwargs):
        staging = Path(kwargs["local_dir"])
        calls.append(staging)
        entered.set()
        assert release.wait(timeout=5)
        (staging / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (staging / "model.safetensors").write_bytes(b"weights")
        return str(staging)

    options = AcquisitionOptions(
        revision="a" * 40,
        target_dir=str(target),
        staging_dir=str(staging_root),
        fast_transfer=False,
    )
    with patch(
        "vllm_mlx.model_workflow.snapshot_download",
        side_effect=fake_snapshot_download,
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(acquire_model, "org/model", options=options)
            assert entered.wait(timeout=5)
            second = executor.submit(acquire_model, "org/model", options=options)
            assert len(calls) == 1
            release.set()
            first_manifest = first.result(timeout=5)
            second_manifest = second.result(timeout=5)

    assert len(calls) == 1
    assert first_manifest["operation_id"] == second_manifest["operation_id"]


def test_different_target_acquisitions_serialize_transfer_environment(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    observed = {}

    def fake_snapshot_download(model_id, *args, **kwargs):
        staging = Path(kwargs["local_dir"])
        observed[model_id] = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        if model_id == "org/first":
            first_entered.set()
            assert release_first.wait(timeout=5)
        else:
            second_entered.set()
        (staging / "config.json").write_text(json.dumps({"model_type": "llama"}))
        return str(staging)

    first_options = AcquisitionOptions(
        revision="a" * 40,
        target_dir=str(tmp_path / "first"),
        staging_dir=str(tmp_path / "stage-first"),
        fast_transfer=True,
    )
    second_options = AcquisitionOptions(
        revision="b" * 40,
        target_dir=str(tmp_path / "second"),
        staging_dir=str(tmp_path / "stage-second"),
        fast_transfer=False,
    )
    with (
        patch("vllm_mlx.model_workflow.find_spec", return_value=object()),
        patch(
            "vllm_mlx.model_workflow.snapshot_download",
            side_effect=fake_snapshot_download,
        ),
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(acquire_model, "org/first", options=first_options)
            assert first_entered.wait(timeout=5)
            second = executor.submit(
                acquire_model, "org/second", options=second_options
            )
            assert not second_entered.wait(timeout=0.1)
            release_first.set()
            first.result(timeout=5)
            second.result(timeout=5)

    assert observed == {"org/first": "1", "org/second": None}
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ


def test_acquire_model_records_cancellation_without_deleting_staging(tmp_path):
    target = tmp_path / "model-final"
    staging_root = tmp_path / "stage"

    with patch(
        "vllm_mlx.model_workflow.snapshot_download",
        side_effect=KeyboardInterrupt(),
    ):
        with pytest.raises(KeyboardInterrupt):
            acquire_model(
                "org/model",
                options=AcquisitionOptions(
                    revision="a" * 40,
                    target_dir=str(target),
                    staging_dir=str(staging_root),
                ),
            )

    journal_path = next(staging_root.glob("*.acquisition.json"))
    journal = json.loads(journal_path.read_text())
    assert journal["status"] == "cancelled"
    assert Path(journal["staging_path"]).exists()


def test_acquire_model_resumes_finalization_after_target_move(tmp_path):
    target = tmp_path / "model-final"
    staging_root = tmp_path / "stage"

    def fake_snapshot_download(*args, **kwargs):
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (staging / "model.safetensors").write_bytes(b"weights")
        return str(staging)

    options = AcquisitionOptions(
        revision="a" * 40,
        target_dir=str(target),
        staging_dir=str(staging_root),
        fast_transfer=False,
    )
    with (
        patch(
            "vllm_mlx.model_workflow.snapshot_download",
            side_effect=fake_snapshot_download,
        ) as mock_download,
        patch(
            "vllm_mlx.model_workflow.inspect_model",
            side_effect=[RuntimeError("inspection interrupted"), {"file_count": 2}],
        ),
    ):
        with pytest.raises(RuntimeError, match="inspection interrupted"):
            acquire_model("org/model", options=options)
        (target / ACQUISITION_MARKER_NAME).unlink()
        manifest = acquire_model("org/model", options=options)

    assert mock_download.call_count == 1
    assert manifest["inspection"] == {"file_count": 2}
    assert (target / MODEL_MANIFEST_NAME).exists()


def test_idempotent_success_removes_matching_leftover_marker(tmp_path):
    target = tmp_path / "model-final"
    staging_root = tmp_path / "stage"

    def fake_snapshot_download(*args, **kwargs):
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text(json.dumps({"model_type": "llama"}))
        return str(staging)

    options = AcquisitionOptions(
        revision="a" * 40,
        target_dir=str(target),
        staging_dir=str(staging_root),
        fast_transfer=False,
    )
    with patch(
        "vllm_mlx.model_workflow.snapshot_download",
        side_effect=fake_snapshot_download,
    ) as mock_download:
        manifest = acquire_model("org/model", options=options)
        marker = target / ACQUISITION_MARKER_NAME
        marker.write_text(json.dumps({"operation_id": manifest["operation_id"]}))

        repeated = acquire_model("org/model", options=options)

    assert repeated["operation_id"] == manifest["operation_id"]
    assert not marker.exists()
    assert mock_download.call_count == 1


def test_convert_model_dry_run_records_mlx_lm_command(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    output = tmp_path / "out"

    payload = convert_model(
        ConversionOptions(
            source_path=str(source),
            output_path=str(output),
            quantize=True,
            q_bits=3,
            q_group_size=64,
            q_mode="affine",
            dry_run=True,
        )
    )

    assert payload["status"] == "dry_run"
    assert payload["backend"] == "mlx-lm"
    assert payload["recipe"]["q_bits"] == 3
    assert "--quantize" in payload["command"]
    assert "--q-bits" in payload["command"]
    assert str(output) in payload["command"]


def test_convert_model_success_writes_manifest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    output = tmp_path / "out"

    def fake_run(*args, **kwargs):
        output.mkdir()
        (output / "config.json").write_text(
            json.dumps({"model_type": "llama", "quantization": {"bits": 4}})
        )
        (output / "model.safetensors").write_bytes(b"weights")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    with patch("vllm_mlx.model_workflow.subprocess.run", side_effect=fake_run):
        payload = convert_model(
            ConversionOptions(source_path=str(source), output_path=str(output))
        )

    assert payload["status"] == "succeeded"
    assert (output / CONVERSION_MANIFEST_NAME).exists()


def test_convert_model_failure_reports_status_and_stderr(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    output = tmp_path / "out"

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="conversion error")

    with patch("vllm_mlx.model_workflow.subprocess.run", side_effect=fake_run):
        payload = convert_model(
            ConversionOptions(source_path=str(source), output_path=str(output))
        )

    assert payload["status"] == "failed"
    assert payload["returncode"] == 1
    assert payload["stderr"] == "conversion error"
    assert "output_inspection" not in payload
    assert "manifest_path" not in payload


def test_inspect_gptq_model_is_not_detected_as_mlx(tmp_path):
    """GPTQ/AWQ quantization_config must not trigger has_mlx_signals."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                },
            }
        )
    )
    (tmp_path / "model.safetensors").write_bytes(b"x" * 1024)

    payload = inspect_model(str(tmp_path))

    assert payload["mlx"]["looks_like_mlx_artifact"] is False
    assert payload["mlx"]["needs_conversion"] is True


def test_register_model_writes_manifest_from_artifact(tmp_path):
    artifact = tmp_path / "mlx-model"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "quantization": {"bits": 4}})
    )
    (artifact / MODEL_MANIFEST_NAME).write_text(
        json.dumps({"kind": "vllm-mlx-model-artifact", "model_id": "org/model"})
    )

    payload = register_model(
        RegistrationOptions(
            artifact_path=str(artifact),
            model_id="qwen-test",
            served_model_name="qwen-test-served",
            preset_alias="fast-qwen",
            mllm=True,
            tool_call_parser="qwen3_coder",
            reasoning_parser="qwen3",
            default_temperature=0.6,
            default_top_p=0.95,
            default_top_k=20,
            default_min_p=0.0,
            default_presence_penalty=0.0,
            default_repetition_penalty=1.0,
            chat_template_kwargs={"enable_thinking": True},
            feature_flags=["prefix_cache"],
        )
    )

    assert payload["kind"] == "vllm-mlx-model-registration"
    assert payload["model_id"] == "qwen-test"
    assert payload["served_model_name"] == "qwen-test-served"
    assert payload["preset_alias"] == "fast-qwen"
    assert payload["mllm"] is True
    assert payload["production_ready"] is False
    assert payload["qualification_required"] is True
    assert payload["serving_defaults"]["top_k"] == 20
    assert payload["serving_defaults"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert payload["parser_policy"]["reasoning_parser"] == "qwen3"
    assert payload["source_manifests"]["acquisition"]["payload"]["model_id"] == (
        "org/model"
    )
    assert (artifact / REGISTRATION_MANIFEST_NAME).exists()


def test_register_model_minimal_defaults(tmp_path):
    """register_model with only artifact_path derives model_id from directory name."""
    artifact = tmp_path / "my-cool-model"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps({"model_type": "llama", "quantization": {"bits": 4}})
    )

    payload = register_model(RegistrationOptions(artifact_path=str(artifact)))

    assert payload["model_id"] == "my-cool-model"
    assert payload["served_model_name"] == "my-cool-model"
    assert payload["preset_alias"] is None
    assert payload["mllm"] is None
    assert payload["serving_defaults"] == {}
    assert payload["parser_policy"] == {}
    assert payload["feature_flags"] == []
    assert payload["qualification_required"] is True
    assert (artifact / REGISTRATION_MANIFEST_NAME).exists()


def test_register_model_requires_local_directory(tmp_path):
    missing = tmp_path / "missing"

    try:
        register_model(RegistrationOptions(artifact_path=str(missing)))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_register_model_rejects_file_as_artifact(tmp_path):
    """register_model raises NotADirectoryError for a file path."""
    file_path = tmp_path / "not-a-dir.safetensors"
    file_path.write_bytes(b"weights")

    try:
        register_model(RegistrationOptions(artifact_path=str(file_path)))
    except NotADirectoryError:
        pass
    else:
        raise AssertionError("expected NotADirectoryError")


def test_qualify_model_dry_run_records_bench_command(tmp_path):
    output = tmp_path / QUALIFICATION_REQUEST_NAME

    payload = qualify_model(
        QualificationOptions(
            model_id="qwen-test",
            server_url="http://127.0.0.1:8090",
            workload_path="/tmp/workload.json",
            output_path=str(output),
            result_path="/tmp/results.json",
            repetitions=3,
            dry_run=True,
            extra_args=["--tag", "nightly"],
        )
    )

    assert payload["status"] == "dry_run"
    assert payload["production_ready"] is False
    assert "--workload" in payload["command"]
    assert "/tmp/workload.json" in payload["command"]
    assert "--tag" in payload["command"]
    assert output.exists()


def test_qualify_model_runs_command_and_records_success(tmp_path):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="all passed", stderr="")

    with patch("vllm_mlx.model_workflow.subprocess.run", side_effect=fake_run):
        payload = qualify_model(
            QualificationOptions(
                model_id="qwen-test",
                workload_path="/tmp/workload.json",
                output_path=str(tmp_path / "result.json"),
            )
        )

    assert payload["status"] == "succeeded"
    assert payload["returncode"] == 0
    assert payload["stdout"] == "all passed"
    assert "completed_at" in payload
    assert (tmp_path / "result.json").exists()


def test_qualify_model_runs_command_and_records_failure(tmp_path):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=7, stdout="", stderr="bad workload")

    with patch("vllm_mlx.model_workflow.subprocess.run", side_effect=fake_run):
        payload = qualify_model(
            QualificationOptions(
                model_id="qwen-test",
                workload_path="/tmp/workload.json",
            )
        )

    assert payload["status"] == "failed"
    assert payload["returncode"] == 7
    assert payload["stderr"] == "bad workload"


def test_drop_none_preserves_zero_and_false_values():
    """_drop_none must keep 0, 0.0, and False -- only drop None."""
    from vllm_mlx.model_workflow import _drop_none

    result = _drop_none(
        {
            "temperature": 0.0,
            "top_k": 0,
            "presence_penalty": 0.0,
            "enabled": False,
            "missing": None,
        }
    )
    assert result == {
        "temperature": 0.0,
        "top_k": 0,
        "presence_penalty": 0.0,
        "enabled": False,
    }
    assert "missing" not in result
