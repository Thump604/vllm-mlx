# SPDX-License-Identifier: Apache-2.0
"""Model acquisition, inspection, and conversion workflow helpers.

The functions in this module intentionally avoid loading model weights. They
collect repository/file metadata, download artifacts, and record manifests so a
model can be qualified before it is served.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from importlib.util import find_spec
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import fcntl

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .model_profile import compute_subject_digest
from .utils.download import LLM_ALLOW_PATTERNS, MLLM_ALLOW_PATTERNS

MODEL_MANIFEST_NAME = "vllm_mlx_model_manifest.json"
CONVERSION_MANIFEST_NAME = "vllm_mlx_conversion_manifest.json"
REGISTRATION_MANIFEST_NAME = "vllm_mlx_registration_manifest.json"
QUALIFICATION_REQUEST_NAME = "vllm_mlx_qualification_request.json"
ACQUISITION_MARKER_NAME = ".vllm_mlx_acquisition_operation.json"
ACQUISITION_JOURNAL_VERSION = 1
CONVERSION_MARKER_NAME = ".vllm_mlx_conversion_operation.json"
CONVERSION_JOURNAL_VERSION = 1

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_IMMUTABLE_HF_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
_METADATA_FILENAMES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "license",
)
_INTERNAL_WORKFLOW_FILENAMES = frozenset(
    {ACQUISITION_MARKER_NAME, CONVERSION_MARKER_NAME, CONVERSION_MANIFEST_NAME}
)
_ACQUISITION_ENV_LOCK = threading.RLock()


@dataclass(frozen=True)
class AcquisitionOptions:
    """Options for Hugging Face model acquisition."""

    revision: str | None = None
    target_dir: str | None = None
    staging_dir: str | None = None
    is_mllm: bool = False
    fast_transfer: bool = True
    local_files_only: bool = False


@dataclass(frozen=True)
class ConversionOptions:
    """Options for the mlx-lm conversion backend."""

    source_path: str
    output_path: str
    quantize: bool = False
    q_bits: int | None = None
    q_group_size: int | None = None
    q_mode: str | None = None
    quant_predicate: str | None = None
    dtype: str | None = None
    trust_remote_code: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class RegistrationOptions:
    """Options for generating a portable model registration manifest."""

    artifact_path: str
    model_id: str | None = None
    served_model_name: str | None = None
    preset_alias: str | None = None
    output_path: str | None = None
    mllm: bool | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    default_presence_penalty: float | None = None
    default_repetition_penalty: float | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    feature_flags: list[str] | None = None


@dataclass(frozen=True)
class QualificationOptions:
    """Options for creating or running a bench-serve qualification handoff."""

    model_id: str
    server_url: str = "http://127.0.0.1:8080"
    workload_path: str | None = None
    output_path: str | None = None
    result_path: str | None = None
    repetitions: int | None = None
    dry_run: bool = False
    extra_args: list[str] | None = None
    profile_path: str | None = None
    evidence_output_path: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bytes_to_gb(size: int | float | None) -> float | None:
    if size is None:
        return None
    return round(float(size) / (1024**3), 3)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_json_strict(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label} at {path}: expected a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_file_record(
    path: Path,
    *,
    source_kind: str,
    source_path: str,
) -> dict[str, Any]:
    """Describe a small inspected metadata file without interpreting weights."""
    return {
        "source_kind": source_kind,
        "source_path": source_path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _hf_source_path(model_id: str, revision: str | None, filename: str) -> str:
    resolved = revision or "unresolved"
    return f"hf://{model_id}@{resolved}/{filename}"


def _available_metadata_files(files: list[dict[str, Any]]) -> set[str]:
    return {
        str(entry["path"])
        for entry in files
        if isinstance(entry.get("path"), str)
        and str(entry["path"]) in _METADATA_FILENAMES
    }


def _local_metadata_files(path: Path) -> dict[str, tuple[Path, str, str]]:
    metadata: dict[str, tuple[Path, str, str]] = {}
    for filename in _METADATA_FILENAMES:
        candidate = path / filename
        if candidate.is_file():
            metadata[filename] = (
                candidate,
                "local_file",
                str(candidate.resolve()),
            )
    return metadata


def _hf_metadata_files(
    model_id: str,
    *,
    revision: str | None,
    local_files_only: bool,
    files: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, tuple[Path, str, str]]:
    metadata: dict[str, tuple[Path, str, str]] = {}
    filenames = _available_metadata_files(files) | {"config.json"}
    for filename in sorted(filenames):
        try:
            cached_path = Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            )
        except Exception as exc:
            warnings.append(f"could not read {filename}: {exc}")
            continue
        metadata[filename] = (
            cached_path,
            "huggingface_hub_file",
            _hf_source_path(model_id, revision, filename),
        )
    return metadata


def _json_metadata(
    record: tuple[Path, str, str] | None,
    *,
    filename: str,
    warnings: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if record is None:
        return None, None
    path, source_kind, source_path = record
    file_record = _metadata_file_record(
        path, source_kind=source_kind, source_path=source_path
    )
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        warnings.append(f"could not parse {filename}: {exc}")
        file_record["parse_error"] = str(exc)
        return None, file_record
    if not isinstance(value, dict):
        detail = "top-level JSON value must be an object"
        warnings.append(f"could not parse {filename}: {detail}")
        file_record["parse_error"] = detail
        return None, file_record
    return value, file_record


def _text_metadata(
    record: tuple[Path, str, str] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    if record is None:
        return None, None
    path, source_kind, source_path = record
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return None, _metadata_file_record(
            path, source_kind=source_kind, source_path=source_path
        )
    return text, _metadata_file_record(
        path, source_kind=source_kind, source_path=source_path
    )


def _chat_template_metadata(
    tokenizer_config: dict[str, Any],
    metadata_files: dict[str, tuple[Path, str, str]],
    file_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    template_record = metadata_files.get("chat_template.jinja")
    template, source = _text_metadata(template_record)
    if template_record is not None and template is None:
        return {
            "value": None,
            "sha256": source["sha256"] if source else None,
            "source_kind": source["source_kind"] if source else None,
            "source_path": source["source_path"] if source else None,
            "source_field": None,
            "error": "chat_template.jinja is not valid UTF-8",
        }
    if template is not None:
        return {
            "value": template,
            "sha256": source["sha256"] if source else None,
            "source_kind": source["source_kind"] if source else None,
            "source_path": source["source_path"] if source else None,
            "source_field": None,
        }

    template = tokenizer_config.get("chat_template")
    if isinstance(template, (str, list, dict)):
        source = file_records.get("tokenizer_config.json")
        encoded = (
            template.encode()
            if isinstance(template, str)
            else json.dumps(
                template, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        )
        return {
            "value": template,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "source_kind": "embedded_json_field",
            "source_path": source["source_path"] if source else None,
            "source_field": "chat_template",
        }
    return {
        "value": None,
        "sha256": None,
        "source_kind": None,
        "source_path": None,
        "source_field": None,
    }


def _license_metadata(
    config: dict[str, Any],
    repository_metadata: dict[str, Any],
    repository_source: str | None,
    resolved_revision: str | None,
    metadata_files: dict[str, tuple[Path, str, str]],
    file_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    repository_license = repository_metadata.get("license")
    if isinstance(repository_license, str) and repository_license:
        return {
            "identifier": repository_license,
            "source_kind": "huggingface_model_card",
            "source_path": (
                _hf_source_path(repository_source, resolved_revision, "README.md")
                if repository_source is not None
                else None
            ),
            "source_field": "license",
        }
    for field in ("license", "license_name"):
        value = config.get(field)
        if isinstance(value, str) and value:
            source = file_records.get("config.json")
            return {
                "identifier": value,
                "source_kind": "embedded_json_field",
                "source_path": source["source_path"] if source else None,
                "source_field": field,
            }

    for filename in ("LICENSE", "LICENSE.md", "LICENSE.txt", "license"):
        record = metadata_files.get(filename)
        if record is not None:
            _, source = _text_metadata(record)
            return {
                "identifier": None,
                "source_kind": source["source_kind"] if source else None,
                "source_path": source["source_path"] if source else None,
                "source_field": None,
            }
    return {
        "identifier": None,
        "source_kind": None,
        "source_path": None,
        "source_field": None,
    }


def _declared_capabilities(
    config: dict[str, Any], repository_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Return only source-declared signals, never runtime capability claims."""
    return {
        "declared_signals": {
            "architectures": _model_family(config)["architectures"],
            "text_config_present": isinstance(config.get("text_config"), dict),
            "vision_config_present": isinstance(config.get("vision_config"), dict),
            "audio_config_present": isinstance(config.get("audio_config"), dict),
            "image_token_id_declared": "image_token_id" in config,
            "video_token_id_declared": "video_token_id" in config,
            "pipeline_tag": repository_metadata.get("pipeline_tag"),
            "library_name": repository_metadata.get("library_name"),
            "repository_tags": repository_metadata.get("tags", []),
        },
        "unknown": {
            "tool_calling": None,
            "reasoning": None,
            "structured_output": None,
            "parser_support": None,
            "runtime_support": None,
            "local_serving_context": None,
            "qualification": None,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _acquisition_identity(
    model_id: str,
    options: AcquisitionOptions,
    *,
    target: Path,
    staging_root: Path,
) -> dict[str, Any]:
    return {
        "version": ACQUISITION_JOURNAL_VERSION,
        "operation": "acquire",
        "model_id": model_id,
        "revision": options.revision,
        "target_path": str(target.resolve()),
        "staging_root": str(staging_root.resolve()),
        "is_mllm": options.is_mllm,
        "fast_transfer": options.fast_transfer,
        "local_files_only": options.local_files_only,
    }


def _acquisition_operation_id(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _acquisition_journal_path(staging_root: Path, target: Path) -> Path:
    target_key = hashlib.sha256(str(target.resolve()).encode()).hexdigest()[:12]
    return staging_root / f".{target.name}.{target_key}.acquisition.json"


def _local_file_inventory(path: Path) -> tuple[list[dict[str, Any]], int]:
    files = []
    total = 0
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        if item.name in _INTERNAL_WORKFLOW_FILENAMES:
            continue
        try:
            size = item.stat().st_size
        except OSError:
            size = 0
        total += size
        files.append({"path": str(item.relative_to(path)), "size": size})
    return files, total


def _hf_file_inventory(
    model_id: str, *, revision: str | None, local_files_only: bool
) -> tuple[list[dict[str, Any]], int | None, str | None, dict[str, Any]]:
    if local_files_only:
        return [], None, None, {}

    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    files = []
    total = 0
    total_known = True
    for sibling in getattr(info, "siblings", []) or []:
        filename = getattr(sibling, "rfilename", None)
        if not filename:
            continue
        size = getattr(sibling, "size", None)
        if size is None:
            total_known = False
        else:
            total += int(size)
        files.append({"path": filename, "size": size})
    card_data = getattr(info, "card_data", None)
    repository_metadata = {
        "license": (
            card_data.get("license")
            if isinstance(card_data, dict)
            else getattr(card_data, "license", None)
        ),
        "library_name": getattr(info, "library_name", None),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": list(getattr(info, "tags", None) or []),
    }
    return (
        files,
        total if total_known else None,
        getattr(info, "sha", revision),
        repository_metadata,
    )


def _config_value(config: dict[str, Any], key: str) -> Any:
    if key in config:
        return config[key]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config.get(key)
    return None


def _model_family(config: dict[str, Any]) -> dict[str, Any]:
    architectures = _config_value(config, "architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    max_context = (
        _config_value(config, "max_position_embeddings")
        or _config_value(config, "max_sequence_length")
        or _config_value(config, "seq_length")
        or _config_value(config, "model_max_length")
    )
    return {
        "model_type": _config_value(config, "model_type"),
        "architectures": architectures,
        "torch_dtype": _config_value(config, "torch_dtype"),
        "max_context": max_context,
        "quantization": config.get("quantization") or config.get("quantization_config"),
        "has_text_config": isinstance(config.get("text_config"), dict),
        "has_vision_config": isinstance(config.get("vision_config"), dict),
        "mtp_num_hidden_layers": _config_value(config, "mtp_num_hidden_layers"),
    }


def _estimate_fit(
    *,
    total_bytes: int | None,
    model_files_bytes: int | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    max_context = _model_family(config).get("max_context")
    warnings: list[str] = []
    if isinstance(max_context, int) and max_context >= 262_144:
        warnings.append(
            "very large advertised context; choose an explicit serving context before loading"
        )

    # Conversion normally needs source weights, output weights, and temporary
    # shards. Keep this conservative without pretending to know architecture
    # residency exactly.
    disk_floor = None
    if total_bytes is not None:
        disk_floor = int(total_bytes * 2.2)

    memory_floor = model_files_bytes or total_bytes
    return {
        "download_size_gb": _bytes_to_gb(total_bytes),
        "model_file_size_gb": _bytes_to_gb(model_files_bytes),
        "estimated_conversion_disk_gb": _bytes_to_gb(disk_floor),
        "rough_load_memory_gb": _bytes_to_gb(memory_floor),
        "warnings": warnings,
    }


def _model_file_bytes(files: list[dict[str, Any]]) -> int | None:
    total = 0
    known = False
    for entry in files:
        path = str(entry.get("path", ""))
        if not path.endswith((".safetensors", ".bin", ".gguf")):
            continue
        size = entry.get("size")
        if size is None:
            return None
        known = True
        total += int(size)
    return total if known else None


_NON_MLX_QUANT_METHODS = frozenset({"gptq", "awq", "squeezellm", "marlin", "fp8"})


def _is_mlx_quantization(quant: Any) -> bool:
    """Return True only when *quant* looks like an mlx-lm quantization config.

    PyTorch quantization configs (GPTQ, AWQ, ...) carry a ``quant_method``
    key that MLX configs never set.  Treating those as MLX-ready is a false
    positive reported in review.
    """
    if not isinstance(quant, dict):
        return False
    method = str(quant.get("quant_method", "")).lower()
    if method in _NON_MLX_QUANT_METHODS:
        return False
    # MLX configs written by mlx-lm always contain "bits".
    return "bits" in quant


def _looks_like_mlx_name(model: str, *, source: str) -> bool:
    name = model.lower() if source == "huggingface" else Path(model).name.lower()
    return (
        name.startswith("mlx-community/")
        or "-mlx" in name
        or "_mlx" in name
        or name.endswith("mlx")
    )


def _is_model_id(value: str) -> bool:
    return bool(_MODEL_ID_RE.fullmatch(value))


def _fast_transfer_env(requested: bool) -> tuple[dict[str, str], dict[str, Any]]:
    if not requested:
        return {}, {"requested": False, "enabled": False, "reason": "disabled"}
    if find_spec("hf_transfer") is None:
        return (
            {},
            {
                "requested": True,
                "enabled": False,
                "reason": "hf_transfer package is not installed",
            },
        )
    return (
        {"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        {"requested": True, "enabled": True, "reason": "enabled"},
    )


def inspect_model(
    model: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Inspect a local model path or Hugging Face model id without loading weights."""
    model_path = Path(model).expanduser()
    warnings: list[str] = []
    total_bytes: int | None

    if model_path.exists():
        files, total_bytes = _local_file_inventory(model_path)
        metadata_files = _local_metadata_files(model_path)
        config_data, config_record = _json_metadata(
            metadata_files.get("config.json"),
            filename="config.json",
            warnings=warnings,
        )
        config = config_data or {}
        resolved_revision = None
        repository_metadata: dict[str, Any] = {}
        source = "local"
        location = str(model_path)
    else:
        if not _is_model_id(model):
            raise ValueError(
                f"{model!r} is not an existing path or a Hugging Face model id"
            )
        source = "huggingface"
        location = model
        (
            files,
            total_bytes,
            resolved_revision,
            repository_metadata,
        ) = _hf_file_inventory(
            model, revision=revision, local_files_only=local_files_only
        )
        metadata_files = _hf_metadata_files(
            model,
            revision=resolved_revision or revision,
            local_files_only=local_files_only,
            files=files,
            warnings=warnings,
        )
        config_data, config_record = _json_metadata(
            metadata_files.get("config.json"),
            filename="config.json",
            warnings=warnings,
        )
        config = config_data or {}

    tokenizer_config_data, tokenizer_config_record = _json_metadata(
        metadata_files.get("tokenizer_config.json"),
        filename="tokenizer_config.json",
        warnings=warnings,
    )
    tokenizer_config = tokenizer_config_data or {}
    generation_config, generation_config_record = _json_metadata(
        metadata_files.get("generation_config.json"),
        filename="generation_config.json",
        warnings=warnings,
    )
    metadata_file_records = {
        filename: _metadata_file_record(
            path, source_kind=source_kind, source_path=source_path
        )
        for filename, (path, source_kind, source_path) in metadata_files.items()
    }
    if config_record is not None:
        metadata_file_records["config.json"] = config_record
    if tokenizer_config_record is not None:
        metadata_file_records["tokenizer_config.json"] = tokenizer_config_record
    if generation_config_record is not None:
        metadata_file_records["generation_config.json"] = generation_config_record
    resolved = resolved_revision if source == "huggingface" else None
    revision_evidence = {
        "requested": revision,
        "resolved": resolved,
        "source_kind": (
            "huggingface_repository" if source == "huggingface" else "local_directory"
        ),
        "resolved_is_immutable": (
            bool(resolved and _IMMUTABLE_HF_REVISION_RE.fullmatch(resolved))
            if source == "huggingface"
            else None
        ),
    }
    metadata_evidence = {
        "revision": revision_evidence,
        "files": metadata_file_records,
        "tokenizer_assets": {
            filename: metadata_file_records[filename]
            for filename in (
                "tokenizer.json",
                "tokenizer_config.json",
                "tokenizer.model",
                "spiece.model",
                "vocab.json",
                "merges.txt",
                "added_tokens.json",
                "special_tokens_map.json",
            )
            if filename in metadata_file_records
        },
        "chat_template": _chat_template_metadata(
            tokenizer_config, metadata_files, metadata_file_records
        ),
        "generation_config": {
            "data": generation_config,
            "sha256": (
                generation_config_record["sha256"]
                if generation_config_record is not None
                else None
            ),
            "source_kind": (
                generation_config_record["source_kind"]
                if generation_config_record is not None
                else None
            ),
            "source_path": (
                generation_config_record["source_path"]
                if generation_config_record is not None
                else None
            ),
        },
        "license": _license_metadata(
            config,
            repository_metadata,
            model if source == "huggingface" else None,
            resolved_revision,
            metadata_files,
            metadata_file_records,
        ),
        "capabilities": _declared_capabilities(config, repository_metadata),
    }

    model_files_bytes = _model_file_bytes(files)
    family = _model_family(config)
    estimate = _estimate_fit(
        total_bytes=total_bytes,
        model_files_bytes=model_files_bytes,
        config=config,
    )
    warnings.extend(estimate.pop("warnings"))
    has_name_signal = _looks_like_mlx_name(model, source=source) and (
        source == "huggingface" or bool(config)
    )
    has_mlx_signals = (
        _is_mlx_quantization(family.get("quantization")) or has_name_signal
    )

    return {
        "model": model,
        "source": source,
        "location": location,
        "revision": resolved_revision or revision,
        "metadata_evidence": metadata_evidence,
        "inspected_at": _now_iso(),
        "file_count": len(files),
        "total_size_bytes": total_bytes,
        "total_size_gb": _bytes_to_gb(total_bytes),
        "model_files_size_gb": _bytes_to_gb(model_files_bytes),
        "model_family": family,
        "mlx": {
            "looks_like_mlx_artifact": has_mlx_signals,
            "needs_conversion": not has_mlx_signals,
        },
        "fit_estimate": estimate,
        "warnings": warnings,
    }


def acquire_model(
    model_id: str,
    *,
    options: AcquisitionOptions | None = None,
) -> dict[str, Any]:
    """Download a model repository and write a finalized artifact manifest."""
    if options is None:
        options = AcquisitionOptions()
    if not _is_model_id(model_id):
        raise ValueError(f"{model_id!r} is not a Hugging Face model id")

    allow_patterns = MLLM_ALLOW_PATTERNS if options.is_mllm else LLM_ALLOW_PATTERNS
    env_updates, fast_transfer = _fast_transfer_env(options.fast_transfer)

    old_env: dict[str, str | None] = {}
    operation_lock = None
    environment_lock_acquired = False
    try:
        if options.target_dir:
            if not options.revision or not _IMMUTABLE_HF_REVISION_RE.fullmatch(
                options.revision
            ):
                raise ValueError(
                    "targeted resumable acquisition requires an immutable 40-64 character revision"
                )
            target = Path(options.target_dir).expanduser()
            staging_root = (
                Path(options.staging_dir).expanduser()
                if options.staging_dir
                else target.parent
            )
            staging_root.mkdir(parents=True, exist_ok=True)
            identity = _acquisition_identity(
                model_id, options, target=target, staging_root=staging_root
            )
            operation_id = _acquisition_operation_id(identity)
            journal_path = _acquisition_journal_path(staging_root, target)
            lock_path = journal_path.with_suffix(f"{journal_path.suffix}.lock")
            operation_lock = lock_path.open("a+")
            fcntl.flock(operation_lock.fileno(), fcntl.LOCK_EX)
            _ACQUISITION_ENV_LOCK.acquire()
            environment_lock_acquired = True
            old_env = {key: os.environ.get(key) for key in env_updates}
            os.environ.update(env_updates)
            journal = (
                _read_json_strict(journal_path, label="acquisition journal")
                if journal_path.exists()
                else {}
            )
            if journal:
                if (
                    journal.get("kind") != "vllm-mlx-acquisition-operation"
                    or journal.get("version") != ACQUISITION_JOURNAL_VERSION
                    or journal.get("operation_id") != operation_id
                    or journal.get("identity") != identity
                ):
                    raise ValueError(
                        f"acquisition journal identity conflict at {journal_path}"
                    )
            staging = staging_root / f".{target.name}.staging-{operation_id[:12]}"
            recorded_staging = journal.get("staging_path")
            if recorded_staging is not None and Path(recorded_staging).resolve() != (
                staging.resolve()
            ):
                raise ValueError(
                    f"acquisition journal staging path conflict at {journal_path}"
                )
            attempt = int(journal.get("attempt", 0)) + 1
            journal = {
                "kind": "vllm-mlx-acquisition-operation",
                "version": ACQUISITION_JOURNAL_VERSION,
                "operation_id": operation_id,
                "identity": identity,
                "status": "running",
                "attempt": attempt,
                "started_at": journal.get("started_at") or _now_iso(),
                "updated_at": _now_iso(),
                "staging_path": str(staging),
                "target_path": str(target),
                "target_published": bool(journal.get("target_published", False)),
                "target_published_at": journal.get("target_published_at"),
            }
            _write_json_atomic(journal_path, journal)
            try:
                if target.exists():
                    existing_manifest = _read_json(target / MODEL_MANIFEST_NAME)
                    if existing_manifest.get("operation_id") == operation_id:
                        marker_path = target / ACQUISITION_MARKER_NAME
                        if _read_json(marker_path).get("operation_id") == operation_id:
                            marker_path.unlink(missing_ok=True)
                        existing_manifest["manifest_path"] = str(
                            target / MODEL_MANIFEST_NAME
                        )
                        journal.update(
                            status="succeeded",
                            updated_at=_now_iso(),
                            completed_at=journal.get("completed_at") or _now_iso(),
                        )
                        _write_json_atomic(journal_path, journal)
                        return existing_manifest
                    marker = _read_json(target / ACQUISITION_MARKER_NAME)
                    if marker.get("operation_id") != operation_id and not journal.get(
                        "target_published"
                    ):
                        raise FileExistsError(f"target path already exists: {target}")
                else:
                    staging.mkdir(parents=True, exist_ok=True)
                    _write_json_atomic(
                        staging / ACQUISITION_MARKER_NAME,
                        {"operation_id": operation_id, "identity": identity},
                    )
                    downloaded = Path(
                        snapshot_download(
                            model_id,
                            revision=options.revision,
                            allow_patterns=allow_patterns,
                            local_dir=str(staging),
                            local_files_only=options.local_files_only,
                        )
                    )
                    _write_json_atomic(
                        downloaded / ACQUISITION_MARKER_NAME,
                        {"operation_id": operation_id, "identity": identity},
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(downloaded), str(target))
                    journal.update(
                        target_published=True,
                        target_published_at=_now_iso(),
                        updated_at=_now_iso(),
                    )
                    _write_json_atomic(journal_path, journal)

                marker_path = target / ACQUISITION_MARKER_NAME
                inspection = inspect_model(str(target), revision=options.revision)
                manifest = {
                    "kind": "vllm-mlx-model-artifact",
                    "operation_id": operation_id,
                    "operation_journal_path": str(journal_path),
                    "model_id": model_id,
                    "revision": options.revision,
                    "path": str(target),
                    "created_at": _now_iso(),
                    "allow_patterns": allow_patterns,
                    "fast_transfer": fast_transfer,
                    "local_files_only": options.local_files_only,
                    "inspection": inspection,
                }
                manifest_path = target / MODEL_MANIFEST_NAME
                _write_json_atomic(manifest_path, manifest)
                marker_path.unlink(missing_ok=True)
                journal.update(
                    status="succeeded",
                    updated_at=_now_iso(),
                    completed_at=_now_iso(),
                    manifest_path=str(manifest_path),
                )
                _write_json_atomic(journal_path, journal)
                manifest["manifest_path"] = str(manifest_path)
                return manifest
            except BaseException as exc:
                status = (
                    "cancelled"
                    if isinstance(exc, (KeyboardInterrupt, SystemExit))
                    else "failed"
                )
                journal.update(
                    status=status,
                    updated_at=_now_iso(),
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                _write_json_atomic(journal_path, journal)
                raise
        else:
            _ACQUISITION_ENV_LOCK.acquire()
            environment_lock_acquired = True
            old_env = {key: os.environ.get(key) for key in env_updates}
            os.environ.update(env_updates)
            final_path = Path(
                snapshot_download(
                    model_id,
                    revision=options.revision,
                    allow_patterns=allow_patterns,
                    local_files_only=options.local_files_only,
                )
            )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if environment_lock_acquired:
            _ACQUISITION_ENV_LOCK.release()
        if operation_lock is not None:
            fcntl.flock(operation_lock.fileno(), fcntl.LOCK_UN)
            operation_lock.close()

    inspection = inspect_model(str(final_path), revision=options.revision)
    manifest = {
        "kind": "vllm-mlx-model-artifact",
        "model_id": model_id,
        "revision": options.revision,
        "path": str(final_path),
        "created_at": _now_iso(),
        "allow_patterns": allow_patterns,
        "fast_transfer": fast_transfer,
        "local_files_only": options.local_files_only,
        "inspection": inspection,
    }
    manifest_path = final_path / MODEL_MANIFEST_NAME
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _conversion_command(options: ConversionOptions) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        options.source_path,
        "--mlx-path",
        options.output_path,
    ]
    if options.quantize:
        command.append("--quantize")
    if options.q_bits is not None:
        command.extend(["--q-bits", str(options.q_bits)])
    if options.q_group_size is not None:
        command.extend(["--q-group-size", str(options.q_group_size)])
    if options.q_mode:
        command.extend(["--q-mode", options.q_mode])
    if options.quant_predicate:
        command.extend(["--quant-predicate", options.quant_predicate])
    if options.dtype:
        command.extend(["--dtype", options.dtype])
    if options.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def _conversion_source_identity(source: Path) -> dict[str, Any]:
    """Bind conversion state to stable source metadata and artifact bytes."""
    if not source.is_dir():
        raise NotADirectoryError(f"conversion source must be a directory: {source}")
    acquisition_manifest = source / MODEL_MANIFEST_NAME
    config = source / "config.json"
    if not config.is_file():
        raise ValueError(f"conversion source is missing config.json: {source}")
    records = _artifact_file_records(source)
    digest_input = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    identity: dict[str, Any] = {
        "path": str(source.resolve()),
        "config_sha256": _sha256_file(config),
        "artifact_sha256": hashlib.sha256(digest_input).hexdigest(),
        "file_count": len(records),
        "total_size_bytes": sum(record["size"] for record in records),
    }
    if acquisition_manifest.is_file():
        manifest = _read_json_strict(
            acquisition_manifest, label="source acquisition manifest"
        )
        identity.update(
            acquisition_manifest_sha256=_sha256_file(acquisition_manifest),
            model_id=manifest.get("model_id"),
            revision=manifest.get("revision"),
            acquisition_operation_id=manifest.get("operation_id"),
        )
    return identity


def _conversion_identity(
    options: ConversionOptions, *, source: Path, output: Path
) -> dict[str, Any]:
    return {
        "version": CONVERSION_JOURNAL_VERSION,
        "operation": "convert",
        "backend": "mlx-lm",
        "source": _conversion_source_identity(source),
        "output_path": str(output.resolve()),
        "recipe": {
            "quantize": options.quantize,
            "q_bits": options.q_bits,
            "q_group_size": options.q_group_size,
            "q_mode": options.q_mode,
            "quant_predicate": options.quant_predicate,
            "dtype": options.dtype,
            "trust_remote_code": options.trust_remote_code,
        },
    }


def _conversion_operation_id(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _conversion_journal_path(output: Path) -> Path:
    output_key = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
    return output.parent / f".{output.name}.{output_key}.conversion.json"


def _artifact_file_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or item.name in _INTERNAL_WORKFLOW_FILENAMES:
            continue
        records.append(
            {
                "path": str(item.relative_to(path)),
                "size": item.stat().st_size,
                "sha256": _sha256_file(item),
            }
        )
    return records


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _validate_weight_file(path: Path) -> None:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        try:
            with safe_open(path, framework="np") as handle:
                if not list(handle.keys()):
                    raise ValueError("contains no tensors")
        except Exception as exc:
            raise ValueError(f"invalid safetensors weight file {path}: {exc}") from exc
        return

    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise ValueError("contains no arrays")
    except Exception as exc:
        raise ValueError(f"invalid NPZ weight file {path}: {exc}") from exc


def validate_converted_artifact(
    path: str | Path,
    *,
    expected_operation_id: str | None = None,
) -> dict[str, Any]:
    """Validate a converted artifact and return content-bound integrity evidence."""
    artifact = Path(path).expanduser()
    if not artifact.is_dir():
        raise NotADirectoryError(f"converted artifact must be a directory: {artifact}")
    config = artifact / "config.json"
    if not config.is_file():
        raise ValueError(f"converted artifact is missing config.json: {artifact}")
    config_payload = _read_json_strict(config, label="converted artifact config")
    if not config_payload.get("model_type") and not config_payload.get("architectures"):
        raise ValueError(
            f"converted artifact config lacks model_type or architectures: {config}"
        )
    weight_files = sorted(
        item
        for pattern in ("*.safetensors", "*.npz")
        for item in artifact.rglob(pattern)
        if item.is_file()
    )
    if not weight_files:
        raise ValueError(f"converted artifact has no MLX weight files: {artifact}")
    for weight_file in weight_files:
        _validate_weight_file(weight_file)

    records = _artifact_file_records(artifact)
    digest_input = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    evidence: dict[str, Any] = {
        "kind": "vllm-mlx-converted-artifact-validation",
        "path": str(artifact.resolve()),
        "validated_at": _now_iso(),
        "file_count": len(records),
        "total_size_bytes": sum(record["size"] for record in records),
        "artifact_sha256": hashlib.sha256(digest_input).hexdigest(),
        "files": records,
        "config_sha256": _sha256_file(config),
        "weight_file_count": len(weight_files),
    }
    manifest_path = artifact / CONVERSION_MANIFEST_NAME
    if manifest_path.is_file():
        manifest = _read_json_strict(manifest_path, label="conversion manifest")
        evidence["manifest_operation_id"] = manifest.get("operation_id")
        if (
            expected_operation_id is not None
            and manifest.get("operation_id") != expected_operation_id
        ):
            raise ValueError(
                "converted artifact manifest operation_id does not match the expected operation"
            )
    elif expected_operation_id is not None:
        raise ValueError(
            f"converted artifact is missing conversion manifest: {artifact}"
        )
    return evidence


def _publish_conversion_output(staging: Path, output: Path) -> None:
    os.replace(staging, output)


def convert_model(options: ConversionOptions) -> dict[str, Any]:
    """Run a resumable, identity-bound mlx-lm conversion operation."""
    command = _conversion_command(options)
    started = _now_iso()
    source_inspection = inspect_model(options.source_path)
    result = {
        "kind": "vllm-mlx-conversion",
        "backend": "mlx-lm",
        "command": command,
        "source_path": options.source_path,
        "output_path": options.output_path,
        "started_at": started,
        "dry_run": options.dry_run,
        "recipe": {
            "quantize": options.quantize,
            "q_bits": options.q_bits,
            "q_group_size": options.q_group_size,
            "q_mode": options.q_mode,
            "quant_predicate": options.quant_predicate,
            "dtype": options.dtype,
            "trust_remote_code": options.trust_remote_code,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mlx_lm": _package_version("mlx-lm"),
        },
        "source_inspection": source_inspection,
    }

    if options.dry_run:
        result["status"] = "dry_run"
        return result

    source_path = Path(options.source_path).expanduser()
    output_path = Path(options.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    identity = _conversion_identity(options, source=source_path, output=output_path)
    operation_id = _conversion_operation_id(identity)
    journal_path = _conversion_journal_path(output_path)
    lock_path = journal_path.with_suffix(f"{journal_path.suffix}.lock")
    staging = output_path.parent / f".{output_path.name}.conversion-{operation_id[:12]}"

    with lock_path.open("a+") as operation_lock:
        fcntl.flock(operation_lock.fileno(), fcntl.LOCK_EX)
        journal = (
            _read_json_strict(journal_path, label="conversion journal")
            if journal_path.exists()
            else {}
        )
        if journal and (
            journal.get("kind") != "vllm-mlx-conversion-operation"
            or journal.get("version") != CONVERSION_JOURNAL_VERSION
            or journal.get("operation_id") != operation_id
            or journal.get("identity") != identity
        ):
            raise ValueError(f"conversion journal identity conflict at {journal_path}")

        if output_path.exists():
            manifest = _read_json(output_path / CONVERSION_MANIFEST_NAME)
            if (
                manifest.get("operation_id") != operation_id
                or manifest.get("identity") != identity
            ):
                raise FileExistsError(
                    f"conversion output path already exists with different identity: {output_path}"
                )
            validation = validate_converted_artifact(
                output_path, expected_operation_id=operation_id
            )
            recorded_validation = manifest.get("artifact_validation")
            recorded_digest = (
                recorded_validation.get("artifact_sha256")
                if isinstance(recorded_validation, dict)
                else None
            )
            journal_digest = journal.get("artifact_sha256")
            expected_digest = recorded_digest or journal_digest
            if expected_digest is None:
                raise ValueError(
                    "published conversion artifact has no recorded integrity digest"
                )
            if validation["artifact_sha256"] != expected_digest:
                raise ValueError(
                    "published conversion artifact bytes differ from the recorded integrity digest"
                )
            if journal_digest is not None and journal_digest != expected_digest:
                raise ValueError(
                    "conversion manifest and journal integrity digests disagree"
                )
            manifest.update(
                manifest_path=str(output_path / CONVERSION_MANIFEST_NAME),
                artifact_validation=validation,
            )
            _write_json_atomic(output_path / CONVERSION_MANIFEST_NAME, manifest)
            (output_path / CONVERSION_MARKER_NAME).unlink(missing_ok=True)
            journal.update(
                kind="vllm-mlx-conversion-operation",
                version=CONVERSION_JOURNAL_VERSION,
                operation_id=operation_id,
                identity=identity,
                status="succeeded",
                attempt=max(1, int(journal.get("attempt", 0))),
                started_at=journal.get("started_at") or manifest.get("started_at"),
                updated_at=_now_iso(),
                completed_at=journal.get("completed_at") or _now_iso(),
                output_published=True,
                manifest_path=str(output_path / CONVERSION_MANIFEST_NAME),
                artifact_sha256=validation["artifact_sha256"],
            )
            _write_json_atomic(journal_path, journal)
            return manifest

        if staging.exists():
            marker = _read_json(staging / CONVERSION_MARKER_NAME)
            if marker.get("operation_id") != operation_id:
                raise FileExistsError(
                    f"conversion staging path is not owned by this operation: {staging}"
                )
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _write_json_atomic(
            staging / CONVERSION_MARKER_NAME,
            {"operation_id": operation_id, "identity": identity},
        )

        attempt = int(journal.get("attempt", 0)) + 1
        journal = {
            "kind": "vllm-mlx-conversion-operation",
            "version": CONVERSION_JOURNAL_VERSION,
            "operation_id": operation_id,
            "identity": identity,
            "status": "running",
            "attempt": attempt,
            "started_at": journal.get("started_at") or started,
            "updated_at": _now_iso(),
            "staging_path": str(staging),
            "output_path": str(output_path),
            "output_published": False,
        }
        _write_json_atomic(journal_path, journal)

        execution_options = replace(options, output_path=str(staging))
        execution_command = _conversion_command(execution_options)
        result.update(
            operation_id=operation_id,
            operation_journal_path=str(journal_path),
            attempt=attempt,
            execution_command=execution_command,
            source_identity=identity["source"],
            identity=identity,
        )
        try:
            completed = subprocess.run(
                execution_command, text=True, capture_output=True, check=False
            )
            result["returncode"] = completed.returncode
            result["stdout"] = completed.stdout
            result["stderr"] = completed.stderr
            result["completed_at"] = _now_iso()
            result["status"] = "succeeded" if completed.returncode == 0 else "failed"
            if completed.returncode != 0:
                journal.update(
                    status="failed",
                    updated_at=_now_iso(),
                    error={
                        "type": "ConversionProcessError",
                        "message": completed.stderr,
                        "returncode": completed.returncode,
                    },
                )
                _write_json_atomic(journal_path, journal)
                return result

            output_inspection = inspect_model(str(staging))
            if _conversion_source_identity(source_path) != identity["source"]:
                raise ValueError(
                    "conversion source changed while conversion was running; output was not published"
                )
            result["output_inspection"] = output_inspection
            manifest_path = staging / CONVERSION_MANIFEST_NAME
            _write_json_atomic(manifest_path, result)
            validation = validate_converted_artifact(
                staging, expected_operation_id=operation_id
            )
            result["artifact_validation"] = validation
            final_manifest_path = output_path / CONVERSION_MANIFEST_NAME
            validation["path"] = str(output_path.resolve())
            output_inspection["model"] = str(output_path)
            output_inspection["location"] = str(output_path)
            result["manifest_path"] = str(final_manifest_path)
            _write_json_atomic(manifest_path, result)
            _publish_conversion_output(staging, output_path)
            (output_path / CONVERSION_MARKER_NAME).unlink(missing_ok=True)
            journal.update(
                status="succeeded",
                updated_at=_now_iso(),
                completed_at=_now_iso(),
                output_published=True,
                manifest_path=str(final_manifest_path),
                artifact_sha256=validation["artifact_sha256"],
            )
            _write_json_atomic(journal_path, journal)
            return result
        except BaseException as exc:
            status = (
                "cancelled"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "failed"
            )
            journal.update(
                status=status,
                updated_at=_now_iso(),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            _write_json_atomic(journal_path, journal)
            raise


def _existing_manifests(path: Path) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    for name, key in (
        (MODEL_MANIFEST_NAME, "acquisition"),
        (CONVERSION_MANIFEST_NAME, "conversion"),
    ):
        manifest_path = path / name
        if manifest_path.exists():
            manifests[key] = {
                "path": str(manifest_path),
                "payload": _read_json(manifest_path),
            }
    return manifests


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def register_model(options: RegistrationOptions) -> dict[str, Any]:
    """Write a portable registration manifest for a finalized local artifact.

    This deliberately does not mutate a production registry. The manifest is a
    handoff artifact that Ops or a deployment tool can apply after qualification.
    """
    artifact = Path(options.artifact_path).expanduser()
    if not artifact.exists():
        raise FileNotFoundError(f"artifact path does not exist: {artifact}")
    if not artifact.is_dir():
        raise NotADirectoryError(f"artifact path must be a directory: {artifact}")

    inspection = inspect_model(str(artifact))
    model_id = options.model_id or artifact.name
    serving_defaults = _drop_none(
        {
            "temperature": options.default_temperature,
            "top_p": options.default_top_p,
            "top_k": options.default_top_k,
            "min_p": options.default_min_p,
            "presence_penalty": options.default_presence_penalty,
            "repetition_penalty": options.default_repetition_penalty,
            "chat_template_kwargs": options.chat_template_kwargs,
        }
    )
    parser_policy = _drop_none(
        {
            "tool_call_parser": options.tool_call_parser,
            "reasoning_parser": options.reasoning_parser,
        }
    )
    payload = {
        "kind": "vllm-mlx-model-registration",
        "schema_version": 1,
        "created_at": _now_iso(),
        "model_id": model_id,
        "served_model_name": options.served_model_name or model_id,
        "preset_alias": options.preset_alias,
        "artifact_path": str(artifact),
        "mllm": options.mllm,
        "feature_flags": options.feature_flags or [],
        "serving_defaults": serving_defaults,
        "parser_policy": parser_policy,
        "inspection": inspection,
        "source_manifests": _existing_manifests(artifact),
        "qualification_required": True,
        "production_ready": False,
    }

    output = (
        Path(options.output_path).expanduser()
        if options.output_path
        else artifact / REGISTRATION_MANIFEST_NAME
    )
    _write_json(output, payload)
    payload["manifest_path"] = str(output)
    return payload


def _qualification_command(options: QualificationOptions) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm_mlx.cli",
        "bench-serve",
        "--url",
        options.server_url,
        "--model",
        options.model_id,
        "--format",
        "json",
    ]
    if options.workload_path:
        command.extend(["--workload", options.workload_path])
    if options.repetitions is not None:
        command.extend(["--repetitions", str(options.repetitions)])
    if options.result_path:
        command.extend(["--output", options.result_path])
    if options.extra_args:
        command.extend(options.extra_args)
    return command


def _validated_profile_subject(profile: Mapping[str, Any]) -> str:
    stored = profile.get("subject_digest")
    computed = str(compute_subject_digest(profile))
    if not isinstance(stored, str) or stored.lower() != computed:
        raise ValueError("model profile subject_digest is missing or stale")
    return computed


def _validated_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("qualification result timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("qualification result timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("qualification result timestamp must include a timezone")
    return value


def _snapshot_qualification_workload(
    workload_path: str | Path, result_path: str | Path
) -> tuple[Path, str]:
    source = Path(workload_path).expanduser().resolve()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"qualification workload is unreadable: {source}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    result = Path(result_path).expanduser().resolve()
    snapshot = (
        result.parent / f"{source.stem}.{digest[:16]}.qualification-workload.json"
    )
    if snapshot.exists():
        if snapshot.read_bytes() != raw:
            raise ValueError(f"qualification workload snapshot conflicts: {snapshot}")
    else:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{snapshot.name}.tmp-", dir=snapshot.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, snapshot)
        finally:
            temporary.unlink(missing_ok=True)
    return snapshot, digest


def normalize_qualification_result(
    profile: Mapping[str, Any],
    result_path: str | Path,
    workload_path: str | Path,
    *,
    expected_workload_sha256: str | None = None,
    expected_result_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind one workload result artifact to an immutable profile subject.

    This records evidence only. It never promotes the profile or marks an
    installation production-ready.
    """
    subject_digest = _validated_profile_subject(profile)
    path = Path(result_path).expanduser().resolve()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"qualification result is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("qualification result must be a JSON object")
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_result_sha256 is not None and artifact_sha256 != expected_result_sha256:
        raise ValueError("qualification result changed before normalization")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("qualification result run_id is missing")
    created_at = _validated_timestamp(payload.get("timestamp"))

    workload = payload.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("qualification result workload is missing")
    workload_name = workload.get("name")
    expected_workload_path = Path(workload_path).expanduser().resolve()
    try:
        workload_sha256 = hashlib.sha256(
            expected_workload_path.read_bytes()
        ).hexdigest()
    except OSError as exc:
        raise ValueError(
            f"qualification workload is unreadable: {expected_workload_path}"
        ) from exc
    if (
        expected_workload_sha256 is not None
        and workload_sha256 != expected_workload_sha256
    ):
        raise ValueError("qualification workload snapshot changed during normalization")
    if not isinstance(workload_name, str) or not workload_name:
        raise ValueError("qualification workload name is missing")
    result_workload_path = workload.get("path")
    if not isinstance(result_workload_path, str) or (
        Path(result_workload_path).expanduser().resolve() != expected_workload_path
    ):
        raise ValueError("qualification result workload path does not match request")

    summary = payload.get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("passed"), bool):
        raise ValueError("qualification result summary.passed must be boolean")
    records = payload.get("results")
    if not isinstance(records, list) or not records:
        raise ValueError("qualification result requires at least one case record")
    expected_model = profile.get("identity", {}).get("served_model_name")
    hardware: dict[str, Any] | None = None
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("qualification case record must be an object")
        if record.get("run_id") != run_id or record.get("workload") != workload_name:
            raise ValueError("qualification case identity does not match its run")
        if _validated_timestamp(record.get("timestamp")) != created_at:
            raise ValueError("qualification case timestamp does not match its run")
        if record.get("model_id") != expected_model:
            raise ValueError("qualification result served model does not match profile")
        runtime = record.get("runtime")
        if not isinstance(runtime, dict) or runtime.get("model_id") != expected_model:
            raise ValueError(
                "qualification runtime-detected model does not match profile"
            )
        quality = record.get("quality")
        if (
            not isinstance(record.get("ok"), bool)
            or not isinstance(quality, dict)
            or not isinstance(quality.get("ok"), bool)
            or record["ok"] != quality["ok"]
        ):
            raise ValueError("qualification case outcome is missing or inconsistent")
        record_hardware = record.get("hardware")
        if not isinstance(record_hardware, dict) or not record_hardware:
            raise ValueError("qualification case hardware fingerprint is missing")
        if hardware is None:
            hardware = record_hardware
        elif record_hardware != hardware:
            raise ValueError("qualification cases have inconsistent hardware")

    assert hardware is not None
    aggregate_passed = all(record["ok"] for record in records)
    if summary["passed"] != aggregate_passed:
        raise ValueError("qualification summary does not match case outcomes")
    if summary.get("case_count") != len(records):
        raise ValueError("qualification summary case_count does not match records")
    if summary.get("failure_count") != sum(not record["ok"] for record in records):
        raise ValueError("qualification summary failure_count does not match records")
    result = "pass" if aggregate_passed else "fail"
    hardware_fingerprint = hashlib.sha256(
        json.dumps(hardware, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    evidence = {
        "evidence_id": f"{run_id}-{artifact_sha256[:16]}",
        "kind": "qualification",
        "location": str(path),
        "artifact_sha256": artifact_sha256,
        "result": result,
        "hardware_fingerprint": hardware_fingerprint,
        "workload_id": f"{workload_name}@sha256:{workload_sha256}",
        "subject_digest": subject_digest,
        "created_at": created_at,
    }
    qualification = profile.get("qualification")
    current_status = (
        qualification.get("status")
        if isinstance(qualification, Mapping)
        else "not_qualified"
    )
    return {
        "kind": "vllm-mlx-normalized-qualification-evidence",
        "schema_version": 1,
        "profile_id": profile.get("profile_id"),
        "profile_revision": profile.get("profile_revision"),
        "subject_digest": subject_digest,
        "qualification_status": current_status,
        "evidence": [evidence],
        "raw_evidence": {"location": str(path), "sha256": artifact_sha256},
        "promotion_required": True,
        "production_ready": False,
    }


def qualify_model(options: QualificationOptions) -> dict[str, Any]:
    """Create or run a bench-serve qualification handoff."""
    profile: dict[str, Any] | None = None
    subject_digest: str | None = None
    run_options = options
    workload_sha256: str | None = None
    private_result_path: Path | None = None
    if options.profile_path:
        profile = _read_json_strict(
            Path(options.profile_path).expanduser(), label="model profile"
        )
        subject_digest = _validated_profile_subject(profile)
        if not options.result_path:
            raise ValueError("profile-bound qualification requires result_path")
        if not options.workload_path:
            raise ValueError("profile-bound qualification requires workload_path")
        artifact_paths = [
            Path(value).expanduser().resolve()
            for value in (
                options.result_path,
                options.output_path,
                options.evidence_output_path,
            )
            if value
        ]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError(
                "qualification result, output, and evidence paths must differ"
            )
        if not options.dry_run:
            workload_snapshot, workload_sha256 = _snapshot_qualification_workload(
                options.workload_path, options.result_path
            )
            final_result = Path(options.result_path).expanduser().resolve()
            final_result.parent.mkdir(parents=True, exist_ok=True)
            descriptor, private_name = tempfile.mkstemp(
                prefix=f".{final_result.name}.run-", dir=final_result.parent
            )
            os.close(descriptor)
            private_result_path = Path(private_name)
            private_result_path.unlink()
            run_options = replace(
                options,
                workload_path=str(workload_snapshot),
                result_path=str(private_result_path),
            )
    command = _qualification_command(run_options)
    payload = {
        "kind": "vllm-mlx-model-qualification",
        "schema_version": 1,
        "created_at": _now_iso(),
        "model_id": options.model_id,
        "server_url": options.server_url,
        "workload_path": run_options.workload_path,
        "workload_source_path": options.workload_path,
        "workload_snapshot_sha256": workload_sha256,
        "result_path": options.result_path,
        "repetitions": options.repetitions,
        "dry_run": options.dry_run,
        "profile_path": options.profile_path,
        "profile_subject_digest": subject_digest,
        "command": command,
        "production_ready": False,
    }

    if not options.dry_run:
        try:
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False
            )
            payload["returncode"] = completed.returncode
            payload["stdout"] = completed.stdout
            payload["stderr"] = completed.stderr
            payload["completed_at"] = _now_iso()
            payload["status"] = "succeeded" if completed.returncode == 0 else "failed"
            if (
                completed.returncode == 0
                and profile is not None
                and options.result_path
            ):
                assert run_options.workload_path is not None
                assert workload_sha256 is not None
                if (
                    hashlib.sha256(
                        Path(run_options.workload_path).read_bytes()
                    ).hexdigest()
                    != workload_sha256
                ):
                    raise ValueError(
                        "qualification workload snapshot changed during run"
                    )
                if private_result_path is None or not private_result_path.is_file():
                    raise ValueError(
                        "qualification command did not create a fresh result"
                    )
                final_result = Path(options.result_path).expanduser().resolve()
                private_result = private_result_path.read_bytes()
                private_result_sha256 = hashlib.sha256(private_result).hexdigest()
                _write_bytes_atomic(final_result, private_result)
                normalized = normalize_qualification_result(
                    profile,
                    final_result,
                    run_options.workload_path,
                    expected_workload_sha256=workload_sha256,
                    expected_result_sha256=private_result_sha256,
                )
                payload["normalized_evidence"] = normalized
                if options.evidence_output_path:
                    evidence_output = Path(options.evidence_output_path).expanduser()
                    _write_json_atomic(evidence_output, normalized)
                    payload["evidence_manifest_path"] = str(evidence_output)
        finally:
            if private_result_path is not None:
                private_result_path.unlink(missing_ok=True)
    else:
        payload["status"] = "dry_run"

    if options.output_path:
        output = Path(options.output_path).expanduser()
        _write_json(output, payload)
        payload["manifest_path"] = str(output)
    return payload
