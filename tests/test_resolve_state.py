# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase 2 registry state resolver."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from vllm_mlx.resolve_state import emit_shell_vars


def _write_registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "policy_defaults": {
                    "memory_budget_gb": 96,
                    "contention_policy": {
                        "strategy": "wait_then_preempt",
                        "wait_timeout_s": 15,
                    },
                },
                "models": [
                    {
                        "id": "qwen3.5-27b",
                        "display_name": "Qwen 3.5 27B",
                        "source": "/Users/David/ai-models/mlx_models/Qwen3.5-27B-VLM-MTP-8bit",
                        "family": "qwen3.5",
                        "architecture": "dense",
                        "execution_class": "shared_candidate",
                        "estimated_memory_gb": 31,
                        "multimodal": True,
                        "supports_mtp": True,
                        "supports_reasoning": True,
                        "supports_tools": True,
                        "supports_specprefill": True,
                        "supports_kv_quant": True,
                        "supports_continuous_batching": True,
                        "draft_model": {
                            "id": "qwen3.5-2b-draft",
                            "source": "/Users/David/ai-models/mlx_models/Qwen3.5-2B-OptiQ-4bit",
                            "estimated_memory_gb": 3,
                        },
                        "serving_profile": {
                            "force_mllm": True,
                            "continuous_batching": True,
                            "prefill_step_size": 256,
                            "tool_call_parser": "qwen3_coder",
                            "reasoning_parser": "qwen3",
                            "enable_auto_tool_choice": True,
                            "enable_thinking_default": True,
                            "specprefill": {
                                "enabled": True,
                                "threshold": 8192,
                                "keep_pct": 0.3,
                            },
                        },
                    },
                    {
                        "id": "gemma-4-26b-a4b-it",
                        "display_name": "Gemma 4 26B",
                        "source": "/Users/David/ai-models/mlx_models/gemma-4-26B-A4B-it-6bit",
                        "family": "gemma4",
                        "architecture": "dense",
                        "execution_class": "shared_candidate",
                        "estimated_memory_gb": 28,
                        "supports_reasoning": True,
                        "supports_tools": True,
                        "supports_specprefill": True,
                        "supports_kv_quant": True,
                        "supports_continuous_batching": True,
                        "serving_profile": {
                            "continuous_batching": True,
                            "prefill_step_size": 256,
                            "tool_call_parser": "gemma4",
                            "reasoning_parser": "gemma4",
                            "enable_auto_tool_choice": True,
                            "enable_thinking_default": True,
                        },
                    },
                ],
                "model_presets": [
                    {
                        "id": "coding-quality",
                        "display_name": "Coding (Quality)",
                        "model_id": "qwen3.5-27b",
                        "priority_class": "interactive",
                        "performance_bias": "quality",
                        "sampling_profile": {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "enable_thinking": True,
                        },
                        "request_policy": {"max_tokens": 32768, "timeout_s": 2400},
                    },
                    {
                        "id": "fast-iteration",
                        "display_name": "Fast Iteration",
                        "model_id": "gemma-4-26b-a4b-it",
                        "priority_class": "interactive",
                        "performance_bias": "speed",
                        "sampling_profile": {
                            "temperature": 0.5,
                            "top_p": 0.9,
                            "enable_thinking": True,
                        },
                        "request_policy": {"max_tokens": 16384, "timeout_s": 1200},
                    },
                ],
                "service_presets": [],
                "discussion_profiles": [],
            },
            sort_keys=False,
        )
    )
    return path


def _write_state(tmp_path: Path, preset: str = "coding-quality") -> Path:
    path = tmp_path / "runtime-state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active_model_preset": preset,
                "active_service_presets": [],
                "active_backend": "vllm-mlx",
                "resident_models": ["qwen3.5-27b"],
                "execution": {
                    "active_model": "qwen3.5-27b",
                    "queue_policy": "default",
                },
                "updated_at": "2026-04-14T22:00:00Z",
                "updated_by": "ops-api",
            }
        )
    )
    return path


def _write_state_no_preset(tmp_path: Path) -> Path:
    path = tmp_path / "runtime-state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active_model_preset": None,
                "active_service_presets": [],
                "active_backend": "vllm-mlx",
                "resident_models": [],
                "execution": {},
                "updated_at": "2026-04-14T22:00:00Z",
                "updated_by": "ops-api",
            }
        )
    )
    return path


def _capture_shell_vars(tmp_path, registry_path, state_path):
    """Capture emitted shell variables as a dict."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        emit_shell_vars(str(registry_path), str(state_path))
    output = buf.getvalue()
    result = {}
    for line in output.strip().splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            # Strip shell quoting
            val = val.strip("'")
            result[key] = val
    return result


def test_qwen_preset_emits_correct_vars(tmp_path):
    """coding-quality preset on qwen3.5-27b produces the right launcher vars."""
    reg = _write_registry(tmp_path)
    state = _write_state(tmp_path, "coding-quality")
    v = _capture_shell_vars(tmp_path, reg, state)

    assert v["MODE_MODEL"] == "Qwen3.5-27B-VLM-MTP-8bit"
    assert v["SERVED_MODEL_NAME"] == "qwen3.5-27b"
    assert v["THINK"] == "true"
    assert v["TOOL_PARSER"] == "qwen3_coder"
    assert v["REASONING_PARSER"] == "qwen3"
    assert v["USE_MLLM"] == "true"
    assert v["USE_CONT_BATCH"] == "true"
    assert v["USE_MTP"] == "true"
    assert v["SPECPREFILL_ENABLED"] == "true"
    assert v["SPECPREFILL_THRESHOLD"] == "8192"
    assert v["SPECPREFILL_KEEP_PCT"] == "0.3"
    assert v["SPECPREFILL_DRAFT_MODEL"] == "Qwen3.5-2B-OptiQ-4bit"
    assert v["MODE_TEMP"] == "0.6"
    assert v["MODE_TOP_P"] == "0.95"
    assert v["KV_QUANTIZE"] == "true"
    assert v["PREFILL_STEP_SIZE"] == "256"


def test_gemma_preset_emits_correct_vars(tmp_path):
    """fast-iteration preset on gemma-4-26b produces different model vars."""
    reg = _write_registry(tmp_path)
    state_path = tmp_path / "runtime-state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active_model_preset": "fast-iteration",
                "active_service_presets": [],
                "active_backend": "vllm-mlx",
                "resident_models": ["gemma-4-26b-a4b-it"],
                "execution": {
                    "active_model": "gemma-4-26b-a4b-it",
                },
            }
        )
    )
    v = _capture_shell_vars(tmp_path, reg, state_path)

    assert v["MODE_MODEL"] == "gemma-4-26B-A4B-it-6bit"
    assert v["SERVED_MODEL_NAME"] == "gemma-4-26b-a4b-it"
    assert v["TOOL_PARSER"] == "gemma4"
    assert v["REASONING_PARSER"] == "gemma4"
    assert v["USE_MTP"] == "false"  # Gemma has no MTP
    assert v["SPECPREFILL_ENABLED"] == "false"  # no specprefill config
    assert v["MODE_TEMP"] == "0.5"
    assert v["MODE_TOP_P"] == "0.9"


def test_no_active_preset_emits_none(tmp_path):
    """No active preset produces MODE_MODEL=NONE."""
    reg = _write_registry(tmp_path)
    state = _write_state_no_preset(tmp_path)
    v = _capture_shell_vars(tmp_path, reg, state)

    assert v["MODE_MODEL"] == "NONE"
    assert v["SERVED_MODEL_NAME"] == ""


def test_invalid_state_raises(tmp_path):
    """Unknown preset in state file raises ValueError."""
    reg = _write_registry(tmp_path)
    state_path = tmp_path / "runtime-state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active_model_preset": "nonexistent-preset",
                "active_service_presets": [],
            }
        )
    )
    with pytest.raises(ValueError, match="unknown preset"):
        from vllm_mlx.runtime_config import load_runtime_state, resolve_runtime_state

        reg_doc = load_registry_document(str(reg))
        state = load_runtime_state(str(state_path))
        resolve_runtime_state(reg_doc, state)


def test_cli_entrypoint(tmp_path):
    """The module can be invoked as a CLI and produces parseable output."""
    reg = _write_registry(tmp_path)
    state = _write_state(tmp_path, "coding-quality")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_mlx.resolve_state",
            str(reg),
            str(state),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    assert "MODE_MODEL=" in result.stdout
    assert "SERVED_MODEL_NAME=" in result.stdout
    assert "TOOL_PARSER=" in result.stdout


# Import needed for test_invalid_state_raises
from vllm_mlx.runtime_config import load_registry_document
