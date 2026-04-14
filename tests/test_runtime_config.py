# SPDX-License-Identifier: Apache-2.0
"""Tests for the runtime registry/state Phase 1 contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from vllm_mlx.runtime_config import (
    load_registry_document,
    load_runtime_state,
    resolve_runtime_state,
)


def _write_registry(
    tmp_path: Path, *, concurrent_model: str = "shared_candidate"
) -> Path:
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
                        "source": "/models/qwen27",
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
                            "source": "/models/qwen2b",
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
                        "source": "/models/gemma26",
                        "family": "gemma4",
                        "architecture": "dense",
                        "execution_class": concurrent_model,
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
                    {
                        "id": "qwen3.5-122b-a10b",
                        "display_name": "Qwen 3.5 122B",
                        "source": "/models/qwen122",
                        "family": "qwen3.5",
                        "architecture": "open_moe",
                        "execution_class": "solo_only",
                        "estimated_memory_gb": 110,
                        "supports_reasoning": True,
                        "supports_tools": True,
                        "supports_specprefill": True,
                        "supports_kv_quant": True,
                        "serving_profile": {
                            "continuous_batching": True,
                            "prefill_step_size": 256,
                            "tool_call_parser": "qwen3_coder",
                            "reasoning_parser": "qwen3",
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
                    }
                ],
                "service_presets": [
                    {
                        "id": "creative-media",
                        "display_name": "Creative Media",
                        "services": {"comfyui": True, "speech": True},
                        "mcp_bundles": ["image-gen", "speech"],
                    }
                ],
                "discussion_profiles": [
                    {
                        "id": "debate",
                        "display_name": "Debate Room",
                        "concurrent_pool": [
                            "qwen3.5-27b",
                            "gemma-4-26b-a4b-it",
                        ],
                        "solo_pool": ["qwen3.5-122b-a10b"],
                        "human_seats": ["moderator"],
                    }
                ],
            },
            sort_keys=False,
        )
    )
    return path


def _write_runtime_state(tmp_path: Path) -> Path:
    path = tmp_path / "runtime-state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active_model_preset": "coding-quality",
                "active_service_presets": ["creative-media"],
                "active_backend": "vllm-mlx",
                "resident_models": ["qwen3.5-27b", "gemma-4-26b-a4b-it"],
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


def test_load_registry_document_and_resolve_runtime_state(tmp_path):
    registry = load_registry_document(_write_registry(tmp_path))
    state = load_runtime_state(_write_runtime_state(tmp_path))

    resolved = resolve_runtime_state(registry, state)

    assert registry.policy_defaults.memory_budget_gb == 96
    assert registry.models["qwen3.5-27b"].draft_model is not None
    assert (
        registry.models["qwen3.5-27b"].serving_profile.tool_call_parser == "qwen3_coder"
    )
    assert registry.discussion_profiles["debate"].concurrent_pool == (
        "qwen3.5-27b",
        "gemma-4-26b-a4b-it",
    )
    assert resolved.model_preset is not None
    assert resolved.model is not None
    assert resolved.model.id == "qwen3.5-27b"
    assert [preset.id for preset in resolved.service_presets] == ["creative-media"]


def test_load_registry_document_rejects_invalid_discussion_pool(tmp_path):
    path = _write_registry(tmp_path, concurrent_model="solo_only")

    with pytest.raises(ValueError, match="concurrent_pool requires shared_candidate"):
        load_registry_document(path)


def test_resolve_runtime_state_rejects_unknown_service_preset(tmp_path):
    registry = load_registry_document(_write_registry(tmp_path))
    state_path = _write_runtime_state(tmp_path)
    state_raw = json.loads(state_path.read_text())
    state_raw["active_service_presets"] = ["missing"]
    state_path.write_text(json.dumps(state_raw))
    state = load_runtime_state(state_path)

    with pytest.raises(
        ValueError, match="active_service_presets references unknown preset"
    ):
        resolve_runtime_state(registry, state)
