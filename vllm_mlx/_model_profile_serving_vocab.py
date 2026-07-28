# SPDX-License-Identifier: Apache-2.0
"""Closed ModelProfile v1 vocabulary used by legacy serving imports."""

from __future__ import annotations

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
    "features": _FEATURE_NAMES,
    "activation_policy": frozenset({"owner_override_fields"}),
    "request_policy": frozenset(
        {"required_fields", "allowed_fields", "forbidden_fields"}
    ),
}
_REGISTRY_FIELDS_REQUIRING_TARGET_CONTRACT = (
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
)
_SERVING_TOP_LEVEL_FIELDS = frozenset(
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
) | frozenset(_REGISTRY_FIELDS_REQUIRING_TARGET_CONTRACT)
