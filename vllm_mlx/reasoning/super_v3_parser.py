# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for NVIDIA Nemotron 3 Super models.

Extends DeepSeek-R1 parser with Nemotron-specific logic:
- When enable_thinking=false, routes content to content field (not reasoning)
- When force_nonempty_content=true, ensures content is never empty

Source: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 on HuggingFace
"""

from .base import DeltaMessage
from .deepseek_r1_parser import DeepSeekR1ReasoningParser


class SuperV3ReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for NVIDIA Nemotron 3 Super v3.

    When thinking is disabled or content would be empty, moves reasoning
    output to the content field. This handles two Nemotron-specific cases:

    1. enable_thinking=false: Model outputs content without <think> tags,
       but the base parser may still route it to reasoning. This parser
       ensures it goes to content.

    2. force_nonempty_content=true: If reasoning exceeds max_tokens and
       </think> is truncated, the base parser returns empty content.
       This parser moves reasoning to content in that case.
    """

    @staticmethod
    def _request_template_kwargs(request) -> dict:
        return getattr(request, "chat_template_kwargs", None) or {}

    def extract_reasoning(
        self,
        model_output: str,
        request=None,
    ) -> tuple[str | None, str | None]:
        reasoning_content, final_content = super().extract_reasoning(model_output)

        # If thinking is disabled or force_nonempty_content is set,
        # and we ended up with no content, swap reasoning into content
        if request is not None and not final_content:
            kwargs = self._request_template_kwargs(request)
            thinking_off = kwargs.get("enable_thinking") is False
            force_content = kwargs.get("force_nonempty_content") is True
            if thinking_off or force_content:
                reasoning_content, final_content = None, reasoning_content

        return reasoning_content, final_content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        request=None,
    ) -> DeltaMessage | None:
        result = super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )
        if result is None:
            return result

        kwargs = self._request_template_kwargs(request) if request is not None else {}
        thinking_off = kwargs.get("enable_thinking") is False
        force_content = kwargs.get("force_nonempty_content") is True

        # Nemotron plain-text output should stream through assistant content,
        # but only when the request explicitly disabled thinking or forced
        # visible content. When thinking is enabled, Nemotron may emit only the
        # closing </think> token because the opening <think> came from the
        # prompt, so untagged deltas before </think> must remain reasoning.
        if (
            (thinking_off or force_content)
            and self.start_token not in current_text
            and self.end_token not in current_text
        ):
            combined = "".join(
                part for part in (result.reasoning, result.content) if part
            )
            return DeltaMessage(content=combined or None)

        if thinking_off:
            combined = "".join(
                part for part in (result.reasoning, result.content) if part
            )
            return DeltaMessage(content=combined or None)

        if force_content and result.content is None and result.reasoning:
            return DeltaMessage(content=result.reasoning)

        return result
