# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for NVIDIA Nemotron 3 Super models.

Extends DeepSeek-R1 parser with Nemotron-specific logic:
- When enable_thinking=false, routes content to content field (not reasoning)
- When force_nonempty_content=true, ensures content is never empty

Source: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 on HuggingFace
"""

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

    def extract_reasoning(
        self,
        model_output: str,
        request=None,
    ) -> tuple[str | None, str | None]:
        reasoning_content, final_content = super().extract_reasoning(model_output)

        # If thinking is disabled or force_nonempty_content is set,
        # and we ended up with no content, swap reasoning into content
        if request is not None and final_content is None:
            kwargs = getattr(request, "chat_template_kwargs", None) or {}
            thinking_off = kwargs.get("enable_thinking") is False
            force_content = kwargs.get("force_nonempty_content") is True
            if thinking_off or force_content:
                reasoning_content, final_content = None, reasoning_content

        return reasoning_content, final_content
