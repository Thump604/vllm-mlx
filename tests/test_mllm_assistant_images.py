# SPDX-License-Identifier: Apache-2.0
"""Assistant images must reach Simple-mode templates and generation together.

Exercise the real Anthropic adapter, MLLM message builder and chat methods.
Only external template rendering, media I/O and model inference are replaced.
"""

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_mlx.api.anthropic_adapter import anthropic_to_openai
from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.models.mllm import MLXMultimodalLM, _build_mllm_chat_messages


@pytest.fixture
def mllm_image_boundary(monkeypatch):
    seen = {}
    prepared_images = [object(), object()]

    def prepare_images(image_urls):
        seen["image_urls"] = list(image_urls)
        return prepared_images

    def get_chat_template(processor, messages, **kwargs):
        seen["messages"] = copy.deepcopy(messages)
        return "image prompt"

    def generate(model, processor, prompt, images, **kwargs):
        seen["prompt"] = prompt
        seen["images"] = images
        return SimpleNamespace(text="ok", prompt_tokens=3, generation_tokens=1)

    vlm = ModuleType("mlx_vlm")
    vlm.generate = generate
    vlm.stream_generate = lambda *args, **kwargs: iter([generate(*args, **kwargs)])
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.get_chat_template = get_chat_template
    cache = ModuleType("mlx_vlm.models.cache")
    cache.make_prompt_cache = lambda *args, **kwargs: []
    models = ModuleType("mlx_vlm.models")
    models.cache = cache
    for module in (vlm, prompt_utils, models, cache):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    wrapper = MLXMultimodalLM("test-model", enable_cache=False)
    wrapper._loaded = True
    wrapper.model = SimpleNamespace(language_model=object())
    wrapper.processor = SimpleNamespace(encode=lambda text: [1, 2, 3])
    monkeypatch.setattr(wrapper, "_prepare_images", prepare_images)
    return wrapper, seen, prepared_images


@pytest.mark.parametrize("route", ["chat", "stream_chat"])
@pytest.mark.parametrize("source_type", ["base64", "url"])
@pytest.mark.parametrize("with_tool_use", [False, True])
@pytest.mark.parametrize("with_text", [False, True])
def test_assistant_images_reach_chat_template_and_generation(
    mllm_image_boundary, route, source_type, with_tool_use, with_text
):
    wrapper, seen, prepared_images = mllm_image_boundary
    if source_type == "base64":
        sources = [
            {"type": "base64", "media_type": "image/png", "data": "YWJj"},
            {"type": "base64", "media_type": "image/jpeg", "data": "ZGVm"},
        ]
        expected_urls = [
            "data:image/png;base64,YWJj",
            "data:image/jpeg;base64,ZGVm",
        ]
    else:
        sources = [
            {"type": "url", "url": "https://example.com/first.png"},
            {"type": "url", "url": "https://example.com/second.jpg"},
        ]
        expected_urls = [
            "https://example.com/first.png",
            "https://example.com/second.jpg",
        ]

    content = [
        {"type": "image", "source": sources[0]},
        {"type": "image", "source": sources[1]},
    ]
    expected_content = [{"type": "image"}, {"type": "image"}]
    if with_text:
        content = [
            {"type": "text", "text": "Before "},
            content[0],
            {"type": "text", "text": " between "},
            content[1],
            {"type": "text", "text": " after."},
        ]
        expected_content = [
            {"type": "text", "text": "Before ", "content": "Before "},
            {"type": "image"},
            {"type": "text", "text": " between ", "content": " between "},
            {"type": "image"},
            {"type": "text", "text": " after.", "content": " after."},
        ]
    expected_assistant = {"role": "assistant", "content": expected_content}
    if with_tool_use:
        content.append(
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "zoom",
                "input": {"scale": 2},
            }
        )
        expected_assistant["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "zoom", "arguments": {"scale": 2}},
            }
        ]

    request = AnthropicRequest(
        model="test-model",
        max_tokens=16,
        messages=[
            {"role": "assistant", "content": content},
            {"role": "user", "content": "Continue."},
        ],
    )
    messages = [
        message.model_dump() for message in anthropic_to_openai(request).messages
    ]
    result = getattr(wrapper, route)(messages=messages, max_tokens=16)
    if route == "stream_chat":
        assert "".join(chunk.text for chunk in result) == "ok"
    else:
        assert result.text == "ok"

    assert seen["messages"] == [
        expected_assistant,
        {
            "role": "user",
            "content": [{"type": "text", "text": "Continue.", "content": "Continue."}],
        },
    ]
    assert seen["image_urls"] == expected_urls
    assert seen["images"] == prepared_images
    assert seen["prompt"] == "image prompt"


def test_assistant_text_parts_keep_string_content():
    image_urls = []
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world."},
            ],
        }
    ]

    assert _build_mllm_chat_messages(
        messages, all_image_urls=image_urls, video_frame_counts={}
    ) == [{"role": "assistant", "content": "Hello world."}]
    assert image_urls == []
