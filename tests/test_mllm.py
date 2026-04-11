# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX Multimodal Language Model (MLLM) wrapper."""

import platform
import sys
import types
from pathlib import Path

import pytest

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_mllm_model():
    """Return a small MLLM model for testing."""
    return "mlx-community/Qwen3-VL-4B-Instruct-3bit"


@pytest.fixture
def test_image_path(tmp_path):
    """Download a real image from Wikimedia Commons for tests."""
    pytest.importorskip("PIL")
    import requests
    from PIL import Image
    import io

    # Use a small dog image from Wikimedia Commons (public domain)
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        path = tmp_path / "test_image.jpg"
        img.save(path)
        return str(path)
    except Exception:
        # Fallback to synthetic image if download fails
        img = Image.new("RGB", (320, 240), color="blue")
        path = tmp_path / "test_image.jpg"
        img.save(path)
        return str(path)


@pytest.fixture
def test_video_path(tmp_path):
    """Download a real video from Wikimedia Commons for tests."""
    import requests

    # Use a short video from Wikimedia Commons (Creative Commons)
    # This is a 3-second sample video
    url = "https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/Big_Buck_Bunny_4K.webm/Big_Buck_Bunny_4K.webm.160p.webm"

    path = tmp_path / "test_video.webm"

    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(path)
    except Exception:
        # Fallback to synthetic video if download fails
        cv2 = pytest.importorskip("cv2")
        import numpy as np

        path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))

        # Create 30 frames (1 second)
        for i in range(30):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:] = (255, 0, 0)  # Blue in BGR
            out.write(frame)

        out.release()
        return str(path)


# =============================================================================
# Unit Tests - No Model Loading Required
# =============================================================================


class TestMLLMHelperFunctions:
    """Test helper functions that don't require model loading."""

    def test_is_base64_image(self):
        """Test base64 image detection."""
        from vllm_mlx.models.mllm import is_base64_image

        assert is_base64_image("data:image/png;base64,iVBORw0KGgo=")
        assert is_base64_image("data:image/jpeg;base64,/9j/4AAQSkZJRg==")
        assert not is_base64_image("https://example.com/image.jpg")
        assert not is_base64_image("/path/to/image.jpg")

    def test_is_base64_video(self):
        """Test base64 video detection."""
        from vllm_mlx.models.mllm import is_base64_video

        assert is_base64_video("data:video/mp4;base64,AAAA")
        assert is_base64_video("data:video/webm;base64,AAAA")
        assert not is_base64_video("https://example.com/video.mp4")
        assert not is_base64_video("/path/to/video.mp4")

    def test_is_url(self):
        """Test URL detection."""
        from vllm_mlx.models.mllm import is_url

        assert is_url("https://example.com/image.jpg")
        assert is_url("http://example.com/video.mp4")
        assert not is_url("/path/to/file.jpg")
        assert not is_url("data:image/png;base64,AAAA")


class TestMLLMThinkingPropagation:
    """Unit tests for thinking-mode propagation in MLLM chat paths."""

    @staticmethod
    def _build_model():
        from types import SimpleNamespace

        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM.__new__(MLXMultimodalLM)
        model._loaded = True
        model._video_native = False
        model._cache_manager = None
        model.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda text: [1, 2, 3])
        )
        model.model = SimpleNamespace(
            config={"model_type": "gemma4"}, language_model=None
        )
        model.model_name = "test-model"
        return model

    def test_chat_passes_enable_thinking_to_template_and_generate(self, monkeypatch):
        from types import SimpleNamespace

        calls = {}

        def fake_generate(*args, **kwargs):
            calls["generate"] = kwargs
            return SimpleNamespace(text="ok", prompt_tokens=12, generation_tokens=3)

        def fake_apply_chat_template(
            processor, config, prompt, add_generation_prompt=True, **kwargs
        ):
            calls["template"] = {
                "processor": processor,
                "config": config,
                "prompt": prompt,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
            return "FORMATTED"

        fake_root = types.ModuleType("mlx_vlm")
        fake_root.generate = fake_generate
        fake_generate_mod = types.ModuleType("mlx_vlm.generate")
        fake_generate_mod.apply_chat_template = fake_apply_chat_template
        fake_models = types.ModuleType("mlx_vlm.models")
        fake_cache = types.ModuleType("mlx_vlm.models.cache")
        fake_cache.make_prompt_cache = lambda *_args, **_kwargs: None

        monkeypatch.setitem(sys.modules, "mlx_vlm", fake_root)
        monkeypatch.setitem(sys.modules, "mlx_vlm.generate", fake_generate_mod)
        monkeypatch.setitem(sys.modules, "mlx_vlm.models", fake_models)
        monkeypatch.setitem(sys.modules, "mlx_vlm.models.cache", fake_cache)

        model = self._build_model()
        output = model.chat(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            max_tokens=32,
            temperature=1.0,
            top_p=0.95,
            chat_template_kwargs={"enable_thinking": True, "foo": "bar"},
        )

        assert output.text == "ok"
        assert calls["template"]["kwargs"]["enable_thinking"] is True
        assert calls["template"]["kwargs"]["foo"] == "bar"
        assert calls["generate"]["enable_thinking"] is True
        assert calls["generate"]["top_p"] == 0.95

    def test_stream_chat_uses_env_thinking_default(self, monkeypatch):
        monkeypatch.setenv("VLLM_MLX_ENABLE_THINKING", "false")
        calls = {}

        def fake_stream_generate(*args, **kwargs):
            calls["stream_generate"] = kwargs
            yield types.SimpleNamespace(text="chunk", prompt_tokens=9)

        def fake_apply_chat_template(
            processor, config, prompt, add_generation_prompt=True, **kwargs
        ):
            calls["template"] = kwargs
            return "FORMATTED"

        fake_root = types.ModuleType("mlx_vlm")
        fake_root.stream_generate = fake_stream_generate
        fake_generate_mod = types.ModuleType("mlx_vlm.generate")
        fake_generate_mod.apply_chat_template = fake_apply_chat_template
        fake_models = types.ModuleType("mlx_vlm.models")
        fake_cache = types.ModuleType("mlx_vlm.models.cache")
        fake_cache.make_prompt_cache = lambda *_args, **_kwargs: None

        monkeypatch.setitem(sys.modules, "mlx_vlm", fake_root)
        monkeypatch.setitem(sys.modules, "mlx_vlm.generate", fake_generate_mod)
        monkeypatch.setitem(sys.modules, "mlx_vlm.models", fake_models)
        monkeypatch.setitem(sys.modules, "mlx_vlm.models.cache", fake_cache)

        model = self._build_model()
        chunks = list(
            model.stream_chat(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
                max_tokens=16,
                temperature=0.7,
            )
        )

        assert chunks[0].text == "chunk"
        assert calls["template"]["enable_thinking"] is False
        assert calls["stream_generate"]["enable_thinking"] is False


class TestMLXMultimodalLMDetection:
    """Unit tests for supported-family helpers."""

    def test_is_mllm_model_detects_gemma4(self):
        from vllm_mlx.models.mllm import MLXMultimodalLM

        assert MLXMultimodalLM.is_mllm_model("mlx-community/gemma-4-27b-it-4bit")
        assert MLXMultimodalLM.is_mllm_model("mlx-community/gemma4-4b-it-4bit")

    def test_supported_families_lists_gemma4(self):
        from vllm_mlx.models.mllm import MLXMultimodalLM

        supported = MLXMultimodalLM.list_supported_model_families()

        assert supported["Gemma 4"] == "Gemma 4 multimodal models"


class TestVideoFrameExtraction:
    """Test video frame extraction functions."""

    def test_get_video_info(self, test_video_path):
        """Test getting video information."""
        cv2 = pytest.importorskip("cv2")

        # Use OpenCV directly since get_video_info may not be exported
        cap = cv2.VideoCapture(test_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Video from Wikimedia will have different properties
        assert total_frames > 0
        assert fps > 0
        assert width > 0
        assert height > 0

    def test_extract_video_frames_smart(self, test_video_path):
        """Test smart frame extraction."""
        cv2 = pytest.importorskip("cv2")
        from vllm_mlx.models.mllm import extract_video_frames_smart

        # Extract frames
        frames = extract_video_frames_smart(test_video_path, fps=2.0, max_frames=10)

        assert len(frames) > 0
        assert len(frames) <= 10
        # Check frame shape (height, width, channels)
        assert len(frames[0].shape) == 3  # Should be 3D array

    def test_extract_frames_respects_max_frames(self, test_video_path):
        """Test that max_frames limit is respected."""
        cv2 = pytest.importorskip("cv2")
        from vllm_mlx.models.mllm import extract_video_frames_smart

        frames = extract_video_frames_smart(test_video_path, fps=30.0, max_frames=5)

        assert len(frames) <= 5

    def test_save_frames_to_temp(self, test_video_path):
        """Test saving frames to temp files."""
        cv2 = pytest.importorskip("cv2")
        from vllm_mlx.models.mllm import extract_video_frames_smart, save_frames_to_temp

        frames = extract_video_frames_smart(test_video_path, fps=1.0, max_frames=2)
        paths = save_frames_to_temp(frames)

        assert len(paths) == len(frames)
        for path in paths:
            assert Path(path).exists()
            assert path.endswith(".jpg")


class TestImageProcessing:
    """Test image processing functions."""

    def test_process_image_input_local_file(self, test_image_path):
        """Test processing local image file."""
        from vllm_mlx.models.mllm import process_image_input

        result = process_image_input(test_image_path)
        assert result == test_image_path

    def test_process_image_input_dict_format(self, test_image_path):
        """Test processing image in dict format."""
        from vllm_mlx.models.mllm import process_image_input

        # OpenAI format
        result = process_image_input({"url": test_image_path})
        assert Path(result).exists()


class TestVideoProcessing:
    """Test video processing functions."""

    def test_process_video_input_local_file(self, test_video_path):
        """Test processing local video file."""
        from vllm_mlx.models.mllm import process_video_input

        result = process_video_input(test_video_path)
        assert result == test_video_path

    def test_process_video_input_dict_format(self, test_video_path):
        """Test processing video in dict format."""
        from vllm_mlx.models.mllm import process_video_input

        # OpenAI format
        result = process_video_input({"url": test_video_path})
        assert Path(result).exists()

    def test_process_video_input_empty_raises(self):
        """Test that empty input raises error."""
        from vllm_mlx.models.mllm import process_video_input

        with pytest.raises(ValueError):
            process_video_input("")

        with pytest.raises(ValueError):
            process_video_input({})


# =============================================================================
# MLLM Model Tests
# =============================================================================


class TestMLLMModelInit:
    """Test MLLM model initialization (no model loading)."""

    def test_model_init(self):
        """Test model initialization."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model")
        assert model.model_name == "test-model"
        assert not model._loaded

    def test_model_info_not_loaded(self):
        """Test model info when not loaded."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model")
        info = model.get_model_info()

        assert info["loaded"] is False
        assert info["model_name"] == "test-model"

    def test_model_repr(self):
        """Test model string representation."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model")
        repr_str = repr(model)

        assert "MLXMultimodalLM" in repr_str
        assert "test-model" in repr_str


# =============================================================================
# Integration Tests - Require Model Loading (Slow)
# =============================================================================


@pytest.mark.slow
class TestMLLMImageGeneration:
    """Integration tests for MLLM image generation."""

    def test_generate_with_image(self, small_mllm_model, test_image_path):
        """Test generation with an image."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        output = model.generate(
            prompt="What animal is in this image?",
            images=[test_image_path],
            max_tokens=30,
        )

        assert output.text is not None
        assert len(output.text) > 0
        assert output.completion_tokens > 0

    def test_describe_image(self, small_mllm_model, test_image_path):
        """Test describe_image convenience method."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        description = model.describe_image(test_image_path, max_tokens=30)

        assert description is not None
        assert len(description) > 0


@pytest.mark.slow
class TestMLLMVideoGeneration:
    """Integration tests for MLLM video generation."""

    def test_generate_with_video(self, small_mllm_model, test_video_path):
        """Test generation with a video."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        output = model.generate(
            prompt="Describe this video.",
            videos=[test_video_path],
            video_fps=1.0,
            video_max_frames=4,
            max_tokens=20,
        )

        assert output.text is not None
        assert len(output.text) > 0

    def test_describe_video(self, small_mllm_model, test_video_path):
        """Test describe_video convenience method."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        description = model.describe_video(
            test_video_path,
            fps=1.0,
            max_frames=4,
            max_tokens=20,
        )

        assert description is not None
        assert len(description) > 0


@pytest.mark.slow
class TestMLLMChat:
    """Integration tests for MLLM chat interface."""

    def test_chat_with_image(self, small_mllm_model, test_image_path):
        """Test chat interface with image."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image_path},
                    {"type": "text", "text": "What animal is this?"},
                ],
            }
        ]

        output = model.chat(messages, max_tokens=30)

        assert output.text is not None
        assert len(output.text) > 0

    def test_chat_with_video(self, small_mllm_model, test_video_path):
        """Test chat interface with video."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": test_video_path},
                    {"type": "text", "text": "Describe the colors in this video."},
                ],
            }
        ]

        output = model.chat(messages, max_tokens=30, video_fps=1.0, video_max_frames=4)

        assert output.text is not None
        assert len(output.text) > 0
