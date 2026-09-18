# SPDX-License-Identifier: Apache-2.0
"""The Omni audio-extraction subprocesses must never touch the terminal.

ffmpeg polls stdin for interactive keys. When the server runs in a
background process group - which every job-control launcher creates - a
background read of the controlling terminal raises SIGTTIN against the
whole group, server included. The symptom is a request frozen at 0% CPU
with no error, until subprocess.run's timeout fires and the audio is
silently dropped. Both the ffmpeg flag and the stdin redirect are asserted
so a refactor cannot quietly reintroduce either half.

Runs without MLX: nothing here needs a model.
"""

from __future__ import annotations

import subprocess
import pytest

from vllm_mlx.models import mllm


@pytest.fixture
def captured_runs(monkeypatch, tmp_path):
    """Route ffmpeg/ffprobe through a recorder instead of the real binaries."""
    calls: list[dict] = []

    def fake_run(argv, **kwargs):
        calls.append({"argv": argv, "kwargs": kwargs})
        # extract_audio_from_video checks the WAV is non-empty afterwards.
        if argv[0] == "ffmpeg":
            open(argv[-1], "wb").write(b"\0" * 64)
            return subprocess.CompletedProcess(argv, 0)
        return subprocess.CompletedProcess(argv, 0, stdout="audio\n", stderr="")

    # mllm imports subprocess/shutil inside the functions, so patch the
    # modules themselves rather than attributes on mllm.
    import shutil

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(mllm._temp_manager, "register", lambda p: p, raising=False)
    return calls


def test_ffmpeg_never_reads_the_terminal(captured_runs, tmp_path):
    video = tmp_path / "clip.mov"
    video.write_bytes(b"\0" * 16)

    out = mllm.extract_audio_from_video(str(video))

    assert out is not None and out.endswith(".wav")
    ffmpeg = [c for c in captured_runs if c["argv"][0] == "ffmpeg"]
    assert len(ffmpeg) == 1
    assert "-nostdin" in ffmpeg[0]["argv"], "ffmpeg must not poll stdin"
    assert (
        ffmpeg[0]["kwargs"].get("stdin") is subprocess.DEVNULL
    ), "ffmpeg must inherit a closed stdin, not the launcher's tty"


def test_ffprobe_never_reads_the_terminal(captured_runs, tmp_path):
    video = tmp_path / "clip.mov"
    video.write_bytes(b"\0" * 16)

    assert mllm._video_has_audio_track(str(video)) is True
    ffprobe = [c for c in captured_runs if c["argv"][0] == "ffprobe"]
    assert len(ffprobe) == 1
    assert ffprobe[0]["kwargs"].get("stdin") is subprocess.DEVNULL
