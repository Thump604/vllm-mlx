# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the narrow qwen_code Thump pilot seam."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from mlx_vlm.generate import PromptCacheState

from vllm_mlx.thump.qwen_pilot import QwenPilotManager


class _FakeLanguageModel:
    def __init__(self):
        self.config = SimpleNamespace(model_type="gemma4")


def _prompt_cache_state(token_ids: list[int], cache: list[object]) -> PromptCacheState:
    state = PromptCacheState()
    state.update(token_ids, cache)
    return state


def test_qwen_pilot_manager_checkpoint_and_restore_round_trip(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    import vllm_mlx.thump.qwen_pilot as pilot

    fake_cache = [object()]
    manager = QwenPilotManager(
        language_model=_FakeLanguageModel(),
        model_path="/tmp/fake-gemma",
    )
    manager.record_finished_prompt_state(
        _prompt_cache_state([11, 12, 13], fake_cache),
        prompt_tokens=2,
        completion_tokens=1,
        route="chat",
    )

    class _FakeSession:
        def __init__(self, root_dir):
            self.root_dir = root_dir

        def initialize_from_live_cache(self, caches, *, total_tokens):
            assert caches is fake_cache
            assert total_tokens == 3

        def checkpoint(
            self,
            manifest_path,
            *,
            model_id_hash,
            session_id,
            sequence_id,
            prompt_tokens,
            generated_tokens,
        ):
            assert model_id_hash == pilot.model_id_hash_for_path("/tmp/fake-gemma")
            assert prompt_tokens == 3
            assert generated_tokens == 0
            manifest_path.write_text("manifest\n", encoding="utf-8")
            return SimpleNamespace(session_id=session_id, sequence_id=sequence_id)

        def close(self):
            return None

    fake_attached_session = SimpleNamespace(
        materialize_prompt_cache=lambda model, upto_tokens: ["restored-cache"],
        close=lambda: None,
    )

    monkeypatch.setattr(
        pilot.SessionSubstrate,
        "from_gemma4_model",
        lambda *args, **kwargs: _FakeSession(kwargs["root_dir"]),
    )
    monkeypatch.setattr(
        pilot.SessionSubstrate,
        "attach_gemma4_checkpoint",
        lambda *args, **kwargs: (fake_attached_session, object()),
    )

    artifact_path = tmp_path / "artifact"
    checkpoint = manager.checkpoint_latest_finished(
        artifact_path,
        qwen_session_id="fixed-session",
        workspace_path="/tmp/workspace",
    )
    assert checkpoint["artifact_path"] == str(artifact_path)
    prompt_state = json.loads((artifact_path / "prompt-state.json").read_text())
    assert prompt_state["token_ids"] == [11, 12, 13]
    assert prompt_state["qwen_session_id"] == "fixed-session"

    arm = manager.arm_restore(artifact_path)
    assert arm["fallback_count"] == 0
    assert arm["restore_mode"] == "thump_hot_restart"

    restored_state = manager.build_request_prompt_cache_state([11, 12, 13, 14])
    assert restored_state.token_ids == [11, 12, 13]
    assert restored_state.cache == ["restored-cache"]
    assert manager.status()["armed_restore"] is None


def test_qwen_pilot_manager_prefix_mismatch_falls_back(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    import vllm_mlx.thump.qwen_pilot as pilot

    manager = QwenPilotManager(
        language_model=_FakeLanguageModel(),
        model_path="/tmp/fake-gemma",
    )
    artifact_path = tmp_path / "artifact"
    artifact_path.mkdir()
    (artifact_path / "session.tsmf").write_text("manifest\n", encoding="utf-8")
    (artifact_path / "prompt-state.json").write_text(
        json.dumps(
            {
                "token_ids": [1, 2, 3],
                "cache_token_count": 3,
                "block_size_tokens": 16,
            }
        ),
        encoding="utf-8",
    )

    fake_attached_session = SimpleNamespace(
        materialize_prompt_cache=lambda model, upto_tokens: ["restored-cache"],
        close=lambda: None,
    )
    monkeypatch.setattr(
        pilot.SessionSubstrate,
        "attach_gemma4_checkpoint",
        lambda *args, **kwargs: (fake_attached_session, object()),
    )

    arm = manager.arm_restore(artifact_path)
    assert arm["fallback_count"] == 0

    fallback_state = manager.build_request_prompt_cache_state([9, 9, 9])
    assert fallback_state.cache is None
    consume = manager.status()["last_consume_status"]
    assert consume["status"] == "cold_fallback"
    assert consume["fallback_reason"] == "prompt_prefix_mismatch"


def test_thump_qwen_pilot_endpoints_use_local_manager(monkeypatch: pytest.MonkeyPatch):
    import vllm_mlx.server as srv

    class _FakeManager:
        def status(self):
            return {"enabled": True, "armed_restore": None}

        def checkpoint_latest_finished(
            self, artifact_path, *, qwen_session_id=None, workspace_path=None
        ):
            return {
                "artifact_path": artifact_path,
                "qwen_session_id": qwen_session_id,
                "workspace_path": workspace_path,
            }

        def arm_restore(self, artifact_path):
            return {
                "artifact_path": artifact_path,
                "fallback_count": 0,
                "fallback_rate": 0.0,
            }

    fake_engine = SimpleNamespace(
        is_mllm=True,
        get_stats=lambda: {"engine_type": "batched"},
        _mllm_instance=SimpleNamespace(
            get_thump_qwen_pilot_manager=lambda: _FakeManager()
        ),
    )

    previous_engine = srv._engine
    previous_model = srv._model_name
    monkeypatch.setenv("VLLM_MLX_ENABLE_THUMP_QWEN_PILOT", "1")
    srv._engine = fake_engine
    srv._model_name = "fake-gemma"
    try:
        client = TestClient(srv.app)
        status = client.get("/_internal/thump/qwen/status")
        assert status.status_code == 200
        assert status.json()["enabled"] is True

        checkpoint = client.post(
            "/_internal/thump/qwen/checkpoint",
            json={
                "artifact_path": "/tmp/artifact",
                "session_id": "fixed-session",
                "workspace_path": "/tmp/workspace",
            },
        )
        assert checkpoint.status_code == 200
        assert checkpoint.json()["qwen_session_id"] == "fixed-session"

        arm = client.post(
            "/_internal/thump/qwen/arm-restore",
            json={"artifact_path": "/tmp/artifact"},
        )
        assert arm.status_code == 200
        assert arm.json()["fallback_count"] == 0
    finally:
        srv._engine = previous_engine
        srv._model_name = previous_model


def test_qwen_pilot_restore_validation_failure_status_persists(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    import vllm_mlx.thump.qwen_pilot as pilot

    manager = QwenPilotManager(
        language_model=_FakeLanguageModel(),
        model_path="/tmp/fake-gemma",
    )
    artifact_path = tmp_path / "artifact"
    artifact_path.mkdir()
    (artifact_path / "session.tsmf").write_text("manifest\n", encoding="utf-8")
    (artifact_path / "prompt-state.json").write_text(
        json.dumps(
            {
                "token_ids": [1, 2, 3],
                "cache_token_count": 3,
                "block_size_tokens": 16,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pilot.SessionSubstrate,
        "attach_gemma4_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("bf16 materialize failed")
        ),
    )

    arm = manager.arm_restore(artifact_path)
    assert arm["fallback_count"] == 1
    assert arm["restore_mode"] == "cold_fallback"

    fresh_state = manager.build_request_prompt_cache_state([9, 9, 9])
    assert fresh_state.cache is None
    consume = manager.status()["last_consume_status"]
    assert consume["status"] == "restore_validation_failed"
    assert consume["fallback_reason"] == "bf16 materialize failed"
