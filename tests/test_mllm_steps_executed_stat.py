# SPDX-License-Identifier: Apache-2.0
"""Focused regression coverage adapted from Don Jacobsmeyer's PR #749."""

try:
    import mlx.core as mx  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

import pytest

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class TestMLLMSchedulerStepsExecuted:
    def _make_scheduler(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        class FakeTokenizer:
            eos_token_id = None

            def encode(self, text):
                return [1, 2, 3]

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        class FakeModel:
            config = None

        scheduler = MLLMScheduler(model=FakeModel(), processor=FakeProcessor())
        scheduler._schedule_waiting = lambda: []
        return scheduler

    def test_get_stats_reports_zero_steps_initially(self):
        scheduler = self._make_scheduler()

        assert scheduler.get_stats()["steps_executed"] == 0

    def test_completed_steps_increment_once(self):
        scheduler = self._make_scheduler()

        scheduler.step()
        scheduler.step()
        scheduler.step()

        assert scheduler.get_stats()["steps_executed"] == 3

    def test_failed_step_is_not_counted(self):
        scheduler = self._make_scheduler()

        class ExplodingBatchGenerator:
            def process_pending_removals(self):
                pass

            def next(self):
                raise RuntimeError("simulated forward-pass failure")

        scheduler.batch_generator = ExplodingBatchGenerator()
        scheduler.running = {"req-exploding": object()}

        with pytest.raises(RuntimeError, match="simulated forward-pass failure"):
            scheduler.step()

        assert scheduler._steps_executed == 0

        scheduler.batch_generator = None
        scheduler.running = {}
        scheduler.step()
        assert scheduler.get_stats()["steps_executed"] == 1


class TestBatchedEngineStepsExecutedPromotion:
    def _make_engine(self, mllm_stats):
        from vllm_mlx.engine.batched import BatchedEngine

        class FakeMLLMScheduler:
            def get_stats(self):
                return mllm_stats

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._mllm_scheduler = FakeMLLMScheduler()
        engine._engine = None
        engine._model_name = "fake-mllm-model"
        engine._created_at = 0.0
        engine._is_mllm = True
        engine._loaded = True
        engine._stream_interval = 1
        engine._mllm_draft_model = None
        return engine

    def test_promotes_steps_executed_to_top_level(self):
        engine = self._make_engine({"steps_executed": 42, "num_waiting": 0})

        assert engine.get_stats()["steps_executed"] == 42

    def test_omits_steps_executed_when_scheduler_lacks_it(self):
        engine = self._make_engine({"num_waiting": 0})

        assert "steps_executed" not in engine.get_stats()
