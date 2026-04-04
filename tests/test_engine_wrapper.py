# bin/test/test_engine_wrapper.py
"""Tests for EngineWrapper — thin abstraction over BatchGenerator."""
from unittest.mock import MagicMock, patch
import pytest


def test_engine_wrapper_has_work_empty():
    """Empty batch reports no work."""
    from vllm_mlx.engine_wrapper import EngineWrapper

    bg = MagicMock()
    bg.__len__ = MagicMock(return_value=0)
    wrapper = EngineWrapper(bg)
    assert wrapper.has_work() is False


def test_engine_wrapper_has_work_active():
    """Active batch reports work."""
    from vllm_mlx.engine_wrapper import EngineWrapper

    bg = MagicMock()
    bg.__len__ = MagicMock(return_value=3)
    wrapper = EngineWrapper(bg)
    assert wrapper.has_work() is True


def test_engine_wrapper_has_work_from_batch_generator_internals():
    """BatchGenerator without __len__ still reports queued/prompt work."""
    from vllm_mlx.engine_wrapper import EngineWrapper

    class BatchLike:
        def __init__(self):
            self._unprocessed_sequences = [("uid",)]
            self._currently_processing = []
            self._prompt_batch = []
            self._generation_batch = []

    wrapper = EngineWrapper(BatchLike())
    assert wrapper.has_work() is True


def test_engine_wrapper_step_returns_tuple():
    """step() returns (prompt_resps, gen_resps) tuple."""
    from vllm_mlx.engine_wrapper import EngineWrapper

    bg = MagicMock()
    bg.next.return_value = (["prompt"], ["gen"])
    wrapper = EngineWrapper(bg)
    prompt_resps, gen_resps = wrapper.step()
    assert prompt_resps == ["prompt"]
    assert gen_resps == ["gen"]


def test_engine_wrapper_insert_delegates():
    """insert() passes through to BatchGenerator."""
    from vllm_mlx.engine_wrapper import EngineWrapper

    bg = MagicMock()
    bg.insert.return_value = [42]
    wrapper = EngineWrapper(bg)
    uids = wrapper.insert([[1, 2, 3]], max_tokens=[100])
    bg.insert.assert_called_once()
    assert uids == [42]


def test_engine_wrapper_insert_passes_extended_args():
    """insert() forwards all_tokens and state_machines for richer scheduling."""
    from vllm_mlx.engine_wrapper import EngineWrapper

    bg = MagicMock()
    bg.insert.return_value = [7]
    wrapper = EngineWrapper(bg)
    state_machine = object()
    wrapper.insert(
        [[1, 2, 3]],
        max_tokens=[10],
        all_tokens=[[1, 2, 3]],
        state_machines=[state_machine],
    )
    _, kwargs = bg.insert.call_args
    assert kwargs["all_tokens"] == [[1, 2, 3]]
    assert kwargs["state_machines"] == [state_machine]


def test_engine_wrapper_insert_segments_delegates():
    """insert_segments() forwards segmented prompts and cache context."""
    from vllm_mlx.engine_wrapper import EngineWrapper

    bg = MagicMock()
    bg.insert_segments.return_value = [8]
    wrapper = EngineWrapper(bg)
    state_machine = object()
    uids = wrapper.insert_segments(
        [[[1, 2, 3], [4, 5]]],
        max_tokens=[10],
        caches=[["cache"]],
        all_tokens=[[1, 2, 3]],
        state_machines=[state_machine],
    )

    bg.insert_segments.assert_called_once()
    _, kwargs = bg.insert_segments.call_args
    assert kwargs["all_tokens"] == [[1, 2, 3]]
    assert kwargs["state_machines"] == [state_machine]
    assert uids == [8]


def test_engine_wrapper_remove_delegates():
    """remove() passes through to BatchGenerator."""
    from vllm_mlx.engine_wrapper import EngineWrapper

    bg = MagicMock()
    wrapper = EngineWrapper(bg)
    wrapper.remove([42])
    bg.remove.assert_called_once_with([42])
