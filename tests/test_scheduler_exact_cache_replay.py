# SPDX-License-Identifier: Apache-2.0
"""Full prompt hits must replay one token from independently rewound state."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
)

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


class _RecordingBatch:
    def __init__(self):
        self.inserts = []

    def insert(self, prompts, **kwargs):
        self.inserts.append((prompts, kwargs))
        return [len(self.inserts)]


def _scheduler(monkeypatch, model=None, max_kv_size=0):
    if model is None:
        model = SimpleNamespace(layers=[object()])
    tokenizer = SimpleNamespace(
        eos_token_id=-1,
        encode=lambda text: list(text),
        decode=lambda tokens: str(tokens),
    )
    scheduler = Scheduler(
        model,
        tokenizer,
        SchedulerConfig(enable_prefix_cache=False, max_kv_size=max_kv_size),
    )
    scheduler.memory_aware_cache = MemoryAwarePrefixCache(
        model, MemoryCacheConfig(max_memory_mb=8, min_prefix_tokens=1)
    )
    scheduler.batch_generator = _RecordingBatch()
    monkeypatch.setattr(scheduler, "_ensure_batch_generator", lambda _params: None)
    return scheduler


def _kv(tokens):
    layer = KVCache()
    data = mx.arange(len(tokens) * 4, dtype=mx.float32).reshape(1, 1, len(tokens), 4)
    layer.update_and_fetch(data, data)
    mx.eval(*layer.state)
    return layer


def _request(tokens, request_id="replay"):
    return Request(
        request_id=request_id,
        prompt=list(tokens),
        sampling_params=SamplingParams(temperature=0, max_tokens=4),
    )


def _llama():
    from mlx_lm.models.llama import Model, ModelArgs

    mx.random.seed(17)
    return Model(
        ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
        )
    )


@pytest.mark.parametrize("extra_tokens", [[], [8, 9]], ids=["exact", "supersequence"])
def test_full_hit_prefills_only_the_last_token(monkeypatch, extra_tokens):
    scheduler = _scheduler(monkeypatch)
    tokens = [1, 2, 3, 4, 5, 6]
    stored_tokens = tokens + extra_tokens
    cache = scheduler.memory_aware_cache
    assert cache.store(stored_tokens, [_kv(stored_tokens)])
    stored = cache._entries[tuple(stored_tokens)].cache[0]
    original_keys = stored.state[0].tolist()

    request = _request(tokens)
    scheduler.add_request(request)
    assert request.remaining_tokens == []

    scheduler._schedule_waiting()

    prompts, kwargs = scheduler.batch_generator.inserts[-1]
    assert prompts == [[tokens[-1]]]
    assert request.cached_tokens == len(tokens) - 1
    restored = kwargs["caches"][0][0]
    assert restored.offset == len(tokens) - 1
    assert restored.state[0].shape[2] == len(tokens) - 1
    assert restored is not stored
    assert stored.offset == len(stored_tokens)
    assert stored.state[0].tolist() == original_keys

    # A mutating replay must not overwrite the resident snapshot's backing.
    update = mx.full((1, 1, 1, 4), 999.0)
    restored.update_and_fetch(update, update)
    mx.eval(*restored.state)
    assert stored.state[0].tolist() == original_keys

    second = _request(tokens, request_id="second-replay")
    scheduler.add_request(second)
    scheduler._schedule_waiting()
    second_prompts, second_kwargs = scheduler.batch_generator.inserts[-1]
    assert second_prompts == [[tokens[-1]]]
    assert second_kwargs["caches"][0][0].offset == len(tokens) - 1
    assert second_kwargs["caches"][0][0] is not restored
    assert stored.state[0].tolist() == original_keys


def test_ssd_full_hit_is_rewound_before_last_token_replay(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    tokens = [1, 2, 3, 4]
    assert scheduler.memory_aware_cache.store(tokens, [_kv(tokens)])
    request = _request(tokens)
    scheduler.add_request(request)
    # Both synchronous and asynchronous SSD promotion publish this request shape.
    request.cache_hit_type = "ssd_hit"

    scheduler._schedule_waiting()

    prompts, kwargs = scheduler.batch_generator.inserts[-1]
    assert prompts == [[tokens[-1]]]
    assert request.cached_tokens == len(tokens) - 1
    assert kwargs["caches"][0][0].offset == len(tokens) - 1


@pytest.mark.parametrize("with_kv", [False, True], ids=["recurrent", "hybrid"])
def test_nonrewindable_full_hit_keeps_cold_prefill(monkeypatch, with_kv):
    scheduler = _scheduler(monkeypatch)
    tokens = [1, 2, 3, 4]
    recurrent = ArraysCache(size=2)
    recurrent.cache = [mx.ones((1, 4)), mx.ones((1, 4))]
    layers = [_kv(tokens), recurrent] if with_kv else [recurrent]
    assert scheduler.memory_aware_cache.store(tokens, layers)
    request = _request(tokens)
    scheduler.add_request(request)

    scheduler._schedule_waiting()

    prompts, kwargs = scheduler.batch_generator.inserts[-1]
    assert prompts == [tokens]
    assert kwargs["caches"] is None
    assert request.cached_tokens == 0
    resident = scheduler.memory_aware_cache._entries[tuple(tokens)].cache
    if with_kv:
        assert resident[0].offset == len(tokens)
    assert resident[-1].state[0].tolist() == [[1.0] * 4]


def test_uncertain_coverage_keeps_cold_prefill(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    tokens = [1, 2, 3, 4]
    assert scheduler.memory_aware_cache.store(tokens, [_kv(tokens[:-1])])
    request = _request(tokens)
    scheduler.add_request(request)

    scheduler._schedule_waiting()

    prompts, kwargs = scheduler.batch_generator.inserts[-1]
    assert prompts == [tokens]
    assert kwargs["caches"] is None
    assert request.cached_tokens == 0


def test_rejected_replay_clone_keeps_cold_prefill(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    tokens = [1, 2, 3, 4]
    assert scheduler.memory_aware_cache.store(tokens, [_kv(tokens)])
    monkeypatch.setattr(
        scheduler.memory_aware_cache, "clone_for_replay", lambda _cache: None
    )
    request = _request(tokens)
    scheduler.add_request(request)

    scheduler._schedule_waiting()

    prompts, kwargs = scheduler.batch_generator.inserts[-1]
    assert prompts == [tokens]
    assert kwargs["caches"] is None
    assert request.cached_tokens == 0


def test_container_cache_keeps_cold_prefill(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    tokens = [1, 2, 3, 4]
    container = CacheList(_kv(tokens), _kv(tokens))
    assert scheduler.memory_aware_cache.store(tokens, [container])
    request = _request(tokens)
    scheduler.add_request(request)

    scheduler._schedule_waiting()

    prompts, kwargs = scheduler.batch_generator.inserts[-1]
    assert prompts == [tokens]
    assert kwargs["caches"] is None
    resident = scheduler.memory_aware_cache._entries[tuple(tokens)].cache[0]
    assert all(child.offset == len(tokens) for child in resident.caches)


@pytest.mark.parametrize("field", ["keys", "values"])
def test_short_backing_does_not_count_as_full_coverage(monkeypatch, field):
    scheduler = _scheduler(monkeypatch)
    tokens = [1, 2, 3, 4]
    assert scheduler.memory_aware_cache.store(tokens, [_kv(tokens)])
    resident = scheduler.memory_aware_cache._entries[tuple(tokens)].cache[0]
    setattr(resident, field, getattr(resident, field)[..., :-2, :])
    request = _request(tokens)
    scheduler.add_request(request)

    scheduler._schedule_waiting()

    prompts, kwargs = scheduler.batch_generator.inserts[-1]
    assert prompts == [tokens]
    assert kwargs["caches"] is None
    assert request.cached_tokens == 0


def test_rewind_failure_keeps_cold_prefill(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    tokens = [1, 2, 3, 4]
    assert scheduler.memory_aware_cache.store(tokens, [_kv(tokens)])

    def fail_trim(*_args):
        raise ValueError("unsupported cache state")

    monkeypatch.setattr(
        "vllm_mlx.scheduler._trim_cache_offset", fail_trim, raising=False
    )
    request = _request(tokens)
    scheduler.add_request(request)

    scheduler._schedule_waiting()

    prompts, kwargs = scheduler.batch_generator.inserts[-1]
    assert prompts == [tokens]
    assert kwargs["caches"] is None
    assert request.cached_tokens == 0


def test_single_token_prompt_remains_one_token_prefill(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    tokens = [1]
    assert scheduler.memory_aware_cache.store(tokens, [_kv(tokens)])
    request = _request(tokens)
    scheduler.add_request(request)

    scheduler._schedule_waiting()

    assert scheduler.batch_generator.inserts[-1][0] == [tokens]
    assert request.cached_tokens == 0


@pytest.mark.parametrize("max_size", [4, 16], ids=["wrapped", "before-wrap"])
def test_rotating_full_hit_replays_only_before_wrap(monkeypatch, max_size):
    scheduler = _scheduler(monkeypatch, max_kv_size=max_size)
    tokens = [1, 2, 3, 4, 5, 6]
    layer = RotatingKVCache(max_size=max_size)
    for token in tokens:
        data = mx.full((1, 1, 1, 4), float(token))
        layer.update_and_fetch(data, data)
    mx.eval(*layer.state)
    assert scheduler.memory_aware_cache.store(tokens, [layer])
    resident = scheduler.memory_aware_cache._entries[tuple(tokens)].cache[0]
    original_keys = resident.state[0].tolist()
    request = _request(tokens)
    scheduler.add_request(request)

    scheduler._schedule_waiting()

    prompts, kwargs = scheduler.batch_generator.inserts[-1]
    if max_size > len(tokens):
        assert prompts == [[tokens[-1]]]
        restored = kwargs["caches"][0][0]
        assert restored.offset == len(tokens) - 1
        assert bool(
            mx.array_equal(
                restored.state[0], mx.array(original_keys)[..., : len(tokens) - 1, :]
            )
        )
        assert request.cached_tokens == len(tokens) - 1
    else:
        assert prompts == [tokens]
        assert kwargs["caches"][0][0].offset == 0
        assert kwargs["caches"][0][0].max_size == max_size
        assert request.cached_tokens == 0
    assert resident.offset == len(tokens)
    assert resident.state[0].tolist() == original_keys


@pytest.mark.parametrize("rotating", [False, True], ids=["flat", "rotating"])
@pytest.mark.parametrize("length", [4, 9])
def test_scheduled_replay_matches_cold_greedy_tokens(monkeypatch, length, rotating):
    model = _llama()
    tokens = list(range(1, length + 1))
    cold_cache = (
        [RotatingKVCache(max_size=16)] if rotating else make_prompt_cache(model)
    )
    cold_logits = model(mx.array([tokens]), cache=cold_cache)
    mx.eval(cold_logits)
    scheduler = _scheduler(monkeypatch, model, max_kv_size=16 if rotating else 0)
    assert scheduler.memory_aware_cache.store(tokens, cold_cache)
    request = _request(tokens)
    scheduler.add_request(request)

    scheduler._schedule_waiting()

    prompts, kwargs = scheduler.batch_generator.inserts[-1]
    assert prompts == [[tokens[-1]]]
    warm_cache = kwargs["caches"][0]
    warm_logits = model(mx.array(prompts), cache=warm_cache)
    for _ in range(4):
        assert bool(
            mx.allclose(cold_logits[0, -1], warm_logits[0, -1], rtol=1e-5, atol=1e-5)
        )
        cold_token = int(mx.argmax(cold_logits[0, -1]))
        warm_token = int(mx.argmax(warm_logits[0, -1]))
        assert warm_token == cold_token
        cold_logits = model(mx.array([[cold_token]]), cache=cold_cache)
        warm_logits = model(mx.array([[warm_token]]), cache=warm_cache)


@pytest.mark.parametrize("full_hit", [False, True], ids=["partial", "exact"])
def test_real_batch_processors_see_the_same_history_after_replay(monkeypatch, full_hit):
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    model = _llama()
    tokens = [1, 2, 3, 4]
    stored_tokens = tokens if full_hit else tokens[:2]
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([stored_tokens]), cache=cache))
    histories = []
    for warm in (False, True):
        scheduler = _scheduler(monkeypatch, model)
        if warm:
            assert scheduler.memory_aware_cache.store(stored_tokens, cache)
        seen = []

        def observe(context, logits):
            seen.append(context.tolist())
            return logits

        request = _request(tokens)
        request.sampling_params.max_tokens = 2
        request.sampling_params.repetition_penalty = 1.2
        request.sampling_params.logits_processors = [observe]
        scheduler.batch_generator = BatchGenerator(
            model=model, max_tokens=2, sampler=make_sampler(temp=0), stop_tokens=set()
        )
        try:
            scheduler.add_request(request)
            scheduler._schedule_waiting()
            for _ in range(6):
                scheduler.batch_generator.next()
                if len(seen) >= 2:
                    break
        finally:
            scheduler.batch_generator.close()
        assert len(seen) >= 2
        assert seen[0] == tokens
        histories.append(seen)
    assert histories[0] == histories[1]


def test_replay_insert_retry_clears_cached_history(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    tokens = [1, 2, 3, 4]
    assert scheduler.memory_aware_cache.store(tokens, [_kv(tokens)])
    batch = scheduler.batch_generator
    original_insert = batch.insert

    def reject_cache(prompts, **kwargs):
        uids = original_insert(prompts, **kwargs)
        if kwargs["caches"] is not None:
            raise ValueError("incompatible restored cache")
        return uids

    monkeypatch.setattr(batch, "insert", reject_cache)
    request = _request(tokens)
    scheduler.add_request(request)

    scheduler._schedule_waiting()

    assert batch.inserts[0][0] == [[tokens[-1]]]
    assert batch.inserts[0][1]["all_tokens"] == [tokens[:-1]]
    assert batch.inserts[1][0] == [tokens]
    assert batch.inserts[1][1]["all_tokens"] == [[]]
    assert batch.inserts[1][1]["caches"] is None
    assert request.cached_tokens == 0
