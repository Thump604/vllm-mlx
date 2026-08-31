# SPDX-License-Identifier: Apache-2.0
"""Pure owner-binding regressions at the cache publication boundary."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import pickle
import threading
from types import MappingProxyType, MethodType, SimpleNamespace

import pytest

import vllm_mlx.cache_owner_identity as owner_module
from vllm_mlx.cache_owner_identity import (
    CacheOwnerIdentity,
    PreparedOwnerBoundCacheEntry,
    VerifiedCacheOwnerContext,
)
from vllm_mlx.memory_cache import MemoryAwarePrefixCache

_RUNTIME_COMPOSITION = "c" * 64
_MODEL_CACHE_IDENTITY = "d" * 64


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _manifest(*, digest: str = "a" * 64):
    return _freeze(
        {
            "schema_version": "specprefill.identity.v1",
            "canonicalization": "rfc8785",
            "digest_algorithm": "sha256",
            "number_profile": "rfc8785-ieee754-safe-integer-v1",
            "digest": digest,
            "role": "target",
            "model_cache_identity": {"model_id": "model-a"},
            "request_protocol_identity": {"version": "1"},
            "draft_compatibility": {"relation": "test"},
        }
    )


def _authority(
    *,
    owner: str = "owner-a",
    namespace: str = "cache-a",
    digest: str = "a" * 64,
) -> CacheOwnerIdentity:
    return CacheOwnerIdentity(
        context=_context(owner=owner, namespace=namespace, digest=digest)
    )


def _context(
    *,
    owner: str = "owner-a",
    namespace: str = "cache-a",
    digest: str = "a" * 64,
    complete_identity: bool = True,
) -> VerifiedCacheOwnerContext:
    return owner_module._issue_verified_cache_owner_context(
        manifest=_manifest(digest=digest),
        model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
        persistence_identity=_freeze(
            {
                "model": "model-fingerprint",
                "tokenizer": "tokenizer-fingerprint" if complete_identity else "",
                "cache_layout": "layout-fingerprint",
            }
        ),
        runtime_composition_digest=_RUNTIME_COMPOSITION,
        cache_namespace=namespace,
        registry_source=f"test:{owner}",
    )


def _cache(
    *,
    complete_identity: bool = True,
) -> MemoryAwarePrefixCache:
    cache = object.__new__(MemoryAwarePrefixCache)
    cache._owner_identity = None
    cache._cache_owner_context = _context(complete_identity=complete_identity)
    cache._memory_lock = threading.RLock()
    cache._owner_prepared_entries = {}
    cache._persistence_identity = {
        "model": "model-fingerprint" if complete_identity else "",
        "tokenizer": "tokenizer-fingerprint",
        "cache_layout": "layout-fingerprint",
    }
    return cache


def test_owner_binding_rejects_same_manifest_on_another_owner():
    owner_a = _authority(owner="same-owner-provenance")
    owner_b = _authority(owner="same-owner-provenance")
    binding_a = owner_a.mint_owner_binding()
    binding_b = owner_b.mint_owner_binding()

    assert binding_a != binding_b
    assert owner_a.validate_owner_binding(binding_a).accepted
    assert owner_b.validate_owner_binding(binding_a).reason == "cache_unsafe"


def test_copied_digest_and_public_fields_do_not_transfer_authority():
    owner = _authority()
    binding = owner.mint_owner_binding()
    copied = replace(binding, _capability=object())

    assert copied.manifest_digest == binding.manifest_digest
    assert copied.owner_provenance_digest == binding.owner_provenance_digest
    assert owner.validate_owner_binding(copied).reason == "cache_unsafe"
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(binding)


def test_reload_requires_new_process_local_binding_for_compatible_identity():
    previous_owner = _authority()
    restored_owner = _authority()
    previous = previous_owner.mint_owner_binding()

    assert restored_owner.validate_owner_binding(previous).reason == "cache_unsafe"
    restored = restored_owner.mint_owner_binding()
    assert restored_owner.validate_owner_binding(restored).accepted
    assert restored.manifest_digest == previous.manifest_digest
    assert restored.owner_provenance_digest == previous.owner_provenance_digest


def test_namespace_and_lifecycle_invalidation_revoke_old_handles():
    owner = _authority()
    old = owner.mint_owner_binding()

    owner.invalidate(cache_namespace="cache-b")

    assert owner.validate_owner_binding(old).reason == "cache_unsafe"
    current = owner.mint_owner_binding()
    assert current.cache_namespace == "cache-b"
    assert current.lifecycle_epoch == old.lifecycle_epoch + 1


def test_request_cancellation_and_release_are_fail_closed():
    owner = _authority()
    cancelled = owner.mint_request_binding(sequence_revision=7)
    released = owner.mint_request_binding(sequence_revision=8)
    forged_revision = replace(released, sequence_revision=9)

    assert owner.validate_request_binding(forged_revision).reason == "cache_unsafe"
    assert owner.cancel_request(cancelled).reason == "cancellation"
    assert owner.validate_request_binding(cancelled).reason == "cancellation"
    assert owner.release_request(cancelled).accepted
    assert owner.validate_request_binding(cancelled).reason == "cache_unsafe"
    assert owner.release_request(released).accepted
    assert owner.validate_request_binding(released).reason == "cache_unsafe"
    assert owner._request_states == {}


def test_cache_owner_accepts_only_its_verified_context():
    with pytest.raises(TypeError, match="issued by verification"):
        VerifiedCacheOwnerContext()
    with pytest.raises(TypeError):
        replace(_context(), cache_namespace="forged")
    with pytest.raises(TypeError, match="immutable"):
        _context().cache_namespace = "forged"
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(_context())

    cache = _cache()
    cache.bind_owner_context(cache._cache_owner_context)
    with pytest.raises(ValueError, match="does not belong"):
        cache.bind_owner_context(_context(owner="other"))


def test_cache_callsite_rejects_wrong_owner_before_preparing_entry():
    cache_a = _cache()
    cache_b = _cache()
    cache_a.bind_owner_context(cache_a._cache_owner_context)
    cache_b.bind_owner_context(cache_b._cache_owner_context)
    request_a = cache_a.mint_owner_request(sequence_revision=1)
    prepared_calls: list[object] = []

    def prepare_store(self, *_args, **_kwargs):
        prepared_calls.append(self)
        return SimpleNamespace(tokens=(1, 2), memory_bytes=16)

    cache_b._prepare_store_unchecked = MethodType(prepare_store, cache_b)
    decision, prepared = cache_b.prepare_owner_bound_store(
        request_a,
        [1, 2],
        [object()],
    )

    assert decision.reason == "cache_unsafe"
    assert prepared is None
    assert prepared_calls == []


def test_cache_callsite_rechecks_cancel_and_release_before_publication():
    cache = _cache()
    cache.bind_owner_context(cache._cache_owner_context)
    committed: list[object] = []

    def prepare_store(self, tokens, *_args, **_kwargs):
        return SimpleNamespace(tokens=tuple(tokens), memory_bytes=16)

    def commit_prepared(self, entry, *, commit_guard, **_kwargs):
        if not commit_guard():
            return False
        committed.append(entry)
        return True

    cache._prepare_store_unchecked = MethodType(prepare_store, cache)
    cache._commit_prepared_unchecked = MethodType(commit_prepared, cache)

    cancelled = cache.mint_owner_request(sequence_revision=1)
    decision, prepared = cache.prepare_owner_bound_store(cancelled, [1, 2], [object()])
    assert decision.accepted and prepared is not None
    assert cache.cancel_owner_request(cancelled).reason == "cancellation"
    assert cache.commit_owner_bound_store(prepared).reason == "cancellation"

    released = cache.mint_owner_request(sequence_revision=2)
    decision, prepared = cache.prepare_owner_bound_store(released, [3, 4], [object()])
    assert decision.accepted and prepared is not None
    assert cache.release_owner_request(released).accepted
    assert cache.commit_owner_bound_store(prepared).reason == "cache_unsafe"
    assert committed == []


def test_prepared_store_rechecks_revocation_before_handle_registration():
    cache = _cache()
    cache.bind_owner_context(cache._cache_owner_context)
    request = cache.mint_owner_request(sequence_revision=1)

    def prepare_store(self, tokens, *_args, **_kwargs):
        self.cancel_owner_request(request)
        return SimpleNamespace(tokens=tuple(tokens), cache=[object()], memory_bytes=16)

    class RecordingPreparedEntries(dict):
        registrations = 0

        def __setitem__(self, key, value):
            self.registrations += 1
            super().__setitem__(key, value)

    prepared_entries = RecordingPreparedEntries()
    cache._owner_prepared_entries = prepared_entries
    cache._prepare_store_unchecked = MethodType(prepare_store, cache)
    decision, prepared = cache.prepare_owner_bound_store(request, [1, 2], [object()])

    assert decision.reason == "cancellation"
    assert prepared is None
    assert cache._owner_prepared_entries == {}
    assert prepared_entries.registrations == 0


def test_prepared_store_removes_handle_when_revoked_during_registration():
    cache = _cache()
    cache.bind_owner_context(cache._cache_owner_context)
    request = cache.mint_owner_request(sequence_revision=1)

    class RevokingPreparedEntries(dict):
        def __setitem__(self, handle_id, value):
            super().__setitem__(handle_id, value)
            cache.cancel_owner_request(request)

    cache._owner_prepared_entries = RevokingPreparedEntries()
    cache._prepare_store_unchecked = MethodType(
        lambda self, tokens, *_args, **_kwargs: SimpleNamespace(
            tokens=tuple(tokens), cache=[object()], memory_bytes=16
        ),
        cache,
    )

    decision, prepared = cache.prepare_owner_bound_store(request, [1, 2], [object()])

    assert decision.reason == "cancellation"
    assert prepared is None
    assert cache._owner_prepared_entries == {}


def test_cache_callsite_publishes_only_current_owner_request():
    cache = _cache()
    binding = cache.bind_owner_context(cache._cache_owner_context)
    request = cache.mint_owner_request(sequence_revision=1)
    committed: list[object] = []

    def prepare_store(self, tokens, *_args, **_kwargs):
        return SimpleNamespace(tokens=tuple(tokens), memory_bytes=16)

    def commit_prepared(self, entry, *, commit_guard, **_kwargs):
        assert commit_guard()
        committed.append(entry)
        return True

    cache._prepare_store_unchecked = MethodType(prepare_store, cache)
    cache._commit_prepared_unchecked = MethodType(commit_prepared, cache)
    decision, prepared = cache.prepare_owner_bound_store(request, [1, 2], [object()])

    assert binding.owner_provenance_digest
    assert decision.accepted and prepared is not None
    assert cache.commit_owner_bound_store(prepared).accepted
    assert len(committed) == 1


def test_bound_cache_rejects_raw_fetch_and_publication_apis():
    cache = _cache()
    cache._memory_lock = threading.RLock()
    cache._copy_lock = threading.Lock()
    cache._entries = OrderedDict()
    cache._sorted_keys = []
    cache._stats = SimpleNamespace(misses=0)
    cache._config = SimpleNamespace(min_prefix_tokens=1)
    cache.bind_owner_context(cache._cache_owner_context)

    with pytest.raises(RuntimeError, match="request lease"):
        cache.fetch([1])
    with pytest.raises(RuntimeError, match="request lease"):
        cache.fetch_exact_auxiliary([1])
    with pytest.raises(RuntimeError, match="request lease"):
        cache.prepare_store([1], [object()])
    with pytest.raises(RuntimeError, match="request lease"):
        cache.store([1], [object()])


def test_prepared_owner_wrapper_never_exposes_private_cache_entry():
    cache = _cache()
    cache.bind_owner_context(cache._cache_owner_context)
    request = cache.mint_owner_request(sequence_revision=1)
    cache._prepare_store_unchecked = MethodType(
        lambda self, tokens, *_args, **_kwargs: SimpleNamespace(
            tokens=tuple(tokens), memory_bytes=16, cache=["state"]
        ),
        cache,
    )
    decision, prepared = cache.prepare_owner_bound_store(request, [1], ["state"])

    assert decision.accepted and prepared is not None
    assert not hasattr(prepared, "cache")
    assert not hasattr(prepared, "_entry")
    cloned_decision, cloned = cache.clone_prepared_owner_bound_cache(
        prepared, lambda value: list(value)
    )
    assert cloned_decision.accepted
    assert cloned == ["state"]
    copied = replace(prepared)
    assert cache.commit_owner_bound_store(copied).reason == "runtime_error"


@pytest.mark.parametrize(
    "lifecycle", ["cancel", "release", "invalidate", "clear", "close"]
)
def test_reentrant_lifecycle_guard_cannot_publish(lifecycle):
    cache = _cache()
    cache._memory_lock = threading.RLock()
    cache._entries = {}
    cache._sorted_keys = []
    cache._current_memory = 0
    cache._max_memory = 1024
    cache.bind_owner_context(cache._cache_owner_context)
    request = cache.mint_owner_request(sequence_revision=1)
    committed: list[object] = []

    def prepare_store(self, tokens, *_args, **_kwargs):
        return SimpleNamespace(tokens=tuple(tokens), memory_bytes=16)

    def commit_prepared(self, entry, *, commit_guard, **_kwargs):
        if not commit_guard():
            return False
        committed.append(entry)
        return True

    cache._prepare_store_unchecked = MethodType(prepare_store, cache)
    cache._commit_prepared_unchecked = MethodType(commit_prepared, cache)
    decision, prepared = cache.prepare_owner_bound_store(request, [1, 2], [object()])
    assert decision.accepted and prepared is not None

    def revoke_during_guard():
        if lifecycle == "cancel":
            cache.cancel_owner_request(request)
        elif lifecycle == "release":
            cache.release_owner_request(request)
        elif lifecycle == "invalidate":
            cache.invalidate_owner_identity()
        elif lifecycle == "clear":
            cache.clear()
        else:
            cache.close_owner_identity()
        return True

    result = cache.commit_owner_bound_store(prepared, commit_guard=revoke_during_guard)

    assert not result.accepted
    assert committed == []


def test_prepared_entry_version_is_fail_closed():
    cache = _cache()
    cache.bind_owner_context(cache._cache_owner_context)
    request = cache.mint_owner_request(sequence_revision=1)
    prepared = SimpleNamespace(tokens=(1, 2), memory_bytes=16)
    wrong_version = PreparedOwnerBoundCacheEntry(
        owner=request.owner,
        request=request,
        handle_id="forged",
        tokens=(1, 2),
        memory_bytes=16,
        version="vllm-mlx.owner-bound-store.v0",
    )

    assert cache.commit_owner_bound_store(wrong_version).reason == "runtime_error"


def test_forged_prepared_handle_is_fail_closed():
    cache = _cache()
    cache.bind_owner_context(cache._cache_owner_context)
    request = cache.mint_owner_request(sequence_revision=1)
    forged = PreparedOwnerBoundCacheEntry(
        owner=request.owner,
        request=request,
        handle_id="forged",
        tokens=(1, 2),
        memory_bytes=16,
    )

    assert cache.commit_owner_bound_store(forged).reason == "runtime_error"
    decision, cloned = cache.clone_prepared_owner_bound_cache(
        forged, lambda value: value
    )
    assert decision.reason == "runtime_error"
    assert cloned is None


def test_cache_replay_rechecks_request_after_clone():
    cache = _cache()
    cache.bind_owner_context(cache._cache_owner_context)
    request = cache.mint_owner_request(sequence_revision=1)

    def cancel_during_clone(value):
        cache.cancel_owner_request(request)
        return list(value)

    cache._clone_for_replay_unchecked = cancel_during_clone
    decision, cloned = cache.clone_owner_bound_for_replay(request, ["state"])

    assert decision.reason == "cancellation"
    assert cloned is None


def test_reentrant_ssd_spill_revocation_blocks_final_publication():
    cache = _cache()
    cache._memory_lock = threading.RLock()
    cache._entries = OrderedDict(
        {(0,): SimpleNamespace(tokens=(0,), cache=[object()], memory_bytes=16)}
    )
    cache._sorted_keys = [(0,)]
    cache._current_memory = 16
    cache._max_memory = 20
    cache._config = SimpleNamespace(max_entries=10)
    cache._stats = SimpleNamespace(
        store_rejections=0,
        evictions=0,
        entry_count=1,
        current_memory_bytes=16,
    )
    cache.bind_owner_context(cache._cache_owner_context)
    request = cache.mint_owner_request(sequence_revision=1)

    class RevokingTier:
        def enqueue_spill(self, *_args):
            cache.cancel_owner_request(request)

    cache._ssd_tier = RevokingTier()
    cache._prepare_store_unchecked = MethodType(
        lambda self, tokens, *_args, **_kwargs: SimpleNamespace(
            tokens=tuple(tokens), cache=[object()], memory_bytes=16
        ),
        cache,
    )
    decision, prepared = cache.prepare_owner_bound_store(request, [1], [object()])
    assert decision.accepted and prepared is not None

    result = cache.commit_owner_bound_store(prepared)

    assert result.reason == "cancellation"
    assert (1,) not in cache._entries
    assert cache.validate_owner_request(request).reason == "cancellation"


def test_in_place_restore_invalidates_live_owner_handles(tmp_path):
    cache = _cache()
    cache.bind_owner_context(cache._cache_owner_context)
    request = cache.mint_owner_request(sequence_revision=1)

    assert cache.restore_hybrid_persistence_snapshot(None) == 0
    assert cache.validate_owner_request(request).reason == "cache_unsafe"

    rebound = cache.mint_owner_request(sequence_revision=2)
    with pytest.raises(RuntimeError, match="strict hybrid persistence"):
        cache.load_from_disk(str(tmp_path))
    assert cache.validate_owner_request(rebound).accepted


@pytest.mark.parametrize("method", ["save_to_disk", "load_from_disk"])
def test_owner_bound_generic_disk_persistence_rejects_before_io(method, tmp_path):
    cache = _cache()

    class ExplodingEntries(dict):
        def __bool__(self):
            raise AssertionError("owner-bound generic persistence touched entries")

        def items(self):
            raise AssertionError("owner-bound generic persistence touched entries")

    cache._entries = ExplodingEntries()
    cache.invalidate_owner_identity = MethodType(
        lambda self: (_ for _ in ()).throw(
            AssertionError("owner-bound generic persistence invalidated owner")
        ),
        cache,
    )

    with pytest.raises(RuntimeError, match="strict hybrid persistence"):
        getattr(cache, method)(str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_cache_binding_requires_complete_provenance_and_closes_permanently():
    with pytest.raises(ValueError, match="complete model/tokenizer/cache"):
        cache = _cache(complete_identity=False)
        cache.bind_owner_context(cache._cache_owner_context)

    cache = _cache()
    cache.bind_owner_context(cache._cache_owner_context)
    cache.close_owner_identity()
    with pytest.raises(RuntimeError, match="closed"):
        cache.mint_owner_request(sequence_revision=1)
