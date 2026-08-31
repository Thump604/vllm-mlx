# SPDX-License-Identifier: Apache-2.0
"""Pure owner-binding regressions at the cache publication boundary."""

from __future__ import annotations

from dataclasses import replace
import pickle
import threading
from types import MappingProxyType, MethodType, SimpleNamespace

import pytest

from vllm_mlx.cache_owner_identity import (
    CacheOwnerIdentity,
    PreparedOwnerBoundCacheEntry,
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
        cache_namespace=namespace,
        actual_provenance={
            "model": "model-fingerprint",
            "tokenizer": "tokenizer-fingerprint",
            "cache_layout": "layout-fingerprint",
            "loaded_owner": owner,
        },
        governed_manifest=_manifest(digest=digest),
        actual_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
        governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
        runtime_composition_digest=_RUNTIME_COMPOSITION,
    )


def _cache(
    *,
    complete_identity: bool = True,
    model_cache_identity_digest: str = _MODEL_CACHE_IDENTITY,
) -> MemoryAwarePrefixCache:
    cache = object.__new__(MemoryAwarePrefixCache)
    cache._owner_identity = None
    cache._loaded_model_cache_identity_digest = model_cache_identity_digest
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


def test_governed_manifest_must_be_frozen_and_match_loaded_cache_identity():
    with pytest.raises(ValueError, match="validated and frozen"):
        _cache().bind_owner_identity(
            dict(_manifest()),
            cache_namespace="cache-a",
            governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
            runtime_composition_digest=_RUNTIME_COMPOSITION,
        )
    with pytest.raises(ValueError, match="does not match the loaded"):
        _cache(model_cache_identity_digest="e" * 64).bind_owner_identity(
            _manifest(),
            cache_namespace="cache-a",
            governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
            runtime_composition_digest=_RUNTIME_COMPOSITION,
        )

    cache = _cache()
    cache.bind_owner_identity(
        _manifest(),
        cache_namespace="cache-a",
        governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
        runtime_composition_digest=_RUNTIME_COMPOSITION,
    )
    with pytest.raises(ValueError, match="does not match bound owner"):
        cache.bind_owner_identity(
            dict(_manifest()),
            cache_namespace="cache-a",
            governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
            runtime_composition_digest=_RUNTIME_COMPOSITION,
        )


def test_cache_callsite_rejects_wrong_owner_before_preparing_entry():
    cache_a = _cache()
    cache_b = _cache()
    cache_a.bind_owner_identity(
        _manifest(),
        cache_namespace="cache-a",
        governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
        runtime_composition_digest=_RUNTIME_COMPOSITION,
    )
    cache_b.bind_owner_identity(
        _manifest(),
        cache_namespace="cache-a",
        governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
        runtime_composition_digest=_RUNTIME_COMPOSITION,
    )
    request_a = cache_a.mint_owner_request(sequence_revision=1)
    prepared_calls: list[object] = []

    def prepare_store(self, *_args, **_kwargs):
        prepared_calls.append(self)
        return SimpleNamespace(tokens=(1, 2), memory_bytes=16)

    cache_b.prepare_store = MethodType(prepare_store, cache_b)
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
    cache.bind_owner_identity(
        _manifest(),
        cache_namespace="cache-a",
        governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
        runtime_composition_digest=_RUNTIME_COMPOSITION,
    )
    committed: list[object] = []

    def prepare_store(self, tokens, *_args, **_kwargs):
        return SimpleNamespace(tokens=tuple(tokens), memory_bytes=16)

    def commit_prepared(self, entry, *, commit_guard, **_kwargs):
        if not commit_guard():
            return False
        committed.append(entry)
        return True

    cache.prepare_store = MethodType(prepare_store, cache)
    cache.commit_prepared = MethodType(commit_prepared, cache)

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


def test_cache_callsite_publishes_only_current_owner_request():
    cache = _cache()
    binding = cache.bind_owner_identity(
        _manifest(),
        cache_namespace="cache-a",
        governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
        runtime_composition_digest=_RUNTIME_COMPOSITION,
    )
    request = cache.mint_owner_request(sequence_revision=1)
    committed: list[object] = []

    def prepare_store(self, tokens, *_args, **_kwargs):
        return SimpleNamespace(tokens=tuple(tokens), memory_bytes=16)

    def commit_prepared(self, entry, *, commit_guard, **_kwargs):
        assert commit_guard()
        committed.append(entry)
        return True

    cache.prepare_store = MethodType(prepare_store, cache)
    cache.commit_prepared = MethodType(commit_prepared, cache)
    decision, prepared = cache.prepare_owner_bound_store(request, [1, 2], [object()])

    assert binding.owner_provenance_digest
    assert decision.accepted and prepared is not None
    assert cache.commit_owner_bound_store(prepared).accepted
    assert len(committed) == 1


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
    cache.bind_owner_identity(
        _manifest(),
        cache_namespace="cache-a",
        governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
        runtime_composition_digest=_RUNTIME_COMPOSITION,
    )
    request = cache.mint_owner_request(sequence_revision=1)
    committed: list[object] = []

    def prepare_store(self, tokens, *_args, **_kwargs):
        return SimpleNamespace(tokens=tuple(tokens), memory_bytes=16)

    def commit_prepared(self, entry, *, commit_guard, **_kwargs):
        if not commit_guard():
            return False
        committed.append(entry)
        return True

    cache.prepare_store = MethodType(prepare_store, cache)
    cache.commit_prepared = MethodType(commit_prepared, cache)
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
    cache.bind_owner_identity(
        _manifest(),
        cache_namespace="cache-a",
        governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
        runtime_composition_digest=_RUNTIME_COMPOSITION,
    )
    request = cache.mint_owner_request(sequence_revision=1)
    prepared = SimpleNamespace(tokens=(1, 2), memory_bytes=16)
    wrong_version = PreparedOwnerBoundCacheEntry(
        owner=request.owner,
        request=request,
        version="vllm-mlx.owner-bound-store.v0",
        _entry=prepared,
    )

    assert cache.commit_owner_bound_store(wrong_version).reason == "runtime_error"


def test_cache_binding_requires_complete_provenance_and_closes_permanently():
    with pytest.raises(ValueError, match="complete model/tokenizer/cache"):
        _cache(complete_identity=False).bind_owner_identity(
            _manifest(),
            cache_namespace="cache-a",
            governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
            runtime_composition_digest=_RUNTIME_COMPOSITION,
        )

    cache = _cache()
    cache.bind_owner_identity(
        _manifest(),
        cache_namespace="cache-a",
        governed_model_cache_identity_digest=_MODEL_CACHE_IDENTITY,
        runtime_composition_digest=_RUNTIME_COMPOSITION,
    )
    cache.close_owner_identity()
    with pytest.raises(RuntimeError, match="closed"):
        cache.mint_owner_request(sequence_revision=1)
