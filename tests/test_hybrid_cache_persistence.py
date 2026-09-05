# SPDX-License-Identifier: Apache-2.0
"""Pure-I/O regressions for exact hybrid prefix-cache restart snapshots."""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pytest

from vllm_mlx.cache_persistence import (
    HYBRID_CACHE_PERSIST_VERSION,
    HybridCachePersistenceError,
    HybridCacheSnapshot,
    HybridEntrySnapshot,
    HybridLayerSnapshot,
    read_hybrid_snapshot,
    write_hybrid_snapshot,
)


def _snapshot(*, model="model-a", tokenizer="tokenizer-a"):
    return HybridCacheSnapshot(
        identity={
            "model": model,
            "tokenizer": tokenizer,
            "cache_layout": "arrays-kv-layout",
        },
        entries=(
            HybridEntrySnapshot(
                tokens=(1, 2, 3, 4),
                memory_bytes=358,
                layers=(
                    HybridLayerSnapshot(
                        "ArraysCache",
                        {
                            "state_0": np.arange(8, dtype=np.float32).reshape(2, 4),
                            "state_1": np.ones((1, 3), dtype=np.float16),
                        },
                        {
                            "num_arrays": 2,
                            "state_original_dtypes": [None, None],
                            "metadata_arrays": [],
                            "metadata_original_dtypes": {},
                            "meta_state": "",
                        },
                    ),
                    HybridLayerSnapshot(
                        "KVCache",
                        {
                            "keys": np.ones((1, 2, 4, 8), dtype=np.float16),
                            "values": np.full((1, 2, 4, 8), 2, dtype=np.float16),
                        },
                        {"offset": 4},
                    ),
                ),
                auxiliary={
                    "last_logits": np.arange(16, dtype=np.float32).reshape(1, 16)
                },
                auxiliary_original_dtypes={"last_logits": None},
            ),
        ),
    )


def _qsa_snapshot():
    return HybridCacheSnapshot(
        identity={
            "model": "qwen4-exp-model",
            "tokenizer": "qwen4-exp-tokenizer",
            "cache_layout": "qwen4-exp-qsa-layout",
        },
        entries=(
            HybridEntrySnapshot(
                tokens=(1, 2, 3, 4),
                memory_bytes=272,
                layers=(
                    HybridLayerSnapshot(
                        "QSAKVCache",
                        {
                            "state_0": np.arange(16, dtype=np.float32).reshape(
                                1, 1, 4, 4
                            ),
                            "state_1": np.ones((1, 1, 4, 4), dtype=np.float32),
                            "state_2": np.arange(16, dtype=np.float32).reshape(1, 4, 4),
                            "state_3": np.arange(4, dtype=np.int32).reshape(1, 4),
                        },
                        {
                            "num_arrays": 4,
                            "state_original_dtypes": [None, None, None, None],
                            "state_container": "tuple",
                        },
                    ),
                ),
                auxiliary={
                    "last_logits": np.arange(16, dtype=np.float32).reshape(1, 16)
                },
                auxiliary_original_dtypes={"last_logits": None},
            ),
        ),
    )


def _qwen4_ple_snapshot():
    snapshot = _qsa_snapshot()
    entry = snapshot.entries[0]
    ple = HybridLayerSnapshot(
        "ArraysCache",
        {
            "state_0": np.zeros((1, 3, 8), dtype=np.float32),
            "state_1": np.zeros((1, 2, 4, 4), dtype=np.float32),
            "state_2": np.zeros((1, 9, 32), dtype=np.float32),
            "state_3": np.zeros((1, 2), dtype=np.int64),
        },
        {
            "num_arrays": 4,
            "state_original_dtypes": [None, None, None, None],
            "metadata_arrays": [],
            "metadata_original_dtypes": {},
            "meta_state": "",
        },
    )
    memory_bytes = sum(value.nbytes for value in ple.tensors.values())
    memory_bytes += sum(
        value.nbytes for layer in entry.layers for value in layer.tensors.values()
    )
    memory_bytes += sum(value.nbytes for value in entry.auxiliary.values())
    return HybridCacheSnapshot(
        snapshot.identity,
        (
            HybridEntrySnapshot(
                entry.tokens,
                memory_bytes,
                (ple, *entry.layers),
                entry.auxiliary,
                entry.auxiliary_original_dtypes,
            ),
        ),
    )


def _generation_dir(tmp_path):
    pointer = json.loads((tmp_path / "index.json").read_text())
    return tmp_path / pointer["generation"]


def _rewrite_manifest(tmp_path, mutate):
    import hashlib

    manifest_path = _generation_dir(tmp_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    pointer_path = tmp_path / "index.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer_path.write_text(json.dumps(pointer, sort_keys=True, separators=(",", ":")))


def test_hybrid_snapshot_round_trip_preserves_layers_tokens_and_logits(tmp_path):
    source = _snapshot()
    assert write_hybrid_snapshot(str(tmp_path), source)

    loaded = read_hybrid_snapshot(str(tmp_path))

    assert loaded is not None
    assert loaded.identity == source.identity
    assert loaded.entries[0].tokens == source.entries[0].tokens
    assert [layer.layer_type for layer in loaded.entries[0].layers] == [
        "ArraysCache",
        "KVCache",
    ]
    np.testing.assert_array_equal(
        loaded.entries[0].auxiliary["last_logits"],
        source.entries[0].auxiliary["last_logits"],
    )


def test_qsa_snapshot_uses_v2_and_round_trips(tmp_path):
    assert HYBRID_CACHE_PERSIST_VERSION == 2
    source = _qsa_snapshot()
    assert write_hybrid_snapshot(str(tmp_path), source)

    pointer = json.loads((tmp_path / "index.json").read_text())
    manifest = json.loads((_generation_dir(tmp_path) / "manifest.json").read_text())
    loaded = read_hybrid_snapshot(str(tmp_path))

    assert pointer["version"] == 2
    assert manifest["version"] == 2
    assert loaded.entries[0].layers[0].layer_type == "QSAKVCache"
    assert loaded.entries[0].layers[0].metadata["num_arrays"] == 4


def test_v2_qwen4_ple_arrays_cache_preserves_integer_token_history(tmp_path):
    source = _qwen4_ple_snapshot()
    assert write_hybrid_snapshot(str(tmp_path), source)
    loaded = read_hybrid_snapshot(str(tmp_path))
    ple = loaded.entries[0].layers[0]
    assert ple.layer_type == "ArraysCache"
    assert ple.metadata["num_arrays"] == 4
    assert ple.tensors["state_3"].dtype == np.int64
    np.testing.assert_array_equal(ple.tensors["state_3"], [[0, 0]])


def test_v1_arrays_and_kv_generation_remains_readable(tmp_path):
    assert write_hybrid_snapshot(str(tmp_path), _snapshot())
    _rewrite_manifest(tmp_path, lambda manifest: manifest.update(version=1))
    pointer_path = tmp_path / "index.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["version"] = 1
    pointer_path.write_text(json.dumps(pointer, sort_keys=True, separators=(",", ":")))

    loaded = read_hybrid_snapshot(str(tmp_path))

    assert [layer.layer_type for layer in loaded.entries[0].layers] == [
        "ArraysCache",
        "KVCache",
    ]


def test_v1_generation_rejects_qsa_layer(tmp_path):
    assert write_hybrid_snapshot(str(tmp_path), _qsa_snapshot())
    _rewrite_manifest(tmp_path, lambda manifest: manifest.update(version=1))
    pointer_path = tmp_path / "index.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["version"] = 1
    pointer_path.write_text(json.dumps(pointer, sort_keys=True, separators=(",", ":")))

    with pytest.raises(HybridCachePersistenceError, match="unsupported cache layer"):
        read_hybrid_snapshot(str(tmp_path))


def test_unknown_generation_version_fails_closed(tmp_path):
    assert write_hybrid_snapshot(str(tmp_path), _snapshot())
    pointer_path = tmp_path / "index.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["version"] = 99
    pointer_path.write_text(json.dumps(pointer, sort_keys=True, separators=(",", ":")))

    with pytest.raises(HybridCachePersistenceError, match="unsupported hybrid"):
        read_hybrid_snapshot(str(tmp_path))


@pytest.mark.parametrize("version", [True, 1.0])
def test_non_integer_generation_version_fails_closed(tmp_path, version):
    assert write_hybrid_snapshot(str(tmp_path), _snapshot())
    _rewrite_manifest(tmp_path, lambda manifest: manifest.update(version=version))
    pointer_path = tmp_path / "index.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["version"] = version
    pointer_path.write_text(json.dumps(pointer, sort_keys=True, separators=(",", ":")))

    with pytest.raises(HybridCachePersistenceError, match="unsupported hybrid"):
        read_hybrid_snapshot(str(tmp_path))


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_arrays", 3),
        ("state_container", "list"),
        ("state_original_dtypes", [None, None, None, "float64"]),
    ],
)
def test_qsa_generation_rejects_invalid_state_metadata(tmp_path, field, value):
    assert write_hybrid_snapshot(str(tmp_path), _qsa_snapshot())

    def mutate(manifest):
        manifest["entries"][0]["layers"][0]["metadata"][field] = value

    _rewrite_manifest(tmp_path, mutate)

    with pytest.raises(HybridCachePersistenceError, match="QSAKVCache"):
        read_hybrid_snapshot(str(tmp_path))


@pytest.mark.parametrize(
    "relative_path",
    [
        "entry-0/tokens.bin",
        "entry-0/layer-0.safetensors",
        "entry-0/auxiliary.safetensors",
    ],
)
def test_hybrid_snapshot_rejects_same_length_tampering(tmp_path, relative_path):
    write_hybrid_snapshot(str(tmp_path), _snapshot())
    path = _generation_dir(tmp_path) / relative_path
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    path.write_bytes(payload)

    with pytest.raises(HybridCachePersistenceError, match="checksum mismatch"):
        read_hybrid_snapshot(str(tmp_path))


def test_hybrid_snapshot_rejects_malformed_or_duplicate_json(tmp_path):
    write_hybrid_snapshot(str(tmp_path), _snapshot())
    (tmp_path / "index.json").write_text(
        '{"version":1,"version":1,"generation":"x","manifest_sha256":"y"}'
    )

    with pytest.raises(HybridCachePersistenceError, match="duplicate JSON key"):
        read_hybrid_snapshot(str(tmp_path))


def test_hybrid_snapshot_requires_last_logits(tmp_path):
    source = _snapshot()
    invalid = HybridCacheSnapshot(
        identity=source.identity,
        entries=(
            HybridEntrySnapshot(
                tokens=source.entries[0].tokens,
                memory_bytes=source.entries[0].memory_bytes,
                layers=source.entries[0].layers,
                auxiliary={},
                auxiliary_original_dtypes={},
            ),
        ),
    )

    with pytest.raises(HybridCachePersistenceError, match="last_logits"):
        write_hybrid_snapshot(str(tmp_path), invalid)


def test_hybrid_snapshot_rejects_wrong_logits_shape(tmp_path):
    source = _snapshot()
    invalid = HybridCacheSnapshot(
        identity=source.identity,
        entries=(
            HybridEntrySnapshot(
                tokens=source.entries[0].tokens,
                memory_bytes=source.entries[0].memory_bytes - 48,
                layers=source.entries[0].layers,
                auxiliary={"last_logits": np.arange(4, dtype=np.float32)},
                auxiliary_original_dtypes={"last_logits": None},
            ),
        ),
    )

    with pytest.raises(HybridCachePersistenceError, match="last_logits shape"):
        write_hybrid_snapshot(str(tmp_path), invalid)


def test_interrupted_publish_keeps_previous_generation_loadable(tmp_path, monkeypatch):
    first = _snapshot(model="model-first")
    write_hybrid_snapshot(str(tmp_path), first)
    original_replace = os.replace

    def interrupt_pointer_publish(source, destination):
        if str(destination).endswith("index.json"):
            raise OSError("simulated interruption")
        return original_replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupt_pointer_publish)
    with pytest.raises(OSError, match="simulated interruption"):
        write_hybrid_snapshot(str(tmp_path), _snapshot(model="model-second"))

    loaded = read_hybrid_snapshot(str(tmp_path))
    assert loaded is not None
    assert loaded.identity["model"] == "model-first"


def test_hybrid_snapshot_rejects_parent_generation_component(tmp_path):
    write_hybrid_snapshot(str(tmp_path), _snapshot())
    pointer = json.loads((tmp_path / "index.json").read_text())
    pointer["generation"] = ".."
    (tmp_path / "index.json").write_text(json.dumps(pointer))

    with pytest.raises(HybridCachePersistenceError, match="invalid generation"):
        read_hybrid_snapshot(str(tmp_path))


def test_hybrid_snapshot_rejects_symlinked_generation(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "manifest.json").write_text("{}")
    pointer = {
        "version": 1,
        "generation": "generation-link",
        "manifest_sha256": "0" * 64,
    }
    (tmp_path / "generation-link").symlink_to(outside, target_is_directory=True)
    (tmp_path / "index.json").write_text(json.dumps(pointer))

    with pytest.raises(
        HybridCachePersistenceError, match="missing or invalid generation"
    ):
        read_hybrid_snapshot(str(tmp_path))


def test_hybrid_snapshot_rejects_truncated_tokens_even_with_updated_checksum(tmp_path):
    import hashlib

    write_hybrid_snapshot(str(tmp_path), _snapshot())
    generation = _generation_dir(tmp_path)
    tokens = generation / "entry-0" / "tokens.bin"
    tokens.write_bytes(tokens.read_bytes()[:-4])
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["tokens_sha256"] = hashlib.sha256(
        tokens.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    pointer = json.loads((tmp_path / "index.json").read_text())
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (tmp_path / "index.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(HybridCachePersistenceError, match="token file length"):
        read_hybrid_snapshot(str(tmp_path))


def test_hybrid_snapshot_rejects_declared_memory_mismatch(tmp_path):
    import hashlib

    write_hybrid_snapshot(str(tmp_path), _snapshot())
    generation = _generation_dir(tmp_path)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["memory_bytes"] += 1
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    pointer = json.loads((tmp_path / "index.json").read_text())
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (tmp_path / "index.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(HybridCachePersistenceError, match="memory byte count"):
        read_hybrid_snapshot(str(tmp_path))


def test_hybrid_snapshot_prunes_stale_generations_after_commit(tmp_path):
    write_hybrid_snapshot(str(tmp_path), _snapshot(model="first"))
    first = _generation_dir(tmp_path)
    write_hybrid_snapshot(str(tmp_path), _snapshot(model="second"))
    second = _generation_dir(tmp_path)

    assert first != second
    assert not first.exists()
    assert second.is_dir()


def test_hybrid_snapshot_serializes_writers_for_same_root(tmp_path, monkeypatch):
    """A second publisher must not enter the generation write concurrently."""
    import vllm_mlx.cache_persistence as persistence

    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    original_write_bytes = persistence._write_bytes

    def controlled_write_bytes(path, payload):
        if threading.current_thread().name == "hybrid-writer-first":
            first_entered.set()
            assert release_first.wait(timeout=5)
        elif threading.current_thread().name == "hybrid-writer-second":
            second_entered.set()
        return original_write_bytes(path, payload)

    monkeypatch.setattr(persistence, "_write_bytes", controlled_write_bytes)
    errors = []

    def publish(model):
        try:
            write_hybrid_snapshot(str(tmp_path), _snapshot(model=model))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(
        target=publish, args=("model-first",), name="hybrid-writer-first"
    )
    second = threading.Thread(
        target=publish, args=("model-second",), name="hybrid-writer-second"
    )
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    try:
        assert not second_entered.wait(timeout=0.25)
    finally:
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    loaded = read_hybrid_snapshot(str(tmp_path))
    assert loaded is not None
    assert loaded.identity["model"] == "model-second"


def test_hybrid_snapshot_reader_waits_for_same_root_writer(tmp_path, monkeypatch):
    """A reader must not traverse generations while publication may prune them."""
    import vllm_mlx.cache_persistence as persistence

    write_hybrid_snapshot(str(tmp_path), _snapshot(model="model-first"))
    writer_entered = threading.Event()
    release_writer = threading.Event()
    reader_entered = threading.Event()
    original_write_bytes = persistence._write_bytes
    original_load_json = persistence._load_json

    def controlled_write_bytes(path, payload):
        if threading.current_thread().name == "hybrid-writer":
            writer_entered.set()
            assert release_writer.wait(timeout=5)
        return original_write_bytes(path, payload)

    def observed_load_json(path):
        if threading.current_thread().name == "hybrid-reader":
            reader_entered.set()
        return original_load_json(path)

    monkeypatch.setattr(persistence, "_write_bytes", controlled_write_bytes)
    monkeypatch.setattr(persistence, "_load_json", observed_load_json)
    errors = []
    loaded = []

    def publish():
        try:
            write_hybrid_snapshot(str(tmp_path), _snapshot(model="model-second"))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def restore():
        try:
            loaded.append(read_hybrid_snapshot(str(tmp_path)))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    writer = threading.Thread(target=publish, name="hybrid-writer")
    reader = threading.Thread(target=restore, name="hybrid-reader")
    writer.start()
    assert writer_entered.wait(timeout=5)
    reader.start()
    try:
        assert not reader_entered.wait(timeout=0.25)
    finally:
        release_writer.set()
        writer.join(timeout=5)
        reader.join(timeout=5)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert errors == []
    assert len(loaded) == 1
    assert loaded[0] is not None
    assert loaded[0].identity["model"] == "model-second"


def test_hybrid_snapshot_rejects_symlinked_lock_file(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-lock"
    outside.write_text("not a lock")
    (tmp_path / ".hybrid-cache.lock").symlink_to(outside)

    with pytest.raises(HybridCachePersistenceError, match="lock must not be a symlink"):
        read_hybrid_snapshot(str(tmp_path))


def test_empty_snapshot_read_does_not_create_lock_file(tmp_path):
    assert read_hybrid_snapshot(str(tmp_path)) is None
    assert not (tmp_path / ".hybrid-cache.lock").exists()


def test_hybrid_snapshot_rejects_invalid_layer_metadata(tmp_path):
    import hashlib

    write_hybrid_snapshot(str(tmp_path), _snapshot())
    generation = _generation_dir(tmp_path)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["layers"][1]["metadata"]["offset"] = 99
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    pointer = json.loads((tmp_path / "index.json").read_text())
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (tmp_path / "index.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(HybridCachePersistenceError, match="KV layer offset"):
        read_hybrid_snapshot(str(tmp_path))


def test_hybrid_snapshot_rejects_symlinked_tensor_file(tmp_path):
    write_hybrid_snapshot(str(tmp_path), _snapshot())
    layer = _generation_dir(tmp_path) / "entry-0" / "layer-0.safetensors"
    outside = tmp_path.parent / f"{tmp_path.name}-layer.safetensors"
    layer.replace(outside)
    layer.symlink_to(outside)

    with pytest.raises(HybridCachePersistenceError, match="missing layer 0 file"):
        read_hybrid_snapshot(str(tmp_path))


def test_hybrid_snapshot_rejects_token_count_above_runtime_limit(tmp_path):
    import hashlib

    write_hybrid_snapshot(str(tmp_path), _snapshot())
    generation = _generation_dir(tmp_path)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["num_tokens"] = 5
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    pointer = json.loads((tmp_path / "index.json").read_text())
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (tmp_path / "index.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(
        HybridCachePersistenceError, match="snapshot exceeds restore token budget"
    ):
        read_hybrid_snapshot(str(tmp_path), max_tokens=4)


def test_hybrid_snapshot_rejects_aggregate_token_count_above_runtime_limit(tmp_path):
    import hashlib

    write_hybrid_snapshot(str(tmp_path), _snapshot())
    generation = _generation_dir(tmp_path)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    second = dict(manifest["entries"][0])
    second["directory"] = "entry-000001"
    manifest["entries"].append(second)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    pointer = json.loads((tmp_path / "index.json").read_text())
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (tmp_path / "index.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":"))
    )

    with pytest.raises(
        HybridCachePersistenceError, match="snapshot exceeds restore token budget"
    ):
        read_hybrid_snapshot(str(tmp_path), max_tokens=7)


def test_hybrid_snapshot_rejects_oversized_manifest_before_json_parse(tmp_path):
    import hashlib

    write_hybrid_snapshot(str(tmp_path), _snapshot())
    generation = _generation_dir(tmp_path)
    manifest_path = generation / "manifest.json"
    manifest_path.write_bytes(b"{" + (b" " * (16 * 1024 * 1024)))
    pointer_path = tmp_path / "index.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer_path.write_text(json.dumps(pointer, sort_keys=True, separators=(",", ":")))

    with pytest.raises(HybridCachePersistenceError, match="invalid JSON file"):
        read_hybrid_snapshot(str(tmp_path))


def test_hybrid_snapshot_rejects_nonempty_arrays_meta_state(tmp_path):
    snapshot = _snapshot()
    snapshot.entries[0].layers[0].metadata["meta_state"] = []

    with pytest.raises(HybridCachePersistenceError, match="ArraysCache meta_state"):
        write_hybrid_snapshot(str(tmp_path), snapshot)
