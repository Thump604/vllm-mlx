# SPDX-License-Identifier: Apache-2.0
"""Real-call-site regressions for MLLM cache-owner lifecycle wiring."""

import ast
import asyncio
from collections import deque
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import MagicMock, call

import pytest

import vllm_mlx.cache_owner_identity as owner_identity_module
from vllm_mlx.cache_owner_identity import (
    CacheOwnerGovernanceTarget,
    OwnerBindingDecision,
    VerifiedCacheOwnerContext,
    build_loaded_cache_owner_digest,
    build_loaded_runtime_composition_digest,
    verify_loaded_model_cache_owner_context,
)
from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest
from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler, MLLMSchedulerConfig
from vllm_mlx.request import RequestStatus
from vllm_mlx.memory_cache import (
    MemoryCacheConfig,
    build_cache_owner_persistence_identity,
)
from vllm_mlx.scheduler import SchedulerConfig
from vllm_mlx.specprefill_contract import (
    build_draft_compatibility,
    build_identity_manifest,
    canonical_identity_bytes,
)


def _generator(*, owner_required: bool = True):
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.prefix_cache = MagicMock()
    generator._cache_owner_required = owner_required
    generator._cache_owner_requests = {}
    generator._prefix_checkpoint_lock = threading.Lock()
    generator._request_prefix_checkpoints = {}
    generator._aborted_request_ids = set()
    generator._pending_removal_lock = threading.Lock()
    generator._pending_removal_uids = set()
    generator.unprocessed_requests = []
    generator.active_batch = None
    generator.uid_counter = 0
    generator.completion_batch_size = 4
    generator._pending_error_responses = []
    return generator


def _request(request_id: str = "request-1") -> MLLMBatchRequest:
    return MLLMBatchRequest(uid=-1, request_id=request_id, prompt="hello")


def test_insert_mints_owner_request_and_abort_revokes_it():
    generator = _generator()
    binding = object()
    generator.prefix_cache.mint_owner_request.return_value = binding

    request = _request()
    assert generator.insert([request]) == [0]
    generator.prefix_cache.mint_owner_request.assert_called_once_with(0)
    assert generator._cache_owner_requests == {request.request_id: binding}

    generator.abort_prefill(request.request_id)
    generator.prefix_cache.cancel_owner_request.assert_called_once_with(binding)


def test_remove_releases_owner_request_after_scheduler_thread_removal():
    generator = _generator()
    binding = object()
    generator.prefix_cache.mint_owner_request.return_value = binding
    request = _request()
    generator.insert([request])

    generator.remove([request.uid])

    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)
    assert generator._cache_owner_requests == {}


def test_partial_insert_failure_releases_every_minted_owner_request():
    generator = _generator()
    first = object()
    generator.prefix_cache.mint_owner_request.side_effect = [
        first,
        RuntimeError("mint failed"),
    ]

    with pytest.raises(RuntimeError, match="mint failed"):
        generator.insert([_request("request-1"), _request("request-2")])

    generator.prefix_cache.release_owner_request.assert_called_once_with(first)
    assert generator._cache_owner_requests == {}
    assert generator.unprocessed_requests == []
    assert generator.uid_counter == 0


def test_preprocess_failure_releases_owner_request_immediately(monkeypatch):
    generator = _generator()
    binding = object()
    generator.prefix_cache.mint_owner_request.return_value = binding
    request = _request()
    generator.insert([request])
    monkeypatch.setattr(
        generator,
        "_preprocess_request",
        MagicMock(side_effect=RuntimeError("bad input")),
    )

    assert generator._process_prompts([request]) is None
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)
    assert generator._cache_owner_requests == {}


def test_inline_failure_removes_request_and_releases_owner_once():
    generator = _generator()
    binding = object()
    generator.prefix_cache.mint_owner_request.return_value = binding
    request = _request()
    generator.insert([request])

    generator._fail_inline_requests([request])

    assert generator.unprocessed_requests == []
    assert generator._cache_owner_requests == {}
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)
    assert [response.request_id for response in generator._pending_error_responses] == [
        request.request_id
    ]


def test_checkpoint_prepare_and_publication_use_same_owner_binding():
    generator = _generator()
    binding = object()
    prepared = SimpleNamespace(tokens=(1, 2, 3))
    cache_state = [object()]
    auxiliary = {"last_logits": object()}
    generator._cache_owner_requests["request-1"] = binding
    generator.prefix_cache.prepare_owner_bound_store.return_value = (
        OwnerBindingDecision(True, "none"),
        prepared,
    )
    generator.prefix_cache.commit_owner_bound_store.return_value = OwnerBindingDecision(
        True, "none"
    )

    entry = generator._prepare_prefix_store(
        "request-1",
        [1, 2, 3],
        cache_state,
        auxiliary=auxiliary,
        persistence_eligible=True,
    )
    assert entry is prepared
    generator.prefix_cache.prepare_owner_bound_store.assert_called_once_with(
        binding,
        [1, 2, 3],
        cache_state,
        auxiliary=auxiliary,
        persistence_eligible=True,
    )

    assert generator._publish_prefill_checkpoint("request-1", entry) is True
    call = generator.prefix_cache.commit_owner_bound_store.call_args
    assert call.args == (prepared,)
    assert call.kwargs["evict_prefixes"] is False
    assert call.kwargs["commit_lock"] is generator._prefix_checkpoint_lock
    assert callable(call.kwargs["commit_guard"])


def test_owner_required_missing_request_binding_fails_closed():
    generator = _generator()

    with pytest.raises(RuntimeError, match="owner-bound request"):
        generator._prepare_prefix_store(
            "missing",
            [1],
            [object()],
        )


def test_legacy_cache_path_remains_explicit_when_owner_binding_is_disabled():
    generator = _generator(owner_required=False)
    entry = SimpleNamespace(tokens=(1, 2, 3))
    generator.prefix_cache.prepare_store.return_value = entry
    generator.prefix_cache.commit_prepared.return_value = True

    prepared = generator._prepare_prefix_store(
        "legacy",
        [1, 2, 3],
        [object()],
    )
    assert prepared is entry
    assert generator._publish_prefill_checkpoint("legacy", prepared) is True
    generator.prefix_cache.prepare_owner_bound_store.assert_not_called()
    generator.prefix_cache.commit_owner_bound_store.assert_not_called()


def test_close_revokes_owner_and_discards_request_handles():
    generator = _generator()
    generator._old_wired_limit = None
    generator._cache_owner_requests["request-1"] = object()

    generator.close()

    generator.prefix_cache.close_owner_identity.assert_called_once_with()
    assert generator._cache_owner_requests == {}


def test_cache_lifecycle_revokes_requests_then_rebinds_verified_context():
    generator = _generator()
    binding = object()
    context = object()
    generator._cache_owner_context = context
    generator._cache_owner_requests["request-1"] = binding

    generator.invalidate_cache_owner_requests()
    generator.rebind_cache_owner_context()

    generator.prefix_cache.cancel_owner_request.assert_called_once_with(binding)
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)
    generator.prefix_cache.bind_owner_context.assert_called_once_with(context)


def test_owner_bound_fetch_and_completion_release_use_live_request_handle():
    generator = _generator()
    binding = object()
    generator._cache_owner_requests["request-1"] = binding
    generator.prefix_cache.fetch_owner_bound.return_value = (
        OwnerBindingDecision(True, "none"),
        ["cache"],
        [3],
    )

    assert generator._fetch_prefix_cache("request-1", [1, 2, 3]) == (
        ["cache"],
        [3],
    )
    generator.prefix_cache.fetch_owner_bound.assert_called_once_with(binding, [1, 2, 3])

    request = _request()
    request.input_ids = None
    batch = SimpleNamespace(requests=[request])
    generator._maybe_store_prefix_cache(batch, [0])
    generator.prefix_cache.release_owner_request.assert_called_once_with(binding)
    assert generator._cache_owner_requests == {}


def test_generator_has_no_direct_cache_publication_or_fetch_bypass():
    source = Path(MLLMBatchGenerator.__module__.replace(".", "/") + ".py")
    source = Path(__file__).parents[1] / source
    tree = ast.parse(source.read_text())
    calls: list[tuple[str, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.functions: list[str] = []

        def visit_FunctionDef(self, node):
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "prepare_store",
                "commit_prepared",
                "store",
                "fetch",
                "fetch_exact_auxiliary",
            }:
                calls.append((self.functions[-1], node.func.attr))
            self.generic_visit(node)

    Visitor().visit(tree)
    assert set(calls) <= {
        ("_prepare_prefix_store", "prepare_store"),
        ("_publish_prefill_checkpoint", "commit_prepared"),
        ("_store_prefix_snapshot", "store"),
        ("_fetch_prefix_cache", "fetch"),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _patch_fixture_tokenizer_version(monkeypatch):
    original = owner_identity_module.importlib.metadata.version

    def version(distribution):
        if distribution == __name__.split(".", 1)[0]:
            return "fixture-version"
        return original(distribution)

    monkeypatch.setattr(owner_identity_module.importlib.metadata, "version", version)


def _write_loaded_identity_fixture(root: Path):
    files = {
        "artifact-source.json": json.dumps(
            {"model_id": "fixture/model", "revision": "fixture-revision"},
            sort_keys=True,
        ),
        "config.json": "{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"weight": "model-00001-of-00001.safetensors"}},
            sort_keys=True,
        ),
        "model-00001-of-00001.safetensors": "fixture-weights",
        "tokenizer_config.json": "{}",
        "chat_template.jinja": "{{ messages }}",
        "tokenizer.json": "{}",
    }
    for name, contents in files.items():
        (root / name).write_text(contents)

    class FixtureLanguageModel:
        config = SimpleNamespace(
            model_type="fixture-type",
            num_hidden_layers=1,
            hidden_size=8,
            vocab_size=16,
            num_key_value_heads=1,
            head_dim=8,
        )

        def make_cache(self):
            class FixtureCache:
                keys = None
                values = None

            return [FixtureCache()]

    class FixtureModel:
        def __init__(self):
            self.language_model = FixtureLanguageModel()
            self.config = SimpleNamespace(model_type="fixture-type")

    class FixtureTokenizer:
        name_or_path = str(root)
        vocab_size = 1
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        chat_template = "{{ messages }}"

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [len(text)]

        def get_vocab(self):
            return {"token": 0}

    model = FixtureModel()
    processor = SimpleNamespace(
        tokenizer=FixtureTokenizer(), chat_template="{{ messages }}"
    )
    cache_config = MemoryCacheConfig(max_memory_mb=1)
    persistence = dict(
        build_cache_owner_persistence_identity(
            model.language_model,
            processor.tokenizer,
            str(root),
            cache_config,
            {"max_kv_size": 1},
            processor,
        )
    )
    mode = {
        "continuous_batching": True,
        "serialized": False,
        "mtp": False,
        "chunked_prefill": True,
        "rotating_cache": True,
        "ple": False,
        "qsa": False,
        "audio": False,
        "capability_modes": ["text"],
    }
    cache_identity = {
        "model_id": "fixture/model",
        "model_revision": "fixture-revision",
        "artifact_id": root.name,
        "artifact_digest": _sha256(root / "artifact-source.json"),
        "weight_index_digest": _sha256(root / "model.safetensors.index.json"),
        "family": "fixture",
        "architecture": "FixtureModel",
        "model_module": type(model).__module__,
        "language_module": type(model.language_model).__module__,
        "model_type": "fixture-type",
        "config_digest": _sha256(root / "config.json"),
        "tokenizer": {
            "files": [
                {"path": "tokenizer.json", "sha256": _sha256(root / "tokenizer.json")}
            ],
            "config_digest": _sha256(root / "tokenizer_config.json"),
            "implementation": type(processor.tokenizer).__module__.split(".", 1)[0],
            "implementation_version": "fixture-version",
            "added_tokens": [],
            "special_tokens": [],
            "encode_probes": [{"text": "abc", "ids": [3]}],
        },
        "chat_template": {
            "name": "fixture",
            "version": "1",
            "sha256": _sha256(root / "chat_template.jinja"),
        },
        "parser": {"name": "fixture", "version": "1", "sha256": "7" * 64},
        "vision": {
            "enabled": False,
            "config_digest": "8" * 64,
            "processor_files": [],
            "media_token_ids": [],
            "media_mapping_digest": "9" * 64,
        },
        "cache_schema": {
            "name": "fixture-cache",
            "version": "1",
            "sha256": persistence["cache_layout"],
        },
        "mode": mode,
    }
    draft_identity = dict(cache_identity)
    draft_identity.update(
        model_id="fixture/draft",
        model_revision="draft-revision",
        artifact_id="draft-artifact",
        artifact_digest="b" * 64,
    )
    relation = build_draft_compatibility(
        cache_identity, draft_identity, relation="fixture-v1"
    )
    manifest = build_identity_manifest(
        cache_identity,
        {
            "template_digest": "c" * 64,
            "parser_digest": "d" * 64,
            "tool_schema_digest": "e" * 64,
            "renderer": "fixture",
            "version": "1",
        },
        role="target",
        draft_compatibility=relation,
    )
    digest = hashlib.sha256(canonical_identity_bytes(cache_identity)).hexdigest()
    expected_fields = (
        ("role", "target"),
        ("request_protocol_identity", manifest["request_protocol_identity"]),
        ("draft_compatibility", manifest["draft_compatibility"]),
    )
    target = CacheOwnerGovernanceTarget(
        manifest=manifest,
        expected_model_cache_identity_digest=digest,
        expected_loaded_owner_digest=build_loaded_cache_owner_digest(
            persistence_identity=persistence,
            artifact_source={
                "model_id": "fixture/model",
                "revision": "fixture-revision",
            },
            loaded_identity={
                "architecture": "FixtureModel",
                "model_module": type(model).__module__,
                "language_module": type(model.language_model).__module__,
                "model_type": "fixture-type",
            },
            runtime_mode=mode,
            tokenizer_implementation=type(processor.tokenizer).__module__.split(".", 1)[
                0
            ],
            tokenizer_implementation_version="fixture-version",
        ),
        expected_identity_fields=expected_fields,
        expected_persistence_identity=persistence,
        runtime_composition_digest=build_loaded_runtime_composition_digest(),
        cache_namespace="fixture-cache",
        registry_source="test:fixture",
        registry_complete=True,
    )
    return model, processor, target, cache_config, mode


def test_loaded_identity_is_derived_from_live_types_and_artifact_bytes(
    tmp_path, monkeypatch
):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    resolved = verify_loaded_model_cache_owner_context(
        model,
        processor,
        str(tmp_path),
        target,
        cache_config=cache_config,
        cache_runtime_identity={"max_kv_size": 1},
        runtime_mode=mode,
    )
    assert resolved.model_cache_identity_digest == target.expected_loaded_owner_digest

    (tmp_path / "config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="config.json"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_loaded_identity_rejects_weight_mutation(tmp_path, monkeypatch):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    (tmp_path / "model-00001-of-00001.safetensors").write_text("mutated")

    with pytest.raises(ValueError, match="provenance"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_loaded_identity_rejects_symlinked_model_root(tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    model_root.mkdir()
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        model_root
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    symlink = tmp_path / "model-link"
    symlink.symlink_to(model_root, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(symlink),
            target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_loaded_identity_rejects_target_only_cache_schema_mutation(
    tmp_path, monkeypatch
):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    cache_identity = _plain(target.manifest["model_cache_identity"])
    cache_identity["cache_schema"]["sha256"] = "0" * 64
    changed_manifest = build_identity_manifest(
        cache_identity,
        _plain(target.manifest["request_protocol_identity"]),
        role="target",
        draft_compatibility=_plain(target.manifest["draft_compatibility"]),
    )
    changed_target = CacheOwnerGovernanceTarget(
        manifest=changed_manifest,
        expected_model_cache_identity_digest=hashlib.sha256(
            canonical_identity_bytes(cache_identity)
        ).hexdigest(),
        expected_loaded_owner_digest=target.expected_loaded_owner_digest,
        expected_identity_fields=(
            ("role", "target"),
            (
                "request_protocol_identity",
                changed_manifest["request_protocol_identity"],
            ),
            ("draft_compatibility", changed_manifest["draft_compatibility"]),
        ),
        expected_persistence_identity=target.expected_persistence_identity,
        runtime_composition_digest=target.runtime_composition_digest,
        cache_namespace=target.cache_namespace,
        registry_source="test:mutated-target",
        registry_complete=True,
    )

    with pytest.raises(ValueError, match="cache schema"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            changed_target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_loaded_identity_rejects_governed_loaded_owner_digest_mismatch(
    tmp_path, monkeypatch
):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    changed_target = CacheOwnerGovernanceTarget(
        manifest=target.manifest,
        expected_model_cache_identity_digest=(
            target.expected_model_cache_identity_digest
        ),
        expected_loaded_owner_digest="0" * 64,
        expected_identity_fields=target.expected_identity_fields,
        expected_persistence_identity=target.expected_persistence_identity,
        runtime_composition_digest=target.runtime_composition_digest,
        cache_namespace=target.cache_namespace,
        registry_source="test:mismatched-loaded-owner",
        registry_complete=True,
    )

    with pytest.raises(ValueError, match="loaded cache owner digest"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            changed_target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_loaded_identity_rejects_runtime_composition_mismatch(tmp_path, monkeypatch):
    model, processor, target, cache_config, mode = _write_loaded_identity_fixture(
        tmp_path
    )
    _patch_fixture_tokenizer_version(monkeypatch)
    changed_target = CacheOwnerGovernanceTarget(
        manifest=target.manifest,
        expected_model_cache_identity_digest=(
            target.expected_model_cache_identity_digest
        ),
        expected_loaded_owner_digest=target.expected_loaded_owner_digest,
        expected_identity_fields=target.expected_identity_fields,
        expected_persistence_identity=target.expected_persistence_identity,
        runtime_composition_digest="0" * 64,
        cache_namespace=target.cache_namespace,
        registry_source="test:mismatched-runtime",
        registry_complete=True,
    )

    with pytest.raises(ValueError, match="runtime composition"):
        verify_loaded_model_cache_owner_context(
            model,
            processor,
            str(tmp_path),
            changed_target,
            cache_config=cache_config,
            cache_runtime_identity={"max_kv_size": 1},
            runtime_mode=mode,
        )


def test_scheduler_threads_verified_owner_context(monkeypatch):
    context = MagicMock(spec=VerifiedCacheOwnerContext)
    config = MLLMSchedulerConfig(
        enable_prefix_cache=True,
        cache_owner_context=context,
    )
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.batch_generator = None
    scheduler.config = config
    scheduler.model = object()
    scheduler.processor = object()
    scheduler.mm_processor = object()
    scheduler.stop_tokens = set()
    scheduler._ssd_tier = None
    captured = {}

    class FakeGenerator:
        prefix_cache = None

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("vllm_mlx.mllm_scheduler.MLLMBatchGenerator", FakeGenerator)
    monkeypatch.setattr("mlx_lm.sample_utils.make_sampler", lambda **_kwargs: object())

    scheduler._ensure_batch_generator()

    assert captured["cache_owner_context"] is context


@pytest.mark.parametrize("insert_result", [RuntimeError("mint failed"), []])
def test_scheduler_insert_failure_leaves_waiting_state_transactional(insert_result):
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request = MLLMRequest(request_id="request-1", prompt="hello")
    request.num_prompt_tokens = 3
    scheduler.waiting = deque([request])
    scheduler.running = {}
    scheduler.config = MLLMSchedulerConfig(max_num_seqs=1)
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.total_prompt_tokens = 0
    scheduler.batch_generator = MagicMock()
    scheduler._ensure_batch_generator = MagicMock()
    if isinstance(insert_result, Exception):
        scheduler.batch_generator.insert.side_effect = insert_result
    else:
        scheduler.batch_generator.insert.return_value = insert_result

    assert scheduler._schedule_waiting() == []

    assert list(scheduler.waiting) == [request]
    assert request.status is RequestStatus.WAITING
    assert request.batch_uid is None
    assert scheduler.running == {}
    assert scheduler.request_id_to_uid == {}
    assert scheduler.uid_to_request_id == {}
    assert scheduler.total_prompt_tokens == 0


def test_batched_engine_derives_and_threads_verified_owner_context(
    tmp_path, monkeypatch
):
    from vllm_mlx.engine.batched import BatchedEngine

    target = MagicMock(spec=CacheOwnerGovernanceTarget)
    context = MagicMock(spec=VerifiedCacheOwnerContext)
    scheduler_config = SchedulerConfig(
        enable_prefix_cache=True,
        enable_mtp=True,
        chunked_prefill_tokens=2048,
        max_kv_size=4096,
        cache_owner_target=target,
    )
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._model = SimpleNamespace(config=SimpleNamespace())
    engine._processor = SimpleNamespace(
        tokenizer=SimpleNamespace(name_or_path=str(tmp_path))
    )
    engine._scheduler_config = scheduler_config
    engine._model_name = str(tmp_path)
    engine._mllm_draft_model = None
    engine._mllm_draft_kind = None
    engine._mllm_draft_block_size = None
    engine._mllm_instance = SimpleNamespace(resolved_model_path=str(tmp_path))
    engine._mllm_scheduler = None
    captured = {}

    class FakeScheduler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            return None

    monkeypatch.setattr("vllm_mlx.mllm_scheduler.MLLMScheduler", FakeScheduler)
    verify = MagicMock(return_value=context)
    monkeypatch.setattr(
        "vllm_mlx.cache_owner_identity.verify_loaded_model_cache_owner_context",
        verify,
    )
    monkeypatch.setattr(
        "vllm_mlx.mllm_batch_generator.normalize_prefix_cache_chat_template",
        lambda _processor: None,
    )

    asyncio.run(engine._start_mllm())

    config = captured["config"]
    assert config.cache_owner_context is context
    assert verify.call_args.kwargs["runtime_mode"] == {
        "continuous_batching": True,
        "serialized": False,
        "mtp": True,
        "chunked_prefill": True,
        "rotating_cache": True,
        "ple": False,
        "qsa": False,
        "audio": False,
    }


def test_mllm_loader_records_and_uses_exact_resolved_model_root(tmp_path, monkeypatch):
    import mlx_vlm
    import mlx_vlm.utils

    from vllm_mlx.models.mllm import MLXMultimodalLM

    observed = []
    model = SimpleNamespace(config=SimpleNamespace())
    processor = object()
    monkeypatch.setattr(mlx_vlm.utils, "get_model_path", lambda _model_name: tmp_path)
    monkeypatch.setattr(
        mlx_vlm,
        "load",
        lambda path: (observed.append(("load", path)) or (model, processor)),
    )
    monkeypatch.setattr(
        mlx_vlm.utils,
        "load_config",
        lambda path: observed.append(("config", path)) or {},
    )

    wrapper = MLXMultimodalLM("fixture/model", enable_cache=False)
    wrapper.load()

    resolved = str(tmp_path.resolve())
    assert wrapper.resolved_model_path == resolved
    assert observed == [("load", resolved), ("config", resolved)]


@pytest.mark.parametrize("operation", ["load", "clear"])
def test_batched_engine_lifecycle_invalidates_then_rebinds_owner(operation):
    from vllm_mlx.engine.batched import BatchedEngine

    events = MagicMock()
    prefix_cache = MagicMock()
    batch_generator = MagicMock(prefix_cache=prefix_cache)
    events.attach_mock(batch_generator.begin_cache_owner_lifecycle_mutation, "begin")
    events.attach_mock(batch_generator.finish_cache_owner_lifecycle_mutation, "finish")
    events.attach_mock(prefix_cache.load_from_disk, "load")
    events.attach_mock(prefix_cache.clear, "clear")
    scheduler = MagicMock(batch_generator=batch_generator)
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._mllm_scheduler = scheduler
    engine._engine = None

    if operation == "load":
        prefix_cache.load_from_disk.return_value = 3
        assert engine.load_cache_from_disk("cache-dir") == 3
        assert events.mock_calls == [
            call.begin(),
            call.load("cache-dir"),
            call.finish(),
        ]
    else:
        engine.clear_prefix_cache()
        assert events.mock_calls == [
            call.begin(),
            call.clear(),
            call.finish(),
        ]


def test_batched_engine_rebinds_owner_when_cache_load_fails():
    from vllm_mlx.engine.batched import BatchedEngine

    prefix_cache = MagicMock()
    prefix_cache.load_from_disk.side_effect = RuntimeError("corrupt cache")
    batch_generator = MagicMock(prefix_cache=prefix_cache)
    scheduler = MagicMock(batch_generator=batch_generator)
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._mllm_scheduler = scheduler
    engine._engine = None

    with pytest.raises(RuntimeError, match="corrupt cache"):
        engine.load_cache_from_disk("cache-dir")

    batch_generator.begin_cache_owner_lifecycle_mutation.assert_called_once_with()
    batch_generator.finish_cache_owner_lifecycle_mutation.assert_not_called()


def test_owner_lifecycle_mutation_rejects_active_generator_work():
    generator = _generator()
    binding = object()
    generator._cache_owner_requests["request-1"] = binding

    with pytest.raises(RuntimeError, match="requires an idle generator"):
        generator.begin_cache_owner_lifecycle_mutation()

    assert generator._cache_owner_requests == {"request-1": binding}
    generator.prefix_cache.cancel_owner_request.assert_not_called()
    generator.prefix_cache.release_owner_request.assert_not_called()


def test_scheduler_cache_clear_stops_before_mutation_when_owner_is_active():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.vision_cache = None
    prefix_cache = MagicMock()
    batch_generator = MagicMock(prefix_cache=prefix_cache)
    batch_generator.begin_cache_owner_lifecycle_mutation.side_effect = RuntimeError(
        "owner active"
    )
    scheduler.batch_generator = batch_generator

    with pytest.raises(RuntimeError, match="owner active"):
        scheduler.clear_runtime_caches()

    prefix_cache.clear.assert_not_called()
    batch_generator.finish_cache_owner_lifecycle_mutation.assert_not_called()
