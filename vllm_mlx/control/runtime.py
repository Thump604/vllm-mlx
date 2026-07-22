# SPDX-License-Identifier: Apache-2.0
"""Managed artifact and residency adapter for the first product workflow."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from vllm_mlx.lifecycle import ModelSpec, ResidencyManager, bind_model_spec_to_profile
from vllm_mlx.model_workflow import AcquisitionOptions, acquire_model
from vllm_mlx.scheduler import SchedulerConfig


class ManagedRuntimeError(RuntimeError):
    """Raised when artifact or residency truth cannot be proven."""


_HASH_FILES = {
    "config_sha256": ("config.json",),
    "tokenizer_sha256": ("tokenizer.json",),
    "chat_template_sha256": ("chat_template.jinja",),
    "generation_config_sha256": ("generation_config.json",),
    "weights_manifest_sha256": ("model.safetensors.index.json", "SHA256SUMS"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_profile_artifact(
    profile: Mapping[str, Any], artifact_path: str | Path
) -> dict[str, Any]:
    """Verify the portable profile hashes against one concrete local artifact."""
    root = Path(artifact_path).expanduser().resolve()
    if not root.is_dir():
        raise ManagedRuntimeError(f"model artifact directory does not exist: {root}")
    verified: dict[str, str] = {}
    for field, candidates in _HASH_FILES.items():
        expected = profile["artifact"]["hashes"].get(field)
        if expected is None:
            continue
        path = next(
            (root / name for name in candidates if (root / name).is_file()), None
        )
        if path is None:
            raise ManagedRuntimeError(
                f"model artifact is missing the file required for {field}"
            )
        actual = _sha256(path)
        if actual != expected:
            raise ManagedRuntimeError(
                f"model artifact {field} does not match profile {profile['profile_id']}"
            )
        verified[field] = actual
    return {"artifact_path": str(root), "verified_hashes": verified}


def profile_to_model_spec(
    profile: Mapping[str, Any],
    artifact_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ModelSpec:
    """Resolve one validated profile into existing residency construction inputs."""
    serving = profile["serving"]
    limits = dict(serving["limits"])
    features = serving["features"]
    applied = dict(overrides or {})
    for name, value in applied.items():
        section, field = name.split(".", 1)
        if section == "limits":
            limits[field] = value

    def enabled(feature: str) -> bool:
        override = applied.get(f"features.{feature}")
        if override is not None:
            return bool(override)
        return str(features[feature]["mode"]) == "enabled_by_default"

    use_batching = enabled("continuous_batching")
    prefix_cache = enabled("prefix_cache")
    kvq4 = enabled("kvq4")
    kvq8 = enabled("kvq8")
    if kvq4 and kvq8:
        raise ManagedRuntimeError("profile cannot enable KVQ4 and KVQ8 together")
    feature_settings = features["continuous_batching"].get("settings", {})
    scheduler = SchedulerConfig(
        max_num_seqs=int(feature_settings.get("max_concurrency", 1)),
        enable_prefix_cache=prefix_cache,
        max_kv_size=int(limits["max_kv_size"]),
        enable_mtp=enabled("mtp"),
        kv_cache_quantization=kvq4 or kvq8,
        kv_cache_quantization_bits=4 if kvq4 else 8,
    )
    spec = ModelSpec(
        model_key="default",
        model_name=str(Path(artifact_path).expanduser().resolve()),
        use_batching=use_batching,
        scheduler_config=scheduler,
        max_tokens=int(limits["max_output_tokens"]),
        force_mllm=serving["route"] == "multimodal",
        mtp=enabled("mtp"),
        specprefill_enabled=enabled("specprefill"),
    )
    return bind_model_spec_to_profile(spec, profile)


class ManagedProductRuntime:
    """Adapt product operations to the existing artifact and residency owners."""

    def __init__(
        self,
        manager: ResidencyManager,
        *,
        model_root: str | Path,
        endpoint: str,
        artifact_bindings: Mapping[str, str | Path] | None = None,
        initial_profile: Mapping[str, Any] | None = None,
        apply_profile: (
            Callable[[Mapping[str, Any], ModelSpec], Awaitable[None] | None] | None
        ) = None,
        clear_profile: Callable[[], None] | None = None,
        sync_runtime: Callable[[], None] | None = None,
    ) -> None:
        self.manager = manager
        self.model_root = Path(model_root).expanduser().resolve()
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint.rstrip("/")
        self.artifact_bindings = {
            profile_id: Path(path).expanduser().resolve()
            for profile_id, path in (artifact_bindings or {}).items()
        }
        self.apply_profile = apply_profile
        self.clear_profile = clear_profile
        self.sync_runtime = sync_runtime
        self._active_profile = (
            deepcopy(dict(initial_profile)) if initial_profile is not None else None
        )
        self._installed: dict[str, Path] = {}

    async def install(self, profile: Mapping[str, Any]) -> Mapping[str, Any]:
        profile_id = str(profile["profile_id"])
        binding = self.artifact_bindings.get(profile_id)
        if binding is not None:
            evidence = await asyncio.to_thread(
                verify_profile_artifact, profile, binding
            )
            self._installed[profile_id] = binding
            return {**evidence, "managed": False, "source": "artifact_binding"}

        if profile["artifact"]["quantization"].get("source") == "conversion_manifest":
            raise ManagedRuntimeError(
                f"profile {profile_id} requires a verified converted-artifact binding"
            )
        target = self._managed_path(profile)
        repository = str(profile["identity"]["repository_id"])
        revision = str(profile["identity"]["resolved_revision"])
        await asyncio.to_thread(
            acquire_model,
            repository,
            options=AcquisitionOptions(
                revision=revision,
                target_dir=str(target),
                is_mllm=profile["serving"]["route"] == "multimodal",
            ),
        )
        evidence = await asyncio.to_thread(verify_profile_artifact, profile, target)
        self._installed[profile_id] = target
        return {**evidence, "managed": True, "source": "immutable_repository"}

    async def activate(
        self, profile: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        artifact = self._artifact_for(profile)
        await asyncio.to_thread(verify_profile_artifact, profile, artifact)
        spec = profile_to_model_spec(profile, artifact, overrides=overrides)
        previous_profile = deepcopy(self._active_profile)
        try:
            previous_spec = self.manager.get_spec("default")
            previous_status = self.manager.get_status("default")
        except KeyError:
            previous_spec = None
            previous_status = {"state": "unloaded"}
        if previous_status["state"] != "unloaded":
            await self.manager.unload("default")
        try:
            self.manager.register_model(spec)
            await self._apply(profile, spec)
            await self.manager.ensure_loaded("default")
        except BaseException:
            if self.manager.get_status("default")["state"] != "unloaded":
                await self.manager.unload("default")
            if previous_spec is not None:
                self.manager.register_model(previous_spec)
                if previous_profile is not None:
                    await self._apply(previous_profile, previous_spec)
                if previous_status["state"] == "loaded":
                    await self.manager.ensure_loaded("default")
            else:
                self.manager.clear_dormant_model("default")
                self._active_profile = None
                self._clear_profile()
            self._sync()
            raise
        self._active_profile = deepcopy(dict(profile))
        self._sync()
        return self.status()

    async def stop(self) -> Mapping[str, Any]:
        try:
            status = self.manager.get_status("default")
        except KeyError:
            return self.status()
        if status["state"] != "unloaded":
            await self.manager.unload("default")
        self._sync()
        return self.status()

    async def remove(self, profile: Mapping[str, Any]) -> Mapping[str, Any]:
        profile_id = str(profile["profile_id"])
        if profile_id in self.artifact_bindings:
            raise ManagedRuntimeError("externally bound artifacts cannot be removed")
        if (
            self._active_profile is not None
            and self._active_profile["profile_id"] == profile_id
            and self.status()["state"] != "unloaded"
        ):
            raise ManagedRuntimeError("active model artifact cannot be removed")
        target = self._managed_path(profile)
        try:
            target.relative_to(self.model_root)
        except ValueError as exc:
            raise ManagedRuntimeError(
                "artifact is outside the managed model root"
            ) from exc
        if target.exists():
            await asyncio.to_thread(verify_profile_artifact, profile, target)
        if (
            self._active_profile is not None
            and self._active_profile["profile_id"] == profile_id
        ):
            self.manager.clear_dormant_model("default")
            self._active_profile = None
            self._clear_profile()
            self._sync()
        if target.exists():
            await asyncio.to_thread(shutil.rmtree, target)
        self._installed.pop(profile_id, None)
        return {"removed": True, "artifact_path": str(target)}

    def status(self) -> Mapping[str, Any]:
        try:
            resident = self.manager.get_status("default")
        except KeyError:
            resident = {
                "model_key": "default",
                "model_name": None,
                "state": "unloaded",
                "active_requests": 0,
                "last_used_at": None,
                "loaded_at": None,
                "last_error": None,
                "estimated_memory_bytes": None,
                "auto_unload_idle_seconds": self.manager.auto_unload_idle_seconds,
            }
        active = self._profile_reference(self._active_profile)
        return {
            "active_profile": active,
            "state": resident["state"],
            "healthy": resident["state"] == "loaded" and resident["last_error"] is None,
            "endpoint": self.endpoint if resident["state"] == "loaded" else None,
            "resident": resident,
        }

    def diagnostics(self) -> Mapping[str, Any]:
        return {
            "status": self.status(),
            "model_root": str(self.model_root),
            "artifact_bindings": sorted(self.artifact_bindings),
            "managed_artifacts": {
                profile_id: str(path)
                for profile_id, path in sorted(self._installed.items())
            },
        }

    def operation_is_cancellable(self, operation_kind: str) -> bool:
        return operation_kind in {"model.activate", "model.stop"}

    def _artifact_for(self, profile: Mapping[str, Any]) -> Path:
        profile_id = str(profile["profile_id"])
        path = self._installed.get(profile_id) or self.artifact_bindings.get(profile_id)
        if path is None:
            target = self._managed_path(profile)
            if target.is_dir():
                path = target
        if path is None:
            raise ManagedRuntimeError(f"profile {profile_id} is not installed")
        return path

    def _managed_path(self, profile: Mapping[str, Any]) -> Path:
        artifact_id = str(profile["identity"]["artifact_id"])
        if not artifact_id or artifact_id in {".", ".."} or "/" in artifact_id:
            raise ManagedRuntimeError(
                "profile artifact_id is not a safe directory name"
            )
        return self.model_root / artifact_id

    async def _apply(self, profile: Mapping[str, Any], spec: ModelSpec) -> None:
        if self.apply_profile is None:
            return
        result = self.apply_profile(profile, spec)
        if asyncio.iscoroutine(result):
            await result

    def _sync(self) -> None:
        if self.sync_runtime is not None:
            self.sync_runtime()

    def _clear_profile(self) -> None:
        if self.clear_profile is not None:
            self.clear_profile()

    @staticmethod
    def _profile_reference(profile: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if profile is None:
            return None
        return {
            "profile_id": profile["profile_id"],
            "profile_revision": profile["profile_revision"],
            "subject_digest": profile["subject_digest"],
        }
