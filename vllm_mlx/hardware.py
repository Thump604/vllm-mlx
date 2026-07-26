# SPDX-License-Identifier: Apache-2.0
"""Read-only, source-attributed Apple Silicon hardware inventory.

This module reports hardware facts needed by later model-fit work.  It does
not load models, estimate performance, choose configuration, or invent a
fallback value when a platform probe is unavailable.
"""

from __future__ import annotations

import json
import platform as _platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class SourceRecord:
    """One sanitized source used to derive an inventory fact."""

    field: str
    source: str
    locator: str
    status: str
    value: Any = None


@dataclass(frozen=True)
class HardwareInventory:
    """Stable source facts about the local host, with no policy decisions."""

    operating_system: str | None
    architecture: str | None
    is_apple_silicon: bool | None
    chip_string: str | None
    machine_model: str | None
    machine_name: str | None
    total_unified_memory_bytes: int | None
    reported_cpu_configuration: str | None
    logical_cpu_cores: int | None
    physical_cpu_cores: int | None
    gpu_core_count: int | None
    macos_version: str | None
    macos_build: str | None
    sources: tuple[SourceRecord, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation without undisclosed raw data."""
        result = asdict(self)
        result["sources"] = [asdict(source) for source in self.sources]
        return result


class PlatformProvider(Protocol):
    """Minimal injectable platform boundary for deterministic inventory tests."""

    def system(self) -> str: ...

    def machine(self) -> str: ...

    def mac_ver(self) -> tuple[str, tuple[str, str, str], str]: ...


CommandRunner = Callable[[list[str]], str]


class _NativePlatform:
    def system(self) -> str:
        return _platform.system()

    def machine(self) -> str:
        return _platform.machine()

    def mac_ver(self) -> tuple[str, tuple[str, str, str], str]:
        return _platform.mac_ver()


def _run_command(command: list[str]) -> str:
    """Run a bounded local probe and return its text output."""
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    )
    return completed.stdout


def _read_sysctl(
    name: str,
    *,
    field: str,
    command_runner: CommandRunner,
    records: list[SourceRecord],
    errors: list[str],
    record_value: bool = True,
) -> str | None:
    try:
        value = command_runner(["sysctl", "-n", name]).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"sysctl {name}: {exc}")
        records.append(SourceRecord(field, "sysctl", name, "error"))
        return None
    except Exception as exc:  # Provider failures must remain visible to callers.
        errors.append(f"sysctl {name}: {exc}")
        records.append(SourceRecord(field, "sysctl", name, "error"))
        return None

    if not value:
        records.append(SourceRecord(field, "sysctl", name, "unknown"))
        return None
    if record_value:
        records.append(SourceRecord(field, "sysctl", name, "reported", value))
    return value


def _positive_integer(
    raw: str | int | None,
    *,
    field: str,
    source: str,
    records: list[SourceRecord],
    errors: list[str],
) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        errors.append(f"{field}: {source} reported a non-integer value")
        records.append(SourceRecord(field, source, field, "invalid"))
        return None
    if value <= 0:
        errors.append(f"{field}: {source} reported a non-positive value")
        records.append(SourceRecord(field, source, field, "invalid"))
        return None
    return value


def _system_profiler_hardware(
    *,
    command_runner: CommandRunner,
    records: list[SourceRecord],
    errors: list[str],
) -> dict[str, Any]:
    """Read only an explicit allowlist from SPHardwareDataType JSON."""
    try:
        raw = command_runner(["system_profiler", "SPHardwareDataType", "-json"])
        payload = json.loads(raw)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        errors.append(f"system_profiler SPHardwareDataType: {exc}")
        records.append(
            SourceRecord(
                "hardware_profile", "system_profiler", "SPHardwareDataType", "error"
            )
        )
        return {}
    except Exception as exc:
        errors.append(f"system_profiler SPHardwareDataType: {exc}")
        records.append(
            SourceRecord(
                "hardware_profile", "system_profiler", "SPHardwareDataType", "error"
            )
        )
        return {}

    entries = payload.get("SPHardwareDataType")
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        errors.append("system_profiler SPHardwareDataType: missing hardware record")
        records.append(
            SourceRecord(
                "hardware_profile", "system_profiler", "SPHardwareDataType", "invalid"
            )
        )
        return {}

    # Deliberately omit serial_number, platform_UUID, provisioning_UDID, and
    # every other field. This inventory is not a device-identity endpoint.
    allowed = {
        "chip_type": "chip_string",
        "machine_model": "machine_model",
        "machine_name": "machine_name",
        "number_processors": "reported_cpu_configuration",
        "total_number_of_cores": "physical_cpu_cores",
    }
    item = entries[0]
    values = {
        key: item[key] for key in allowed if key in item and item[key] not in (None, "")
    }
    for key, value in values.items():
        if key in {"number_processors", "total_number_of_cores"}:
            continue
        records.append(
            SourceRecord(
                allowed[key],
                "system_profiler",
                f"SPHardwareDataType.{key}",
                "reported",
                value,
            )
        )
    return values


def _system_profiler_gpu_cores(
    *,
    command_runner: CommandRunner,
    records: list[SourceRecord],
    errors: list[str],
) -> int | None:
    """Return a GPU count only when SPDisplays reports one directly."""
    try:
        raw = command_runner(["system_profiler", "SPDisplaysDataType", "-json"])
        payload = json.loads(raw)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        errors.append(f"system_profiler SPDisplaysDataType: {exc}")
        records.append(
            SourceRecord(
                "gpu_core_count", "system_profiler", "SPDisplaysDataType", "error"
            )
        )
        return None
    except Exception as exc:
        errors.append(f"system_profiler SPDisplaysDataType: {exc}")
        records.append(
            SourceRecord(
                "gpu_core_count", "system_profiler", "SPDisplaysDataType", "error"
            )
        )
        return None

    displays = payload.get("SPDisplaysDataType")
    if not isinstance(displays, list):
        records.append(
            SourceRecord(
                "gpu_core_count", "system_profiler", "SPDisplaysDataType", "unknown"
            )
        )
        return None
    for display in displays:
        if not isinstance(display, dict):
            continue
        for key in ("sppci_cores", "gpu_cores"):
            value = _positive_integer(
                display.get(key),
                field="gpu_core_count",
                source=f"system_profiler {key}",
                records=records,
                errors=errors,
            )
            if value is not None:
                records.append(
                    SourceRecord(
                        "gpu_core_count",
                        "system_profiler",
                        f"SPDisplaysDataType.{key}",
                        "reported",
                        value,
                    )
                )
                return value
    records.append(
        SourceRecord(
            "gpu_core_count", "system_profiler", "SPDisplaysDataType", "unknown"
        )
    )
    return None


def collect_hardware_inventory(
    *,
    platform_provider: PlatformProvider | None = None,
    command_runner: CommandRunner | None = None,
) -> HardwareInventory:
    """Collect source facts without loading MLX or model artifacts.

    ``platform_provider`` and ``command_runner`` are intentional boundary
    injection points. They keep parser tests deterministic without replacing
    module internals or invoking the host's subprocesses.
    """
    provider = platform_provider or _NativePlatform()
    runner = command_runner or _run_command
    records: list[SourceRecord] = []
    errors: list[str] = []

    operating_system = provider.system() or None
    architecture = provider.machine() or None
    is_apple_silicon = (
        None
        if operating_system is None or architecture is None
        else operating_system == "Darwin"
        and architecture.lower() in {"arm64", "aarch64"}
    )
    records.extend(
        (
            SourceRecord(
                "operating_system",
                "platform",
                "platform.system",
                "reported" if operating_system is not None else "unknown",
                operating_system,
            ),
            SourceRecord(
                "architecture",
                "platform",
                "platform.machine",
                "reported" if architecture is not None else "unknown",
                architecture,
            ),
            SourceRecord(
                "is_apple_silicon",
                "derived",
                "platform.system+machine",
                "reported" if is_apple_silicon is not None else "unknown",
                is_apple_silicon,
            ),
        )
    )

    macos_version: str | None = None
    if operating_system == "Darwin":
        version = provider.mac_ver()[0] or None
        macos_version = version
        records.append(
            SourceRecord(
                "macos_version",
                "platform",
                "platform.mac_ver",
                "reported" if version else "unknown",
                version,
            )
        )

    if operating_system != "Darwin":
        return HardwareInventory(
            operating_system=operating_system,
            architecture=architecture,
            is_apple_silicon=is_apple_silicon,
            chip_string=None,
            machine_model=None,
            machine_name=None,
            total_unified_memory_bytes=None,
            reported_cpu_configuration=None,
            logical_cpu_cores=None,
            physical_cpu_cores=None,
            gpu_core_count=None,
            macos_version=None,
            macos_build=None,
            sources=tuple(records),
            errors=tuple(errors),
        )

    hardware = _system_profiler_hardware(
        command_runner=runner, records=records, errors=errors
    )
    memory = _positive_integer(
        _read_sysctl(
            "hw.memsize",
            field="total_unified_memory_bytes",
            command_runner=runner,
            records=records,
            errors=errors,
            record_value=False,
        ),
        field="total_unified_memory_bytes",
        source="sysctl hw.memsize",
        records=records,
        errors=errors,
    )
    logical_cpu_cores = _positive_integer(
        _read_sysctl(
            "hw.ncpu",
            field="logical_cpu_cores",
            command_runner=runner,
            records=records,
            errors=errors,
            record_value=False,
        ),
        field="logical_cpu_cores",
        source="sysctl hw.ncpu",
        records=records,
        errors=errors,
    )
    if memory is not None:
        records.append(
            SourceRecord(
                "total_unified_memory_bytes",
                "sysctl",
                "hw.memsize",
                "reported",
                memory,
            )
        )
    if logical_cpu_cores is not None:
        records.append(
            SourceRecord(
                "logical_cpu_cores",
                "sysctl",
                "hw.ncpu",
                "reported",
                logical_cpu_cores,
            )
        )
    build = _read_sysctl(
        "kern.osversion",
        field="macos_build",
        command_runner=runner,
        records=records,
        errors=errors,
    )
    chip = hardware.get("chip_type") or _read_sysctl(
        "machdep.cpu.brand_string",
        field="chip_string",
        command_runner=runner,
        records=records,
        errors=errors,
    )
    model = hardware.get("machine_model") or _read_sysctl(
        "hw.model",
        field="machine_model",
        command_runner=runner,
        records=records,
        errors=errors,
    )
    physical_cpu_cores = _positive_integer(
        hardware.get("total_number_of_cores"),
        field="physical_cpu_cores",
        source="system_profiler total_number_of_cores",
        records=records,
        errors=errors,
    )
    if physical_cpu_cores is not None:
        records.append(
            SourceRecord(
                "physical_cpu_cores",
                "system_profiler",
                "SPHardwareDataType.total_number_of_cores",
                "reported",
                physical_cpu_cores,
            )
        )

    reported_cpu_configuration = hardware.get("number_processors")
    if reported_cpu_configuration is not None:
        records.append(
            SourceRecord(
                "reported_cpu_configuration",
                "system_profiler",
                "SPHardwareDataType.number_processors",
                "reported",
                reported_cpu_configuration,
            )
        )

    return HardwareInventory(
        operating_system=operating_system,
        architecture=architecture,
        is_apple_silicon=is_apple_silicon,
        chip_string=chip,
        machine_model=model,
        machine_name=hardware.get("machine_name"),
        total_unified_memory_bytes=memory,
        reported_cpu_configuration=reported_cpu_configuration,
        logical_cpu_cores=logical_cpu_cores,
        physical_cpu_cores=physical_cpu_cores,
        gpu_core_count=_system_profiler_gpu_cores(
            command_runner=runner, records=records, errors=errors
        ),
        macos_version=macos_version,
        macos_build=build,
        sources=tuple(records),
        errors=tuple(errors),
    )
