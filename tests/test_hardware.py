# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for the read-only hardware inventory boundary."""

from __future__ import annotations

import json
import platform

import pytest

from vllm_mlx.hardware import collect_hardware_inventory


class StaticPlatform:
    def __init__(
        self, system: str = "Darwin", machine: str = "arm64", version: str = "15.5"
    ):
        self._system = system
        self._machine = machine
        self._version = version

    def system(self) -> str:
        return self._system

    def machine(self) -> str:
        return self._machine

    def mac_ver(self) -> tuple[str, tuple[str, str, str], str]:
        return (self._version, ("", "", ""), "")


def command_map(values: dict[tuple[str, ...], str]):
    def run(command: list[str]) -> str:
        return values[tuple(command)]

    return run


def complete_commands(
    *, hardware: dict | None = None, displays: dict | None = None
) -> dict[tuple[str, ...], str]:
    return {
        ("sysctl", "-n", "hw.memsize"): "137438953472\n",
        ("sysctl", "-n", "hw.ncpu"): "16\n",
        ("sysctl", "-n", "kern.osversion"): "24F74\n",
        ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M4 Max\n",
        ("sysctl", "-n", "hw.model"): "Mac16,3\n",
        ("system_profiler", "SPHardwareDataType", "-json"): json.dumps(
            hardware
            or {
                "SPHardwareDataType": [
                    {
                        "chip_type": "Apple M4 Max",
                        "machine_model": "Mac16,3",
                        "machine_name": "MacBook Pro",
                        "number_processors": "1",
                        "total_number_of_cores": "16",
                        "serial_number": "must-not-leak",
                        "platform_UUID": "must-not-leak",
                    }
                ]
            }
        ),
        ("system_profiler", "SPDisplaysDataType", "-json"): json.dumps(
            displays
            or {
                "SPDisplaysDataType": [
                    {"sppci_cores": "40", "spdisplays_vendor": "Apple"}
                ]
            }
        ),
    }


def test_inventory_collects_only_direct_source_facts():
    inventory = collect_hardware_inventory(
        platform_provider=StaticPlatform(),
        command_runner=command_map(complete_commands()),
    )

    assert inventory.operating_system == "Darwin"
    assert inventory.architecture == "arm64"
    assert inventory.is_apple_silicon is True
    assert inventory.chip_string == "Apple M4 Max"
    assert inventory.machine_model == "Mac16,3"
    assert inventory.machine_name == "MacBook Pro"
    assert inventory.total_unified_memory_bytes == 137438953472
    assert inventory.reported_cpu_configuration == "1"
    assert inventory.logical_cpu_cores == 16
    assert inventory.physical_cpu_cores == 16
    assert inventory.gpu_core_count == 40
    assert inventory.macos_version == "15.5"
    assert inventory.macos_build == "24F74"
    assert inventory.errors == ()

    records_by_field = {}
    for record in inventory.sources:
        records_by_field.setdefault(record.field, []).append(record)
    for field, value in inventory.to_dict().items():
        if field in {"sources", "errors"} or value is None:
            continue
        assert any(
            record.status == "reported" and record.value == value
            for record in records_by_field[field]
        ), field

    assert any(
        record.field == "total_unified_memory_bytes" and record.locator == "hw.memsize"
        for record in inventory.sources
    )


def test_inventory_whitelists_system_profiler_and_does_not_emit_device_identifiers():
    inventory = collect_hardware_inventory(
        platform_provider=StaticPlatform(),
        command_runner=command_map(complete_commands()),
    )
    encoded = json.dumps(inventory.to_dict())

    assert "must-not-leak" not in encoded
    assert "serial_number" not in encoded
    assert "platform_UUID" not in encoded


def test_inventory_preserves_unknowns_and_reports_command_failures():
    commands = complete_commands(
        hardware={"SPHardwareDataType": [{"serial_number": "secret"}]},
        displays={"SPDisplaysDataType": [{}]},
    )
    del commands[("sysctl", "-n", "hw.memsize")]

    inventory = collect_hardware_inventory(
        platform_provider=StaticPlatform(), command_runner=command_map(commands)
    )

    assert inventory.total_unified_memory_bytes is None
    assert inventory.physical_cpu_cores is None
    assert inventory.gpu_core_count is None
    assert any(error.startswith("sysctl hw.memsize:") for error in inventory.errors)
    assert any(
        record.field == "gpu_core_count" and record.status == "unknown"
        for record in inventory.sources
    )


def test_inventory_rejects_invalid_direct_numeric_facts_without_defaults():
    commands = complete_commands()
    commands[("sysctl", "-n", "hw.ncpu")] = "not-a-number\n"
    commands[("system_profiler", "SPDisplaysDataType", "-json")] = json.dumps(
        {"SPDisplaysDataType": [{"sppci_cores": "-1"}]}
    )

    inventory = collect_hardware_inventory(
        platform_provider=StaticPlatform(), command_runner=command_map(commands)
    )

    assert inventory.logical_cpu_cores is None
    assert inventory.gpu_core_count is None
    assert any("logical_cpu_cores" in error for error in inventory.errors)
    assert any("gpu_core_count" in error for error in inventory.errors)


def test_inventory_reports_malformed_system_profiler_payloads_without_fallbacks():
    commands = complete_commands()
    commands[("system_profiler", "SPHardwareDataType", "-json")] = "not-json"
    commands[("system_profiler", "SPDisplaysDataType", "-json")] = "not-json"

    inventory = collect_hardware_inventory(
        platform_provider=StaticPlatform(), command_runner=command_map(commands)
    )

    assert inventory.chip_string == "Apple M4 Max"
    assert inventory.machine_model == "Mac16,3"
    assert inventory.machine_name is None
    assert inventory.physical_cpu_cores is None
    assert inventory.gpu_core_count is None
    assert any("SPHardwareDataType" in error for error in inventory.errors)
    assert any("SPDisplaysDataType" in error for error in inventory.errors)
    assert any(
        record.field == "chip_string"
        and record.locator == "machdep.cpu.brand_string"
        and record.value == "Apple M4 Max"
        for record in inventory.sources
    )
    assert any(
        record.field == "machine_model"
        and record.locator == "hw.model"
        and record.value == "Mac16,3"
        for record in inventory.sources
    )


def test_non_apple_host_is_not_classified_as_apple_silicon():
    calls: list[list[str]] = []

    def runner(command: list[str]) -> str:
        calls.append(command)
        raise AssertionError("non-macOS inventory must not run macOS probes")

    inventory = collect_hardware_inventory(
        platform_provider=StaticPlatform(system="Linux", machine="aarch64", version=""),
        command_runner=runner,
    )

    assert inventory.is_apple_silicon is False
    assert inventory.macos_version is None
    assert inventory.total_unified_memory_bytes is None
    assert calls == []


@pytest.mark.skipif(
    platform.system() != "Darwin", reason="requires native macOS probes"
)
def test_native_macos_inventory_smoke():
    inventory = collect_hardware_inventory()

    assert inventory.operating_system == "Darwin"
    assert inventory.architecture
    assert inventory.macos_version
    assert isinstance(inventory.sources, tuple)
