# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import pytest

from vllm_mlx.catalog import load_catalog


@pytest.fixture
def product_catalog(tmp_path):
    root = Path(__file__).parents[2]
    profile = copy.deepcopy(
        load_catalog(root / "catalog" / "profiles").get("qwen3.6-35b-a3b-8bit", 1)
    )
    profile["qualification"] = {
        "status": "qualified",
        "reason": "Test-only qualification overlay for workflow behavior.",
        "evidence": [
            {
                "evidence_id": "qwen-catalog-workflow-test",
                "kind": "workflow_test",
                "location": "tests/product_workflows",
                "artifact_sha256": "1" * 64,
                "result": "pass",
                "hardware_fingerprint": "test-host",
                "workload_id": "install-to-code",
                "subject_digest": profile["subject_digest"],
                "created_at": "2026-07-21T00:00:00Z",
            }
        ],
    }
    (tmp_path / "qwen3.6-35b-a3b-8bit.json").write_text(json.dumps(profile))
    return load_catalog(tmp_path)


class FakeProductClient:
    def __init__(self, profile_reference):
        self.profile_reference = profile_reference
        self.calls = []
        self.idempotency_ledger = {}
        self.operation_creations = []
        self.operations = {
            "install-1": [
                {
                    "operation_id": "install-1",
                    "status": "running",
                    "profile": profile_reference,
                },
                {
                    "operation_id": "install-1",
                    "status": "succeeded",
                    "profile": profile_reference,
                },
            ],
            "activate-1": [
                {
                    "operation_id": "activate-1",
                    "status": "succeeded",
                    "profile": profile_reference,
                }
            ],
        }

    def install(self, profile, idempotency_key):
        self.calls.append(("install", profile, idempotency_key))
        ledger_key = ("install", idempotency_key)
        if ledger_key in self.idempotency_ledger:
            operation_id = self.idempotency_ledger[ledger_key]
            return dict(self.operations[operation_id][-1])
        self.idempotency_ledger[ledger_key] = "install-1"
        self.operation_creations.append("install-1")
        return {
            "operation_id": "install-1",
            "status": "queued",
            "profile": profile,
        }

    def activate(self, profile, idempotency_key, *, overrides=None):
        self.calls.append(("activate", profile, idempotency_key, overrides))
        ledger_key = ("activate", idempotency_key)
        if ledger_key in self.idempotency_ledger:
            operation_id = self.idempotency_ledger[ledger_key]
            return dict(self.operations[operation_id][-1])
        self.idempotency_ledger[ledger_key] = "activate-1"
        self.operation_creations.append("activate-1")
        return {
            "operation_id": "activate-1",
            "status": "running",
            "profile": profile,
        }

    def operation(self, operation_id):
        self.calls.append(("operation", operation_id))
        records = self.operations[operation_id]
        return records.pop(0) if len(records) > 1 else records[0]

    def cancel_operation(self, operation_id, idempotency_key):
        self.calls.append(("cancel", operation_id, idempotency_key))
        return {
            "operation_id": operation_id,
            "status": "cancelled",
            "profile": self.profile_reference,
        }

    def status(self):
        self.calls.append(("status",))
        return {
            "active_profile": self.profile_reference,
            "state": "loaded",
            "healthy": True,
            "endpoint": "http://127.0.0.1:8080",
        }

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        return {"choices": [{"message": {"content": "ready"}}]}


@pytest.fixture
def profile_reference(product_catalog):
    profile = product_catalog.get("qwen3.6-35b-a3b-8bit", 1)
    return {
        "profile_id": profile["profile_id"],
        "profile_revision": profile["profile_revision"],
        "subject_digest": profile["subject_digest"],
    }


@pytest.fixture
def product_client(profile_reference):
    return FakeProductClient(profile_reference)
