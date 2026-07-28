# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_mlx.product_workflows import ProductWorkflowError, install_to_code


def test_golden_install_to_code_emits_auth_aware_configuration(
    product_catalog, product_client
):
    result = install_to_code(
        product_client,
        product_catalog,
        profile_id="qwen3.6-35b-a3b-8bit",
        profile_revision=1,
        install_idempotency_key="install-golden-1",
        activate_idempotency_key="activate-golden-1",
        coding_client="openai",
        runtime_api_key_configured=True,
    )

    assert result["workflow"] == "install_to_code"
    assert result["coding_configuration"]["environment"] == {
        "OPENAI_BASE_URL": "http://127.0.0.1:8080/v1"
    }
    assert result["coding_configuration"]["authentication"]["required"] is True
    assert all(call[0] != "chat" for call in product_client.calls)


def test_golden_workflow_stops_on_failed_operation(product_catalog, product_client):
    product_client.operations["install-1"] = [
        {
            "operation_id": "install-1",
            "status": "failed",
            "profile": product_client.profile_reference,
        }
    ]
    with pytest.raises(ProductWorkflowError, match="status failed"):
        install_to_code(
            product_client,
            product_catalog,
            profile_id="qwen3.6-35b-a3b-8bit",
            profile_revision=1,
            install_idempotency_key="install-golden-1",
            activate_idempotency_key="activate-golden-1",
            coding_client="openai",
        )
    assert all(call[0] != "activate" for call in product_client.calls)


def test_golden_workflow_stops_when_activation_fails(product_catalog, product_client):
    product_client.operations["activate-1"] = [
        {
            "operation_id": "activate-1",
            "status": "failed",
            "profile": product_client.profile_reference,
        }
    ]
    with pytest.raises(ProductWorkflowError, match="status failed"):
        install_to_code(
            product_client,
            product_catalog,
            profile_id="qwen3.6-35b-a3b-8bit",
            profile_revision=1,
            install_idempotency_key="install-golden-1",
            activate_idempotency_key="activate-golden-1",
            coding_client="openai",
        )
    assert all(call[0] != "status" for call in product_client.calls)


def test_golden_workflow_rejects_substituted_operation(product_catalog, product_client):
    product_client.operations["install-1"] = [
        {
            "operation_id": "other-operation",
            "status": "succeeded",
            "profile": product_client.profile_reference,
        }
    ]
    with pytest.raises(ProductWorkflowError, match="substituted"):
        install_to_code(
            product_client,
            product_catalog,
            profile_id="qwen3.6-35b-a3b-8bit",
            profile_revision=1,
            install_idempotency_key="install-golden-1",
            activate_idempotency_key="activate-golden-1",
            coding_client="openai",
        )


def test_golden_workflow_rejects_wrong_profile_operation(
    product_catalog, product_client
):
    product_client.operations["install-1"] = [
        {
            "operation_id": "install-1",
            "status": "succeeded",
            "profile": {
                "profile_id": "other",
                "profile_revision": 1,
                "subject_digest": "0" * 64,
            },
        }
    ]
    with pytest.raises(ProductWorkflowError, match="not bound"):
        install_to_code(
            product_client,
            product_catalog,
            profile_id="qwen3.6-35b-a3b-8bit",
            profile_revision=1,
            install_idempotency_key="install-golden-1",
            activate_idempotency_key="activate-golden-1",
            coding_client="openai",
        )


def test_golden_workflow_cancels_and_reports_recovery_on_poll_exhaustion(
    product_catalog, product_client
):
    product_client.operations["install-1"] = [
        {
            "operation_id": "install-1",
            "status": "running",
            "profile": product_client.profile_reference,
        }
    ]
    with pytest.raises(ProductWorkflowError, match="did not finish") as caught:
        install_to_code(
            product_client,
            product_catalog,
            profile_id="qwen3.6-35b-a3b-8bit",
            profile_revision=1,
            install_idempotency_key="install-golden-1",
            activate_idempotency_key="activate-golden-1",
            coding_client="openai",
            max_operation_polls=1,
        )
    assert caught.value.recovery["operation_id"] == "install-1"
    assert caught.value.recovery["cancellation"]["status"] == "cancelled"
    assert any(call[0] == "cancel" for call in product_client.calls)


def test_golden_workflow_validates_final_polled_record_before_cancellation(
    product_catalog, product_client
):
    product_client.operations["install-1"] = [
        {
            "operation_id": "substituted",
            "status": "running",
            "profile": product_client.profile_reference,
        }
    ]
    with pytest.raises(ProductWorkflowError, match="substituted"):
        install_to_code(
            product_client,
            product_catalog,
            profile_id="qwen3.6-35b-a3b-8bit",
            profile_revision=1,
            install_idempotency_key="install-golden-1",
            activate_idempotency_key="activate-golden-1",
            coding_client="openai",
            max_operation_polls=1,
        )
    assert all(call[0] != "cancel" for call in product_client.calls)


def test_golden_workflow_rejects_unbound_cancellation_record(
    product_catalog, product_client
):
    product_client.operations["install-1"] = [
        {
            "operation_id": "install-1",
            "status": "running",
            "profile": product_client.profile_reference,
        }
    ]

    def wrong_cancel(operation_id, idempotency_key):
        return {
            "operation_id": "other",
            "status": "cancelled",
            "profile": product_client.profile_reference,
        }

    product_client.cancel_operation = wrong_cancel
    with pytest.raises(ProductWorkflowError, match="did not finish") as caught:
        install_to_code(
            product_client,
            product_catalog,
            profile_id="qwen3.6-35b-a3b-8bit",
            profile_revision=1,
            install_idempotency_key="install-golden-1",
            activate_idempotency_key="activate-golden-1",
            coding_client="openai",
            max_operation_polls=1,
        )
    assert caught.value.recovery["cancellation_error"]["type"] == (
        "ProductWorkflowError"
    )


def test_golden_workflow_replay_uses_same_idempotency_keys(
    product_catalog, product_client
):
    arguments = {
        "profile_id": "qwen3.6-35b-a3b-8bit",
        "profile_revision": 1,
        "install_idempotency_key": "install-golden-1",
        "activate_idempotency_key": "activate-golden-1",
        "coding_client": "openai",
    }
    first = install_to_code(product_client, product_catalog, **arguments)
    second = install_to_code(product_client, product_catalog, **arguments)
    install_calls = [call for call in product_client.calls if call[0] == "install"]
    activate_calls = [call for call in product_client.calls if call[0] == "activate"]
    assert [call[2] for call in install_calls] == ["install-golden-1"] * 2
    assert [call[2] for call in activate_calls] == ["activate-golden-1"] * 2
    assert product_client.operation_creations == ["install-1", "activate-1"]
    assert second["install_operation"] == first["install_operation"]
    assert second["activate_operation"] == first["activate_operation"]


def test_golden_workflow_rejects_missing_active_endpoint(
    product_catalog, product_client
):
    original_status = product_client.status

    def status_without_endpoint():
        status = original_status()
        status.pop("endpoint")
        return status

    product_client.status = status_without_endpoint
    with pytest.raises(ProductWorkflowError, match="inference endpoint"):
        install_to_code(
            product_client,
            product_catalog,
            profile_id="qwen3.6-35b-a3b-8bit",
            profile_revision=1,
            install_idempotency_key="install-golden-1",
            activate_idempotency_key="activate-golden-1",
            coding_client="openai",
        )
