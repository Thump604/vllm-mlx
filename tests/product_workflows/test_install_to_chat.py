# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_mlx.product_workflows import ProductWorkflowError, install_to_chat


def test_golden_install_to_chat_uses_exact_identity_and_waits_for_activation(
    product_catalog, product_client, profile_reference
):
    result = install_to_chat(
        product_client,
        product_catalog,
        profile_id="golden-model",
        profile_revision=1,
        install_idempotency_key="install-golden-1",
        activate_idempotency_key="activate-golden-1",
        message="hello",
    )

    assert result["profile_reference"] == profile_reference
    assert result["install_operation"]["status"] == "succeeded"
    assert result["activate_operation"]["status"] == "succeeded"
    assert result["chat"]["choices"][0]["message"]["content"] == "ready"
    assert product_client.calls[0] == (
        "install",
        profile_reference,
        "install-golden-1",
    )
    assert product_client.calls[-1] == (
        "chat",
        {
            "model": result["profile"]["identity"]["served_model_name"],
            "message": "hello",
            "stream": False,
        },
    )


def test_golden_chat_fails_before_inference_when_active_identity_drifts(
    product_catalog, product_client
):
    product_client.profile_reference = {
        "profile_id": "other",
        "profile_revision": 1,
        "subject_digest": "0" * 64,
    }
    with pytest.raises(ProductWorkflowError, match="active profile"):
        install_to_chat(
            product_client,
            product_catalog,
            profile_id="golden-model",
            profile_revision=1,
            install_idempotency_key="install-golden-1",
            activate_idempotency_key="activate-golden-1",
            message="must not send",
        )
    assert all(call[0] != "chat" for call in product_client.calls)


@pytest.mark.parametrize(
    "status_change",
    [{"state": "failed"}, {"healthy": False}],
)
def test_golden_chat_requires_loaded_healthy_runtime(
    product_catalog, product_client, status_change
):
    original_status = product_client.status

    def unhealthy_status():
        return {**original_status(), **status_change}

    product_client.status = unhealthy_status
    with pytest.raises(ProductWorkflowError, match="loaded and healthy"):
        install_to_chat(
            product_client,
            product_catalog,
            profile_id="golden-model",
            profile_revision=1,
            install_idempotency_key="install-golden-1",
            activate_idempotency_key="activate-golden-1",
            message="must not send",
        )
    assert all(call[0] != "chat" for call in product_client.calls)
