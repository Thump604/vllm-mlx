# SPDX-License-Identifier: Apache-2.0
"""Tests for the Gemma 4 LiteRT MTP adapter contract builder."""

from vllm_mlx.patches.gemma4_litert_mtp import build_gemma4_litert_mtp_contract


def test_build_gemma4_litert_mtp_contract():
    interface = {
        "signature_defs": [
            {
                "key": "mtp_drafter",
                "subgraph_index": 0,
            }
        ],
        "main_subgraph": {
            "inputs": [
                {"name": "mtp_drafter_input_pos:0", "shape": [1], "type": "INT32"},
                {
                    "name": "mtp_drafter_activations:0",
                    "shape": [1, 1, 3072],
                    "type": "FLOAT32",
                },
                {
                    "name": "mtp_drafter_param_tensor:0",
                    "shape": [1, 1, 1, 7],
                    "type": "INT32",
                },
                {
                    "name": "mtp_drafter_kv_cache_k_14:0",
                    "shape": [1, 1, 32003, 512],
                    "type": "INT8",
                },
                {
                    "name": "mtp_drafter_kv_cache_k_13:0",
                    "shape": [1, 1, 32003, 256],
                    "type": "INT8",
                },
                {
                    "name": "mtp_drafter_mask:0",
                    "shape": [1, 1, 1, 32003],
                    "type": "BOOL",
                },
                {
                    "name": "mtp_drafter_kv_cache_v_13:0",
                    "shape": [1, 1, 256, 32003],
                    "type": "INT8",
                },
                {
                    "name": "mtp_drafter_kv_cache_v_14:0",
                    "shape": [1, 1, 512, 32003],
                    "type": "INT8",
                },
            ],
            "outputs": [
                {
                    "name": "StatefulPartitionedCall:0",
                    "shape": [1, 1, 262144],
                    "type": "FLOAT32",
                },
                {
                    "name": "StatefulPartitionedCall:1",
                    "shape": [1, 1, 1536],
                    "type": "FLOAT32",
                },
            ],
        },
    }
    model_config = {
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "vocab_size": 262144,
            "num_hidden_layers": 35,
            "hidden_size_per_layer_input": 256,
        }
    }

    contract = build_gemma4_litert_mtp_contract(interface, model_config)

    assert contract.signature_key == "mtp_drafter"
    assert contract.model_type == "gemma4_text"
    assert contract.activations_input_size == 3072
    assert contract.projected_activations_size == 1536
    assert contract.logits_vocab_size == 262144
    assert contract.next_token_ids_required is True
    assert contract.requires_draft_token_embedding_lookup is True
    assert contract.requires_projected_activations_feedback is True
    assert contract.requires_explicit_kv_cache_adapter is True
    assert contract.requires_external_verifier_loop is True
    assert (
        contract.activations_formula
        == "concat(next_token_embedding, projected_activations)"
    )
    assert contract.projected_state_cache_field == "projected_activations"
    assert len(contract.kv_cache_specs) == 2
    assert [spec.layer_index for spec in contract.kv_cache_specs] == [13, 14]
    assert [spec.runtime_source_layer_index for spec in contract.kv_cache_specs] == [
        13,
        14,
    ]
    assert [spec.source_layer_type for spec in contract.kv_cache_specs] == [
        "sliding_attention",
        "full_attention",
    ]
    assert contract.kv_cache_specs[0].head_dim == 256
    assert contract.kv_cache_specs[1].head_dim == 512
    assert contract.kv_cache_specs[0].time_capacity == 32003
    assert contract.kv_cache_specs[0].key_layout == "BHTD"
    assert contract.kv_cache_specs[0].value_layout == "BHDT"
