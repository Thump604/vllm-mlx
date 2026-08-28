import json

from vllm_mlx.server import _coerce_tool_arguments


def _tool_schema(properties):
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "parameters": {"type": "object", "properties": properties},
            },
        }
    ]


def test_coerces_json_encoded_array_and_integer_to_declared_types():
    arguments = json.dumps(
        {
            "argv": '["/usr/bin/printf", "transport-ok\\n"]',
            "timeout_seconds": "1",
        }
    )
    tools = _tool_schema(
        {
            "argv": {"type": "array", "items": {"type": "string"}},
            "timeout_seconds": {"type": "integer"},
        }
    )

    assert json.loads(_coerce_tool_arguments(arguments, "terminal", tools)) == {
        "argv": ["/usr/bin/printf", "transport-ok\n"],
        "timeout_seconds": 1,
    }


def test_preserves_invalid_value_when_conversion_is_not_lossless():
    arguments = json.dumps({"argv": "not-json", "timeout_seconds": "1.5"})
    tools = _tool_schema(
        {
            "argv": {"type": "array"},
            "timeout_seconds": {"type": "integer"},
        }
    )

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_preserves_value_that_already_matches_any_union_type():
    arguments = json.dumps({"timeout_seconds": "1"})
    tools = _tool_schema({"timeout_seconds": {"type": ["integer", "string"]}})

    assert _coerce_tool_arguments(arguments, "terminal", tools) == arguments


def test_preserves_existing_object_to_string_normalization():
    arguments = json.dumps({"content": {"answer": 42}})
    tools = _tool_schema({"content": {"type": "string"}})

    normalized = json.loads(_coerce_tool_arguments(arguments, "terminal", tools))
    assert json.loads(normalized["content"]) == {"answer": 42}


def test_coerces_json_boolean_without_treating_it_as_integer():
    arguments = json.dumps({"enabled": "true", "count": True})
    tools = _tool_schema({"enabled": {"type": "boolean"}, "count": {"type": "integer"}})

    assert json.loads(_coerce_tool_arguments(arguments, "terminal", tools)) == {
        "enabled": True,
        "count": True,
    }
