# Copyright (c) Microsoft. All rights reserved.

from typing import Literal

import pytest
from pydantic import TypeAdapter

import agentlightning.types.resources as resources
from agentlightning.types import LLM, PromptTemplate, Resource


def test_resource_registry_contains_builtin_types() -> None:
    registered = resources.get_registered_resource_types()

    assert registered["llm"] is LLM
    assert registered["proxy_llm"] is resources.ProxyLLM
    assert registered["prompt_template"] is PromptTemplate
    assert resources.get_resource_type("llm") is LLM

    registered.clear()

    assert resources.get_resource_type("llm") is LLM


def test_register_resource_type_rejects_conflicting_resource_type() -> None:
    class AlternateLLM(Resource):
        resource_type: Literal["llm"] = "llm"

    with pytest.raises(ValueError, match="already registered"):
        resources.register_resource_type(AlternateLLM)


def test_register_resource_type_adds_model_to_registry() -> None:
    class ToolResource(Resource):
        resource_type: Literal["tool"] = "tool"
        command: str

    resources.register_resource_type(ToolResource)

    assert resources.get_resource_type("tool") is ToolResource


def test_named_resources_deserializes_registered_builtin_types() -> None:
    named_resources = TypeAdapter(resources.NamedResources).validate_python(
        {
            "main": {"resource_type": "llm", "endpoint": "http://localhost", "model": "test"},
            "prompt": {"resource_type": "prompt_template", "template": "Hello {name}", "engine": "f-string"},
        }
    )

    assert isinstance(named_resources["main"], LLM)
    assert isinstance(named_resources["prompt"], PromptTemplate)
