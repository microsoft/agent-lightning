# Copyright (c) Microsoft. All rights reserved.

from pydantic import TypeAdapter

import agentlightning.types.resources as resources
from agentlightning.types import LLM, PromptTemplate


def test_resources_do_not_expose_an_unconnected_runtime_registry() -> None:
    assert not hasattr(resources, "register_resource_type")
    assert not hasattr(resources, "get_resource_type")
    assert not hasattr(resources, "get_registered_resource_types")


def test_named_resources_deserializes_builtin_types() -> None:
    named_resources = TypeAdapter(resources.NamedResources).validate_python(
        {
            "main": {"resource_type": "llm", "endpoint": "http://localhost", "model": "test"},
            "prompt": {"resource_type": "prompt_template", "template": "Hello {name}", "engine": "f-string"},
        }
    )

    assert isinstance(named_resources["main"], LLM)
    assert isinstance(named_resources["prompt"], PromptTemplate)
