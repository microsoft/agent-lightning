# Copyright (c) Microsoft. All rights reserved.

"""Tests for PromptResourceCodec round-trip, engine preservation, and error cases."""

from __future__ import annotations

import pytest

from agentlightning.algorithm.gepa.resources import PromptResourceCodec
from agentlightning.types import LLM, NamedResources, PromptTemplate


def _make_resources(template: str = "Hello {name}", engine: str = "f-string") -> NamedResources:
    return {"system_prompt": PromptTemplate(template=template, engine=engine)}  # type: ignore[dict-item]


class TestPromptResourceCodec:
    def test_round_trip(self):
        resources = _make_resources()
        codec = PromptResourceCodec(resource_name="system_prompt", engine="f-string")

        candidate = codec.resources_to_candidate(resources)
        assert candidate == {"system_prompt": "Hello {name}"}

        rebuilt = codec.candidate_to_resources(candidate)
        assert isinstance(rebuilt["system_prompt"], PromptTemplate)
        assert rebuilt["system_prompt"].template == "Hello {name}"  # type: ignore[union-attr]
        assert rebuilt["system_prompt"].engine == "f-string"  # type: ignore[union-attr]

    def test_engine_preservation(self):
        resources: NamedResources = {
            "jinja_prompt": PromptTemplate(template="Hello {{ name }}", engine="jinja"),  # type: ignore[dict-item]
        }
        codec = PromptResourceCodec(resource_name="jinja_prompt", engine="jinja")
        candidate = codec.resources_to_candidate(resources)
        rebuilt = codec.candidate_to_resources(candidate)
        assert rebuilt["jinja_prompt"].engine == "jinja"  # type: ignore[union-attr]

    def test_resources_to_candidate_missing_key(self):
        codec = PromptResourceCodec(resource_name="missing", engine="f-string")
        with pytest.raises(KeyError):
            codec.resources_to_candidate({})

    def test_resources_to_candidate_wrong_type(self):
        resources: NamedResources = {
            "llm_resource": LLM(endpoint="http://localhost", model="test"),
        }
        codec = PromptResourceCodec(resource_name="llm_resource", engine="f-string")
        with pytest.raises(TypeError, match="not a PromptTemplate"):
            codec.resources_to_candidate(resources)

    def test_candidate_to_resources_missing_key(self):
        codec = PromptResourceCodec(resource_name="prompt", engine="f-string")
        with pytest.raises(KeyError):
            codec.candidate_to_resources({"wrong_key": "text"})


class TestFromInitialResources:
    def test_auto_detect(self):
        resources = _make_resources()
        codec, seed = PromptResourceCodec.from_initial_resources(resources)
        assert codec.resource_name == "system_prompt"
        assert codec.engine == "f-string"
        assert seed == {"system_prompt": "Hello {name}"}

    def test_explicit_resource_name(self):
        resources: NamedResources = {
            "first": LLM(endpoint="http://localhost", model="test"),
            "second": PromptTemplate(template="Greet {user}", engine="f-string"),  # type: ignore[dict-item]
        }
        codec, seed = PromptResourceCodec.from_initial_resources(resources, resource_name="second")
        assert codec.resource_name == "second"
        assert seed == {"second": "Greet {user}"}

    def test_no_prompt_template_raises(self):
        resources: NamedResources = {
            "llm": LLM(endpoint="http://localhost", model="test"),
        }
        with pytest.raises(ValueError, match="No PromptTemplate found"):
            PromptResourceCodec.from_initial_resources(resources)

    def test_explicit_name_not_found_raises(self):
        resources = _make_resources()
        with pytest.raises(ValueError, match="not found"):
            PromptResourceCodec.from_initial_resources(resources, resource_name="nonexistent")

    def test_explicit_name_wrong_type_raises(self):
        resources: NamedResources = {
            "llm": LLM(endpoint="http://localhost", model="test"),
        }
        with pytest.raises(ValueError, match="not a PromptTemplate"):
            PromptResourceCodec.from_initial_resources(resources, resource_name="llm")
