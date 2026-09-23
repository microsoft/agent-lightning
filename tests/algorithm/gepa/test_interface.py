# Copyright (c) Microsoft. All rights reserved.

"""Tests for the GEPA Algorithm subclass."""

from __future__ import annotations

import pytest

from agentlightning.algorithm.gepa.config import GEPAConfig
from agentlightning.algorithm.gepa.interface import GEPA
from agentlightning.types import LLM, NamedResources, PromptTemplate


class TestInit:
    def test_default_config(self):
        algo = GEPA()
        assert algo.config.max_metric_calls is None
        assert algo.config.candidate_selection_strategy == "pareto"

    def test_custom_config(self):
        config = GEPAConfig(max_metric_calls=20, seed=42)
        algo = GEPA(config=config)
        assert algo.config.max_metric_calls == 20
        assert algo.config.seed == 42

    def test_resource_name_stored(self):
        algo = GEPA(resource_name="my_prompt")
        assert algo._resource_name == "my_prompt"

    def test_result_is_none_before_run(self):
        algo = GEPA()
        assert algo.result is None


class TestGetBestPrompt:
    def test_raises_before_run(self):
        algo = GEPA()
        with pytest.raises(ValueError, match="run\\(\\) has not been called"):
            algo.get_best_prompt()

    def test_returns_prompt_after_setting(self):
        algo = GEPA()
        prompt = PromptTemplate(template="Best prompt", engine="f-string")
        algo._best_prompt = prompt
        assert algo.get_best_prompt() is prompt


class TestRunValidation:
    @pytest.mark.asyncio
    async def test_requires_train_dataset(self):
        algo = GEPA()
        algo.set_store(_make_mock_store())
        algo.set_initial_resources(_make_initial_resources())
        with pytest.raises(ValueError, match="train_dataset is required"):
            await algo.run(train_dataset=None)

    @pytest.mark.asyncio
    async def test_requires_initial_resources(self):
        algo = GEPA()
        algo.set_store(_make_mock_store())
        with pytest.raises(ValueError, match="initial_resources are not set"):
            await algo.run(train_dataset=["task1", "task2"])

    def test_codec_auto_detect(self):
        """Verify that the codec is properly built from initial resources."""
        from agentlightning.algorithm.gepa.resources import PromptResourceCodec

        resources = _make_initial_resources()
        codec, seed = PromptResourceCodec.from_initial_resources(resources)
        assert codec.resource_name == "system_prompt"
        assert seed == {"system_prompt": "You are helpful."}

    def test_codec_explicit_name(self):
        """Verify explicit resource name selection works."""
        from agentlightning.algorithm.gepa.resources import PromptResourceCodec

        resources: NamedResources = {
            "llm": LLM(endpoint="http://localhost", model="test"),
            "prompt": PromptTemplate(template="Greet {user}", engine="f-string"),  # type: ignore[dict-item]
        }
        codec, seed = PromptResourceCodec.from_initial_resources(resources, resource_name="prompt")
        assert codec.resource_name == "prompt"
        assert seed == {"prompt": "Greet {user}"}


class TestIsAsync:
    def test_gepa_run_is_async(self):
        algo = GEPA()
        assert algo.is_async() is True


# ---------- Helpers ----------


def _make_mock_store():
    from unittest.mock import MagicMock

    from agentlightning.store.base import LightningStore

    return MagicMock(spec=LightningStore)


def _make_initial_resources() -> NamedResources:
    return {
        "system_prompt": PromptTemplate(template="You are helpful.", engine="f-string"),  # type: ignore[dict-item]
    }
