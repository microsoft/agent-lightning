# Copyright (c) Microsoft. All rights reserved.

from types import ModuleType

import pytest
from omegaconf import OmegaConf

from examples.gsm8k import train_gsm8k_agent
from examples.search_r1 import train_search_r1_agent


@pytest.mark.parametrize(
    "module,api_parameter,tokenizer_key,default_model",
    [
        (train_gsm8k_agent, "api", "GSM8K_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        (train_search_r1_agent, "api_type", "SEARCH_R1_TOKENIZER_MODEL", "meta-llama/Llama-3.2-3B-Instruct"),
    ],
    ids=["gsm8k", "search-r1"],
)
@pytest.mark.parametrize(
    "model,model_override,tokenizer_override",
    [
        (None, None, None),
        ("test/flag-model", None, None),
        (None, "test/override-model", None),
        ("test/flag-model", "test/override-model", None),
        ("test/flag-model", "test/override-model", "test/custom-tokenizer"),
    ],
    ids=["default", "model-flag", "model-override", "override-after-flag", "explicit-tokenizer"],
)
def test_completion_tokenizer_follows_final_model(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    api_parameter: str,
    tokenizer_key: str,
    default_model: str,
    model: str | None,
    model_override: str | None,
    tokenizer_override: str | None,
) -> None:
    # Isolate example overrides from VERL's optional training configuration.
    monkeypatch.setattr(module, "compose", lambda **kwargs: OmegaConf.create({}))
    config_overrides: list[str] = []
    if model_override is not None:
        config_overrides.append(f"actor_rollout_ref.model.path={model_override}")
    if tokenizer_override is not None:
        config_overrides.append(f"agentlightning.local.env_map.{tokenizer_key}={tokenizer_override}")

    config = module.build_config(
        model=model,
        config_overrides=config_overrides,
        **{api_parameter: "completion"},
    )

    expected_model = model_override or model or default_model
    assert config.actor_rollout_ref.model.path == expected_model
    env_map = OmegaConf.to_container(config.agentlightning.local.env_map, resolve=True)
    assert isinstance(env_map, dict)
    assert env_map[tokenizer_key] == (tokenizer_override or expected_model)
