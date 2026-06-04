from __future__ import annotations

from examples.search_r1 import train_search_r1_agent


def test_build_config_ci_sets_local_agent_and_small_batches() -> None:
    config = train_search_r1_agent.build_config(
        model="test-model",
        agl_base_url="http://agl",
        agl_key="secret",
        run_name="unit",
        ci=True,
    )

    assert config.actor_rollout_ref.model.path == "test-model"
    assert config.agentlightning.agl_base_url == "http://agl"
    assert config.agentlightning.agl_key == "secret"
    assert config.trainer.experiment_name == "search_r1_ci_unit"
    assert config.trainer.n_gpus_per_node == 1
    assert config.trainer.logger == ["console"]
    assert config.trainer.total_training_steps == 1
    assert config.data.train_batch_size == 2
    assert config.actor_rollout_ref.rollout.n == 2
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 2
    assert config.agentlightning.trace_aggregator.level == "trajectory"
    assert config.agentlightning.trace_aggregator.trajectory_max_prompt_length == 2048
    assert config.agentlightning.trace_aggregator.trajectory_max_response_length == 2048
    assert config.agentlightning.local.agent_class == "examples.search_r1.agents.search_r1_agent:SearchR1Agent"
    assert config.agentlightning.local.env_map.QUESTION == "input.question"
    assert config.agentlightning.local.env_map.GOLDEN_ANSWERS == "input.golden_answers"


def test_build_config_applies_dotlist_overrides() -> None:
    config = train_search_r1_agent.build_config(
        config_overrides=("trainer.total_training_steps=7", "actor_rollout_ref.rollout.n=3"),
    )

    assert config.trainer.total_training_steps == 7
    assert config.actor_rollout_ref.rollout.n == 3
