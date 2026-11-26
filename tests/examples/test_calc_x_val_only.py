# Copyright (c) Microsoft. All rights reserved.

"""The Calc-X example has to be able to score a checkpoint without training it.

`fit()` only reaches its `val_only` exit from the validation pass it runs before
the first step, so turning one knob without the other silently trains instead of
evaluating. These tests pin that pair down, along with the two ways the mode can
be undone from outside: the checkpoint has to travel through `resume_from_path`
rather than the base model path the gateway keys on, and the CLI overrides merge
last, so they must not be able to switch the validation pass back off.
"""

from __future__ import annotations

import pytest

from examples.calc_x.train_calc_agent import build_config, verl_default_config

CHECKPOINT = "/runs/calc_x/global_step_128"


def test_training_still_skips_the_pre_training_validation() -> None:
    trainer = verl_default_config()["trainer"]
    assert trainer["val_only"] is False
    assert trainer["val_before_train"] is False


def test_val_only_switches_on_the_validation_pass() -> None:
    trainer = verl_default_config(val_only=True)["trainer"]
    assert trainer["val_only"] is True
    assert trainer["val_before_train"] is True


def test_evaluation_restores_the_checkpoint_through_the_resume_path() -> None:
    # Only the trainer's own exit matters here; Hydra needs verl to compose.
    pytest.importorskip("verl")

    config = build_config(resume_path=CHECKPOINT, val_only=True)
    assert config.trainer.val_only is True
    assert config.trainer.val_before_train is True
    assert config.trainer.resume_mode == "resume_path"
    assert config.trainer.resume_from_path == CHECKPOINT
    assert config.trainer.experiment_name == "calc_x_eval"


def test_run_name_is_still_appended_after_the_evaluation_marker() -> None:
    pytest.importorskip("verl")

    config = build_config(val_only=True, run_name="step128")
    assert config.trainer.experiment_name == "calc_x_eval_step128"


def test_the_checkpoint_does_not_replace_the_base_model() -> None:
    # The gateway contract is default_proxy.model_name == actor_rollout_ref.model.path
    # (docs/25-api-gateway-configuration.md), so aiming either at a checkpoint makes the
    # proxy answer "model not found" before the validation pass gets anywhere.
    pytest.importorskip("verl")

    config = build_config(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        resume_path=CHECKPOINT,
        val_only=True,
    )
    assert config.actor_rollout_ref.model.path == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.trainer.resume_from_path == CHECKPOINT


def test_cli_overrides_cannot_turn_evaluation_back_into_training() -> None:
    # The overrides merge last, and in this mode the held-out split is also what run_ppo
    # sizes its sampler from, so a val_before_train that got through would optimise on the
    # data being scored, silently.
    pytest.importorskip("verl")

    config = build_config(
        resume_path=CHECKPOINT,
        val_only=True,
        config_overrides=["trainer.val_before_train=false", "trainer.val_only=false"],
    )
    assert config.trainer.val_only is True
    assert config.trainer.val_before_train is True
    assert config.trainer.resume_mode == "resume_path"
    assert config.trainer.resume_from_path == CHECKPOINT
