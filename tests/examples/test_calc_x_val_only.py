# Copyright (c) Microsoft. All rights reserved.

"""The Calc-X example has to be able to score a checkpoint without training it.

`fit()` only reaches its `val_only` exit from the validation pass it runs before
the first step, so turning one knob without the other silently trains instead of
evaluating. These tests pin that pair down.
"""

from __future__ import annotations

import pytest

from examples.calc_x.train_calc_agent import build_config, verl_default_config


def test_training_still_skips_the_pre_training_validation() -> None:
    trainer = verl_default_config()["trainer"]
    assert trainer["val_only"] is False
    assert trainer["val_before_train"] is False


def test_val_only_switches_on_the_validation_pass() -> None:
    trainer = verl_default_config(val_only=True)["trainer"]
    assert trainer["val_only"] is True
    assert trainer["val_before_train"] is True


def test_evaluation_run_is_named_apart_from_training() -> None:
    # Only the trainer's own exit matters here; Hydra needs verl to compose.
    pytest.importorskip("verl")

    config = build_config(model="/checkpoints/global_step_128", val_only=True)
    assert config.trainer.val_only is True
    assert config.trainer.val_before_train is True
    assert config.actor_rollout_ref.model.path == "/checkpoints/global_step_128"
    assert config.trainer.experiment_name == "calc_x_eval"


def test_run_name_is_still_appended_after_the_evaluation_marker() -> None:
    pytest.importorskip("verl")

    config = build_config(val_only=True, run_name="step128")
    assert config.trainer.experiment_name == "calc_x_eval_step128"
