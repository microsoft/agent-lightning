# Copyright (c) Microsoft. All rights reserved.

"""Compose native VERL configuration; training requires the documented GPU stack."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def build_config(model: str, overrides=()):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "agentlightning/verl"), version_base=None):
        config = compose(config_name="config")
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(Path(__file__).with_name("config.yaml")))
    config.actor_rollout_ref.model.path = model
    return OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))


def main():
    parser = argparse.ArgumentParser(description="Sealed process-credit arithmetic example")
    parser.add_argument(
        "--model", required=True, help="Local model with zero dropout and matching serving/training weights"
    )
    parser.add_argument("--print-config", action="store_true")
    args, overrides = parser.parse_known_args()
    config = build_config(args.model, overrides)
    from importlib.metadata import version

    from agentlightning.verl.process_credit import validate_process_config

    validate_process_config(config, verl_version=version("verl"))
    if args.print_config:
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    from agentlightning.verl.entrypoint import run_ppo

    tasks = [{"a": a, "b": 2, "c": 3, "d": 1} for a in range(4)]
    run_ppo(config, train_dataset=tasks, val_dataset=tasks)


if __name__ == "__main__":
    main()
