# Copyright (c) Microsoft. All rights reserved.

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

"""VERL entrypoint for agl-lite — thin wrapper around verl's run_ppo.

The only customization is:
  1. Injecting AglLiteAgentLoopManager via ``agent_loop_manager_class``
  2. Supporting pre-loaded in-memory datasets (verl's TaskRunner loads from files)
"""

from __future__ import annotations

import os
import socket
from typing import Any, Sequence

import hydra
import ray
from omegaconf import OmegaConf

from .dataset import LoadedDataset

__all__ = [
    "main",
    "run_ppo",
]

# FQN of our custom AgentLoopManager.
_AGL_LITE_AGENT_LOOP_MANAGER = "agl_lite.verl.agent_loop.AglLiteAgentLoopManager"


@hydra.main(config_path="pkg://agl_lite/verl", config_name="config", version_base=None)
def main(config: Any):
    run_ppo(config, train_dataset=None, val_dataset=None)


def run_ppo(
    config: Any,
    train_dataset: Sequence[Any] | None,
    val_dataset: Sequence[Any] | None,
) -> None:
    """Launch VERL PPO training with agl-lite agent orchestration.

    Injects ``AglLiteAgentLoopManager`` into the config and delegates to
    verl's standard training loop. Datasets can be passed as in-memory
    sequences or loaded from files (via ``config.data.train_files``).
    """
    # Ensure our agent loop manager is set in the config.
    OmegaConf.set_struct(config, False)
    if not config.actor_rollout_ref.rollout.get("agent"):
        config.actor_rollout_ref.rollout.agent = {}
    config.actor_rollout_ref.rollout.agent.agent_loop_manager_class = _AGL_LITE_AGENT_LOOP_MANAGER
    OmegaConf.set_struct(config, True)

    from verl.trainer.main_ppo import get_ppo_ray_runtime_env

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = OmegaConf.to_container(
            config.ray_kwargs.get("ray_init", OmegaConf.create({}))
        )
        runtime_env_kwargs = ray_init_kwargs.pop("runtime_env", {})
        runtime_env = {**default_runtime_env, **runtime_env_kwargs}
        _temp_dir = os.environ.get("RAY_tmpdir")
        ray.init(
            runtime_env=runtime_env,
            **({"_temp_dir": _temp_dir} if _temp_dir else {}),
            **ray_init_kwargs,
        )

    # Wrap in-memory datasets.
    train_ds = LoadedDataset(train_dataset) if train_dataset is not None else None
    val_ds = LoadedDataset(val_dataset) if val_dataset is not None else None

    runner = _AglTaskRunner.remote()
    ray.get(runner.run.remote(config, train_ds, val_ds))


@ray.remote(num_cpus=1)
class _AglTaskRunner:
    """TaskRunner that extends verl's TaskRunner with pre-loaded dataset support."""

    def __init__(self):
        from verl.trainer.main_ppo import TaskRunner
        self._delegate = TaskRunner()

    def run(self, config, train_dataset_ref, val_dataset_ref):
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.trainer.main_ppo import (
            create_rl_sampler,
            need_critic,
            need_reference_policy,
            validate_config,
        )
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.utils.tokenizer import hf_processor, hf_tokenizer

        print(f"AglTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Worker setup — delegated to verl's TaskRunner
        d = self._delegate
        actor_rollout_cls, ray_worker_group_cls = d.add_actor_rollout_worker(config)
        d.add_critic_worker(config)
        d.add_reward_model_resource_pool(config)
        d.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = d.init_resource_pool_mgr(config)

        # Datasets — use pre-loaded if available, else load from files
        if train_dataset_ref is not None:
            train_dataset = train_dataset_ref
        else:
            from verl.trainer.main_ppo import create_rl_dataset
            train_dataset = create_rl_dataset(
                config.data.train_files, config.data, tokenizer, processor, is_train=True,
            )

        if val_dataset_ref is not None:
            val_dataset = val_dataset_ref
        else:
            from verl.trainer.main_ppo import create_rl_dataset
            val_dataset = create_rl_dataset(
                config.data.val_files, config.data, tokenizer, processor, is_train=False,
            )

        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=d.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()
