"""VERL entrypoint for agl-lite — wraps verl's PPO setup with a custom trainer.

Customizations:
  1. Use AglLiteRayPPOTrainer (subclass of RayPPOTrainer) that drives rollouts
     through the agl-lite HTTP API instead of stock VERL agent loop workers.
  2. Support pre-loaded in-memory datasets (verl's TaskRunner loads from files).
"""

# VERL's PPO entrypoint helpers are imported from its runtime module because
# they are not exposed as a stable public API in verl 0.7.1.
# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import os
import socket
from collections.abc import Sequence
from typing import Any, cast

import hydra
import ray
from omegaconf import OmegaConf

from .dataset import LoadedDataset

__all__ = [
    "main",
    "run_ppo",
]


@hydra.main(config_path="pkg://agl_lite/verl", config_name="config", version_base=None)
def main(config: Any):
    run_ppo(config, train_dataset=None, val_dataset=None)


def run_ppo(
    config: Any,
    train_dataset: Sequence[Any] | None,
    val_dataset: Sequence[Any] | None,
) -> None:
    """Launch VERL PPO training with agl-lite agent orchestration.

    Datasets can be passed as in-memory sequences or loaded from files
    (via ``config.data.train_files``).
    """
    from verl.trainer.main_ppo import get_ppo_ray_runtime_env

    if not ray.is_initialized():
        default_runtime_env = cast(dict[str, Any], get_ppo_ray_runtime_env())
        ray_init_config = OmegaConf.to_container(config.ray_kwargs.get("ray_init", OmegaConf.create({})), resolve=True)
        ray_init_kwargs: dict[str, Any] = (
            {str(key): value for key, value in ray_init_config.items()} if isinstance(ray_init_config, dict) else {}
        )
        runtime_env_config = ray_init_kwargs.pop("runtime_env", {})
        runtime_env_kwargs = dict(runtime_env_config) if isinstance(runtime_env_config, dict) else {}
        runtime_env = {**default_runtime_env, **runtime_env_kwargs}
        # Pass agl-lite env vars to Ray workers.
        env_vars = runtime_env.setdefault("env_vars", {})
        if not isinstance(env_vars, dict):
            env_vars = {}
            runtime_env["env_vars"] = env_vars
        for var in (
            "AGL_KEY",
            "AGL_BASE_URL",
            "AGL_MODEL_ENDPOINT",
            "AGL_NAMESPACE",
            "WANDB_API_KEY",
            "WANDB_ENTITY",
            "WANDB_PROJECT",
            "WANDB_DIR",
            "WANDB_MODE",
            "WANDB_RUN_ID",
            "WANDB_RESUME",
        ):
            val = os.environ.get(var)
            if val:
                env_vars[var] = val
        _temp_dir = os.environ.get("RAY_TMPDIR")
        ray.init(
            runtime_env=runtime_env,
            **({"_temp_dir": _temp_dir} if _temp_dir else {}),
            **ray_init_kwargs,
        )

    # Wrap in-memory datasets.
    train_ds = LoadedDataset(train_dataset) if train_dataset is not None else None
    val_ds = LoadedDataset(val_dataset) if val_dataset is not None else None

    runner = cast(Any, _AglTaskRunner).remote()
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
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.utils.tokenizer import hf_processor, hf_tokenizer

        from agl_lite.verl.trainer import AglLiteRayPPOTrainer

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
                config.data.train_files,
                config.data,
                tokenizer,
                processor,
                is_train=True,
            )

        if val_dataset_ref is not None:
            val_dataset = val_dataset_ref
        else:
            from verl.trainer.main_ppo import create_rl_dataset

            val_dataset = create_rl_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                is_train=False,
            )

        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = AglLiteRayPPOTrainer(
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

        # Entry dispatch: async-rollout path is a fully independent training
        # loop. When async_rollout.enabled=false (default) the new code is
        # never called and sync RL behavior is byte-level equivalent to the
        # pre-async version.
        async_cfg = config.agentlightning.get("async_rollout", None)
        if async_cfg is not None and async_cfg.get("enabled", False):
            trainer.async_fit()
        else:
            trainer.fit()
