# Copyright (c) Microsoft. All rights reserved.

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

"""VERL entrypoint for agl-lite — thin wrapper around verl's run_ppo.

The only customization is injecting our AglLiteAgentLoopManager via
the ``agent_loop_manager_class`` config field, so the standard
RayPPOTrainer uses agl-lite for agent execution while keeping VERL's
internal vLLM servers for inference (with weight updates after PPO steps).
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import hydra
import ray
from omegaconf import OmegaConf

from .dataset import LoadedDataset

__all__ = [
    "main",
    "run_ppo",
]

# FQN of our custom AgentLoopManager — set in config automatically.
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
    verl's standard ``main_ppo.run_ppo``.  Datasets are wrapped in
    ``LoadedDataset`` if provided as in-memory sequences.
    """
    # Ensure our agent loop manager is set in the config.
    OmegaConf.set_struct(config, False)
    if not config.actor_rollout_ref.rollout.get("agent"):
        config.actor_rollout_ref.rollout.agent = {}
    config.actor_rollout_ref.rollout.agent.agent_loop_manager_class = _AGL_LITE_AGENT_LOOP_MANAGER
    OmegaConf.set_struct(config, True)

    # Wrap in-memory datasets for verl compatibility.
    if train_dataset is not None:
        train_dataset = LoadedDataset(train_dataset)
    if val_dataset is not None:
        val_dataset = LoadedDataset(val_dataset)

    # Use verl's standard run_ppo with a custom TaskRunner that passes datasets.
    from verl.trainer.main_ppo import run_ppo as verl_run_ppo

    # verl's run_ppo doesn't accept datasets directly — it expects them to be
    # loaded from files inside TaskRunner. We use a custom TaskRunner subclass
    # that injects the pre-loaded datasets.
    if train_dataset is not None or val_dataset is not None:
        _run_ppo_with_datasets(config, train_dataset, val_dataset)
    else:
        verl_run_ppo(config)


def _run_ppo_with_datasets(config: Any, train_dataset: Any, val_dataset: Any) -> None:
    """Start Ray and run PPO with pre-loaded datasets."""
    from verl.trainer.main_ppo import TaskRunner as VerlTaskRunner, get_ppo_ray_runtime_env

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = OmegaConf.to_container(
            config.ray_kwargs.get("ray_init", OmegaConf.create({}))
        )
        runtime_env_kwargs = ray_init_kwargs.pop("runtime_env", {})
        runtime_env = {**default_runtime_env, **runtime_env_kwargs}

        # On shared machines, RAY_tmpdir isolates from other users' clusters.
        _temp_dir = os.environ.get("RAY_tmpdir")

        ray.init(
            runtime_env=runtime_env,
            **({"_temp_dir": _temp_dir} if _temp_dir else {}),
            **ray_init_kwargs,
        )

    @ray.remote(num_cpus=1)
    class AglTaskRunner(VerlTaskRunner):
        """TaskRunner that uses pre-loaded datasets instead of loading from files."""

        def run(self, config):
            from pprint import pprint
            from omegaconf import OmegaConf
            from verl.utils.fs import copy_to_local
            from verl.utils.tokenizer import hf_processor, hf_tokenizer
            from verl.utils.dataset.rl_dataset import collate_fn
            from verl.trainer.main_ppo import create_rl_sampler
            from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager

            pprint(OmegaConf.to_container(config, resolve=True))
            OmegaConf.resolve(config)

            local_path = copy_to_local(config.actor_rollout_ref.model.path)
            trust_remote_code = config.data.get("trust_remote_code", False)
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(local_path, use_fast=True)

            # Worker setup — delegated to parent's add_* methods
            actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.add_reward_model_worker(config)
            self.add_reference_policy_worker(config, actor_rollout_cls)

            global_pool_id = "global_pool"
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
            mapping = {role: global_pool_id for role in self.role_worker_mapping}
            resource_pool_manager = ResourcePoolManager(
                resource_pool_spec=resource_pool_spec, mapping=mapping
            )

            # Use pre-loaded datasets (passed via ray.put)
            td = ray.get(train_dataset_ref)
            vd = ray.get(val_dataset_ref)

            train_sampler = create_rl_sampler(config.data, td)
            trainer = RayPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                train_dataset=td,
                val_dataset=vd,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
            )
            trainer.init_workers()
            trainer.fit()

    # Put datasets in Ray object store
    train_dataset_ref = ray.put(train_dataset)
    val_dataset_ref = ray.put(val_dataset)

    runner = AglTaskRunner.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
