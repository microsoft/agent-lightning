# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Type, cast

from agentlightning.adapter import TraceAdapter
from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.base import LightningStore
from agentlightning.types import Dataset

from .dataset import AgentDataset, LoadedDataset

if TYPE_CHECKING:
    from .daemon import AgentModeDaemon
    from .trainer import AgentLightningTrainer

hydra: Any = import_module("hydra")
ray: Any = import_module("ray")
create_rl_sampler: Callable[..., Any] = getattr(import_module("verl.trainer.main_ppo"), "create_rl_sampler")
load_reward_manager: Callable[..., Any] = getattr(import_module("verl.trainer.ppo.reward"), "load_reward_manager")

__all__ = [
    "main",
    "run_ppo",
    "TaskRunner",
]


def _import_attr(module_name: str, attr_name: str) -> Any:
    return getattr(import_module(module_name), attr_name)


def _main(config: Any):
    from .daemon import AgentModeDaemon
    from .trainer import AgentLightningTrainer

    run_ppo(
        config,
        train_dataset=None,
        val_dataset=None,
        store=None,
        llm_proxy=None,
        adapter=None,
        trainer_cls=AgentLightningTrainer,
        daemon_cls=AgentModeDaemon,
    )


main: Callable[[], Any] = hydra.main(
    config_path="pkg://agentlightning/verl",
    config_name="config",
    version_base=None,
)(_main)


def run_ppo(
    config: Any,
    train_dataset: Dataset[Any] | None,
    val_dataset: Dataset[Any] | None,
    store: LightningStore | None,
    llm_proxy: LLMProxy | None,
    adapter: TraceAdapter[Any] | None,
    trainer_cls: Type[AgentLightningTrainer],
    daemon_cls: Type[AgentModeDaemon],
) -> None:
    if store is None:
        raise ValueError("VERL execution requires a store and does not support v0 fallback mode.")

    if not ray.is_initialized():
        # this is for local ray cluster
        try:
            # verl >= 0.6.0
            num_cpus = config.ray_kwargs.ray_init.num_cpus
        except AttributeError:
            # verl < 0.6.0
            num_cpus = config.ray_init.num_cpus
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
            },
            num_cpus=num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(
        runner.run.remote(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            store=store,
            llm_proxy=llm_proxy,
            adapter=adapter,
            trainer_cls=trainer_cls,
            daemon_cls=daemon_cls,
        )
    )


class _TaskRunner:
    def run(
        self,
        config: Any,
        train_dataset: Dataset[Any] | None,
        val_dataset: Dataset[Any] | None,
        store: LightningStore | None,
        llm_proxy: LLMProxy | None,
        adapter: TraceAdapter[Any] | None,
        trainer_cls: Type[AgentLightningTrainer],
        daemon_cls: Type[AgentModeDaemon],
    ):
        # print initial config
        from pprint import pprint

        omega_conf = _import_attr("omegaconf", "OmegaConf")
        copy_to_local = _import_attr("verl.utils.fs", "copy_to_local")

        pprint(omega_conf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        omega_conf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        hf_processor = _import_attr("verl.utils.tokenizer", "hf_processor")
        hf_tokenizer = _import_attr("verl.utils.tokenizer", "hf_tokenizer")

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            ray_worker_group_cls = _import_attr("verl.single_controller.ray", "RayWorkerGroup")
            fsdp_workers = import_module("verl.workers.fsdp_workers")
            actor_rollout_ref_worker = getattr(fsdp_workers, "ActorRolloutRefWorker")
            async_actor_rollout_ref_worker = getattr(fsdp_workers, "AsyncActorRolloutRefWorker")
            critic_worker = getattr(fsdp_workers, "CriticWorker")

            actor_rollout_cls = (
                async_actor_rollout_ref_worker
                if config.actor_rollout_ref.rollout.mode == "async"
                else actor_rollout_ref_worker
            )

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            megatron_module = import_module("verl.single_controller.ray.megatron")
            megatron_workers = import_module("verl.workers.megatron_workers")
            actor_rollout_cls = getattr(megatron_workers, "ActorRolloutRefWorker")
            critic_worker = getattr(megatron_workers, "CriticWorker")
            ray_worker_group_cls = getattr(megatron_module, "NVMegatronRayWorkerGroup")

        else:
            raise NotImplementedError

        resource_pool_manager_cls = _import_attr("verl.trainer.ppo.ray_trainer", "ResourcePoolManager")

        try:
            # verl >= 0.6.0
            role = _import_attr("verl.trainer.ppo.utils", "Role")
        except ImportError:
            # Fallback for verl <= 0.5.0
            role = _import_attr("verl.trainer.ppo.ray_trainer", "Role")

        role_worker_mapping: dict[Any, Any] = {
            role.ActorRollout: ray.remote(actor_rollout_cls),
            role.Critic: ray.remote(critic_worker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            role.ActorRollout: global_pool_id,
            role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                reward_model_worker = _import_attr("verl.workers.fsdp_workers", "RewardModelWorker")
            elif config.reward_model.strategy == "megatron":
                reward_model_worker = _import_attr("verl.workers.megatron_workers", "RewardModelWorker")
            else:
                raise NotImplementedError
            role_worker_mapping[role.RewardModel] = ray.remote(reward_model_worker)
            mapping[role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = resource_pool_manager_cls(resource_pool_spec=resource_pool_spec, mapping=mapping)

        collate_fn = _import_attr("verl.utils.dataset.rl_dataset", "collate_fn")

        # Use our special dataset
        if train_dataset is None:
            train_dataset = AgentDataset(
                data_files=config.data.train_files,
                tokenizer=tokenizer,
                processor=processor,
                config=config.data,
            )
        else:
            train_dataset = LoadedDataset(train_dataset)

        if val_dataset is None:
            val_dataset = AgentDataset(
                data_files=config.data.val_files,
                tokenizer=tokenizer,
                processor=processor,
                config=config.data,
            )
        else:
            val_dataset = LoadedDataset(val_dataset)

        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = cast(
            Any,
            trainer_cls(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
                store=store,
                llm_proxy=llm_proxy,
                adapter=adapter,
                daemon_cls=daemon_cls,
            ),
        )
        trainer.init_workers()
        trainer.fit()


TaskRunner: Any = ray.remote(num_cpus=1)(_TaskRunner)  # please make sure main_task is not scheduled on head


if __name__ == "__main__":
    main()
