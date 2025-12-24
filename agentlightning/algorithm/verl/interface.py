# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Type

from hydra import compose, initialize
from omegaconf import OmegaConf

from agentlightning.algorithm.base import Algorithm
from agentlightning.client import AgentLightningClient
from agentlightning.types import Dataset
from agentlightning.verl.entrypoint import run_ppo  # type: ignore

if TYPE_CHECKING:
    from agentlightning.verl.daemon import AgentModeDaemon
    from agentlightning.verl.trainer import AgentLightningTrainer


class VERL(Algorithm):
    """VERL-powered algorithm that delegates training to the VERL PPO runner.

    !!! warning
        Advanced customisation currently requires copying the VERL source and
        modifying it directly. Native hooks for overriding training behaviour
        will land in a future release.

    Args:
        config: Dictionary mirroring the overrides passed to the VERL CLI. The
            overrides are merged with VERL's packaged defaults via Hydra before
            launching training.
        trainer_cls: Optional override for the trainer class. Experimental.
        daemon_cls: Optional override for the daemon class. Experimental.

    !!! note "Trajectory aggregation (experimental)"

        Trajectory-level aggregation merges an entire multi-turn rollout into a single,
        masked training sample so GPU time is spent once per trajectory rather than N times
        per turn. Enable it via:

        ```python
        config["agentlightning"]["trace_aggregator"] = {
            "level": "trajectory",
            "trajectory_max_prompt_length": 4096,
            "trajectory_max_response_length": 34384,
        }
        ```

        Keep conversations structured (message lists rather than manual string
        concatenation) so prefix matching can stitch traces. `trajectory_max_prompt_length`
        should be set to the maximum length of the prompt for the first turn, and
        `trajectory_max_response_length` should be set to the maximum cumulative
        length of agent responses in the full trajectory.
        Toggle `debug=True` plus `mismatch_log_dir` when you need to inspect
        retokenization or chat-template mismatches. See
        [this blog post](https://agent-lightning.github.io/posts/trajectory_level_aggregation/)
        for more details.

    Examples:
        ```python
        from agentlightning.algorithm.verl import VERL

        algorithm = VERL(
            config={
                "algorithm": {
                    "adv_estimator": "grpo",
                    "use_kl_in_reward": False,
                },
                "data": {
                    "train_batch_size": 32,
                    "max_prompt_length": 4096,
                    "max_response_length": 2048,
                },
                "actor_rollout_ref": {
                    "rollout": {
                        "tensor_model_parallel_size": 1,
                        "n": 4,
                        "log_prob_micro_batch_size_per_gpu": 4,
                        "multi_turn": {"format": "hermes"},
                        "name": "vllm",
                        "gpu_memory_utilization": 0.6,
                    },
                    "actor": {
                        "ppo_mini_batch_size": 32,
                        "ppo_micro_batch_size_per_gpu": 4,
                        "optim": {"lr": 1e-6},
                        "use_kl_loss": False,
                        "kl_loss_coef": 0.0,
                        "entropy_coeff": 0,
                        "clip_ratio_low": 0.2,
                        "clip_ratio_high": 0.3,
                        "fsdp_config": {
                            "param_offload": True,
                            "optimizer_offload": True,
                        },
                    },
                    "ref": {
                        "log_prob_micro_batch_size_per_gpu": 8,
                        "fsdp_config": {"param_offload": True},
                    },
                    "model": {
                        "path": "Qwen/Qwen2.5-1.5B-Instruct",
                        "use_remove_padding": True,
                        "enable_gradient_checkpointing": True,
                    },
                },
                "trainer": {
                    "n_gpus_per_node": 1,
                    "val_before_train": True,
                    "critic_warmup": 0,
                    "logger": ["console", "wandb"],
                    "project_name": "AgentLightning",
                    "experiment_name": "calc_x",
                    "nnodes": 1,
                    "save_freq": 64,
                    "test_freq": 32,
                    "total_epochs": 2,
                },
            }
        )
        trainer.fit(algorithm, train_dataset=my_train_dataset)
        ```
    """

    def __init__(
        self,
        config: dict[str, Any],
        trainer_cls: Optional[Type[AgentLightningTrainer]] = None,
        daemon_cls: Optional[Type[AgentModeDaemon]] = None,
    ):
        super().__init__()

        # Compose the base config exactly like your decorator:
        with initialize(version_base=None, config_path="pkg://agentlightning/verl"):
            base_cfg = compose(config_name="config")

        # Merge your dict overrides
        override_conf = OmegaConf.create(config)
        # Allow adding new fields
        OmegaConf.set_struct(base_cfg, False)
        self.config = OmegaConf.merge(base_cfg, override_conf)
        self.trainer_cls = trainer_cls
        self.daemon_cls = daemon_cls

    def run(
        self,
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
    ) -> None:
        """Launch the VERL PPO entrypoint with the configured runtime context.

        Args:
            train_dataset: Optional dataset forwarded to VERL for training.
            val_dataset: Optional dataset forwarded to VERL for evaluation.

        Raises:
            ValueError: If required dependencies such as the store, LLM proxy, or
                adapter have been garbage-collected when using the V1 execution
                mode.
        """
        from agentlightning.verl.daemon import AgentModeDaemon
        from agentlightning.verl.trainer import AgentLightningTrainer

        trainer_cls = self.trainer_cls or AgentLightningTrainer
        daemon_cls = self.daemon_cls or AgentModeDaemon
        try:
            store = self.get_store()
        except Exception:
            print("Store is not set. Assuming v0 execution mode.")
            run_ppo(
                self.config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                store=None,
                llm_proxy=None,
                adapter=None,
                trainer_cls=trainer_cls,
                daemon_cls=daemon_cls,
            )
        else:
            print("Store is set. Assuming v1 execution mode.")
            llm_proxy = self.get_llm_proxy()
            adapter = self.get_adapter()
            run_ppo(
                self.config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                store=store,
                llm_proxy=llm_proxy,
                adapter=adapter,
                trainer_cls=trainer_cls,
                daemon_cls=daemon_cls,
            )

    def get_client(self) -> AgentLightningClient:
        """Create a client bound to the VERL-managed Agent Lightning server.

        Deprecated:
            Since v0.2.
        """
        port = self.config.agentlightning.port
        return AgentLightningClient(endpoint=f"http://localhost:{port}")
