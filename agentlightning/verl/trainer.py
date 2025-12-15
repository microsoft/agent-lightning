# Copyright (c) Microsoft. All rights reserved.

# type: ignore

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pprint import pprint
from typing import Dict, Tuple, Type
from pathlib import Path

import numpy as np
import torch
import verl
from codetiming import Timer
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    _compute_response_info,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from agentlightning.adapter import TraceAdapter, TraceToTripletBase
from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.base import LightningStore

from .daemon import AgentModeDaemon

__all__ = [
    "AgentLightningTrainer",
]


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


# This function is adapted from verl.
# We introduce a new parameter `suffix` to distinguish between metrics computed
# before and after AgentLightning’s post-processing.
# - "Before" refers to raw reward and advantage values.
# - "After" refers to values computed following post-processing, which involves:
#     (1) Dropping prompts that exceed the maximum allowed length.
#     (2) Adjusting the batch size to be a multiple of the mini PPO size.
# Different suffixes are used to label these two stages accordingly.
def compute_data_metrics(batch: DataProto, use_critic: bool = True, suffix: str = "") -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    def _quantiles(tensor, probs):
        return [torch.quantile(tensor.float(), p).detach().item() for p in probs]

    prompt_q50, prompt_q95 = _quantiles(prompt_length, [0.5, 0.95])
    response_q50, response_q95 = _quantiles(response_length, [0.5, 0.95])

    metrics = {
        # score
        "critic/score/mean" + suffix: torch.mean(sequence_score).detach().item(),
        "critic/score/max" + suffix: torch.max(sequence_score).detach().item(),
        "critic/score/min" + suffix: torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean" + suffix: torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max" + suffix: torch.max(sequence_reward).detach().item(),
        "critic/rewards/min" + suffix: torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean" + suffix: torch.mean(valid_adv).detach().item(),
        "critic/advantages/max" + suffix: torch.max(valid_adv).detach().item(),
        "critic/advantages/min" + suffix: torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean" + suffix: torch.mean(valid_returns).detach().item(),
        "critic/returns/max" + suffix: torch.max(valid_returns).detach().item(),
        "critic/returns/min" + suffix: torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean" + suffix: torch.mean(valid_values).detach().item(),
                "critic/values/max" + suffix: torch.max(valid_values).detach().item(),
                "critic/values/min" + suffix: torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var" + suffix: (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean" + suffix: torch.mean(response_length).detach().item(),
        "response_length/max" + suffix: torch.max(response_length).detach().item(),
        "response_length/min" + suffix: torch.min(response_length).detach().item(),
        "response_length/clip_ratio"
        + suffix: torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        "response_length/p50" + suffix: response_q50,
        "response_length/p95" + suffix: response_q95,
        # prompt length
        "prompt_length/mean" + suffix: torch.mean(prompt_length).detach().item(),
        "prompt_length/max" + suffix: torch.max(prompt_length).detach().item(),
        "prompt_length/min" + suffix: torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio"
        + suffix: torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        "prompt_length/p50" + suffix: prompt_q50,
        "prompt_length/p95" + suffix: prompt_q95,
    }
    return metrics


class AgentLightningTrainer(RayPPOTrainer):
    """
    Specialized PPO trainer for agent-based reinforcement learning.

    This trainer is designed specifically for scenarios where the model interacts with
    external environments, tools, or APIs through an AgentLightningServer. It simplifies
    the training loop by removing the complex conditional logic present in the original
    RayPPOTrainer and focusing on the agent mode workflow.

    Key differences from RayPPOTrainer:

    1. Uses AgentModeDaemon for server communication
    2. Simplified data flow without pop/union operations
    3. Direct batch processing through agent daemon
    4. Streamlined validation using agent_mode validation
    """

    def __init__(
        self,
        store: LightningStore | None,
        llm_proxy: LLMProxy | None,
        adapter: TraceAdapter | None,
        daemon_cls: Type[AgentModeDaemon],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.store = store
        self.llm_proxy = llm_proxy
        self.adapter = adapter
        self.daemon_cls = daemon_cls
        self._sequence_log_file = getattr(self.config.trainer, "sequence_log_file", None)
        if self._sequence_log_file:
            try:
                path = Path(self._sequence_log_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    with path.open("w", encoding="utf-8") as f:
                        f.write(
                            "step,prompt_mean,prompt_p50,prompt_p95,prompt_max,"
                            "response_mean,response_p50,response_p95,response_max\n"
                        )
            except Exception:
                self._sequence_log_file = None

    def _validate(self):
        assert len(self.val_dataloader) == 1, "Please set val_batch_size to None for better throughput."

        test_data = next(iter(self.val_dataloader))
        test_batch = DataProto.from_single_dict(test_data)

        self.async_rollout_manager.wake_up()
        self.agent_mode_daemon.set_up_data_and_server(
            test_batch.non_tensor_batch,
            self.async_rollout_manager.server_addresses,
            is_train=False,
        )
        self.agent_mode_daemon.run_until_all_finished()
        test_metrics = self.agent_mode_daemon.get_test_metrics()
        self.agent_mode_daemon.clear_data_and_server()
        self.async_rollout_manager.sleep()
        return test_metrics

    def _compute_reference_log_prob(self, batch: DataProto) -> DataProto:
        """Compute reference log probability using the correct worker based on LoRA configuration.

        In verl 0.6.0+, when LoRA is detected (indicated by ref_in_actor=True),
        the reference policy is computed by the actor rollout worker instead of a separate
        ref policy worker. This method handles both scenarios by checking the ref_in_actor flag.
        Note: verl sets ref_in_actor=True when it detects LoRA configuration (e.g., lora_rank > 0 or lora_adapter_path is set).

        Args:
            batch: The data batch to compute reference log probabilities for.

        Returns:
            DataProto with reference log probabilities added.

        Raises:
            RuntimeError: If the required worker is not available.
        """
        if getattr(self, "ref_in_actor", False):
            actor_worker = getattr(self, "actor_rollout_wg", None)
            if actor_worker is None:
                raise RuntimeError("actor_rollout_wg is required when ref_in_actor is True.")
            return actor_worker.compute_ref_log_prob(batch)

        ref_worker = getattr(self, "ref_policy_wg", None)
        if ref_worker is None:
            raise RuntimeError(
                "Reference policy worker was not initialized. "
                "Ensure `use_reference_policy` is enabled and the VERL config exposes the ref worker."
            )
        return ref_worker.compute_ref_log_prob(batch)

    def _select_val_metric(self, val_metrics: dict) -> tuple[str, float | None]:
        """Pick a representative validation metric for logging."""

        if not val_metrics:
            return "val_reward", None
        preferred_keys = [k for k in val_metrics if k.startswith("val-core/")]
        fallback_keys = [k for k in val_metrics if k.startswith("val-aux/")]
        candidate_keys = preferred_keys or fallback_keys or list(val_metrics.keys())
        key = candidate_keys[0]
        return key, val_metrics.get(key)

    def _log_sequence_window(self, metrics: dict) -> None:
        """Persist compact prompt/response length stats for the current logging window."""

        if not self._sequence_log_file:
            return
        prompt_mean = metrics.get("prompt_length/mean_after_processing", metrics.get("prompt_length/mean"))
        prompt_p50 = metrics.get("prompt_length/p50_after_processing", metrics.get("prompt_length/p50"))
        prompt_p95 = metrics.get("prompt_length/p95_after_processing", metrics.get("prompt_length/p95"))
        prompt_max = metrics.get("prompt_length/max_after_processing", metrics.get("prompt_length/max"))
        response_mean = metrics.get("response_length/mean_after_processing", metrics.get("response_length/mean"))
        response_p50 = metrics.get("response_length/p50_after_processing", metrics.get("response_length/p50"))
        response_p95 = metrics.get("response_length/p95_after_processing", metrics.get("response_length/p95"))
        response_max = metrics.get("response_length/max_after_processing", metrics.get("response_length/max"))

        if prompt_mean is None or response_mean is None:
            return

        try:
            with open(self._sequence_log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{self.global_steps},{prompt_mean},{prompt_p50},{prompt_p95},{prompt_max},"
                    f"{response_mean},{response_p50},{response_p95},{response_max}\n"
                )
        except Exception:
            return

    def _train_step(self, batch_dict: dict) -> dict:
        # Isolate in a separate method to automatically recycle the variables before validation.
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        metrics = {}
        timing_raw = {}

        with _timer("step", timing_raw):

            # When agent mode is enabled, we read the batch as it is.
            gen_batch = batch

            # generate a batch
            with _timer("gen", timing_raw):
                self.async_rollout_manager.wake_up()
                self.agent_mode_daemon.set_up_data_and_server(
                    gen_batch.non_tensor_batch, self.async_rollout_manager.server_addresses
                )
                self.agent_mode_daemon.run_until_all_finished()
                batch, agent_metrics = self.agent_mode_daemon.get_train_data_batch(
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    device=gen_batch.batch["fake_ids"].device,
                )
                metrics.update(agent_metrics)
                self.agent_mode_daemon.clear_data_and_server()
                self.async_rollout_manager.sleep()

            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                with _timer("gen_max", timing_raw):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                    batch = batch.union(gen_baseline_output)
                    reward_baseline_tensor = self.reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            # uid is used for algorithm like GRPO, should be aligned to data id
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]

            batch.batch["response_mask"] = compute_response_mask(batch)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with _timer("reward", timing_raw):
                # compute reward model score
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                reward_extra_infos_dict = {}

            # for agent mode, pad the lengths to calculate old log prob, ref, and values
            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    ref_log_prob = self._compute_reference_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with _timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            # for agent mode, unpad to calculate adv
            # it is important, as adv should be based on the raw traces
            batch = unpad_dataproto(batch, pad_size=pad_size)

            with _timer("adv", timing_raw):
                # if agent_mode is enabled, there is already token_level_scores
                # token_level_scores is not needed to compute here

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                )  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            # Calculate the metrics before processing. Refer to the comments of function `compute_data_metrics` for details.
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, suffix="_before_processing"))

            # after advantages are assinged, we begin to drop (1) long prompt (2) floor to ppo minisize
            keep_indices = (~batch.batch["is_drop_mask"]).nonzero(as_tuple=True)[0]
            metrics["training/n_triplets_prompt_too_long"] = (
                batch.batch["is_drop_mask"].shape[0] - keep_indices.shape[0]
            )
            batch = batch[keep_indices]
            # next, round to minibatch size
            mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            n_transition = len(batch)
            random_indices = list(range(n_transition))
            random.shuffle(random_indices)
            batch.reorder(torch.tensor(random_indices).type(torch.int32))
            n_remained_transition = n_transition // mini_batch_size * mini_batch_size
            batch = batch[list(range(n_remained_transition))]
            metrics["training/n_triplets_dropped_remainder"] = n_transition - n_remained_transition

            # Agent mode note: Change the order of balance batch;
            #     1. first calculate advantage
            #     2. then drop the samples (too long prompt & floor to ppo minisize)
            #     3. balance
            # balance the number of valid tokens on each dp rank.
            # Note that this breaks the order of data inside the batch.
            # Please take care when you implement group based adv computation such as GRPO and rloo
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            # update critic
            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with _timer("update_actor", timing_raw):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            # Log rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                with _timer("dump_rollout_generations", timing_raw):
                    print(batch.batch.keys())
                    inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    self._dump_generations(
                        inputs=inputs,
                        outputs=outputs,
                        scores=scores,
                        reward_extra_infos_dict=reward_extra_infos_dict,
                        dump_path=rollout_data_dir,
                    )

        # compute training metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, suffix="_after_processing"))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        # TODO: implement actual tflpo and theoretical tflpo
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        return metrics

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        assert self.async_rollout_mode, "If agent mode is enabled, async server must be enabled"
        if self.adapter is not None and not isinstance(self.adapter, TraceToTripletBase):
            raise ValueError("Adapter must be a TraceToTripletBase for currently VERL implementation.")
        verl_version = verl.__version__
        if verl_version == "0.5.0":
            # Note (Zhiyuan): To avoid further patch into vllm async server, using the same sentence to get the naming here.
            # However, it is possible that verl updates the naming and causes incompatibility.
            # Reference: https://github.com/volcengine/verl/blob/5b5e09d9cc20625e436d01f69d9cc739ff681c54/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L217
            model = "/".join(self.config.actor_rollout_ref.model.path.split("/")[-2:])
        else:
            # For other versions (e.g., 0.6.0), we use the full path to the model.
            model = self.config.actor_rollout_ref.model.path
        self.agent_mode_daemon = self.daemon_cls(
            self.config.agentlightning.port,
            self.config.actor_rollout_ref.rollout.n,
            train_information={
                "model": model,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            },
            tokenizer=self.tokenizer,
            mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            pad_token_id=self.tokenizer.pad_token_id,
            mode="v1" if self.store is not None else "v0",
            store=self.store,
            llm_proxy=self.llm_proxy,
            adapter=self.adapter,
            processor=self.processor,  # For Qwen2-VL mrope position_ids
            image_base_dir=getattr(self.config.data, "image_base_dir", None),
        )
        self.agent_mode_daemon.start()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        n_gpus = self.resource_pool_manager.get_n_gpus()
        self._cumulative_step_time = 0.0
        self._cumulative_step_count = 0

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                # train step
                metrics = self._train_step(batch_dict)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with _timer("validate", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    # write validation summary to progress log
                    ts_val = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    key, val = self._select_val_metric(val_metrics)
                    self._progress_log(
                        f"[{ts_val}] [val] step {self.global_steps}/{self.total_training_steps} | {key}={val}"
                    )

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                # step metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # Lightweight console/file summary (mirrors Ray trainer behaviour)
                self._log_window.append(
                    {
                        "tokens": metrics.get("perf/total_num_tokens", 0),
                        "step_time": metrics.get("perf/time_per_step", 0.0),
                        "reward": metrics.get("reward/mean", metrics.get("training/reward", None)),
                    }
                )
                # update cumulative step time to stabilize ETA
                step_time = metrics.get("perf/time_per_step", None)
                if step_time is not None:
                    self._cumulative_step_time += float(step_time)
                    self._cumulative_step_count += 1

                if self._log_interval > 0 and (self.global_steps % self._log_interval == 0 or is_last_step):
                    window = list(self._log_window)
                    total_tokens = sum(m.get("tokens", 0) or 0 for m in window)
                    total_time = sum(m.get("step_time", 0.0) or 0.0 for m in window)
                    avg_step_window = total_time / len(window) if window else 0.0
                    avg_step_overall = (
                        self._cumulative_step_time / self._cumulative_step_count
                        if self._cumulative_step_count > 0
                        else 0.0
                    )
                    throughput = total_tokens / total_time if total_time > 0 else 0.0
                    throughput_per_gpu = throughput / n_gpus if n_gpus else throughput
                    reward_vals = [m["reward"] for m in window if m.get("reward") is not None]
                    reward_avg = sum(reward_vals) / len(reward_vals) if reward_vals else None
                    eta_sec = (
                        (self.total_training_steps - self.global_steps) * avg_step_overall
                        if avg_step_overall > 0
                        else float("inf")
                    )
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec)) if eta_sec != float("inf") else "n/a"
                    lr_val = self.config.actor_rollout_ref.actor.optim.lr
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    line = (
                        f"[{ts}] [train] step {self.global_steps}/{self.total_training_steps} | "
                        f"eta {eta_str} | step_win {avg_step_window:.2f}s | step_avg {avg_step_overall:.2f}s | "
                        f"toks/s {throughput/1e3:.1f}k ({throughput_per_gpu/1e3:.1f}k/gpu) | "
                        f"reward {reward_avg if reward_avg is not None else 'n/a'} | lr {lr_val}"
                    )
                    self._progress_log(line)
                    self._log_sequence_window(metrics)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()

                    # This exit logic is to ensure a robust CI.
                    pprint(f"Flush the logger...")
                    del logger  # Make sure the loggers are flushed and closed properly
                    pprint(f"Training finished at step {self.global_steps}.")
                    return

                progress_bar.update(1)
                self.global_steps += 1
