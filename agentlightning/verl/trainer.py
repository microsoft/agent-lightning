# Copyright (c) Microsoft. All rights reserved.

# type: ignore

from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from pprint import pprint
from typing import Dict, Tuple
from typing import Optional
import numpy as np
import ray
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
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.ray_trainer import (
    # AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    # compute_advantage,
    compute_response_mask,
)
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from agentlightning.adapter import TraceAdapter, TraceToTripletBase
from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.base import LightningStore
from . import core_algos
from .daemon import AgentModeDaemon
from .core_algos import AdvantageEstimator



def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["data_id_list"],
            traj_index=data.non_tensor_batch["rollout_id_list"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "data_id_list" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["data_id_list"]
        if "rollout_id_list" in data.non_tensor_batch:  # optional
            adv_kwargs["traj_index"] = data_proto.non_tensor_batch["rollout_id_list"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data






__all__ = [
    "AgentLightningTrainer",
]







@ray.remote
def decode_and_dump_async(
    prompts_list,
    responses_list,
    scores,
    reward_extra_infos_dict,
    dump_path,
    global_steps,
    tokenizer_name_or_path,
    pad_token_id,
    rollout_id_list=None,
):
    """Asynchronously decode token IDs and dump to JSONL file.
    
    This function runs in a separate Ray worker to avoid blocking the main training loop.
    """
    from transformers import AutoTokenizer
    
    # Load tokenizer in the worker (avoids serialization issues)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    # Decode: remove padding and convert to text
    inputs_ids_wo_padding = [[tid for tid in ids if tid != pad_token_id] for ids in prompts_list]
    outputs_ids_wo_padding = [[tid for tid in ids if tid != pad_token_id] for ids in responses_list]
    inputs = tokenizer.batch_decode(inputs_ids_wo_padding, skip_special_tokens=False)
    outputs = tokenizer.batch_decode(outputs_ids_wo_padding, skip_special_tokens=False)
    
    # Dump to JSONL
    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"{global_steps}.jsonl")
    
    n = len(inputs)
    base_data = {
        "input": inputs,
        "input_ids": inputs_ids_wo_padding,
        "output": outputs,
        "output_ids": outputs_ids_wo_padding,
        "score": scores,
        "step": [global_steps] * n,
    }
    
    # Add rollout_id if available
    if rollout_id_list is not None and len(rollout_id_list) == n:
        base_data["rollout_id"] = rollout_id_list
    
    for k, v in reward_extra_infos_dict.items():
        if len(v) == n:
            base_data[k] = v
    
    lines = []
    for i in range(n):
        entry = {k: v[i] for k, v in base_data.items()}
        lines.append(json.dumps(entry, ensure_ascii=False))
    
    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")
    
    return filename  # Return filename for logging


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
        # prompt length
        "prompt_length/mean" + suffix: torch.mean(prompt_length).detach().item(),
        "prompt_length/max" + suffix: torch.max(prompt_length).detach().item(),
        "prompt_length/min" + suffix: torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio"
        + suffix: torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
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
        self, store: LightningStore | None, llm_proxy: LLMProxy | None, adapter: TraceAdapter | None, **kwargs
    ):
        super().__init__(**kwargs)
        self.store = store
        self.llm_proxy = llm_proxy
        self.adapter = adapter
        self._pending_dump_future = None  # Track pending async dump task

    def _validate(self):
        # assert len(self.val_dataloader) == 1, "Please set val_batch_size to None for better throughput."
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        test_metrics_all = {}
        # Wait for any pending training dump to complete before validation
        if self._pending_dump_future is not None:
            try:
                prev_filename = ray.get(self._pending_dump_future)
                print(f"[Async] Waiting for training dump before validation: {prev_filename}")
            except Exception as e:
                print(f"[Async] Training dump failed before validation: {e}")
            self._pending_dump_future = None
        # test_data = next(iter(self.val_dataloader))
        # Log validation generations if enabled (sync execution is fine for validation)
        val_data_dir = self.config.trainer.get("validation_data_dir", None)

        for test_data in self.val_dataloader:    
            test_batch = DataProto.from_single_dict(test_data)

            self.async_rollout_manager.wake_up()
            self.agent_mode_daemon.set_up_data_and_server(
                test_batch.non_tensor_batch,
                self.async_rollout_manager.server_addresses,
                is_train=False,
            )
            self.agent_mode_daemon.run_until_all_finished()
            test_metrics = self.agent_mode_daemon.get_test_metrics()
            for k, v in test_metrics.items():
                if not (k in test_metrics_all):
                    test_metrics_all[k] = []
                if type(v) is list:
                    test_metrics_all[k] += v
                else:
                    test_metrics_all[k].append(float(v))

            if val_data_dir:
                # Get validation batch data for logging
                if "fake_ids" in test_batch.batch:
                    device = test_batch.batch["fake_ids"].device
                elif "input_ids" in test_batch.batch:
                    device = test_batch.batch["input_ids"].device
                else:
                    raise ValueError("valid device not found")
                val_batch, _ = self.agent_mode_daemon.get_val_data_batch(
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    device=device,
                )
                # Remove padding tokens before decoding for better readability
                inputs_ids_wo_padding = [[token_id for token_id in ids if token_id != self.tokenizer.pad_token_id] 
                                        for ids in val_batch.batch["prompts"]]
                outputs_ids_wo_padding = [[token_id for token_id in ids if token_id != self.tokenizer.pad_token_id] 
                                        for ids in val_batch.batch["responses"]]
                inputs = self.tokenizer.batch_decode(inputs_ids_wo_padding, skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(outputs_ids_wo_padding, skip_special_tokens=False)
                scores = val_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                sample_inputs.extend(inputs)
                sample_outputs.extend(outputs)
                sample_scores.extend(scores)
            # iterate to the next
            self.agent_mode_daemon.clear_data_and_server()
            self.async_rollout_manager.sleep()

        # recalculate the metrics one by one
        test_metrics = {}
        for k, v in test_metrics_all.items():
            test_metrics[k] = np.mean(v)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict={},
                dump_path=val_data_dir,
            )
            val_steps = self.global_steps
            val_jsonl_path = os.path.join(val_data_dir, f"validation_metrics_step-{val_steps}.jsonl")
            with open(val_jsonl_path, "w") as fw:
                fw.write(json.dumps(test_metrics, ensure_ascii=False, indent=4))
        return test_metrics

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
                if "fake_ids" in gen_batch.batch:
                    device = gen_batch.batch["fake_ids"].device
                elif "input_ids" in gen_batch.batch:
                    device = gen_batch.batch["input_ids"].device
                else:
                    raise ValueError("valid device not found")
                batch, agent_metrics = self.agent_mode_daemon.get_train_data_batch(
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    device=device,
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
            if "response_mask" not in batch.batch:
                batch.batch["response_mask"] = compute_response_mask(batch)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with _timer("reward", timing_raw):
                # compute reward model score
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)
                reward_extra_infos_dict = {}
            print("🎃 返回生成的总样本长度", len(batch.batch))
            assert(len(batch.batch) > 0)
            # for agent mode, pad the lengths to calculate old log prob, ref, and values
            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
            print("📙 padding后的总样本长度", len(batch.batch))
            assert(len(batch.batch))
            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                print(">>> 计算log-prob/loss等")
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
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
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

            # after advantages are assinged, we begin to drop (1) long prompt (2) floor to ppo minisize
            keep_indices = (~batch.batch["is_drop_mask"]).nonzero(as_tuple=True)[0]
            metrics["training/n_triplets_prompt_too_long"] = (
                batch.batch["is_drop_mask"].shape[0] - keep_indices.shape[0]
            )
            num_before_drop = len(batch)
            batch = batch[keep_indices]
            num_after_drop = len(batch)
            print("🚨 过滤过长样本之前的总样本数：", num_before_drop, "过滤过长样本之后的总样本数：", num_after_drop)
            # next, round to minibatch size
            mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            num_gpus = self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node
            mini_batch_size = max(mini_batch_size, num_gpus)  # must can be divided by total number of gpus
            n_transition = len(batch)
            random_indices = list(range(n_transition))
            random.shuffle(random_indices)
            # batch.reorder(torch.tensor(random_indices).type(torch.int32))
            n_remained_transition = (n_transition // mini_batch_size) * mini_batch_size
            print(f"🐊 过滤样本至{mini_batch_size}整除后的总样本数：", n_remained_transition)
            assert (n_remained_transition >= mini_batch_size)
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

            # Start async dump BEFORE critic/actor update to maximize parallelism
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                with _timer("prepare_async_dump", timing_raw):
                    # Wait for previous dump to complete (if any)
                    if self._pending_dump_future is not None:
                        try:
                            prev_filename = ray.get(self._pending_dump_future)
                            print(f"[Async] Previous dump completed: {prev_filename}")
                        except Exception as e:
                            print(f"[Async] Previous dump failed: {e}")
                        self._pending_dump_future = None
                    
                    # Launch async dump (will run in parallel with critic/actor updates)
                    # Extract rollout_id_list if available
                    rollout_id_list = None
                    if "rollout_id_list" in batch.non_tensor_batch:
                        rollout_id_list = batch.non_tensor_batch["rollout_id_list"].tolist()
                    
                    self._pending_dump_future = decode_and_dump_async.remote(
                        prompts_list=batch.batch["prompts"].cpu().tolist(),
                        responses_list=batch.batch["responses"].cpu().tolist(),
                        scores=batch.batch["token_level_scores"].sum(-1).cpu().tolist(),
                        reward_extra_infos_dict=deepcopy(reward_extra_infos_dict),
                        dump_path=rollout_data_dir,
                        global_steps=self.global_steps,
                        tokenizer_name_or_path=self.config.actor_rollout_ref.model.path,
                        pad_token_id=self.tokenizer.pad_token_id,
                        rollout_id_list=rollout_id_list,
                    )
                    print(f"[Async] Launched async dump for step {self.global_steps}")

            # update critic (runs in parallel with async dump)
            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor (runs in parallel with async dump)
                with _timer("update_actor", timing_raw):
                    batch.meta_info["global_steps"] = self.global_steps
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

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
                
        # Extract validation kwargs from config
        
        verl_version = verl.__version__
        model_path = self.config.actor_rollout_ref.model.path
        model = "/".join(model_path.split("/")[-2:]) 
        # if verl_version == "0.5.0":
        #     # Note (Zhiyuan): To avoid further patch into vllm async server, using the same sentence to get the naming here.
        #     # However, it is possible that verl updates the naming and causes incompatibility.
        #     # Reference: https://github.com/volcengine/verl/blob/5b5e09d9cc20625e436d01f69d9cc739ff681c54/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L217
        #     model = "/".join(self.config.actor_rollout_ref.model.path.split("/")[-2:])
        # else:
        #     # For other versions (e.g., 0.6.0), we use the full path to the model.
        #     model = self.config.actor_rollout_ref.model.path
        val_kwargs = self.config.actor_rollout_ref.rollout.get("val_kwargs", {})
        val_information = {
            "model": model,
            "temperature": val_kwargs.get("temperature", 0.0),
            "top_p": val_kwargs.get("top_p", 1.0),
            "max_turns": self.config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns,
            "format": self.config.actor_rollout_ref.rollout.multi_turn.format,
        }
        val_rollout_n = val_kwargs.get("n", 1)
        
        self.agent_mode_daemon = AgentModeDaemon(
            self.config.agentlightning.port,
            self.config.actor_rollout_ref.rollout.n,
            train_information={
                "model": model,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
                "top_p": self.config.actor_rollout_ref.rollout.top_p,
                "max_turns": self.config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns,
                "format": self.config.actor_rollout_ref.rollout.multi_turn.format,
            },
            val_information=val_information,
            val_rollout_n=val_rollout_n,
            tokenizer=self.tokenizer,
            mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            pad_token_id=self.tokenizer.pad_token_id,
            mode="v1" if self.store is not None else "v0",
            store=self.store,
            llm_proxy=self.llm_proxy,
            adapter=self.adapter,
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

                # Wait for any pending dump before checkpoint or finish
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    if self._pending_dump_future is not None:
                        try:
                            prev_filename = ray.get(self._pending_dump_future)
                            print(f"[Async] Waiting for dump before checkpoint: {prev_filename}")
                        except Exception as e:
                            print(f"[Async] Dump failed before checkpoint: {e}")
                        self._pending_dump_future = None
                    
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

                if is_last_step:
                    # Ensure all async dumps are completed before finishing
                    if self._pending_dump_future is not None:
                        try:
                            prev_filename = ray.get(self._pending_dump_future)
                            print(f"[Async] Final dump completed: {prev_filename}")
                        except Exception as e:
                            print(f"[Async] Final dump failed: {e}")
                        self._pending_dump_future = None
                    
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()

                    # This exit logic is to ensure a robust CI.
                    pprint(f"Flush the logger...")
                    del logger  # Make sure the loggers are flushed and closed properly
                    pprint(f"Training finished at step {self.global_steps}.")
                    return

                progress_bar.update(1)
                self.global_steps += 1
