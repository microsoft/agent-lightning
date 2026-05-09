"""AglLiteRayPPOTrainer — RayPPOTrainer subclass that drives rollouts via agl-lite.

Modeled on Agent Lightning's AgentLightningTrainer, adapted for VERL 0.7.1.
The vLLM wake/sleep API moved in 0.7.1: use self.checkpoint_manager.update_weights()
and .sleep_replicas() instead of async_rollout_manager.wake_up()/sleep().

Per training step:
    1. (already awake, weights synced from prior step or initial _load_checkpoint)
    2. rollout_bridge.set_up_data_and_server   — register vLLM endpoints + enqueue rollouts
    3. rollout_bridge.run_until_all_finished   — poll until terminal
    4. rollout_bridge.get_train_data_batch     — assemble DataProto
    5. rollout_bridge.cleanup_agent_jobs       — delete tracked agent Jobs when enabled
    6. rollout_bridge.clear_data_and_server    — reset local bridge state
    7. self.checkpoint_manager.sleep_replicas()  — offload vLLM
    8. log-prob / KL / advantage / actor+critic update (stock VERL helpers)
    9. self.checkpoint_manager.update_weights(global_steps)  — wake + sync for next step
"""

# pyright: reportPrivateImportUsage=false
# pyright: reportUnusedCoroutine=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportReturnType=false

from __future__ import annotations

import logging
import uuid
from pprint import pprint
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import marked_timer
from verl.utils.tracking import Tracking

from .rollout_bridge import AglLiteRolloutBridge

log = logging.getLogger(__name__)


class AglLiteRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer that drives rollouts via the agl-lite HTTP API.

    Inherits VERL worker/checkpoint/init machinery and overrides the training
    and validation flow where rollout generation must go through agl-lite.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rollout_bridge: AglLiteRolloutBridge | None = None

    def _ensure_rollout_bridge(self) -> AglLiteRolloutBridge:
        if self._rollout_bridge is not None:
            return self._rollout_bridge
        al = self.config.agentlightning
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        trace_aggregator = al.get("trace_aggregator", None)
        if trace_aggregator is not None:
            trace_aggregator = OmegaConf.to_container(trace_aggregator, resolve=True)
        self._rollout_bridge = AglLiteRolloutBridge(
            agl_base_url=al.get("agl_base_url", "http://localhost:8080"),
            agl_key=al.get("agl_key", ""),
            train_rollout_n=self.config.actor_rollout_ref.rollout.n,
            train_information={
                "model": self.config.actor_rollout_ref.model.path,
                "resources_id": al.get("resources_id"),
            },
            tokenizer=self.tokenizer,
            mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            pad_token_id=pad_token_id,
            reward_fillna_value=al.get("reward_fillna_value", 0.0),
            timeout_seconds=al.get("timeout_seconds", 1200.0),
            processor=self.processor,
            image_base_dir=self.config.data.get("image_base_dir"),
            trace_aggregator=trace_aggregator,
            cleanup_agent_jobs=al.get("cleanup_agent_jobs", False),
            cleanup_namespace=al.get("cleanup_namespace", None),
        )
        return self._rollout_bridge

    def _rollout(self, gen_batch: DataProto, is_train: bool) -> tuple[DataProto, dict[str, Any]]:
        """Run the agl-lite rollout flow and return (DataProto, metrics).

        Training returns the DataProto assembled from agl-lite triplets plus
        rollout metrics. Validation returns metrics from ``get_test_metrics()``
        and an empty DataProto placeholder. In both paths, optional Job cleanup
        runs after results are extracted and before local bridge state is reset.
        """
        rollout_bridge = self._ensure_rollout_bridge()
        server_addresses = list(self.async_rollout_manager.server_addresses)
        data_dict = dict(gen_batch.non_tensor_batch)
        rollout_bridge.set_up_data_and_server(
            data=data_dict,
            server_addresses=server_addresses,
            is_train=is_train,
        )
        poll_timeout = self.config.agentlightning.get("poll_timeout_seconds", None)
        rollout_bridge.run_until_all_finished(verbose=True, timeout_seconds=poll_timeout)
        if is_train:
            trace_aggregator = self.config.agentlightning.get("trace_aggregator", {})
            level = trace_aggregator.get("level", "transition")
            max_prompt_length = (
                trace_aggregator.get("trajectory_max_prompt_length", self.config.data.max_prompt_length)
                if str(level).startswith("trajectory")
                else self.config.data.max_prompt_length
            )
            max_response_length = (
                trace_aggregator.get("trajectory_max_response_length", self.config.data.max_response_length)
                if str(level).startswith("trajectory")
                else self.config.data.max_response_length
            )
            out, metrics = rollout_bridge.get_train_data_batch(
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                device=torch.device("cpu"),
                global_steps=self.global_steps,
            )
            rollout_bridge.cleanup_agent_jobs()
            rollout_bridge.clear_data_and_server()
            return out, metrics
        # validation: caller will pull metrics via rollout_bridge.get_test_metrics()
        metrics = rollout_bridge.get_test_metrics()
        rollout_bridge.cleanup_agent_jobs()
        rollout_bridge.clear_data_and_server()
        return DataProto(batch=None), metrics

    def _train_step(
        self,
        batch_dict: dict[str, Any],
        timing_raw: dict[str, float],
        curr_step_profile: bool,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        rollout_n = self.config.actor_rollout_ref.rollout.n

        batch: DataProto = DataProto.from_single_dict(batch_dict)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps

        # ── 1. Rollout via agl-lite bridge ─────────────────────────────────
        with marked_timer("gen", timing_raw, color="red"):
            if curr_step_profile:
                self.async_rollout_manager.start_profile()

            gen_batch_output, agent_metrics = self._rollout(gen_batch, is_train=True)
            self.checkpoint_manager.sleep_replicas()
            if curr_step_profile:
                self.async_rollout_manager.stop_profile()
            metrics.update(agent_metrics)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("REMAX baseline not yet supported in AglLiteRayPPOTrainer")

        # Agent Lightning agent-mode output can contain one row per transition
        # or merged trajectory, so it is not positionally aligned with the
        # source dataloader batch. Use bridge output as the training batch.
        batch = gen_batch_output
        if "data_id_list" in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]
        else:
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        assert "token_level_scores" in batch.batch, "rollout bridge must populate token_level_scores"
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # ── 2. pad → log-probs / values → unpad ────────────────────────────
        divisor = self.actor_rollout_wg.world_size
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, divisor)

        with marked_timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob, _old_log_prob_mfu = self._compute_old_log_prob(batch_padded)
            if "entropys" in old_log_prob.batch:
                old_log_prob.batch.pop("entropys")
            batch_padded = batch_padded.union(old_log_prob)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                ref_log_prob = self._compute_ref_log_prob(batch_padded)
                batch_padded = batch_padded.union(ref_log_prob)

        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self._compute_values(batch_padded)
                batch_padded = batch_padded.union(values)

        batch = unpad_dataproto(batch_padded, pad_size=pad_size)

        # ── 3. KL → advantage (drop is_drop_mask AFTER advantage) ──────────
        with marked_timer("adv", timing_raw, color="brown"):
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=rollout_n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                config=self.config.algorithm,
            )

        # ── 4. Drop is_drop_mask + floor to ppo_mini_batch_size ────────────
        if "is_drop_mask" in batch.batch:
            keep = (~batch.batch["is_drop_mask"].bool()).nonzero(as_tuple=True)[0].tolist()
            metrics["training/n_triplets_prompt_too_long"] = len(batch) - len(keep)
            batch = batch[keep]
        mini_bs = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        trunc = (len(batch) // mini_bs) * mini_bs
        metrics["training/n_triplets_dropped_remainder"] = len(batch) - trunc
        batch = batch[:trunc] if trunc > 0 else batch[:0]
        if len(batch) == 0:
            metrics["agent/zero_after_drop"] = 1
            log.warning("batch empty after drop+floor; skipping update this step")
            return metrics

        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # ── 5. update critic / actor ───────────────────────────────────────
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self._update_critic(batch)
            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self._update_actor(batch)
            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

        # ── 6. wake + sync weights for next step ───────────────────────────
        with marked_timer("update_weights", timing_raw, color="red"):
            self.checkpoint_manager.update_weights(self.global_steps)

        return metrics

    def fit(self):
        """Training loop driven by AglLiteRolloutBridge for rollouts."""
        self._ensure_rollout_bridge()

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        # Wake vLLM + push checkpoint weights so the first rollout/validation
        # sees the loaded weights.
        self.checkpoint_manager.update_weights(self.global_steps)

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                timing_raw: dict[str, float] = {}
                is_last_step = self.global_steps >= self.total_training_steps

                curr_step_profile = (
                    self.global_steps in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )

                with marked_timer("step", timing_raw):
                    metrics = self._train_step(batch_dict, timing_raw, curr_step_profile)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("validate", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

    def _validate(self, merged: bool = False):
        """Validation via agl-lite rollout bridge.

        Wake/sleep contract:
          - vLLM is expected to be awake on current weights when this is called
            (fit() calls update_weights before val_before_train and after each
            actor update).
          - This method does NOT wake or sleep vLLM. It only registers/enqueues/
            polls/clears via the bridge. The next training step will sleep
            replicas at the end of its rollout block.
          - If a future change inserts sleep_replicas() between actor-update and
            validation, that branch must call update_weights again before
            entering _validate.
        """
        self._ensure_rollout_bridge()
        assert self.async_rollout_manager.server_addresses, (
            "_validate called before async_rollout_manager has server addresses"
        )

        merged_metrics: dict[str, Any] = {}
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            val_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
            if val_n > 1:
                test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            _, val_metrics_step = self._rollout(test_gen_batch, is_train=False)
            for k, v in val_metrics_step.items():
                merged_metrics[k] = v

        return merged_metrics
