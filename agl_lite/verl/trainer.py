"""AglLiteRayPPOTrainer — RayPPOTrainer subclass that drives rollouts via agl-lite.

Modeled on Agent Lightning's AgentLightningTrainer, adapted for VERL 0.7.1.
The vLLM wake/sleep API moved in 0.7.1: use self.checkpoint_manager.update_weights()
and .sleep_replicas() instead of async_rollout_manager.wake_up()/sleep().

Per training step:
    1. (already awake, weights synced from prior step or initial _load_checkpoint)
    2. rollout replicas resume_generation()  — accept new requests after prior abort
    3. rollout_bridge.set_up_data_and_server   — register vLLM endpoints + enqueue rollouts
    4. rollout_bridge.run_until_all_finished   — poll until terminal
    5. rollout_bridge.get_train_data_batch     — assemble DataProto
    6. rollout replicas abort_all_requests()  — kill residual requests; may pause vLLM generation
    7. rollout_bridge.clear_data_and_server    — reset local bridge state
    8. self.checkpoint_manager.sleep_replicas()  — offload vLLM
    9. log-prob / KL / advantage / actor+critic update (stock VERL helpers)
    10. self.checkpoint_manager.update_weights(global_steps)  — wake + sync for next step
"""

# pyright: reportPrivateImportUsage=false
# pyright: reportUnusedCoroutine=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportReturnType=false

from __future__ import annotations

import asyncio
import logging
import os
import random
import uuid
from pprint import pprint
from typing import Any, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import marked_timer
from verl.utils.ray_utils import auto_await
from verl.utils.tracking import Tracking

from agl_lite.hooks import RolloutHooks, load_hooks

from .rollout_bridge import AglLiteRolloutBridge
from .sample_iterator import SampleIterator

log = logging.getLogger(__name__)


def _tracking_backends_with_wandb(configured: Any, wandb_mode: str | None = None) -> list[str]:
    if configured is None:
        backends: list[str] = ["console"]
    elif isinstance(configured, str):
        backends = [configured]
    else:
        backends = [str(backend) for backend in configured]

    normalized_mode = (wandb_mode if wandb_mode is not None else os.environ.get("WANDB_MODE", "")).lower()
    if normalized_mode == "disabled":
        return [backend for backend in backends if backend != "wandb"] or ["console"]
    if "wandb" not in backends:
        backends.append("wandb")
    return backends


def _suffix_metrics(metrics: dict[str, Any], suffix: str) -> dict[str, Any]:
    return {f"{key}{suffix}": value for key, value in metrics.items()}


def _n_gpus_for_metrics(trainer: Any) -> int:
    resource_pool_manager = getattr(trainer, "resource_pool_manager", None)
    if resource_pool_manager is not None and hasattr(resource_pool_manager, "get_n_gpus"):
        try:
            return int(resource_pool_manager.get_n_gpus())
        except Exception:
            pass
    trainer_config = getattr(trainer.config, "trainer", {})
    n_gpus_per_node = int(trainer_config.get("n_gpus_per_node", 1))
    nnodes = int(trainer_config.get("nnodes", 1))
    return max(1, n_gpus_per_node * nnodes)


def _batch_dict_len(batch: dict[str, Any]) -> int:
    """Leading-dim length of a dataloader batch dict (0 if empty)."""
    if not batch:
        return 0
    for v in batch.values():
        if hasattr(v, "__len__"):
            return len(v)
    return 0


class AglLiteRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer that drives rollouts via the agl-lite HTTP API.

    Inherits VERL worker/checkpoint/init machinery and overrides the training
    and validation flow where rollout generation must go through agl-lite.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rollout_bridge: AglLiteRolloutBridge | None = None
        self._hooks: RolloutHooks | None = None

    def _ensure_hooks(self) -> RolloutHooks | None:
        if self._hooks is not None:
            return self._hooks
        hooks_path = self.config.agentlightning.get("hooks", None)
        if not hooks_path:
            return None
        self._hooks = load_hooks(hooks_path)
        self._hooks.on_startup()
        return self._hooks

    def _ensure_rollout_bridge(self) -> AglLiteRolloutBridge:
        if self._rollout_bridge is not None:
            return self._rollout_bridge
        al = self.config.agentlightning
        hooks = self._ensure_hooks()
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
            },
            tokenizer=self.tokenizer,
            mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            pad_token_id=pad_token_id,
            reward_fillna_value=al.get("reward_fillna_value", 0.0),
            timeout_seconds=al.get("timeout_seconds", 1200.0),
            processor=self.processor,
            image_base_dir=self.config.data.get("image_base_dir"),
            trace_aggregator=trace_aggregator,
            hooks=hooks,
            local_agent_class=al.get("local", {}).get("agent_class", None),
            local_env_map=OmegaConf.to_container(al.get("local", {}).get("env_map", {}), resolve=True),
            k8s_job_template_path=al.get("k8s", {}).get("job_template_path", None),
            cleanup_agent_jobs=al.get("cleanup_agent_jobs", False),
            cleanup_namespace=al.get("cleanup_namespace", None),
            cleanup_k8s_client=al.get("cleanup_k8s_client", None),
        )
        return self._rollout_bridge

    @auto_await
    async def _abort_all_rollout_requests(self) -> None:
        await asyncio.gather(*[replica.abort_all_requests() for replica in self.async_rollout_manager.rollout_replicas])

    @auto_await
    async def _resume_all_rollout_generation(self) -> None:
        await asyncio.gather(*[replica.resume_generation() for replica in self.async_rollout_manager.rollout_replicas])

    def _rollout(self, gen_batch: DataProto, is_train: bool) -> tuple[DataProto, dict[str, Any]]:
        """Run the agl-lite rollout flow and return (DataProto, metrics).

        Training returns the DataProto assembled from agl-lite triplets plus
        rollout metrics. Validation returns metrics from ``get_test_metrics()``
        and an empty DataProto placeholder.
        """
        rollout_bridge = self._ensure_rollout_bridge()
        server_addresses = list(self.async_rollout_manager.server_addresses)
        self._resume_all_rollout_generation()
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
            print("AglLiteRayPPOTrainer: aborting residual vLLM requests.")
            self._abort_all_rollout_requests()
            print("AglLiteRayPPOTrainer: residual vLLM requests aborted.")
            rollout_bridge.finish_sync_rollout_batch()
            return out, metrics
        # validation: caller will pull metrics via rollout_bridge.get_test_metrics()
        metrics = rollout_bridge.get_test_metrics()
        print("AglLiteRayPPOTrainer: aborting residual vLLM requests.")
        self._abort_all_rollout_requests()
        print("AglLiteRayPPOTrainer: residual vLLM requests aborted.")
        rollout_bridge.finish_sync_rollout_batch()
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
            print("AglLiteRayPPOTrainer: sleeping rollout replicas.")
            self.checkpoint_manager.sleep_replicas()
            print("AglLiteRayPPOTrainer: rollout replicas slept.")
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

        metrics.update(
            _suffix_metrics(
                compute_data_metrics(batch=batch, use_critic=self.use_critic),
                "_before_processing",
            )
        )

        # ── 4. Drop is_drop_mask + floor to ppo_mini_batch_size ────────────
        metrics["critic/n_transition_before_dropping"] = len(batch)
        if "is_drop_mask" in batch.batch:
            keep = (~batch.batch["is_drop_mask"].bool()).nonzero(as_tuple=True)[0].tolist()
            metrics["training/n_triplets_prompt_too_long"] = len(batch) - len(keep)
            batch = batch[keep]
        mini_bs = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        n_transition = len(batch)
        random_indices = list(range(n_transition))
        random.shuffle(random_indices)
        batch.reorder(torch.tensor(random_indices).type(torch.int32))
        n_remained_transition = n_transition // mini_bs * mini_bs
        metrics["training/n_triplets_dropped_remainder"] = n_transition - n_remained_transition
        batch = batch[list(range(n_remained_transition))]
        metrics["critic/n_transition_after_dropping"] = len(batch)
        if len(batch) == 0:
            metrics["agent/zero_after_drop"] = 1
            log.warning("batch empty after drop+floor; skipping update this step")
            return metrics

        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        metrics.update(
            _suffix_metrics(
                compute_data_metrics(batch=batch, use_critic=self.use_critic),
                "_after_processing",
            )
        )

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

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = _n_gpus_for_metrics(self)
        if n_gpus > 0 and "step" in timing_raw:
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        return metrics

    def fit(self):
        """Training loop driven by AglLiteRolloutBridge for rollouts."""
        self._ensure_rollout_bridge()

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=_tracking_backends_with_wandb(self.config.trainer.logger),
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

        for _epoch in range(self.config.trainer.total_epochs):
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

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

    # ────────────────────────────────────────────────────────────────────
    # Async-rollout training path — independent of fit() / _train_step() /
    # _rollout(). When ``agentlightning.async_rollout.enabled == False`` none
    # of the code below is loaded into the call stack, so sync RL behavior
    # remains byte-level equivalent to the pre-async version.
    # ────────────────────────────────────────────────────────────────────

    def _async_rollout(
        self,
        new_samples_dict: dict[str, Any],
        async_train_batch_size: int,
        gateway_retry_after_seconds: int,
        gateway_drain_timeout_seconds: float,
        rollout_n: int,
    ) -> tuple[DataProto, dict[str, Any]]:
        """Async rollout phase — only called from async_fit().

        Differs from _rollout():
          - Caller (async_fit) has already resumed the gateway at the top of
            this step's rollout phase; we do NOT resume here.
          - Uses run_until_groups_finished (group-finish early stop) instead
            of run_until_all_finished. The bridge pauses + drains the gateway
            internally before returning, so the caller can sleep_replicas()
            immediately after.
          - Carry-over rids stay alive across steps; we do NOT clear bridge
            state here. cleanup_agent_jobs() runs only for selected (consumed)
            rids — carry-over Job/pods are preserved.
          - Validation continues to use the legacy _rollout(is_train=False).
        """
        rollout_bridge = self._ensure_rollout_bridge()
        server_addresses = list(self.async_rollout_manager.server_addresses)
        self._resume_all_rollout_generation()

        # 1. Register + enqueue only the new samples for this step.
        rollout_bridge.async_set_up_data_and_server(
            data=new_samples_dict,
            server_addresses=server_addresses,
            async_train_batch_size=async_train_batch_size,
        )

        # 2. Poll until ``train_batch_size`` complete groups exist inside the
        #    larger active pool, then pause + drain the gateway.
        poll_timeout = self.config.agentlightning.get("poll_timeout_seconds", None)
        selected_rids, unselected_rids, async_poll_metrics = rollout_bridge.run_until_groups_finished(
            target_groups=self.config.data.train_batch_size,
            rollout_n=rollout_n,
            drain_timeout=gateway_drain_timeout_seconds,
            timeout_seconds=poll_timeout,
            retry_after_seconds=gateway_retry_after_seconds,
            step_label=f"step={self.global_steps}",
            verbose=True,
        )

        # 3. Commit the selection (updates carry-over pool, birth-step tracking).
        carry_over_metrics = rollout_bridge.commit_async_step_selection(
            selected_rids=selected_rids,
            unselected_rids=unselected_rids,
            current_step=self.global_steps,
        )

        # 4. Assemble the training batch from the selected rids (mirrors the
        #    sync trace-aggregator path).
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
        out, assembly_metrics = rollout_bridge.async_get_train_data_batch(
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            device=torch.device("cpu"),
            global_steps=self.global_steps,
        )

        # 5. Clean up K8s Jobs for the selected (consumed) rids ONLY.
        #    Carry-over rids' Jobs/pods MUST survive across steps.
        rollout_bridge.async_cleanup_consumed(consumed_rids=selected_rids)

        metrics: dict[str, Any] = {}
        metrics.update(assembly_metrics)
        metrics.update(async_poll_metrics)
        metrics.update(carry_over_metrics)
        return out, metrics

    def _async_train_step(
        self,
        new_samples_dict: dict[str, Any],
        timing_raw: dict[str, float],
        curr_step_profile: bool,
        async_train_batch_size: int,
        gateway_retry_after_seconds: int,
        gateway_drain_timeout_seconds: float,
    ) -> dict[str, Any]:
        """One async-rollout training step.

        Mirrors _train_step() but uses _async_rollout. The rollout phase
        consumes the **selected** rids only; carry-over rids stay alive in
        the bridge across steps.
        """
        metrics: dict[str, Any] = {}
        rollout_n = self.config.actor_rollout_ref.rollout.n

        # ── 1. Rollout via agl-lite bridge (group-finish + pause/drain) ────
        with marked_timer("gen", timing_raw, color="red"):
            if curr_step_profile:
                self.async_rollout_manager.start_profile()

            gen_batch_output, agent_metrics = self._async_rollout(
                new_samples_dict=new_samples_dict,
                async_train_batch_size=async_train_batch_size,
                gateway_retry_after_seconds=gateway_retry_after_seconds,
                gateway_drain_timeout_seconds=gateway_drain_timeout_seconds,
                rollout_n=rollout_n,
            )
            self.checkpoint_manager.sleep_replicas()
            if curr_step_profile:
                self.async_rollout_manager.stop_profile()
            metrics.update(agent_metrics)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("REMAX baseline not yet supported in AglLiteRayPPOTrainer")

        batch = gen_batch_output
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        batch.meta_info["global_steps"] = self.global_steps
        if "data_id_list" in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]
        else:
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )

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

        # ── 3. KL → advantage ──────────────────────────────────────────────
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
        batch_sliceable = cast(Any, batch)
        batch = batch_sliceable[:trunc] if trunc > 0 else batch_sliceable[:0]
        if len(batch) == 0:
            metrics["agent/zero_after_drop"] = 1
            log.warning("async batch empty after drop+floor; skipping update this step")
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

    def _validate_preserving_async_carry_over(self) -> dict[str, Any]:
        """Run legacy validation without corrupting async carry-over state.

        Validation still uses the sync rollout path, whose bridge cleanup clears
        shared per-rid bookkeeping. In async mode those same structures also
        contain carry-over rollouts that must be polled or consumed again next
        step, so snapshot the carry-over slice and put it back after validation
        finishes.
        """
        bridge = self._ensure_rollout_bridge()
        carry_over_rids = set(getattr(bridge, "_carry_over_rids", set()))
        if not carry_over_rids:
            return self._validate()

        carry_over_dids = {
            did for rid in carry_over_rids
            if (did := getattr(bridge, "_rid_to_data_id", {}).get(rid)) is not None
        }

        def by_rid(name: str) -> dict[Any, Any]:
            return {
                rid: value for rid, value in getattr(bridge, name, {}).items()
                if rid in carry_over_rids
            }

        def set_by_rid(name: str) -> set[Any]:
            return set(getattr(bridge, name, set())) & carry_over_rids

        enqueue_order = [rid for rid in getattr(bridge, "_enqueue_order", []) if rid in carry_over_rids]
        snapshots = {
            "_completed_rollouts": by_rid("_completed_rollouts"),
            "_task_id_to_original_sample": by_rid("_task_id_to_original_sample"),
            "_rollout_status": by_rid("_rollout_status"),
            "_rollout_error": by_rid("_rollout_error"),
            "_rollout_start_time": by_rid("_rollout_start_time"),
            "_rollout_end_time": by_rid("_rollout_end_time"),
            "_raw_events_by_rollout": by_rid("_raw_events_by_rollout"),
            "_triplet_events_by_rollout": by_rid("_triplet_events_by_rollout"),
            "_carry_over_birth_step": by_rid("_carry_over_birth_step"),
        }
        timeout_rids = set_by_rid("_timeout_rids")
        step_new_rids = set_by_rid("_step_new_rids")
        selected_rids = set_by_rid("_selected_rids")
        group_finish_time = {
            did: value for did, value in getattr(bridge, "_group_finish_time", {}).items()
            if did in carry_over_dids
        }

        try:
            return self._validate()
        finally:
            existing_order = [rid for rid in getattr(bridge, "_enqueue_order", []) if rid not in carry_over_rids]
            bridge._enqueue_order = enqueue_order + existing_order
            for name, values in snapshots.items():
                getattr(bridge, name).update(values)
            bridge._timeout_rids.update(timeout_rids)
            bridge._step_new_rids.update(step_new_rids)
            bridge._selected_rids.update(selected_rids)
            bridge._group_finish_time.update(group_finish_time)

    def async_fit(self):
        """Async-rollout training loop — distinct from fit().

        Driven by ``global_steps < total_training_steps``, NOT by
        ``for batch_dict in self.train_dataloader``. Each step:

          1. n_new = async_train_batch_size - bridge.n_carry_over_data_ids()
          2. samples = sample_iterator.take(n_new)
          3. resume gateway (it was paused at the end of the previous step)
          4. _async_train_step(samples, ...) — which calls _async_rollout
             (run_until_groups_finished → pause+drain) then sleep_replicas →
             training → update_weights.

        Carry-over rids stay alive in the bridge across steps so stateful
        agent pods (Claude Code/Codex/Cursor) preserve their working tree.
        """
        from agl_lite.client import AglLiteSyncClient

        self._ensure_rollout_bridge()

        async_cfg = self.config.agentlightning.async_rollout
        async_train_batch_size = async_cfg.async_train_batch_size
        gateway_retry_after_seconds = int(async_cfg.get("gateway_retry_after_seconds", 5))
        gateway_drain_timeout_seconds = float(async_cfg.get("gateway_drain_timeout_seconds", 30.0))
        max_carry_over_age_steps = async_cfg.get("max_carry_over_age_steps", None)
        allow_equal_batch_size_for_debug = bool(async_cfg.get("allow_equal_batch_size_for_debug", False))

        train_batch_size = self.config.data.train_batch_size

        # Startup constraint checks (§3.6).
        if async_train_batch_size is None:
            raise ValueError(
                "agentlightning.async_rollout.async_train_batch_size must be set when "
                "async_rollout.enabled=true."
            )
        if async_train_batch_size < train_batch_size:
            raise ValueError(
                f"async_train_batch_size ({async_train_batch_size}) must be >= "
                f"data.train_batch_size ({train_batch_size})."
            )
        if async_train_batch_size == train_batch_size and not allow_equal_batch_size_for_debug:
            raise ValueError(
                f"async_train_batch_size == data.train_batch_size ({train_batch_size}) — "
                "no overshoot, defeats the purpose of async rollout. Set "
                "async_rollout.allow_equal_batch_size_for_debug=true to override (debug only)."
            )

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Async Training")

        sample_iter = SampleIterator(self.train_dataloader)
        self.global_steps += 1
        last_val_metrics = None

        # Short-lived sync client for resume_gateway between steps.
        # Built per call (cheap) inside _resume_gateway_sync to keep async_fit
        # itself free of asyncio plumbing.
        def _resume_gateway_sync() -> None:
            with AglLiteSyncClient(
                base_url=self.config.agentlightning.agl_base_url,
                key=self.config.agentlightning.get("agl_key", ""),
                timeout=30.0,
            ) as client:
                response = client.post("/proxy/resume")
                response.raise_for_status()

        while self.global_steps <= self.total_training_steps:
            timing_raw: dict[str, float] = {}
            is_last_step = self.global_steps >= self.total_training_steps

            curr_step_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )

            with marked_timer("step", timing_raw):
                # 1. Compute how many NEW data_ids this step needs.
                assert self._rollout_bridge is not None
                n_carry_over = self._rollout_bridge.n_carry_over_data_ids()
                n_new = async_train_batch_size - n_carry_over
                if n_new <= 0:
                    raise RuntimeError(
                        "async rollout carry-over saturated: "
                        f"async_train_batch_size={async_train_batch_size}, "
                        f"n_carry_over_data_ids={n_carry_over}. "
                        "Carry-over-only steps are not supported; increase async_train_batch_size "
                        "or reduce long-tail carry-over before continuing."
                    )

                # 2. Pull n_new samples from the dataloader. take(n) may cross
                #    epoch boundaries to fill n. n_new must be positive; carry-
                #    over-only steps are rejected above.
                new_samples_dict, cross_epoch = sample_iter.take(n_new)
                # If we have neither carry-over nor new samples, training is done.
                if not new_samples_dict and n_carry_over == 0:
                    log.info("async_fit: dataloader exhausted and no carry-over — ending training.")
                    progress_bar.close()
                    return

                # 3. Resume the gateway (it was paused by the previous step's
                #    rollout phase). The gateway starts un-paused on the first
                #    step too — this resume is a no-op then.
                _resume_gateway_sync()

                # 4. Async training step.
                metrics = self._async_train_step(
                    new_samples_dict=new_samples_dict,
                    timing_raw=timing_raw,
                    curr_step_profile=curr_step_profile,
                    async_train_batch_size=async_train_batch_size,
                    gateway_retry_after_seconds=gateway_retry_after_seconds,
                    gateway_drain_timeout_seconds=gateway_drain_timeout_seconds,
                )

            metrics.setdefault(
                "training/async/n_new_data_ids", _batch_dict_len(new_samples_dict)
            )
            metrics["training/async/n_carry_over_data_ids_in"] = n_carry_over
            metrics["training/async/sample_iterator_epoch"] = sample_iter.epoch
            metrics["training/async/sample_iterator_consumed"] = sample_iter.consumed
            metrics["training/async/cross_epoch_boundary"] = int(cross_epoch)

            # Optional: warn on stale carry-over rids.
            if max_carry_over_age_steps is not None:
                age = metrics.get("training/async/carry_over_age_max_steps", 0)
                if age and age > int(max_carry_over_age_steps):
                    log.warning(
                        "async_fit: carry-over rid age %d > max_carry_over_age_steps %d — "
                        "possible stuck rollout. step=%d",
                        age, int(max_carry_over_age_steps), self.global_steps,
                    )

            if self.config.trainer.test_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.test_freq == 0
            ):
                with marked_timer("validate", timing_raw, color="green"):
                    # _validate uses the legacy _rollout path — first resume
                    # the gateway (rollout phase paused it).
                    _resume_gateway_sync()
                    val_metrics = self._validate_preserving_async_carry_over()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)
                metrics["training/async/val_to_train_engine_reset"] = 1

            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                with marked_timer("save_checkpoint", timing_raw):
                    self._save_checkpoint()

            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/epoch": sample_iter.epoch,
                }
            )
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1

        progress_bar.close()

    def _validate(self, merged: bool = False):
        """Validation via agl-lite rollout bridge.

        Wake/sleep contract:
                    - vLLM is expected to be awake on current weights when this is called
                        (fit() calls update_weights before val_before_train and after each
                        actor update).
                    - _rollout() resumes generation before enqueueing requests, because
                        abort_all_requests() can leave newer vLLM engines paused.
                    - This method does NOT wake, sleep, or sync vLLM weights. It only
                        registers/enqueues/polls/clears via the bridge. The next training
                        step will sleep replicas at the end of its rollout block.
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
