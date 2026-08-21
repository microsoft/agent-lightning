# Copyright (c) Microsoft. All rights reserved.

"""AgentLightningRayPPOTrainer drives VERL rollouts through Agent Lightning."""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Any, TypeVar

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import marked_timer
from verl.utils.ray_utils import auto_await
from verl.utils.tracking import Tracking

from agentlightning.client import AgentLightningSyncClient
from agentlightning.hooks import RolloutHooks, load_hooks

from .agl_rollout_manager import (
    AglAsyncRolloutManager,
    AglRolloutManager,
    AglRolloutManagerBase,
    CompletedRollout,
    EnqueuedRollout,
)
from .per_rollout_loss import PER_ROLLOUT_MEAN_LOSS_MODE, normalize_advantages_by_rollout
from .rollout_adapter import RolloutAdapter
from .rollout_level_advantage import compute_rollout_level_advantage

log = logging.getLogger(__name__)

RolloutManagerT = TypeVar("RolloutManagerT", bound=AglRolloutManagerBase)


def _batch_dict_len(batch: dict[str, Any] | None) -> int:
    """Leading-dim length of a dataloader batch dict (0 if absent/empty)."""
    if batch is None or not batch:
        return 0
    return len(next(iter(batch.values())))


def _grpo_group_metrics(batch: Any) -> dict[str, int]:
    """Count GRPO groups and how many have zero intra-group reward variance."""
    uids = batch.non_tensor_batch.get("uid")
    scores = batch.batch.get("token_level_scores")
    if uids is None or scores is None:
        return {}
    sequence_score = scores.sum(-1).detach().float().cpu()
    groups: dict[Any, list[float]] = defaultdict(list)
    for uid, score in zip(uids, sequence_score.tolist(), strict=False):
        groups[uid].append(score)
    n_zero_adv = sum(1 for vals in groups.values() if max(vals) - min(vals) == 0.0)
    return {
        "training/n_groups": len(groups),
        "training/n_zero_adv_groups": n_zero_adv,
    }


def _same_reward_uid_indices(batch: DataProto) -> list[int]:
    if "uid" not in batch.non_tensor_batch:
        return []

    rewards = batch.batch["token_level_scores"].sum(dim=-1)
    uid_to_indices: dict[Any, list[int]] = {}
    for sample_idx, uid in enumerate(batch.non_tensor_batch["uid"]):
        uid_to_indices.setdefault(uid, []).append(sample_idx)

    same_reward_indices: list[int] = []
    for indices in uid_to_indices.values():
        group_rewards = rewards[indices]
        if torch.allclose(group_rewards, group_rewards[0].expand_as(group_rewards)):
            same_reward_indices.extend(indices)

    return same_reward_indices


class AgentLightningRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer that drives train and validation rollouts via Agent Lightning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_async = self.config.agentlightning.async_rollout.enabled
        self.epoch = 0
        async_train_batch_size = self.config.agentlightning.async_rollout.async_train_batch_size
        train_batch_size = self.config.data.train_batch_size
        if self.is_async and async_train_batch_size <= train_batch_size:
            raise ValueError(
                f"async_train_batch_size ({async_train_batch_size}) must be > "
                f"data.train_batch_size ({train_batch_size})."
            )
        self._hooks: RolloutHooks | None = None
        self._agl_client: AgentLightningSyncClient | None = None
        self._carry_over_rollouts: list[EnqueuedRollout] = []
        self._train_dataloader_iter: Any | None = None

    def _ensure_hooks(self) -> RolloutHooks | None:
        if self._hooks is not None:
            return self._hooks
        hooks_path = self.config.agentlightning.hooks
        if not hooks_path:
            return None
        self._hooks = load_hooks(hooks_path)
        self._hooks.on_startup()
        return self._hooks

    def _ensure_agl_client(self) -> AgentLightningSyncClient:
        if self._agl_client is not None:
            return self._agl_client
        self._agl_client = AgentLightningSyncClient(
            base_url=self.config.agentlightning.agl_base_url,
            key=self.config.agentlightning.agl_key,
            timeout=300,
        )
        return self._agl_client

    def _make_rollout_manager(self, manager_cls: type[RolloutManagerT]) -> RolloutManagerT:
        al = self.config.agentlightning
        return manager_cls(
            agl_base_url=al.agl_base_url,
            agl_key=al.agl_key,
            model=self.config.actor_rollout_ref.model.path,
            step=self.global_steps,
            train_rollout_n=self.config.actor_rollout_ref.rollout.n,
            rollout_timeout_seconds=al.rollout_timeout_seconds,
            hooks=self._ensure_hooks(),
            local_agent_class=al.local.agent_class,
            local_env_map=al.local.env_map,
            k8s_job_template_path=al.k8s.job_template_path,
        )

    def _rollout_replicas(self) -> list[Any]:
        if hasattr(self, "llm_server_manager"):
            return list(self.llm_server_manager.get_replicas())
        return list(self.async_rollout_manager.rollout_replicas)

    @auto_await
    async def _abort_all_rollout_requests(self) -> None:
        await asyncio.gather(*[replica.abort_all_requests() for replica in self._rollout_replicas()])

    @auto_await
    async def _resume_all_rollout_generation(self) -> None:
        await asyncio.gather(*[replica.resume_generation() for replica in self._rollout_replicas()])

    def _resume_gateway(self) -> None:
        # Resuming is idempotent and safe to retry.
        self._ensure_agl_client().post_with_retry("/proxy/resume")

    def _pause_and_drain_gateway(self, *, reason: str) -> dict[str, Any]:
        client = self._ensure_agl_client()

        # Pausing is idempotent; allow five minutes when in-flight requests saturate the server.
        response = client.post_with_retry("/proxy/pause", json={"reason": reason}, timeout=300.0)
        paused_payload = response.json()
        inflight_on_pause = int(paused_payload.get("inflight", 0))

        drain_started_at = time.perf_counter()
        residual = inflight_on_pause
        while True:
            response = client.get("/proxy/state")
            response.raise_for_status()
            residual = int(response.json().get("inflight", 0))
            if residual <= 0:
                break
            time.sleep(0.25)
        drain_seconds = time.perf_counter() - drain_started_at

        return {
            "training/async/proxy_inflight_at_pause": inflight_on_pause,
            "training/async/proxy_drain_seconds": drain_seconds,
        }

    def _compute_async_rollout_metrics(
        self,
        *,
        previous_carry_over_rollouts: list[EnqueuedRollout],
        completed_rollouts: list[CompletedRollout],
        new_carry_over_rollouts: list[EnqueuedRollout],
    ) -> dict[str, Any]:
        max_carry_over_age = max(
            (self.global_steps - rollout.step for rollout in new_carry_over_rollouts),
            default=0,
        )
        return {
            "training/async/n_prev_carry_over_rollouts": len(previous_carry_over_rollouts),
            "training/async/n_completed_rollouts": len(completed_rollouts),
            "training/async/n_new_carry_over_rollouts": len(new_carry_over_rollouts),
            "training/async/new_carry_over_age_max_steps": max_carry_over_age,
        }

    def _rollout_lifecycle_metrics(
        self,
        completed_rollouts: list[CompletedRollout],
    ) -> dict[str, Any]:
        """Per-rollout pod queue/run timing for the current step.

        Emits scalar aggregates of the queue wait / run duration / total so the
        "pods launched in batches" effect shows up as a trend curve. Pod startup
        is approximated by running_at - submitted (queue + init), which is
        exactly the wait we want to watch under CPU-limited batched launch.

        The per-rollout wandb Table (rollout_lifecycle/step_N) is disabled: its
        one-row-per-rollout payload was large and slow to log.
        """
        if not completed_rollouts:
            return {}

        queue_waits: list[float] = []
        run_durations: list[float] = []
        totals: list[float] = []
        n_missing_running = 0

        for rollout in completed_rollouts:
            submitted = rollout.enqueue_time
            running_at = rollout.running_at
            finished_at = rollout.finished_at
            queue_wait = (running_at - submitted) if running_at is not None else None
            run_duration = finished_at - running_at if (running_at is not None and finished_at is not None) else None
            total = (finished_at - submitted) if finished_at is not None else None
            if queue_wait is not None:
                queue_waits.append(queue_wait)
            else:
                n_missing_running += 1
            if run_duration is not None:
                run_durations.append(run_duration)
            if total is not None:
                totals.append(total)

        metrics: dict[str, Any] = {}
        # Keep scalar timings; per-rollout W&B tables are too large and slow.

        def _agg(prefix: str, values: list[float]) -> None:
            if not values:
                return
            arr = np.asarray(values, dtype=float)
            metrics[f"{prefix}/mean"] = float(arr.mean())
            metrics[f"{prefix}/p50"] = float(np.percentile(arr, 50))
            metrics[f"{prefix}/p90"] = float(np.percentile(arr, 90))
            metrics[f"{prefix}/max"] = float(arr.max())

        _agg("timing/rollout_queue_wait_s", queue_waits)
        _agg("timing/rollout_run_duration_s", run_durations)
        _agg("timing/rollout_total_s", totals)
        metrics["timing/rollout_n_missing_running_ts"] = n_missing_running
        return metrics

    def _next_train_batch_dict_for_rollout(self) -> dict[str, Any]:
        if not self.is_async:
            return self._next_train_batch_dict()

        async_cfg = self.config.agentlightning.async_rollout
        async_train_batch_size = async_cfg.async_train_batch_size
        n_carry_over = len({rollout.data_id for rollout in self._carry_over_rollouts})
        n_new = int(async_train_batch_size) - n_carry_over
        return self._next_train_batch_dict_with_size(n_new)

    def _next_train_batch_dict(self) -> dict[str, Any]:
        while True:
            if self._train_dataloader_iter is None:
                self._train_dataloader_iter = iter(self.train_dataloader)

            for batch_dict in self._train_dataloader_iter:
                return batch_dict

            self.epoch += 1
            self._train_dataloader_iter = None

    def _next_train_batch_dict_with_size(self, size: int) -> dict[str, Any]:
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")

        def split_batch(batch: dict[str, Any], head_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
            head: dict[str, Any] = {}
            tail: dict[str, Any] = {}
            for key, value in batch.items():
                head[key] = value[:head_size]
                tail[key] = value[head_size:]
            return head, tail

        def concat_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
            if len(batches) == 1:
                return batches[0]

            output: dict[str, Any] = {}
            for key in batches[0]:
                values = [batch[key] for batch in batches]
                sample = values[0]
                if isinstance(sample, torch.Tensor):
                    output[key] = torch.cat(values, dim=0)
                elif isinstance(sample, np.ndarray):
                    output[key] = np.concatenate(values, axis=0)
                elif isinstance(sample, list):
                    merged: list[Any] = []
                    for value in values:
                        merged.extend(value)
                    output[key] = merged
                else:
                    raise TypeError(
                        f"unsupported batch value type for key {key!r}: {type(sample).__name__}. "
                        "Expected torch.Tensor, numpy.ndarray, or list."
                    )
            return output

        collected: list[dict[str, Any]] = []
        remaining = size
        buffered_batch = getattr(self, "_train_dataloader_buf", None)

        if buffered_batch is not None:
            buffered_size = _batch_dict_len(buffered_batch)
            if buffered_size <= remaining:
                collected.append(buffered_batch)
                remaining -= buffered_size
                self._train_dataloader_buf = None
            else:
                head, tail = split_batch(buffered_batch, remaining)
                collected.append(head)
                self._train_dataloader_buf = tail
                remaining = 0

        while remaining > 0:
            if self._train_dataloader_iter is None:
                self._train_dataloader_iter = iter(self.train_dataloader)

            try:
                batch_dict = next(self._train_dataloader_iter)
            except StopIteration:
                self.epoch += 1
                self._train_dataloader_iter = None
                continue

            batch_size = _batch_dict_len(batch_dict)
            if batch_size <= remaining:
                collected.append(batch_dict)
                remaining -= batch_size
            else:
                head, tail = split_batch(batch_dict, remaining)
                collected.append(head)
                self._train_dataloader_buf = tail
                remaining = 0

        return concat_batches(collected)

    def _rollout(self, gen_batch: DataProto, is_train: bool) -> tuple[DataProto, dict[str, Any]]:
        """Run Agent Lightning rollouts and return the resulting DataProto plus metrics."""
        # verl 0.8.0 moved rollout server state behind llm_server_manager.
        has_llm_server_manager = hasattr(self, "llm_server_manager")
        if has_llm_server_manager:
            server_addresses = list(self.llm_server_manager.get_addresses())
        else:
            server_addresses = list(self.async_rollout_manager.server_addresses)
        self._resume_all_rollout_generation()
        if self.is_async:
            self._resume_gateway()
        data_dict = dict(gen_batch.non_tensor_batch)

        async_rollout_metrics: dict[str, Any] = {}
        if self.is_async and is_train:
            rollout_manager = self._make_rollout_manager(AglAsyncRolloutManager)
            rollout_manager.delete_model()
            rollout_manager.register_model(server_addresses)
            previous_carry_over_rollouts = list(self._carry_over_rollouts)
            completed_rollouts, new_carry_over_rollouts = rollout_manager.enqueue_and_wait_until_group_completed(
                data_dict,
                previous_carry_over_rollouts,
                is_train=True,
                target_finished_group_num=self.config.data.train_batch_size,
            )
            self._carry_over_rollouts = new_carry_over_rollouts
            async_rollout_metrics = self._compute_async_rollout_metrics(
                previous_carry_over_rollouts=previous_carry_over_rollouts,
                completed_rollouts=completed_rollouts,
                new_carry_over_rollouts=new_carry_over_rollouts,
            )
        else:
            rollout_manager = self._make_rollout_manager(AglRolloutManager)
            rollout_manager.delete_model()
            rollout_manager.register_model(server_addresses)
            completed_rollouts = rollout_manager.enqueue_and_wait_until_completed(data_dict, is_train=is_train)

        trace_aggregator = self.config.agentlightning.trace_aggregator
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
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        rollout_adapter = RolloutAdapter(
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            device=torch.device("cpu"),
            pad_token_id=pad_token_id,
            reward_fillna_value=self.config.agentlightning.reward_fillna_value,
            trace_aggregator_level=level,
            tokenizer=self.tokenizer,
        )

        if is_train:
            out, metrics = rollout_adapter.get_train_data_batch(
                completed_rollouts,
                global_steps=self.global_steps,
            )
            metrics.update(async_rollout_metrics)
            metrics.update(self._rollout_lifecycle_metrics(completed_rollouts))
        else:
            metrics = rollout_adapter.get_test_metrics(completed_rollouts, global_steps=self.global_steps)
            out = DataProto(batch=None)

        if self.is_async:
            print("AgentLightningRayPPOTrainer: pausing and draining agl gateway.")
            metrics.update(self._pause_and_drain_gateway(reason=f"rollout_done step={self.global_steps}"))
            print("AgentLightningRayPPOTrainer: agl gateway paused and drained.")
        else:
            print("AgentLightningRayPPOTrainer: aborting residual vLLM requests.")
            self._abort_all_rollout_requests()
            print("AgentLightningRayPPOTrainer: residual vLLM requests aborted.")
        return out, metrics

    def _train_step(
        self,
        timing_raw: dict[str, float],
        curr_step_profile: bool,
    ) -> tuple[dict[str, Any], DataProto] | None:
        metrics: dict[str, Any] = {}
        self._step_start_wall = time.time()
        metrics["timing/step_start_wall"] = self._step_start_wall
        rollout_n = self.config.actor_rollout_ref.rollout.n

        batch_dict = self._next_train_batch_dict_for_rollout()

        batch: DataProto = DataProto.from_single_dict(batch_dict)

        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        # verl 0.8.0 moved rollout profiling behind llm_server_manager.
        has_llm_server_manager = hasattr(self, "llm_server_manager")

        with marked_timer("gen", timing_raw, color="red"):
            if curr_step_profile:
                if has_llm_server_manager:
                    self.llm_server_manager.start_profile()
                else:
                    self.async_rollout_manager.start_profile()

            gen_batch_output, agent_metrics = self._rollout(gen_batch, is_train=True)
            if curr_step_profile:
                if has_llm_server_manager:
                    self.llm_server_manager.stop_profile()
                else:
                    self.async_rollout_manager.stop_profile()
            metrics.update(agent_metrics)
        metrics["timing/rollout_phase_end_wall"] = time.time()

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("REMAX baseline not yet supported in AgentLightningRayPPOTrainer")

        batch = gen_batch_output
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        if "data_id_list" in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]
        else:
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        assert "token_level_scores" in batch.batch, "rollout bridge must populate token_level_scores"
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        metrics["training/n_sample_collected"] = len(batch)
        if "is_drop_mask" in batch.batch:
            keep = (~batch.batch["is_drop_mask"].bool()).nonzero(as_tuple=True)[0].tolist()
            metrics["training/n_sample_dropped/marked"] = len(batch) - len(keep)
            batch = batch[keep]

        mini_bs = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        n_transition = len(batch)
        max_ppo_update_times = self.config.agentlightning.get("max_ppo_update_times", None)
        n_remained_transition = n_transition // mini_bs * mini_bs
        if max_ppo_update_times is not None:
            n_remained_transition = min(n_remained_transition, mini_bs * max_ppo_update_times)

        n_to_drop = n_transition - n_remained_transition
        n_dropped_same_reward = 0
        n_dropped_random = 0
        if n_to_drop > 0:
            same_reward_indices = _same_reward_uid_indices(batch)
            random.shuffle(same_reward_indices)
            same_reward_drop_indices = same_reward_indices[:n_to_drop]
            same_reward_drop_set = set(same_reward_drop_indices)
            n_dropped_same_reward = len(same_reward_drop_indices)

            n_random_to_drop = n_to_drop - n_dropped_same_reward
            random_drop_indices: list[int] = []
            if n_random_to_drop > 0:
                random_candidates = [
                    sample_idx for sample_idx in range(n_transition) if sample_idx not in same_reward_drop_set
                ]
                random.shuffle(random_candidates)
                random_drop_indices = random_candidates[:n_random_to_drop]
                n_dropped_random = len(random_drop_indices)

            drop_indices = same_reward_drop_set | set(random_drop_indices)
            keep_indices = [sample_idx for sample_idx in range(n_transition) if sample_idx not in drop_indices]
            batch = batch[keep_indices]
        metrics["training/n_sample_dropped/same_reward"] = n_dropped_same_reward
        metrics["training/n_sample_dropped/random"] = n_dropped_random
        metrics["training/n_sample_trained"] = len(batch)
        if len(batch) == 0:
            print("WARNING: no trainable batch after drop+floor; skipping this training step.")
            return None

        print("AgentLightningRayPPOTrainer: sleeping rollout replicas.")
        self.checkpoint_manager.sleep_replicas()
        print("AgentLightningRayPPOTrainer: rollout replicas slept.")

        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_mode = bool(rollout_corr_config and rollout_corr_config.get("bypass_mode", False))

        if bypass_mode:
            if "rollout_log_probs" not in batch.batch:
                raise RuntimeError("bypass_mode requires rollout_log_probs in batch")
            if not torch.isfinite(batch.batch["rollout_log_probs"]).all():
                raise RuntimeError("bypass_mode requires finite rollout_log_probs everywhere")
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                apply_bypass_mode(
                    batch,
                    rollout_corr_config,
                    self.config.actor_rollout_ref.actor.policy_loss,
                )
        else:
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                metrics["perf/mfu/actor_infer"] = old_log_prob_mfu
                if "entropys" in old_log_prob.batch:
                    old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self._compute_values(batch)
                batch = batch.union(values)

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

            if rollout_corr_config is not None and not bypass_mode and "rollout_log_probs" in batch.batch:
                from verl.trainer.ppo.rollout_corr_helper import (
                    compute_rollout_correction_and_add_to_batch,
                )

                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                metrics.update(is_metrics)

            adv_kwargs = {
                "adv_estimator": self.config.algorithm.adv_estimator,
                "gamma": self.config.algorithm.gamma,
                "lam": self.config.algorithm.lam,
                "num_repeat": rollout_n,
                "norm_adv_by_std_in_grpo": self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                "config": self.config.algorithm,
            }
            if self.config.algorithm.get("enable_rollout_level_advantage", False):
                batch, rollout_adv_metrics = compute_rollout_level_advantage(batch, **adv_kwargs)
                metrics.update(rollout_adv_metrics)
            else:
                batch = compute_advantage(batch, **adv_kwargs)

        # Count GRPO groups and zero-advantage groups in the trained batch.
        metrics.update(_grpo_group_metrics(batch))
        metrics["critic/n_transition_after_dropping"] = len(batch)
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))

        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode == PER_ROLLOUT_MEAN_LOSS_MODE:
            rollout_ids = batch.non_tensor_batch.get("rollout_id_list")
            if rollout_ids is None:
                raise RuntimeError("per_rollout_mean loss requires rollout_id_list")
            batch.batch["advantages"] = normalize_advantages_by_rollout(
                batch.batch["advantages"],
                batch.batch["response_mask"],
                rollout_ids,
                num_trained_rows=len(batch),
            )

        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self._update_critic(batch)
            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                actor_output = self._update_actor(batch)
            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

        with marked_timer("update_weights", timing_raw, color="red"):
            self.checkpoint_manager.update_weights(self.global_steps)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        # Return the batch so fit() can compute throughput after the step timer closes.
        return metrics, batch

    def fit(self):
        """Training loop driven by AglRolloutManager for rollouts."""

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.epoch = 0
        self._train_dataloader_iter = None
        self._carry_over_rollouts = []
        self._load_checkpoint()
        # Push loaded weights before the first rollout or validation.
        self.checkpoint_manager.update_weights(self.global_steps)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        self.epoch += 1
        last_val_metrics = None

        while True:
            if self.global_steps > self.total_training_steps or self.epoch > self.config.trainer.total_epochs:
                progress_bar.close()
                return

            timing_raw: dict[str, float] = {}

            curr_step_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )

            with marked_timer("step", timing_raw):
                result = self._train_step(timing_raw, curr_step_profile)
            if result is None:
                print("AgentLightningRayPPOTrainer: train step returned no batch; advancing step.")
                self.global_steps += 1
                continue
            metrics, step_batch = result

            # Compute timing and throughput after the step timer records its total duration.
            metrics.update(compute_timing_metrics(batch=step_batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            if n_gpus > 0 and "step" in timing_raw:
                metrics.update(compute_throughout_metrics(batch=step_batch, timing_raw=timing_raw, n_gpus=n_gpus))

            is_last_step = self.global_steps >= self.total_training_steps

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

    def _validate(self, merged: bool = False):
        """Run validation through the Agent Lightning rollout manager."""
        # verl 0.8.0 moved rollout server state behind llm_server_manager.
        has_llm_server_manager = hasattr(self, "llm_server_manager")
        if has_llm_server_manager:
            server_addresses = list(self.llm_server_manager.get_addresses())
        else:
            server_addresses = list(self.async_rollout_manager.server_addresses)
        assert server_addresses, "_validate called before rollout server addresses are available"

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
                "temperature": self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
                "validate": True,
                "global_steps": self.global_steps,
            }

            _, val_metrics_step = self._rollout(test_gen_batch, is_train=False)
            for k, v in val_metrics_step.items():
                merged_metrics[k] = v

        return merged_metrics
