# Copyright (c) Microsoft. All rights reserved.

"""Score-centering policy gradients and their VERL FSDP worker integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from types import MethodType
from typing import Any, cast

import torch
from torch.utils.checkpoint import checkpoint
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import tensordict_utils as tu
from verl.utils import torch_functional as verl_F
from verl.utils.metric import AggregationType, Metric
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import no_padding_2_padding


@dataclass(frozen=True)
class ScoreCenteringConfig:
    centering: bool = True
    is_mode: str = "none"
    top_logprobs_num: int = 128
    mis_min: float = 0.5
    mis_max: float = 5.0
    tis_max: float = 2.0

    def __post_init__(self) -> None:
        if self.is_mode not in {"none", "mis", "tis"}:
            raise ValueError(f"Unknown importance sampling mode: {self.is_mode}")
        if not (math.isfinite(self.mis_min) and math.isfinite(self.mis_max) and 0 < self.mis_min <= self.mis_max):
            raise ValueError("MIS bounds must be finite, positive and ordered")
        if not math.isfinite(self.tis_max) or self.tis_max <= 0 or self.top_logprobs_num < 1:
            raise ValueError("TIS cap and candidate count must be positive")


def importance_weights(log_ratio: torch.Tensor, config: ScoreCenteringConfig) -> torch.Tensor:
    if config.is_mode == "none":
        return torch.ones_like(log_ratio)
    if config.is_mode == "tis":
        return log_ratio.clamp(max=math.log(config.tis_max)).exp()
    keep = (log_ratio >= math.log(config.mis_min)) & (log_ratio <= math.log(config.mis_max))
    return log_ratio.masked_fill(~keep, -torch.inf).exp()


def support_scores(
    head_log_probs: torch.Tensor,
    support_ids: torch.Tensor,
    sampling_log_probs: torch.Tensor,
    sampled_log_probs: torch.Tensor,
    sampled_sampling_log_probs: torch.Tensor,
    config: ScoreCenteringConfig,
) -> dict[str, torch.Tensor]:
    valid = support_ids >= 0
    log_p = head_log_probs.masked_fill(~valid, 0.0)
    with torch.no_grad():
        log_q = sampling_log_probs.masked_fill(~valid, 0.0)
        p = log_p.detach().exp().masked_fill(~valid, 0.0)
        q = log_q.exp().masked_fill(~valid, 0.0)
        rho = (1 - q.sum(-1)).clamp_min(1e-6) / (1 - p.sum(-1)).clamp_min(1e-6)
        if config.is_mode == "none":
            head_mass, alpha = q, rho
        elif config.is_mode == "tis":
            head_mass, alpha = torch.minimum(p, config.tis_max * q), (config.tis_max * rho).clamp(max=1)
        else:
            keep = (log_p.detach() - log_q >= math.log(config.mis_min)) & (
                log_p.detach() - log_q <= math.log(config.mis_max)
            )
            head_mass = p * keep
            alpha = ((rho >= 1 / config.mis_max) & (rho <= 1 / config.mis_min)).to(p.dtype)
        coefficients = (head_mass - alpha.unsqueeze(-1) * p).masked_fill(~valid, 0.0)
        sample_weight = importance_weights(sampled_log_probs.detach() - sampled_sampling_log_probs, config)
    correction = (coefficients * log_p.masked_fill(coefficients == 0, 0.0)).sum(-1)
    return {
        "log_probs": sampled_log_probs,
        "sc_correction": correction,
        "sc_sample_weight": sample_weight,
        "sc_support_size": valid.sum(-1).to(log_p.dtype),
    }


def token_loss(scores: dict[str, torch.Tensor], advantages: torch.Tensor, config: ScoreCenteringConfig) -> torch.Tensor:
    weight = scores["sc_sample_weight"].detach()
    score = weight * scores["log_probs"].masked_fill(weight == 0, 0.0)
    if config.centering:
        score = score - scores["sc_correction"]
    return -advantages.detach() * score


def global_token_mean(
    losses: torch.Tensor, active: torch.Tensor, global_token_count: int, dp_size: int
) -> torch.Tensor:
    if global_token_count <= 0:
        raise ValueError("An optimizer minibatch must contain generated tokens")
    return losses.masked_fill(~active.bool(), 0.0).sum() * dp_size / global_token_count


def pad_active_support(rows, active, device):
    candidates, probabilities = [], []
    for row, mask in zip(rows, active.cpu().tolist(), strict=True):
        for index, keep in enumerate(mask):
            if keep:
                candidates.append(torch.tensor(row["token_ids"][index], dtype=torch.long))
                probabilities.append(torch.tensor(row["log_probs"][index], dtype=torch.float32))
    ids = torch.nn.utils.rnn.pad_sequence(candidates, batch_first=True, padding_value=-1).to(device)
    log_q = torch.nn.utils.rnn.pad_sequence(probabilities, batch_first=True, padding_value=0.0).to(device)
    return ids, log_q


def prepare_support_outputs(engine, output, output_args, micro_batch, logits_processor_func):
    """Map padded response support to packed *prediction* positions per sequence."""
    if engine.use_ulysses_sp:
        raise ValueError("Score centering currently requires FSDP without sequence parallelism")
    if tu.get(micro_batch, "use_fused_kernels", False):
        raise ValueError("Score centering needs unfused LM-head logits")
    input_ids = micro_batch["input_ids"]
    offsets = input_ids.offsets()
    logits = output.logits.reshape(-1, output.logits.shape[-1])
    active = micro_batch["response_mask"].bool()
    prompt_width = micro_batch["prompts"].shape[1]
    prompt_lengths = micro_batch["attention_mask"][:, :prompt_width].sum(-1)
    positions = offsets[:-1, None] + prompt_lengths[:, None] - 1 + torch.arange(active.shape[1], device=logits.device)
    logit_positions = positions
    if not tu.get(micro_batch, "use_remove_padding", True):
        row_starts = torch.arange(active.shape[0], device=logits.device) * output.logits.shape[1]
        logit_positions = row_starts[:, None] + positions - offsets[:-1, None]
    positions = positions[active].long()
    logit_positions = logit_positions[active].long()
    ids, log_q = pad_active_support(tu.get(micro_batch, "score_centering"), active, logits.device)
    selected_ids = torch.cat((ids.clamp_min(0), micro_batch["responses"][active, None]), dim=-1)
    scaled_logits = logits if engine.score_centering_temperature == 1 else logits / engine.score_centering_temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    labels[logit_positions] = micro_batch["responses"][active]
    if logits.is_cuda:
        sampled_log_probs = verl_F.logprobs_from_logits(logits=scaled_logits, labels=labels, inplace_backward=False)[
            logit_positions
        ]
    else:
        sampled_log_probs = verl_F.logprobs_from_logits_v2(scaled_logits, labels)[logit_positions]
    selected_logits = scaled_logits[logit_positions[:, None], selected_ids].to(
        torch.float64 if logits.dtype == torch.float64 else torch.float32
    )
    head_log_probs = sampled_log_probs[:, None] + (selected_logits[:, :-1] - selected_logits[:, -1:])
    scores = support_scores(
        head_log_probs,
        ids,
        log_q,
        sampled_log_probs,
        micro_batch["rollout_log_probs"][active],
        engine.score_centering_config,
    )
    if engine.score_centering_entropy:
        with torch.set_grad_enabled(torch.is_grad_enabled() and engine.score_centering_entropy_coeff != 0):
            scores["entropy"] = cast(
                torch.Tensor, checkpoint(verl_F.entropy_from_logits, scaled_logits, use_reentrant=False)
            )[logit_positions]
    result = {}
    for key, values in scores.items():
        packed = values.new_zeros(input_ids.values().shape[0]).index_copy(0, positions, values)
        result[key] = torch.nested.nested_tensor_from_jagged(packed, offsets)
    return result


def score_centering_loss(config, model_output, data, dp_group=None, actor_config=None):
    scores = {key: no_padding_2_padding(value, data) for key, value in model_output.items()}
    active = data["response_mask"].bool()
    losses = token_loss(scores, data["advantages"], config)
    count, dp_size = tu.get(data, "batch_num_tokens"), tu.get(data, "dp_size")
    loss = global_token_mean(losses, active, count, dp_size)

    def metric(value):
        return Metric(value=value.detach(), aggregation=AggregationType.SUM)

    def mean(value):
        return global_token_mean(value.detach(), active, count, dp_size)

    log_ratio = (scores["log_probs"].detach() - data["rollout_log_probs"]).masked_fill(~active, 0.0)
    metrics = {
        "pg_loss": metric(loss),
        "sc_correction": metric(mean(scores["sc_correction"])),
        "sc_support_size": metric(mean(scores["sc_support_size"])),
        "sc_support_size_max": Metric(value=scores["sc_support_size"].max(), aggregation=AggregationType.MAX),
        "sc_weight": metric(mean(scores["sc_sample_weight"])),
        "sc_is_kept": metric(mean((scores["sc_sample_weight"] > 0).float())),
        "train_rollout_logprob_diff": metric(mean(scores["log_probs"] - data["rollout_log_probs"])),
        "rollout_k3_kl": metric(mean(log_ratio.expm1() - log_ratio)),
    }
    if "entropy" in scores:
        entropy = global_token_mean(scores["entropy"], active, count, dp_size)
        if actor_config is not None:
            loss = loss - actor_config.entropy_coeff * entropy
        metrics["entropy_loss"] = metric(entropy)
    if actor_config is not None and actor_config.use_kl_loss:
        from verl.trainer.ppo.core_algos import kl_penalty

        per_token_kl = kl_penalty(
            # VERL annotates Tensor inputs using the legacy FloatTensor class.
            logprob=cast(torch.FloatTensor, scores["log_probs"]),
            ref_logprob=data["ref_log_prob"],
            kl_penalty=actor_config.kl_loss_type,
        )
        kl_loss = global_token_mean(per_token_kl, active, count, dp_size)
        loss = loss + actor_config.kl_loss_coef * kl_loss
        metrics["kl_loss"] = metric(kl_loss)
    return loss, metrics


def configure_actor_worker(config, role_worker_mapping):
    import ray
    from omegaconf import open_dict
    from verl.trainer.ppo.utils import Role

    if config.algorithm.use_kl_in_reward or config.algorithm.rollout_correction is not None:
        raise ValueError("SC requires reward-side KL disabled and no outer rollout correction")
    with open_dict(config.actor_rollout_ref):
        config.actor_rollout_ref.score_centering = config.agentlightning.score_centering
    for role in (Role.ActorRollout, Role.ActorRolloutRef):
        if role in role_worker_mapping:
            role_worker_mapping[role] = ray.remote(ScoreCenteringWorker)
    return ScoreCenteringWorker


class ScoreCenteringWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)  # pyright: ignore[reportAttributeAccessIssue]
    def init_model(self):
        super().init_model()
        if self._is_actor:
            config = ScoreCenteringConfig(**dict(self.config.score_centering))
            engine = cast(Any, self.actor.engine)
            engine.score_centering_config = config
            engine.score_centering_temperature = float(self.config.rollout.temperature)
            engine.score_centering_entropy_coeff = float(self.config.actor.entropy_coeff)
            engine.score_centering_entropy = bool(
                self.config.actor.calculate_entropy or self.config.actor.entropy_coeff
            )
            engine.prepare_model_outputs = MethodType(prepare_support_outputs, engine)
            self.set_loss_fn(partial(score_centering_loss, config, actor_config=self.config.actor))
