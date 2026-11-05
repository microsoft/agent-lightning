# Copyright (c) Microsoft. All rights reserved.

"""Complete original-action projection without truncation or reward-dependent filtering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentlightning.credit.artifacts import replay_credit
from agentlightning.credit.state import validate_snapshot
from agentlightning.credit.types import IncompletePolicyTraceError, validate_sampling_parameters

if TYPE_CHECKING:
    from agentlightning.verl.agl_rollout_manager import CompletedRollout


@dataclass(frozen=True)
class TokenOrigin:
    """Identity/offset of one complete generated token, including action EOS."""

    action_event_id: str
    action_index: int
    offset: int


@dataclass(frozen=True)
class ProjectionLimits:
    """Predeclared row limits; exceeding them aborts instead of training a prefix."""

    max_prompt_length: int
    max_response_length: int
    max_total_length: int | None = None
    eos_token_id: int | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for value in (self.max_prompt_length, self.max_response_length, self.max_total_length):
            if value is not None and (type(value) is not int or value < 1):
                raise IncompletePolicyTraceError("projection limits must be positive integers")


@dataclass(frozen=True)
class ProjectedCreditRow:
    """One native row with masks derived exclusively from original token provenance."""

    rollout_id: str
    data_id: str
    snapshot_hash: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    log_probs: tuple[float, ...]
    origins: tuple[TokenOrigin | None, ...]
    image_urls: tuple[str, ...] | None = None

    @property
    def response_mask(self) -> tuple[int, ...]:
        return tuple(int(origin is not None) for origin in self.origins)


def _tokens(value, description: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or any(type(token) is not int or token < 0 for token in value):
        raise IncompletePolicyTraceError(f"invalid original {description} token geometry")
    return tuple(value)


def _check_limits(row: ProjectedCreditRow, limits: ProjectionLimits) -> None:
    if len(row.prompt_ids) > limits.max_prompt_length or len(row.response_ids) > limits.max_response_length:
        raise IncompletePolicyTraceError(
            "complete process action exceeds row limit; bound generation before sampling or increase limits"
        )
    if limits.max_total_length is not None and len(row.prompt_ids) + len(row.response_ids) > limits.max_total_length:
        raise IncompletePolicyTraceError("complete process context exceeds total row limit")


def project_process_rollout(
    completed: CompletedRollout, *, level: str, limits: ProjectionLimits
) -> list[ProjectedCreditRow]:
    """Project sealed successful actions into native transition or merged text rows.

    Tool/prompt/pad positions receive no origin. Prefix mismatch starts a new
    complete row; merging never changes action time, evidence cutoff or token
    identity. No action is inferred from matching token strings.
    """
    artifact = completed.process_credit
    if artifact is None:
        raise IncompletePolicyTraceError("process projection requires sealed original evidence")
    resolved = replay_credit(artifact)
    actions, _ = validate_snapshot(artifact.snapshot)
    triplets = completed.triplets or []
    if level not in {"transition", "trajectory"} or len(triplets) != len(actions):
        raise IncompletePolicyTraceError("process action count or aggregation level is inconsistent")
    rows = []
    expected_origins = set()
    for index, (triplet, action) in enumerate(zip(triplets, actions, strict=True)):
        envelope = action.envelope
        metadata = triplet.metadata.get("process_credit", {})
        if metadata.get("action_event_id") != envelope.event_id or metadata.get("action_index") != index:
            raise IncompletePolicyTraceError("native triplet lost its original action identity")
        validate_sampling_parameters(action.data.get("adjusted_params") or action.data.get("request", {}))
        source = action.data["response"]
        choice = source["choices"][0]
        prompt = _tokens(triplet.prompt.get("token_ids"), "prompt")
        response = _tokens(triplet.response.get("token_ids"), "response")
        original_prompt = source.get("prompt_token_ids") or choice.get("prompt_token_ids", [])
        if prompt != _tokens(original_prompt, "captured prompt") or response != _tokens(
            choice.get("token_ids"), "captured response"
        ):
            raise IncompletePolicyTraceError("native token projection differs from original sampled response")
        log_probs = triplet.response.get("log_probs")
        if not response or not isinstance(log_probs, list | tuple) or len(log_probs) != len(response):
            raise IncompletePolicyTraceError("complete action requires aligned captured token logprobs")
        if any(
            isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value > 0
            for value in log_probs
        ):
            raise IncompletePolicyTraceError("nonfinite/invalid captured action logprob")
        finish = choice.get("finish_reason")
        eos = limits.eos_token_id
        eos_ids = (eos,) if isinstance(eos, int) else (eos or ())
        if finish != "length" and (finish != "stop" or response[-1] not in eos_ids):
            raise IncompletePolicyTraceError("complete action EOS is absent or unverifiable")
        usage = source.get("usage", {})
        if "completion_tokens" in usage and usage["completion_tokens"] != len(response):
            raise IncompletePolicyTraceError("captured response omitted sampled tokens including EOS")
        origins = tuple(TokenOrigin(envelope.event_id, index, offset) for offset in range(len(response)))
        expected_origins.update(origins)
        row = ProjectedCreditRow(
            completed.rollout_id,
            completed.data_id,
            resolved.snapshot_hash,
            prompt,
            response,
            tuple(log_probs),
            origins,
            tuple(triplet.image_urls) if triplet.image_urls else None,
        )
        _check_limits(row, limits)
        if level == "trajectory" and row.image_urls:
            raise IncompletePolicyTraceError("trajectory-level images remain unsupported; use transition mode")
        if (
            level == "trajectory"
            and rows
            and prompt[: len(rows[-1].prompt_ids + rows[-1].response_ids)]
            == rows[-1].prompt_ids + rows[-1].response_ids
        ):
            previous = rows.pop()
            observation = prompt[len(previous.prompt_ids) + len(previous.response_ids) :]
            row = ProjectedCreditRow(
                row.rollout_id,
                row.data_id,
                row.snapshot_hash,
                previous.prompt_ids,
                previous.response_ids + observation + response,
                previous.log_probs + (0.0,) * len(observation) + tuple(log_probs),
                previous.origins + (None,) * len(observation) + origins,
            )
            _check_limits(row, limits)
        rows.append(row)
    actual = [origin for row in rows for origin in row.origins if origin is not None]
    if len(actual) != len(set(actual)) or set(actual) != expected_origins:
        raise IncompletePolicyTraceError("projection must retain every sampled action token exactly once")
    return rows


def build_process_batch(completed_rollouts, adapter, scheduled_groups, config):
    """Build real native tensors from all scheduled trajectories, with no row selection."""
    from dataclasses import asdict

    import numpy as np
    import torch
    from tensordict import TensorDict
    from verl import DataProto

    from agentlightning.credit.artifacts import make_credit_artifact
    from agentlightning.credit.types import canonical_json
    from agentlightning.verl.process_credit import compute_process_coefficients
    from agentlightning.verl.rollout_adapter import (
        _build_multi_modal_inputs,
        _compute_mrope_position_ids,
        _is_mrope_processor,
        get_left_padded_ids_and_attention_mask,
        get_right_padded_ids_and_attention_mask,
    )

    if any(rollout.process_credit is None for rollout in completed_rollouts):
        raise IncompletePolicyTraceError("process collection cannot mix unsealed outcome trajectories")
    resolved = [replay_credit(rollout.process_credit) for rollout in completed_rollouts]
    coefficients = compute_process_coefficients(resolved, scheduled_groups, config)
    by_id = {item.rollout_id: item for item in resolved}
    limits = ProjectionLimits(
        adapter.max_prompt_length,
        adapter.max_response_length,
        adapter.max_total_length,
        getattr(adapter.tokenizer, "eos_token_id", None),
    )
    rows = []
    per_rollout = {}
    for rollout in completed_rollouts:
        projected = project_process_rollout(rollout, level=adapter.trace_aggregator_level, limits=limits)
        per_rollout[rollout.rollout_id] = projected
        rows.extend(projected)
    outcomes = [item.terminal_outcome for item in resolved]
    metrics = {
        "training/reward": float(np.mean(outcomes)) if outcomes else 0,
        "training/n_rollouts": coefficients.scheduled_count,
        "training/n_sample": len(rows),
        "process/zero_action_rollouts": sum(not item.action_event_ids for item in resolved),
        "process/scheduled_rollouts": coefficients.scheduled_count,
        "process/normalization_tokens": config.normalization_tokens,
    }
    meta = {
        "process_scheduled_count": coefficients.scheduled_count,
        "process_config_digest": config.digest,
        "process_policy_version": scheduled_groups.rollouts[0].policy_version,
        "process_snapshot_hashes": [item.snapshot_hash for item in resolved],
        "process_zero_action": not rows,
    }
    if not rows:
        return DataProto(batch=None, meta_info=meta), metrics
    prompts, responses, attention, masks, log_probs, advantages, returns, scores = ([] for _ in range(8))
    origins = []
    row_ids = []
    projection_records = {rollout.rollout_id: [] for rollout in completed_rollouts}
    action_lengths = {
        rollout.rollout_id: [len(triplet.response["token_ids"]) for triplet in rollout.triplets]
        for rollout in completed_rollouts
    }
    for row in rows:
        prompt, prompt_mask = get_left_padded_ids_and_attention_mask(
            list(row.prompt_ids), adapter.max_prompt_length, adapter.pad_token_id
        )
        response, response_attention = get_right_padded_ids_and_attention_mask(
            list(row.response_ids), adapter.max_response_length, adapter.pad_token_id
        )
        row_coefficients = [
            coefficients.coefficients[row.rollout_id][origin.action_index] if origin else 0 for origin in row.origins
        ]
        row_returns = [
            coefficients.returns[row.rollout_id][origin.action_index] if origin else 0 for origin in row.origins
        ]
        row_scores = [
            by_id[row.rollout_id].rewards[origin.action_index]
            if origin is not None and origin.offset == action_lengths[row.rollout_id][origin.action_index] - 1
            else 0
            for origin in row.origins
        ]
        padding = adapter.max_response_length - len(row.response_ids)
        prompts.append(prompt)
        responses.append(response)
        attention.append(prompt_mask + response_attention)
        masks.append(list(row.response_mask) + [0] * padding)
        log_probs.append(list(row.log_probs) + [0.0] * padding)
        advantages.append(row_coefficients + [0.0] * padding)
        returns.append(row_returns + [0.0] * padding)
        scores.append(row_scores + [0.0] * padding)
        row_origins = [asdict(origin) if origin is not None else None for origin in row.origins]
        origins.append(row_origins + [None] * padding)
        row_id = f"{row.rollout_id}:{len(projection_records[row.rollout_id])}"
        row_ids.append(row_id)
        projection_records[row.rollout_id].append(
            {
                "row_id": row_id,
                "prompt_token_ids": row.prompt_ids,
                "response_token_ids": row.response_ids,
                "origins": row_origins,
                "response_mask": row.response_mask,
                "coefficients": row_coefficients,
                "returns": row_returns,
            }
        )
    device = adapter.device
    prompt_tensor = torch.tensor(prompts, dtype=torch.long, device=device)
    response_tensor = torch.tensor(responses, dtype=torch.long, device=device)
    input_ids = torch.cat((prompt_tensor, response_tensor), dim=-1)
    attention_tensor = torch.tensor(attention, dtype=torch.long, device=device)
    positions = torch.clamp(attention_tensor.cumsum(-1) - 1, min=0)
    tensors = {
        "prompts": prompt_tensor,
        "responses": response_tensor,
        "input_ids": input_ids,
        "attention_mask": attention_tensor,
        "position_ids": positions,
        "response_mask": torch.tensor(masks, dtype=torch.long, device=device),
        "rollout_log_probs": torch.tensor(log_probs, dtype=torch.float32, device=device),
        "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
        "returns": torch.tensor(returns, dtype=torch.float32, device=device),
        "token_level_scores": torch.tensor(scores, dtype=torch.float32, device=device),
    }
    multimodal = None
    if any(row.image_urls for row in rows):
        if adapter.processor is None:
            raise IncompletePolicyTraceError("complete image-bearing process actions require the original processor")
        multimodal = []
        position_rows = []
        use_mrope = _is_mrope_processor(adapter.processor)
        for index, row in enumerate(rows):
            inputs, has_types = (
                _build_multi_modal_inputs(adapter.processor, list(row.image_urls)) if row.image_urls else (None, False)
            )
            multimodal.append(inputs)
            if use_mrope:
                if row.image_urls and (inputs is None or "image_grid_thw" not in inputs):
                    raise IncompletePolicyTraceError("process image geometry lacks required image_grid_thw")
                position_rows.append(
                    _compute_mrope_position_ids(
                        adapter.processor,
                        input_ids[index],
                        attention_tensor[index],
                        inputs.get("image_grid_thw") if inputs is not None else None,
                        has_types,
                    )
                )
        if use_mrope:
            tensors["position_ids"] = torch.stack(position_rows)
    batch = DataProto(batch=TensorDict(tensors, batch_size=len(rows)), meta_info=meta)

    def object_vector(values):
        vector = np.empty(len(values), dtype=object)
        vector[:] = values
        return vector

    batch.non_tensor_batch.update(
        {
            "rollout_id_list": np.array([row.rollout_id for row in rows]),
            "data_id_list": np.array([row.data_id for row in rows]),
            "uid": np.array([row.data_id for row in rows]),
            "process_snapshot_hash": np.array([row.snapshot_hash for row in rows]),
            "process_group_id": np.array([row.data_id for row in rows]),
            "process_row_id": np.array(row_ids),
            "process_padding": np.zeros(len(rows), dtype=bool),
            "process_token_origins": object_vector(origins),
            "process_credit_action_index": object_vector(
                [[origin["action_index"] if origin else None for origin in row] for row in origins]
            ),
        }
    )
    if multimodal is not None:
        batch.non_tensor_batch["multi_modal_inputs"] = object_vector(multimodal)
    scheduling = tuple(
        {
            **item.model_dump(),
            "episode_return": coefficients.episode_returns[item.rollout_id],
            "snapshot_hash": by_id[item.rollout_id].snapshot_hash,
        }
        for item in scheduled_groups.rollouts
    )
    for rollout in completed_rollouts:
        updated = make_credit_artifact(
            rollout.process_credit.snapshot,
            config,
            projection=tuple(projection_records[rollout.rollout_id]),
            scheduled_groups=scheduling,
        )
        if len(canonical_json(updated.model_dump(mode="json"))) > config.max_artifact_bytes:
            raise IncompletePolicyTraceError("complete projected artifact exceeds configured size limit")
        rollout.process_credit = updated
    return batch, metrics
