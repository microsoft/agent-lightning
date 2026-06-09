"""Adapters from completed agl-lite rollouts to VERL training data."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from agl_lite.verl.agl_rollout_manager import CompletedRollout


def ids_startswith(full_ids: list[int], prefix_ids: list[int]) -> bool:
    return full_ids[: len(prefix_ids)] == prefix_ids


def get_left_padded_ids_and_attention_mask(
    ids: list[int], max_length: int, pad_token_id: int
) -> tuple[list[int], list[int]]:
    seq_len = len(ids)
    if seq_len >= max_length:
        return ids[-max_length:], [1] * max_length

    pad_len = max_length - seq_len
    return [pad_token_id] * pad_len + ids, [0] * pad_len + [1] * seq_len


def get_right_padded_ids_and_attention_mask(
    ids: list[int], max_length: int, pad_token_id: int
) -> tuple[list[int], list[int]]:
    seq_len = len(ids)
    if seq_len >= max_length:
        return ids[:max_length], [1] * max_length

    pad_len = max_length - seq_len
    return ids + [pad_token_id] * pad_len, [1] * seq_len + [0] * pad_len


class RolloutAdapter:
    """Convert completed rollout results into VERL data structures."""

    def __init__(
        self,
        *,
        max_prompt_length: int,
        max_response_length: int,
        device: torch.device,
        pad_token_id: int,
        reward_fillna_value: float = 0.0,
        trace_aggregator_level: str = "transition",
    ) -> None:
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.device = device
        self.pad_token_id = pad_token_id
        self.reward_fillna_value = reward_fillna_value
        self.trace_aggregator_level = trace_aggregator_level

    def get_train_data_batch(self, completed_rollouts: list[CompletedRollout]) -> tuple[DataProto, dict[str, Any]]:
        """Build a VERL training batch from completed rollouts."""
        level = self.trace_aggregator_level
        if level not in {"transition", "trajectory"}:
            raise ValueError(f"Unknown trace_aggregator_level: {level}")

        # Keep rollout randomness within each sample instead of ordering samples by completion time.
        sorted_rollouts = sorted(completed_rollouts, key=lambda rollout: (rollout.step, rollout.sample_idx_in_step))

        final_rewards: list[float] = []
        sample_with_reward_count = 0
        sample_with_trace_count = 0

        input_ids_list: list[list[int]] = []
        input_attention_mask_list: list[list[int]] = []
        response_ids_list: list[list[int]] = []
        response_attention_mask_list: list[list[int]] = []
        response_mask_list: list[list[int]] = []
        reward_list: list[float] = []
        data_id_list: list[str] = []
        rollout_id_list: list[str] = []
        turn_index_list: list[int] = []
        is_drop_list: list[bool] = []
        response_log_probs_list: list[list[float] | None] = []
        n_trunc_sample_because_of_response = 0
        unmerged_count = 0
        response_len_per_turn_list: list[int] = []

        def append_training_row(
            *,
            rollout_id: str,
            data_id: str,
            turn_index: int,
            prompt_ids: list[int],
            response_ids: list[int],
            reward: float,
            response_mask: list[int] | None = None,
            response_log_probs: list[float] | None = None,
        ) -> None:
            nonlocal n_trunc_sample_because_of_response
            if len(prompt_ids) > self.max_prompt_length:
                prompt_ids = prompt_ids[: self.max_prompt_length]
                is_drop_list.append(True)
            else:
                is_drop_list.append(False)

            if len(response_ids) > self.max_response_length:
                response_ids = response_ids[: self.max_response_length]
                if response_mask is not None:
                    response_mask = response_mask[: self.max_response_length]
                if response_log_probs is not None:
                    response_log_probs = response_log_probs[: self.max_response_length]
                n_trunc_sample_because_of_response += 1

            if response_log_probs is not None and len(response_log_probs) != len(response_ids):
                response_log_probs = None

            one_input_ids, one_input_attention_mask = get_left_padded_ids_and_attention_mask(
                prompt_ids, self.max_prompt_length, self.pad_token_id
            )
            one_response_ids, one_response_attention_mask = get_right_padded_ids_and_attention_mask(
                response_ids, self.max_response_length, self.pad_token_id
            )
            input_ids_list.append(one_input_ids)
            input_attention_mask_list.append(one_input_attention_mask)
            response_ids_list.append(one_response_ids)
            response_attention_mask_list.append(one_response_attention_mask)
            if response_mask is not None:
                one_response_mask, _ = get_right_padded_ids_and_attention_mask(
                    response_mask, self.max_response_length, 0
                )
                response_mask_list.append(one_response_mask)

            response_log_probs_list.append(response_log_probs)

            reward_list.append(reward)
            data_id_list.append(data_id)
            rollout_id_list.append(rollout_id)
            if level == "transition":
                turn_index_list.append(turn_index)

        for rollout in sorted_rollouts:
            final_reward = self._fillna_reward(rollout)
            final_rewards.append(final_reward)
            if rollout.final_reward is not None:
                sample_with_reward_count += 1

            if not rollout.triplets:
                print(f"Warning: No triplets found for training rollout {rollout.rollout_id}, skipping.")
                continue
            sample_with_trace_count += 1

            if level == "transition":
                for turn_index, triplet in enumerate(rollout.triplets):
                    response_ids = triplet.response["token_ids"]
                    log_probs = triplet.response["log_probs"]
                    response_len_per_turn_list.append(len(response_ids))
                    append_training_row(
                        rollout_id=rollout.rollout_id,
                        data_id=rollout.data_id,
                        turn_index=turn_index,
                        prompt_ids=triplet.prompt["token_ids"],
                        response_ids=response_ids,
                        reward=final_reward,
                        response_log_probs=log_probs,
                    )
                continue
            else:
                first_triplet = rollout.triplets[0]
                group_start_turn_index = 0
                current_prompt_ids = list(first_triplet.prompt["token_ids"])
                current_response_ids = list(first_triplet.response["token_ids"])
                current_context = current_prompt_ids + current_response_ids
                current_response_mask = [1] * len(current_response_ids)
                current_response_log_probs: list[float] | None = first_triplet.response["log_probs"]
                response_len_per_turn_list.append(len(current_response_ids))
                merged_group_count = 0

                for turn_index, triplet in enumerate(rollout.triplets[1:], start=1):
                    prompt_ids = triplet.prompt["token_ids"]
                    response_ids = triplet.response["token_ids"]
                    log_probs = triplet.response["log_probs"]
                    response_len_per_turn_list.append(len(response_ids))
                    next_context = prompt_ids + response_ids

                    if ids_startswith(prompt_ids, current_context):
                        if len(prompt_ids) > len(current_context):
                            observation_ids = prompt_ids[len(current_context) :]
                            current_response_ids += observation_ids
                            current_response_mask += [0] * len(observation_ids)
                            if current_response_log_probs is not None:
                                current_response_log_probs += [0.0] * len(observation_ids)
                        current_response_ids += response_ids
                        current_response_mask += [1] * len(response_ids)
                        if current_response_log_probs is not None:
                            if log_probs is None or len(log_probs) != len(response_ids):
                                current_response_log_probs = None
                            else:
                                current_response_log_probs += list(log_probs)
                        current_context = next_context
                        continue

                    append_training_row(
                        rollout_id=rollout.rollout_id,
                        data_id=rollout.data_id,
                        turn_index=group_start_turn_index,
                        prompt_ids=current_prompt_ids,
                        response_ids=current_response_ids,
                        reward=final_reward,
                        response_mask=current_response_mask,
                        response_log_probs=current_response_log_probs,
                    )
                    merged_group_count += 1

                    group_start_turn_index = turn_index
                    current_context = next_context
                    current_prompt_ids = list(prompt_ids)
                    current_response_ids = list(response_ids)
                    current_response_mask = [1] * len(response_ids)
                    current_response_log_probs = log_probs

                append_training_row(
                    rollout_id=rollout.rollout_id,
                    data_id=rollout.data_id,
                    turn_index=group_start_turn_index,
                    prompt_ids=current_prompt_ids,
                    response_ids=current_response_ids,
                    reward=final_reward,
                    response_mask=current_response_mask,
                    response_log_probs=current_response_log_probs,
                )
                merged_group_count += 1

                if merged_group_count > 1:
                    unmerged_count += 1

        n_sample = len(input_ids_list)
        if n_sample == 0:
            raise RuntimeError("get_train_data_batch emitted zero training rows.")

        batch_input_ids = torch.LongTensor(input_ids_list).to(self.device)
        input_attention_mask = torch.LongTensor(input_attention_mask_list).to(self.device)
        batch_response_ids = torch.LongTensor(response_ids_list).to(self.device)
        response_attention_mask = torch.LongTensor(response_attention_mask_list).to(self.device)
        batch_response_mask = torch.LongTensor(response_mask_list).to(self.device) if level == "trajectory" else None

        batch_seq = torch.cat([batch_input_ids, batch_response_ids], dim=-1)
        attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=-1)
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

        row_has_log_probs_list = [log_probs is not None for log_probs in response_log_probs_list]
        emit_rollout_log_probs = all(row_has_log_probs_list)
        if not emit_rollout_log_probs and any(row_has_log_probs_list):
            print("Warning: Mixed rollout log_probs availability, omitting rollout_log_probs from batch.")

        is_drop_mask = torch.BoolTensor(is_drop_list).to(self.device)
        scores = torch.tensor(reward_list, dtype=torch.bfloat16).to(self.device)

        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
        token_positions = torch.arange(attention_mask.shape[-1], device=attention_mask.device).unsqueeze(0)
        eos_mask_idx = torch.argmax(token_positions * attention_mask, dim=-1)
        token_level_scores[torch.arange(n_sample), eos_mask_idx] = scores
        token_level_scores = token_level_scores[:, -self.max_response_length :]

        batch_dict = {
            "prompts": batch_input_ids,
            "responses": batch_response_ids,
            "input_ids": batch_seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "is_drop_mask": is_drop_mask,
            "token_level_scores": token_level_scores.contiguous(),
        }
        if level == "trajectory":
            assert batch_response_mask is not None
            batch_dict["response_mask"] = batch_response_mask
        if emit_rollout_log_probs:
            padded_log_probs_list = [
                log_probs + [0.0] * (self.max_response_length - len(log_probs))
                for log_probs in response_log_probs_list
                if log_probs is not None
            ]
            batch_dict["rollout_log_probs"] = torch.tensor(padded_log_probs_list, dtype=torch.float32).to(self.device)

        batch = TensorDict(batch_dict, batch_size=n_sample)  # type: ignore[arg-type]
        data_proto = DataProto(batch=batch)
        data_proto.non_tensor_batch["data_id_list"] = np.array(data_id_list)
        data_proto.non_tensor_batch["rollout_id_list"] = np.array(rollout_id_list)
        if level == "transition":
            data_proto.non_tensor_batch["turn_index_list"] = np.array(turn_index_list)

        n_response_turns = len(response_len_per_turn_list)
        data_metrics = {
            "training/reward": float(np.mean(final_rewards)) if final_rewards else 0.0,
            "training/n_sample": n_sample,
            "training/n_rollouts": len(sorted_rollouts),
            "training/n_rollouts_w_trace": sample_with_trace_count,
            "training/n_rollouts_w_reward": sample_with_reward_count,
            "training/n_truncated_sample": n_trunc_sample_because_of_response,
            "training/n_turns": n_response_turns,
            "response_length/training/avg_by_turn": float(np.mean(response_len_per_turn_list)),
            "response_length/training/max_by_turn": int(np.max(response_len_per_turn_list)),
            "response_length/training/min_by_turn": int(np.min(response_len_per_turn_list)),
        }
        if level == "trajectory":
            data_metrics["training/n_unmerged_rollouts"] = unmerged_count

        return data_proto, data_metrics

    def get_test_metrics(self, completed_rollouts: list[CompletedRollout]) -> dict[str, Any]:
        """Build validation metrics from completed rollouts."""
        sample_stat_list: list[dict[str, Any]] = []

        for rollout in completed_rollouts:
            final_reward = self._fillna_reward(rollout)
            sample_stat: dict[str, Any] = {
                "reward": final_reward,
                "has_reward": rollout.final_reward is not None,
            }
            if rollout.triplets:
                response_length_list = [
                    len(triplet.response.get("token_ids") or [])
                    for triplet in rollout.triplets
                ]
                sample_stat.update(
                    {
                        "total_response_length": np.sum(response_length_list),
                        "mean_response_length": np.mean(response_length_list) if response_length_list else 0,
                        "turn_count": len(rollout.triplets),
                    }
                )
            sample_stat_list.append(sample_stat)

        stats_w_trace = [stat for stat in sample_stat_list if "total_response_length" in stat]
        if not stats_w_trace:
            raise RuntimeError("get_test_metrics received zero completed rollouts with trace.")

        return {
            "val/reward": float(np.mean([stat["reward"] for stat in sample_stat_list])),
            "val/n_rollouts": len(sample_stat_list),
            "val/n_rollouts_w_trace": len(stats_w_trace),
            "val/n_rollouts_w_reward": len([stat for stat in sample_stat_list if stat["has_reward"]]),
            "val/mean_response_length_per_turn": float(
                np.mean([stat["mean_response_length"] for stat in stats_w_trace])
            ),
            "val/mean_total_response_length_per_rollout": float(
                np.mean([stat["total_response_length"] for stat in stats_w_trace])
            ),
            "val/turn_count": float(np.mean([stat["turn_count"] for stat in stats_w_trace])),
        }

    def _fillna_reward(self, rollout: CompletedRollout) -> float:
        if rollout.final_reward is not None:
            return rollout.final_reward
        return self.reward_fillna_value


__all__ = ["RolloutAdapter"]
