"""Custom AgentLoopManager that uses agl-lite for agent orchestration.

Uses VERL's internal vLLM servers (managed by the parent AgentLoopManager)
for inference — this is essential because VERL pushes updated model weights
to these servers after each PPO step via CheckpointEngineManager.

Instead of running agents as Ray actors (AgentLoopWorker), this manager
delegates agent execution to the agl-lite HTTP API:
  - Agents run as K8s pods (managed by agl-lite controller)
  - LLM calls go through the agl-lite gateway → VERL's internal vLLM
  - The gateway captures token IDs transparently
  - Rewards are computed by hooks on the agl-lite server

The standard RayPPOTrainer calls ``generate_sequences(prompts)`` and gets
back a ``DataProto`` with all the fields it needs.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

import torch
from tensordict import TensorDict
from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.utils.ray_utils import auto_await

from agl_lite.client import AglLiteClient
from agl_lite.schemas.api import EnqueueRolloutRequest, RegisterModelRequest
from agl_lite.schemas.rollout import RolloutStatus

from .daemon import (
    _to_native,
    get_left_padded_ids_and_attention_mask,
    get_right_padded_ids_and_attention_mask,
)


class AglLiteAgentLoopManager(AgentLoopManager):
    """AgentLoopManager that delegates agent execution to agl-lite.

    Inherits VERL's vLLM server management (so model weight updates work)
    but replaces AgentLoopWorker-based agent execution with agl-lite's
    K8s-based agent runner.

    The flow:
      1. Parent creates and manages internal vLLM inference servers
      2. generate_sequences() registers those servers with the agl-lite gateway
      3. agl-lite enqueues rollouts → K8s agent pods → gateway → internal vLLM
      4. Gateway captures token IDs; hooks compute rewards
      5. Triplets fetched via HTTP, assembled into DataProto for PPO
    """

    def __init__(self, config, worker_group=None, rollout_resource_pool=None,
                 reward_loop_worker_handles=None):
        # Parent __init__ sets: config, rollout_config (OmegaConf), model_config (OmegaConf),
        # worker_group, rollout_resource_pool, rollout_replica_class, agent_loop_workers_class
        super().__init__(config, worker_group, rollout_resource_pool, reward_loop_worker_handles)

        # Load tokenizer — the parent AgentLoopManager doesn't set self.tokenizer
        # (that's done in AgentLoopWorker). We need it for DataProto construction.
        from transformers import AutoTokenizer
        model_path = config.actor_rollout_ref.model.path
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # agl-lite connection — create a fresh client for each generate_sequences
        # call to avoid stale TCP connections (VERL pauses between val/train).
        self._agl_base_url = config.agentlightning.get("agl_base_url", "http://localhost:8080")
        self._agl_key = config.agentlightning.get("agl_key", "")
        self.timeout_seconds = config.agentlightning.get("timeout_seconds", 1200.0)

        # Training config for tensor construction
        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self._pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0

        # Track whether we've registered models with the gateway
        self._models_registered = False

    async def _init_agent_loop_workers(self):
        """Override: skip creating Ray AgentLoopWorker actors.

        Agents run as K8s pods via agl-lite, not as Ray actors.
        We keep an empty list so any code checking len(self.agent_loop_workers) works.
        """
        self.agent_loop_workers = []

    async def _register_models_if_needed(self, client: AglLiteClient):
        """Register VERL's internal vLLM servers with the agl-lite gateway."""
        if self._models_registered:
            return
        if not self.server_addresses:
            return

        model_name = self.config.actor_rollout_ref.model.path
        regs = []
        for addr in self.server_addresses:
            endpoint = addr if addr.startswith("http") else f"http://{addr}/v1"
            regs.append(RegisterModelRequest(model=model_name, endpoint=endpoint))
        await client.register_models(regs)
        self._models_registered = True
        print(f"AglLiteAgentLoopManager: registered {len(regs)} model server(s) with gateway")

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Enqueue rollouts via agl-lite, wait, fetch triplets, build DataProto.

        Called by RayPPOTrainer.fit() for both training and validation.
        Uses a fresh HTTP client per call to avoid stale TCP connections
        (VERL has long pauses between calls for FSDP weight sync).
        """
        start_time = time.time()

        async with AglLiteClient(
            base_url=self._agl_base_url, agl_key=self._agl_key
        ) as client:
            # 1. Register VERL's internal vLLM servers with agl-lite gateway
            await self._register_models_if_needed(client)

            # 2. Build rollout requests from batch
            num_samples = len(prompts)
            rollout_requests: List[EnqueueRolloutRequest] = []
            sample_map: Dict[str, Dict[str, Any]] = {}

            for i in range(num_samples):
                original = {k: _to_native(v[i]) for k, v in prompts.non_tensor_batch.items()}
                original["_sample_idx"] = i

                rollout_requests.append(EnqueueRolloutRequest(
                    input=original,
                    config={"timeout": int(self.timeout_seconds)},
                ))

            # 3. Enqueue
            created = await client.enqueue_rollouts(rollout_requests)
            rollout_ids = [r.rollout_id for r in created]
            for r in created:
                sample_map[r.rollout_id] = r.input if isinstance(r.input, dict) else {}

            print(f"AglLiteAgentLoopManager: enqueued {len(rollout_ids)} rollouts, polling...")

            # 4. Poll until all complete
            terminal = {RolloutStatus.SUCCEEDED, RolloutStatus.TERMINAL_FAILED, RolloutStatus.CANCELLED}
            poll_start = time.time()
            while time.time() - poll_start < self.timeout_seconds:
                rollouts = await client.query_rollouts(ids=rollout_ids, limit=len(rollout_ids))
                done = all(r.status in terminal for r in rollouts)
                if done:
                    succeeded = sum(1 for r in rollouts if r.status == RolloutStatus.SUCCEEDED)
                    failed = sum(1 for r in rollouts if r.status != RolloutStatus.SUCCEEDED)
                    print(f"AglLiteAgentLoopManager: {succeeded} succeeded, {failed} failed")
                    break
                await asyncio.sleep(5)
            else:
                print(f"WARNING: rollouts timed out after {self.timeout_seconds}s")

            # 5. Fetch triplets and rewards
            completed_data: List[Dict[str, Any]] = []
            for rid in rollout_ids:
                events = await client.get_events(rid, format="triplet")
                triplets = []
                final_reward: Optional[float] = None
                for evt in events:
                    if evt.event_type == "model_request":
                        d = evt.data
                        triplets.append({
                            "prompt_ids": d.get("prompt_token_ids", []),
                            "response_ids": d.get("response_token_ids", []),
                        })
                    elif evt.event_type == "reward":
                        final_reward = evt.data.get("value")

                completed_data.append({
                    "rollout_id": rid,
                    "triplets": triplets,
                    "reward": final_reward if final_reward is not None else 0.0,
                    "original": sample_map.get(rid, {}),
                })

        # 6. Build DataProto (outside the client context — no more HTTP needed)
        output = self._build_data_proto(completed_data, prompts)

        elapsed = time.time() - start_time
        output.meta_info["timing"] = {
            "agent_loop/generate_sequences/mean": elapsed,
            "agent_loop/generate_sequences/min": elapsed,
            "agent_loop/generate_sequences/max": elapsed,
            "agent_loop/tool_calls/mean": 0.0,
            "agent_loop/tool_calls/min": 0.0,
            "agent_loop/tool_calls/max": 0.0,
            "agent_loop/num_preempted/mean": 0.0,
            "agent_loop/num_preempted/min": 0.0,
            "agent_loop/num_preempted/max": 0.0,
            "agent_loop/slowest/generate_sequences": elapsed,
            "agent_loop/slowest/tool_calls": 0.0,
            "agent_loop/slowest/prompt_length": 0,
            "agent_loop/slowest/response_length": 0,
            "agent_loop/slowest/num_preempted": 0,
        }
        return output

    def _build_data_proto(self, completed_data: List[Dict], prompts: DataProto) -> DataProto:
        """Convert completed rollouts with triplets into a DataProto batch.

        Builds the same tensor structure that AgentLoopWorker._postprocess creates:
          - prompts: [bsz, prompt_length]
          - responses: [bsz, response_length]
          - input_ids: [bsz, prompt_length + response_length]
          - attention_mask: [bsz, prompt_length + response_length]
          - position_ids: [bsz, prompt_length + response_length]
          - response_mask: [bsz, response_length]
          - rm_scores: [bsz, response_length]
        """
        max_prompt_length = self.max_prompt_length
        max_response_length = self.max_response_length
        pad_token_id = self._pad_token_id

        prompt_ids_list = []
        response_ids_list = []
        response_mask_list = []
        input_ids_list = []
        attention_mask_list = []
        scores_list = []

        for item in completed_data:
            triplets = item["triplets"]
            reward = item["reward"]

            if not triplets:
                # No triplets (agent failed before any LLM call).
                # Insert a single EOS token so VERL doesn't treat this as
                # an aborted sequence (which would crash compute_data_metrics
                # if ALL sequences are aborted).
                eos_id = (self._tokenizer.eos_token_id if self._tokenizer else None) or pad_token_id
                prompt_ids_list.append([pad_token_id] * max_prompt_length)
                resp = [eos_id] + [pad_token_id] * (max_response_length - 1)
                resp_mask = [1] + [0] * (max_response_length - 1)
                response_ids_list.append(resp)
                response_mask_list.append(resp_mask)
                input_ids_list.append([pad_token_id] * max_prompt_length + resp)
                attention_mask_list.append([0] * max_prompt_length + resp_mask)
                scores_list.append(reward)
                continue

            # Use last triplet for single-turn; for multi-turn, concatenate
            last = triplets[-1]
            p_ids = last["prompt_ids"]
            r_ids = last["response_ids"]

            # Pad/truncate
            padded_prompt, prompt_mask = get_left_padded_ids_and_attention_mask(
                p_ids, max_prompt_length, pad_token_id
            )
            padded_resp, resp_mask = get_right_padded_ids_and_attention_mask(
                r_ids, max_response_length, pad_token_id
            )

            prompt_ids_list.append(padded_prompt)
            response_ids_list.append(padded_resp)
            response_mask_list.append(resp_mask)
            input_ids_list.append(padded_prompt + padded_resp)
            attention_mask_list.append(prompt_mask + resp_mask)
            scores_list.append(reward)

        n = len(completed_data)

        prompts_t = torch.tensor(prompt_ids_list, dtype=torch.long)
        responses_t = torch.tensor(response_ids_list, dtype=torch.long)
        input_ids_t = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask_list, dtype=torch.long)
        response_mask_t = torch.tensor(response_mask_list, dtype=torch.long)
        position_ids_t = torch.clamp(torch.cumsum(attention_mask_t, dim=-1) - 1, min=0)

        # rm_scores: place reward at the last valid response token
        rm_scores = torch.zeros_like(response_mask_t, dtype=torch.float32)
        response_lengths = torch.clamp(response_mask_t.sum(dim=1) - 1, min=0)
        rm_scores[torch.arange(n), response_lengths] = torch.tensor(scores_list, dtype=torch.float32)

        batch = TensorDict(
            {
                "prompts": prompts_t,
                "responses": responses_t,
                "input_ids": input_ids_t,
                "attention_mask": attention_mask_t,
                "position_ids": position_ids_t,
                "response_mask": response_mask_t,
                "rm_scores": rm_scores,
            },
            batch_size=n,
        )

        # Non-tensor batch — pass through from input + required fields
        non_tensor_batch: Dict[str, Any] = {}
        if prompts.non_tensor_batch:
            for k, v in prompts.non_tensor_batch.items():
                if len(v) == n:
                    non_tensor_batch[k] = v
        non_tensor_batch["__num_turns__"] = np.array(
            [max(len(d["triplets"]), 1) for d in completed_data], dtype=np.int32
        )
        non_tensor_batch.setdefault(
            "multi_modal_inputs", np.array([{} for _ in range(n)], dtype=object)
        )

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={},
        )
