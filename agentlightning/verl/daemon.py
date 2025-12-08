# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import random
import socket
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, Tuple
import json
import os
import numpy as np
import requests
import torch
from flask import Flask, Response, abort, request
from tensordict import TensorDict
from verl import DataProto
from transformers import AutoTokenizer

from agentlightning import LLM, AgentLightningServer, NamedResources, RolloutLegacy, configure_logger
from agentlightning.adapter.triplet import TracerTraceToTriplet, TraceToTripletBase
from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.store.base import LightningStore
from agentlightning.types import Rollout, RolloutConfig, Task
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
import asyncio


configure_logger()

__all__ = [
    "AgentModeDaemon",
    "get_left_padded_ids_and_attention_mask",
    "get_right_padded_ids_and_attention_mask",
]


def lcp_len(a, b):
    L = min(len(a), len(b))
    i = 0
    while i < L and a[i] == b[i]:
        i += 1
    return i

def get_left_padded_ids_and_attention_mask(
    ids: List[int], max_length: int, pad_token_id: int
) -> Tuple[List[int], List[int]]:
    """
    Left-pad (or truncate) a sequence of token IDs to a fixed length,
    and build the corresponding attention mask.

    Args:
        ids:             the original list of token IDs.
        max_length:      desired total length after padding/truncation.
        pad_token_id:    ID to use for padding.

    Returns:
        padded_ids (any):      list of length == max_length.
        attention_mask (any):  list of same length: 1 for non-pad tokens, 0 for pads.
    """
    seq_len = len(ids)

    if seq_len >= max_length:
        # too long → truncate from the left, keep the last max_length tokens
        trimmed = ids[-max_length:]
        attention_mask = [1] * max_length
        return trimmed, attention_mask

    # too short → pad on the left
    pad_len = max_length - seq_len
    padded_ids = [pad_token_id] * pad_len + ids
    attention_mask = [0] * pad_len + [1] * seq_len
    return padded_ids, attention_mask


def get_right_padded_ids_and_attention_mask(
    ids: List[int], max_length: int, pad_token_id: int
) -> Tuple[List[int], List[int]]:
    """
    Right-pad (or truncate) a sequence of token IDs to a fixed length,
    and build the corresponding attention mask.

    Args:
        ids:            the original list of token IDs.
        max_length:     desired total length after padding/truncation.
        pad_token_id:   ID to use for padding.

    Returns:
        padded_ids (any):     list of length == max_length.
        attention_mask (any): list of same length: 1 for non-pad tokens, 0 for pads.
    """
    seq_len = len(ids)

    if seq_len >= max_length:
        # too long → truncate to the first max_length tokens
        trimmed = ids[:max_length]
        attention_mask = [1] * max_length
        return trimmed, attention_mask

    # too short → pad on the right
    pad_len = max_length - seq_len
    padded_ids = ids + [pad_token_id] * pad_len
    attention_mask = [1] * seq_len + [0] * pad_len
    return padded_ids, attention_mask


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _to_native(obj: Any) -> Any:
    """Convert data retrieved from Parquet to data usable in AGL server."""
    # 1) Arrays -> list (then recurse)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())

    # 2) NumPy scalar types -> Python scalars
    if isinstance(obj, np.generic):
        return _to_native(obj.item())

    # 3) Dict-like -> dict
    if isinstance(obj, Mapping):
        return {_to_native(k): _to_native(v) for k, v in obj.items()}  # type: ignore

    # 4) Lists/Tuples/Sets -> list
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(x) for x in obj]  # type: ignore

    # 5) Anything else: leave as-is
    return obj


class AgentModeDaemon:
    """
    AgentModeDaemon using the AgentLightningServer SDK.

    This class manages the server lifecycle, task queueing, and results
    retrieval, while also running a proxy server for LLM requests. It maintains
    the original interface for compatibility with the RayPPOTrainer.
    """

    def __init__(
        self,
        port: Optional[int],
        train_rollout_n: int,
        train_information: Dict[str, Any],
        tokenizer: Any,
        mini_batch_size: int,
        pad_token_id: int,
        reward_fillna_value: float = 0.0,
        llm_timeout_seconds: float = 1200.0,
        mode: Literal["v0", "v1"] = "v1",
        llm_proxy: LLMProxy | None = None,
        store: LightningStore | None = None,
        adapter: TraceToTripletBase | None = None,
        val_information: Dict[str, Any] | None = None,
        val_rollout_n: int = 1,
        trace_agg_mode: Literal["trajectory", "transition","use_last_trace"] = "transition",
    ):
        self.mode = mode
        self.llm_timeout_seconds = llm_timeout_seconds

        # Server and Task Configuration
        if mode == "v0":
            assert port is not None
            self.server_port = port
            self.server = AgentLightningServer(
                host="0.0.0.0", port=self.server_port, task_timeout_seconds=self.llm_timeout_seconds
            )
            self.proxy_port = _find_available_port()  # Run proxy on a different port
        else:
            assert store is not None
            self.store = store
            if llm_proxy is None:
                self.llm_proxy = LLMProxy(
                    port=_find_available_port(),
                    model_list=[],
                    store=store,
                )
            else:
                # Reuse the existing LLM proxy (probably configured by user)
                self.llm_proxy = llm_proxy
            if adapter is None:
                self.adapter = TracerTraceToTriplet()
            else:
                # Reuse the one from trainer
                self.adapter = adapter
            self._internal_loop: Optional[asyncio.AbstractEventLoop] = None
            self._internal_loop_thread = threading.Thread(target=self._internal_loop_runner, daemon=True)
            self._internal_loop_thread.start()

        # Training and Data Configuration
        self.train_rollout_n = train_rollout_n
        self.train_information = train_information
        self.val_rollout_n = val_rollout_n
        self.val_information = val_information if val_information is not None else {
            "model": train_information.get("model", "default-model"),
            "temperature": 0.0,
            "top_p": 1.0,
        }
        self.mini_batch_size = mini_batch_size
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.reward_fillna_value = reward_fillna_value

        # Internal State
        self.backend_llm_server_addresses: List[str] = []
        self._total_tasks_queued = 0
        self._completed_rollouts_v0: Dict[str, RolloutLegacy] = {}
        self._task_id_to_original_sample: Dict[str, Dict[str, Any]] = {}
        self._server_thread: Optional[threading.Thread] = None
        self._proxy_thread: Optional[threading.Thread] = None
        self.is_train = True
        self.trace_agg_mode = trace_agg_mode
        ### 目前仅支持千问；后面可以再调整
        toolcall_format = train_information.get("format", "hermes")
        self.tool_parser = ToolParser.get_tool_parser(toolcall_format, tokenizer)
        tools_examples = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location", "unit"],
                },
            },
        ]
        toolcall_message_examples = [
            {"role": "user", "content": "What's the weather like in Paris?"},
            {"role": "assistant", "content": "", "tool_calls":[
                {
                    "id": "fc_1234xyz",
                    "type": "function",
                    "function":{
                        "name": "get_weather",
                        "arguments": "{\"location\":\"Paris, France\"}"
                    }
                }
            ]
            }
        ]
        toolcall_example_chat_template = tokenizer.apply_chat_template(
            toolcall_message_examples, tools=tools_examples,\
                add_generation_prompt=False, tokenize=False,
        )
        toolcall_example_chat_template_token_last2 = tokenizer.encode(toolcall_example_chat_template.strip())[-2:]
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        toolcall_candidate_token_last2_list = [toolcall_example_chat_template_token_last2]
        if toolcall_example_chat_template_token_last2[-1] != eos_token_id:
            toolcall_candidate_token_last2_list.append([toolcall_example_chat_template_token_last2[0], eos_token_id])
        if toolcall_example_chat_template_token_last2[-1] != pad_token_id:
            toolcall_candidate_token_last2_list.append([toolcall_example_chat_template_token_last2[0], pad_token_id])
        self.toolcall_candidate_token_last2_list = toolcall_candidate_token_last2_list
        print(f">>> {eos_token_id=}, {pad_token_id=}, {self.toolcall_candidate_token_last2_list=}")


    def _internal_loop_runner(self):
        """Run the internal loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._internal_loop = loop
        loop.run_forever()
        loop.close()

    def _start_proxy_server_v0(self):
        """
        Initializes and runs a Flask-based proxy server in a separate thread.
        This proxy load-balances requests to the actual backend LLM servers.
        """
        app = Flask(__name__)

        num_requests = 0
        last_request_time = 0

        @app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
        def proxy(path: str):  # type: ignore
            if not self.backend_llm_server_addresses:
                abort(503, description="No backend LLM servers available.")

            # Randomly choose a backend server for load balancing
            target_server = random.choice(self.backend_llm_server_addresses)
            target_url = f"http://{target_server}/v1/{path}"

            # Copy client request headers, removing the Host header
            headers = {key: value for key, value in request.headers if key.lower() != "host"}

            # Log the request for debugging
            nonlocal num_requests, last_request_time
            current_time = time.time()
            num_requests += 1
            if current_time - last_request_time > 60 or num_requests == 1 or num_requests % 100 == 0:
                print(f"Proxying {request.method} request to {target_server}. Request data: {request.get_data()}")
            last_request_time = current_time

            try:
                # Forward the request to the target backend
                resp = requests.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=request.args,  # type: ignore
                    data=request.get_data(),
                    cookies=request.cookies,
                    allow_redirects=False,
                    timeout=self.llm_timeout_seconds,
                )
                # Filter out hop-by-hop headers before returning the response
                excluded_headers = [
                    "content-encoding",
                    "content-length",
                    "transfer-encoding",
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "upgrade",
                ]
                response_headers = [
                    (name, value) for name, value in resp.raw.headers.items() if name.lower() not in excluded_headers
                ]
                if resp.status_code == 200:
                    # NOTE: from Zhiyuan's code.
                    # https://github.com/hzy46/verl_agent_mode/blob/2db65ea9858f645a914120357412a7540f8bd82d/verl/trainer/ppo/ray_trainer.py#L692-L711
                    # request_json = json.loads(request.get_data().decode("utf-8"))
                    response_json = json.loads(resp.content.decode("utf-8"))
                    # response_message = ChatCompletion(**response_json).choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
                    # tool_schemas = request_json.get("tools", None)
                    # prompt_ids = self.tokenizer.apply_chat_template(request_json["messages"], tools=tool_schemas, add_generation_prompt=True, tokenize=True)
                    # full_ids = self.tokenizer.apply_chat_template(request_json["messages"] + [response_message], tools=tool_schemas, add_generation_prompt=False, tokenize=True)
                    # TBD: response_ids sometimes ends with "<eos_id>\n", shall we keep the extra "\n"?
                    # sometimes it has some differences with the hacky method in the end, but this should align with ToolCompletionCallback
                    # response_ids = full_ids[len(prompt_ids):]

                    # NOTE (yuge): They are different. Don't know why.
                    # assert response_json['prompt_token_ids'] == prompt_ids
                    # patched_response_ids = response_json['response_token_ids'][0]
                    # assert patched_response_ids == response_ids[:len(patched_response_ids)], f"{patched_response_ids} != {response_ids[:len(patched_response_ids)]}"
                    # response_json['prompt_token_ids'] = prompt_ids
                    # response_json['response_token_ids'] = [response_ids]
                    replaced_return_content = json.dumps(response_json).encode("utf-8")
                    return Response(replaced_return_content, status=resp.status_code, headers=response_headers)
                return Response(resp.content, resp.status_code, response_headers)
            except requests.exceptions.RequestException as e:
                abort(500, description=f"Error proxying request: {e}")

        def run_app():
            app.run(host="0.0.0.0", port=self.proxy_port, threaded=True, debug=False)

        self._proxy_thread = threading.Thread(target=run_app, daemon=True)
        self._proxy_thread.start()
        print(f"Proxy server running on port {self.proxy_port}")

    async def _update_proxy_server_v1(self):
        model_name = self.train_information.get("model")
        if not model_name:
            raise ValueError("Model name is not set.")
        self.llm_proxy.update_model_list(
            [
                ModelConfig(
                    {
                        "model_name": model_name,
                        "litellm_params": {
                            "model": "hosted_vllm/" + model_name,
                            "api_base": f"http://{address}/v1/",
                            "custom_llm_proxy": "vllm",
                            "timeout": 10,  # 增加超时时间，避免过早断开连接
                            "max_retries": 5,  # 增加重试次数
                            "num_retries": 5,  # LiteLLM 的重试参数
                        },
                    }
                )
                for address in self.backend_llm_server_addresses
            ],
        )

    ## 增大重试量和时间 
    async def _update_proxy_server_v1(self):
            model_name = self.train_information.get("model")
            if not model_name:
                raise ValueError("Model name is not set.")
            self.llm_proxy.update_model_list(
                [
                    ModelConfig(
                        {
                            "model_name": model_name,
                            "litellm_params": {
                                "model": "hosted_vllm/" + model_name,
                                "api_base": f"http://{address}/v1/",
                                "custom_llm_proxy": "vllm",
                                "timeout": 30,  # 增加超时时间，避免过早断开连接
                                "max_retries": 5,  # 增加重试次数
                                "num_retries": 5,  # LiteLLM 的重试参数
                            },
                        }
                    )
                    for address in self.backend_llm_server_addresses
                ],
            )
            await self.llm_proxy.restart()

    def start(self):
        """Starts the main AgentLightningServer and the proxy server."""

        if self.mode == "v0":

            def run_server():
                """Run the AgentLightningServer in a separate thread."""
                asyncio.run(self.server.run_forever())

            self._server_thread = threading.Thread(target=run_server, daemon=True)
            self._server_thread.start()

            # Wait for the server's internal startup event to be set.
            print("Waiting for AgentLightningServer to start...")
            is_ready = self.server.startup_event.wait(timeout=20.0)  # Wait up to 20s
            if not is_ready:
                raise RuntimeError("AgentLightningServer failed to start within the timeout period.")

            print(f"AgentLightningServer control plane running on port {self.server_port}")

            self._start_proxy_server_v0()
        else:
            # Agent lightning server is no longer needed;
            # Start proxy server in _async_set_up
            pass

    async def _async_set_up(self, data: Dict[str, Any], server_addresses: List[str], is_train: bool = True):
        """Async helper to set up data and resources on the server."""
        self.clear_data_and_server()
        if server_addresses != self.backend_llm_server_addresses:
            self.backend_llm_server_addresses = server_addresses
            if self.mode == "v1" and not self.llm_proxy.is_running():
                await self._update_proxy_server_v1()
        self.is_train = is_train


        # Select information based on training or validation mode
        info = self.train_information if is_train else self.val_information
        rollouts_per_sample = self.train_rollout_n if is_train else self.val_rollout_n
        
        # Build sampling parameters
        sampling_parameters = {
            "temperature": info.get("temperature", 0.7 if is_train else 0.0),
            "max_turns": info.get("max_turns", 5)
        }
        # Add top_p if specified
        if "top_p" in info:
            sampling_parameters["top_p"] = info["top_p"]
        # Add max_turns if specified
        if "max_turns" in info:
            sampling_parameters["max_turns"] = info["max_turns"]
        
        print(f"[AgentModeDaemon] Mode: {'train' if is_train else 'val'}, sampling_parameters: {sampling_parameters}, rollouts_per_sample: {rollouts_per_sample}")
        # 1. Update resources on the server for clients to use
        if self.mode == "v0":
            llm_resource = LLM(
                endpoint=f"http://127.0.0.1:{self.proxy_port}/v1",
                model=info.get("model", "default-model"),
                sampling_parameters=sampling_parameters,
            )
        else:
            llm_resource = self.llm_proxy.as_resource(
                model=info.get("model"),
                sampling_parameters=sampling_parameters,
            )

        resources: NamedResources = {"main_llm": llm_resource}

        if self.mode == "v0":
            resources_id = await self.server.update_resources(resources)
        else:
            resources_update = await self.store.add_resources(resources)
            resources_id = resources_update.resources_id

        # 2. Queue tasks for agents to process
        keys = list(data.keys())
        num_samples = len(data[keys[0]])

        for i in range(num_samples):
            data_id = str(uuid.uuid4())
            original_sample = {key: data[key][i] for key in keys}
            original_sample["data_id"] = data_id

            # For training, each sample is rolled out multiple times
            for _ in range(rollouts_per_sample):
                task_metadata = {"data_id": data_id, "is_train": is_train}

                # Data ID is different from Rollout ID, as one data can have multiple rollouts.
                if self.mode == "v0":
                    rollout_id = await self.server.queue_task(
                        sample=_to_native(original_sample),
                        mode="train" if is_train else "val",
                        resources_id=resources_id,
                        metadata=task_metadata,
                    )
                else:
                    rollout = await self.store.enqueue_rollout(
                        input=_to_native(original_sample),
                        mode="train" if is_train else "val",
                        resources_id=resources_id,
                        metadata=task_metadata,
                    )
                    await self.store.update_rollout(
                        rollout_id=rollout.rollout_id,
                        config=RolloutConfig(
                            unresponsive_seconds=self.llm_timeout_seconds,
                            timeout_seconds=self.llm_timeout_seconds,
                        ),
                    )
                    rollout_id = rollout.rollout_id

                # Store original sample data to reconstruct batch information later
                self._task_id_to_original_sample[rollout_id] = original_sample
                self._total_tasks_queued += 1

    def set_up_data_and_server(self, data: Dict[str, Any], server_addresses: List[str], is_train: bool = True):
        """Synchronous wrapper for setting up data and server resources."""
        coro = self._async_set_up(data, server_addresses, is_train)

        if self.mode == "v0":
            if not self.server.loop or not self.server.startup_event.is_set():
                raise RuntimeError("Server is not running or ready.")

            future = asyncio.run_coroutine_threadsafe(coro, self.server.loop)

        else:
            if self._internal_loop is None:
                raise RuntimeError("Internal loop is not running.")
            future = asyncio.run_coroutine_threadsafe(coro, self._internal_loop)
        try:
            future.result(timeout=600)  # Wait for completion with a timeout
        except Exception as e:
            print(f"Failed to set up data on server: {e}")
            raise

    def _validate_data(self, rollout: RolloutLegacy):
        if rollout.final_reward is None:
            print(
                f"Warning: Reward is None for rollout {rollout.rollout_id}, will be auto-set to {self.reward_fillna_value}."
            )
        if rollout.triplets is None:
            print(f"Warning: Triplet is None for rollout {rollout.rollout_id}.")
        elif len(rollout.triplets) == 0:
            print(f"Warning: Length of triplets is 0 for rollout {rollout.rollout_id}.")
        elif any(not r.response.get("token_ids", []) for r in rollout.triplets):
            print(f"Warning: Rollout {rollout.rollout_id} contains empty response: {rollout.triplets}")
        elif any(not r.prompt.get("token_ids", []) for r in rollout.triplets):
            print(f"Warning: Rollout {rollout.rollout_id} contains empty prompt: {rollout.triplets}")

    async def _validate_data_v1(self, rollout: Rollout) -> RolloutLegacy:
        """Convert Rollout to RolloutLegacy and validate.

        1. Task: construct from Rollout
        2. Triplets: obtained by querying spans and feeding into the adapter
        3. Final reward: extracted from last triplet's reward, searching backwards if not found
        4. Extract acc from reward span attributes and add to metadata
        """
        # Query spans for this rollout (latest attempt)
        spans = await self.store.query_spans(rollout.rollout_id, attempt_id="latest")

        # Convert spans to triplets using the adapter
        if not spans:
            # No triplets found, will emit a warning later.
            triplets = []
        else:
            triplets = self.adapter.adapt(spans)

        # Extract final reward and acc from spans
        final_reward: Optional[float] = None
        acc_value: Optional[float] = None
        
        if spans:
            # Search for reward span and extract both reward and acc from attributes
            for span in reversed(spans):
                # Check if this is a reward span (exact match or contains 'reward')
                is_reward_span = (span.name == "agentlightning.reward" or 
                                 span.name == "reward" or 
                                 "reward" in span.name.lower())
                
                if is_reward_span and span.attributes:
                    if final_reward is None and "reward" in span.attributes:
                        final_reward = float(span.attributes["reward"])
                    if acc_value is None and "acc" in span.attributes:
                        acc_value = float(span.attributes["acc"])
                        print(f"[Daemon] Extracted acc={acc_value} from span attributes for rollout {rollout.rollout_id}")
                    if final_reward is not None and acc_value is not None:
                        break
            
            # Debug: print if acc was not found
            if acc_value is None:
                print(f"[Daemon] Warning: acc not found in spans for rollout {rollout.rollout_id}")
        
        # If not found in spans, try triplets as fallback
        if final_reward is None and triplets:            # Search backwards through triplets for the first non-None reward
            for triplet in reversed(triplets):
                if triplet.reward is not None:
                    final_reward = triplet.reward
                    break

        # Merge acc into metadata
        metadata = rollout.metadata or {}
        if acc_value is not None:
            metadata["acc"] = acc_value
        else:
            metadata["acc"] = 0

        # Construct the Task object from Rollout
        task = Task(
            rollout_id=rollout.rollout_id,
            input=rollout.input,
            mode=rollout.mode,
            resources_id=rollout.resources_id,
            metadata=metadata,
        )

        # Create the Rollout object (without trace and logs as per user's note)
        result_rollout = RolloutLegacy(
            rollout_id=rollout.rollout_id,
            task=task,
            final_reward=final_reward,
            triplets=triplets,
            metadata=metadata,
        )

        # Run the same validation as v0
        self._validate_data(result_rollout)

        return result_rollout

    async def _async_run_until_finished(self, verbose: bool = True):
        """Async helper to wait for all tasks to complete."""
        while len(self._completed_rollouts_v0) < self._total_tasks_queued:
            if self.mode == "v0":
                completed_batch = await self.server.retrieve_completed_rollouts()
            else:
                completed_batch = await self.store.wait_for_rollouts(
                    rollout_ids=list(self._task_id_to_original_sample.keys()), timeout=0
                )
            for rollout in completed_batch:
                if rollout.rollout_id in self._completed_rollouts_v0:
                    # Already processed, skip
                    continue
                if isinstance(rollout, Rollout):
                    rollout = await self._validate_data_v1(rollout)
                else:
                    self._validate_data(rollout)
                if rollout.rollout_id not in self._task_id_to_original_sample:
                    print(f"Warning: Received unknown rollout ID {rollout.rollout_id}, skipping.")
                else:
                    self._completed_rollouts_v0[rollout.rollout_id] = rollout
            if verbose:
                print(f"Completed {len(self._completed_rollouts_v0)}/{self._total_tasks_queued} tasks...")
            await asyncio.sleep(5)

        print("All tasks finished.")

    def run_until_all_finished(self, verbose: bool = True):
        """Synchronously waits for all queued tasks to be completed and reported."""
        if self._total_tasks_queued == 0:
            print("Warning: No tasks were queued.")
            return

        if self.mode == "v0":
            if not self.server.loop or not self.server.startup_event.is_set():
                raise RuntimeError("Server is not running or ready.")
            loop = self.server.loop
        else:
            loop = self._internal_loop
            assert loop is not None

        coro = self._async_run_until_finished(verbose)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            future.result()  # Wait indefinitely for all tasks to complete
        except Exception as e:
            print(f"Error while waiting for tasks to finish: {e}")
            raise

    def get_test_metrics(self):
        """Calculates and returns metrics for a validation run."""
        assert not self.is_train, "This method should only be called during validation."
        assert len(self._completed_rollouts_v0) == self._total_tasks_queued

        sample_stat_list: List[Dict[str, Any]] = []
        sample_stat_list_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(
            list
        )  # FIXME: Evaluate whether grouping stats by source is actually needed.
        finished_id_to_acc: Dict[str, float] = {}

        for rollout_id, rollout in self._completed_rollouts_v0.items():
            final_reward = self._fillna_reward(rollout)
            
            # Extract accuracy from metadata
            acc_value = float(rollout.metadata.get("acc", 0))
            finished_id_to_acc[rollout_id] = acc_value
            
            if not rollout.triplets:
                print(f"Warning: No triplets found for test rollout {rollout.rollout_id}.")
                sample_stat_list.append({"reward": final_reward, "acc": acc_value})
                continue
            response_length_list = [len(triplet.response.get("token_ids", [])) for triplet in rollout.triplets]
            if "data_source" in self._task_id_to_original_sample[rollout_id]:
                # When a test sample includes a 'data_source' field, record per-source statistics for test results.
                data_source = self._task_id_to_original_sample[rollout_id]["data_source"]
                sample_stat_list_by_source[data_source].append(
                    {
                        "sum_response_length": np.sum(response_length_list),
                        "mean_response_length": np.mean(response_length_list) if response_length_list else 0,
                        "turn_count": len(rollout.triplets),
                        "reward": final_reward,
                        "acc": acc_value,
                    }
                )
            sample_stat_list.append(
                {
                    "sum_response_length": np.sum(response_length_list),
                    "mean_response_length": np.mean(response_length_list) if response_length_list else 0,
                    "turn_count": len(rollout.triplets),
                    "reward": final_reward,
                    "acc": acc_value,
                }
            )
        metric_dict: Dict[str, Any] = {}

        stats_w_trace = [stat for stat in sample_stat_list if "sum_response_length" in stat]
        stats_w_trace_by_source = {
            data_source: [stat for stat in sample_stats if "sum_response_length" in stat]
            for data_source, sample_stats in sample_stat_list_by_source.items()
        }
        for data_source, sample_stats in sample_stat_list_by_source.items():
            metric_dict.update(
                {
                    f"val/{data_source}/n_rollouts": len(sample_stats),
                    f"val/{data_source}/n_rollouts_w_trace": len(stats_w_trace_by_source[data_source]),
                    f"val/{data_source}/reward": np.mean(
                        [stat["reward"] for stat in sample_stats]
                    ),  # each rollout must have a reward (fillna if missing)
                    f"val/{data_source}/mean_response_length": np.mean(
                        [stat["mean_response_length"] for stat in stats_w_trace_by_source[data_source]]
                    ),
                    f"val/{data_source}/sum_response_length": np.mean(
                        [stat["sum_response_length"] for stat in stats_w_trace_by_source[data_source]]
                    ),
                    f"val/{data_source}/turn_count": np.mean(
                        [stat["turn_count"] for stat in stats_w_trace_by_source[data_source]]
                    ),
                    f"val/{data_source}/acc": np.sum(
                        [float(stat["acc"]) for stat in sample_stats]
                    )/len(sample_stats),
                    f"val-core/{data_source}/acc": [float(stat["acc"]) for stat in sample_stats],
                    f"val-core/{data_source}/reward": [float(stat["reward"]) for stat in sample_stats],
                }
            )
        metric_dict.update(
            {
                "val/n_rollouts": len(sample_stat_list),
                "val/n_rollouts_w_trace": len(stats_w_trace),
                "val/reward": np.mean(
                    [stat["reward"] for stat in sample_stat_list]
                ),  # each rollout must have a reward (fillna if missing)
                "val/acc": np.mean(
                    [stat["acc"] for stat in sample_stat_list]
                ),
                "val/mean_response_length": np.mean([stat["mean_response_length"] for stat in stats_w_trace]),
                "val/sum_response_length": np.mean([stat["sum_response_length"] for stat in stats_w_trace]),
                "val/turn_count": np.mean([stat["turn_count"] for stat in stats_w_trace]),
                "val-core/all_source/reward": [float(stat["reward"]) for stat in sample_stat_list],
                "val-core/all_source/acc": [float(stat["acc"]) for stat in sample_stat_list],
            }
        )
        metric_dict["val/acc"] = np.sum(list(finished_id_to_acc.values()))/len(sample_stat_list)
        print(f"[Daemon] Validation: computed val/acc={metric_dict['val/acc']:.4f} from {len(finished_id_to_acc)} rollouts")
        
        return metric_dict

    def get_val_data_batch(self, max_prompt_length: int, max_response_length: int, device: torch.device):
        """
        Processes completed rollouts to generate a validation data batch.
        Similar to get_train_data_batch but for validation mode.
        """
        assert not self.is_train, "This method should only be called during validation."
        return self._get_data_batch(max_prompt_length, max_response_length, device)

    def get_train_data_batch(self, max_prompt_length: int, max_response_length: int, device: torch.device):
        """
        Processes completed rollouts to generate a training data batch.

        This function reconstructs the logic from the original AgentModeDaemon,
        using data retrieved from the new server architecture. It handles padding,
        truncation, and tensor creation for the PPO training loop.
        """
        assert self.is_train, "This method should only be called during training."
        return self._get_data_batch(max_prompt_length, max_response_length, device)

    def _get_data_batch(self, max_prompt_length: int, max_response_length: int, device: torch.device):
        """
        Internal method to process completed rollouts into a data batch.
        Used by both training and validation.
        """
        assert len(self._completed_rollouts_v0) == self._total_tasks_queued

        # 1. Reconstruct the `finished_id_to_sample_info` structure from completed rollouts
        finished_id_to_sample_info: Dict[str, Dict[str, Any]] = {}
        finished_id_to_final_reward: Dict[str, float] = {}
        finished_id_to_acc: Dict[str, float] = {}
        
        for rollout_id, rollout in self._completed_rollouts_v0.items():
            original_sample = self._task_id_to_original_sample[rollout_id]

            final_reward = self._fillna_reward(rollout)
            
            # Extract accuracy from metadata
            acc_value = float(rollout.metadata["acc"])
            if not rollout.triplets:
                finished_id_to_final_reward[rollout_id] = final_reward
                finished_id_to_acc[rollout_id] = acc_value
                print(f"Warning: No triplets found for training rollout {rollout.rollout_id}, skipping.")
                continue
            # The client should report triplets that contain prompt_ids and response_ids.
            # Example triplet.prompt: {"token_ids": [...]}
            # Example triplet.response: {"token_ids": [...]}
            trace_list = [
                {"prompt_ids": t.prompt.get("token_ids", []), "response_ids": t.response.get("token_ids", [])}
                for t in rollout.triplets
            ]
            trace_list = [trace_list_item for trace_list_item in trace_list if len(trace_list_item["prompt_ids"]) and len(trace_list_item["response_ids"])]
            # 这里额外添加一些处理逻辑扔掉所有不符合要求的坏输出
            trace_list_valid = []
            for trace_list_item in trace_list:
                response_ids = trace_list_item["response_ids"]
                content_wo_tool_calls, tool_calls = asyncio.run(self.tool_parser.extract_tool_calls(response_ids))
                if len(tool_calls):
                    is_valid = False
                    # 如果有工具要求必须得工具调用之后没有奇怪输出才加入训练
                    for toolcall_candidate_token_last2 in self.toolcall_candidate_token_last2_list:
                        if len(response_ids) > 2 and (response_ids[-2] == toolcall_candidate_token_last2[0]) and (response_ids[-1] == toolcall_candidate_token_last2[1]):
                            trace_list_valid.append(trace_list_item)
                            is_valid = True
                            break
                    # if len(response_ids) > 2 and response_ids[-1] == 151645 and response_ids[-2] == 151658:
                    #     # </tool_call><|im_end|>
                    #     is_valid = True
                    #     trace_list_valid.append(trace_list_item)
                    # elif len(response_ids) > 2 and response_ids[-1] == 151643 and response_ids[-2] == 151658:
                    #     # </tool_call><|endoftext|>
                    #     is_valid = True
                    #     trace_list_valid.append(trace_list_item)
                    if not is_valid:
                        response_text = self.tokenizer.decode(response_ids)
                        print(f"Warning: Invalid response_id (w/ tool calls) found for training rollout: {response_text}")
                        continue
                else:
                    trace_list_valid.append(trace_list_item)
            # 如果清理后的轨迹输出都只有1条的话就直接扔掉整个
            if len(trace_list_valid) <= 1:
                trace_list_valid = []
            trace_list = trace_list_valid
            info = {
                "reward": final_reward,
                "trace_list": trace_list,
                "data_id": original_sample["data_id"],
            }
            finished_id_to_sample_info[rollout_id] = info
            finished_id_to_final_reward[rollout_id] = final_reward
            finished_id_to_acc[rollout_id] = acc_value
        # --- Data processing and tensor creation logic ---
        # Get all the reported data.
        # prompt_ids are left-padded.
        # response_ids are right-padded.
        # They are concatenated in the middle.
        # Discard handling:
        #   - Those exceeding max_prompt_length will be marked for discard, but not
        #     discarded here. They are only truncated and marked, to be discarded later.
        #     This is for the correctness of the advantage calculation.
        #   - The discard for the PPO mini-batch should also be handled this way.
        input_ids_list: List[List[int]] = []
        input_attention_mask_list: List[List[int]] = []
        response_ids_list: List[List[int]] = []
        response_attention_mask_list: List[List[int]] = []
        reward_list: List[float] = []
        data_id_list: List[str] = []
        rollout_id_list: List[str] = []
        turn_index_list: List[int] = []
        is_drop_list: List[bool] = []
        n_trunc_sample_because_of_response = 0
        turn_count = 0

        if self.trace_agg_mode == "transition":
            for rollout_id, sample_info in finished_id_to_sample_info.items():
                for turn_index, trace in enumerate(sample_info["trace_list"]):
                    reward_list.append(sample_info["reward"])
                    prompt_ids, response_ids = trace["prompt_ids"], trace["response_ids"]

                    # Mark samples with prompts exceeding max_prompt_length to be dropped later
                    if len(prompt_ids) > max_prompt_length:
                        prompt_ids = prompt_ids[:max_prompt_length]
                        is_drop_list.append(True)
                    else:
                        is_drop_list.append(False)

                    # Truncate responses that exceed max_response_length
                    if len(response_ids) > max_response_length:
                        response_ids = response_ids[:max_response_length]
                        n_trunc_sample_because_of_response += 1

                    # Pad prompts to the left and responses to the right
                    one_input_ids, one_input_attention_mask = get_left_padded_ids_and_attention_mask(
                        prompt_ids, max_prompt_length, self.pad_token_id
                    )
                    one_response_ids, one_response_attention_mask = get_right_padded_ids_and_attention_mask(
                        response_ids, max_response_length, self.pad_token_id
                    )

                    input_ids_list.append(one_input_ids)
                    input_attention_mask_list.append(one_input_attention_mask)
                    response_ids_list.append(one_response_ids)
                    response_attention_mask_list.append(one_response_attention_mask)
                    data_id_list.append(sample_info["data_id"])
                    rollout_id_list.append(rollout_id)
                    turn_index_list.append(turn_index)
                    turn_count += 1

        elif self.trace_agg_mode == "trajectory":
            response_mask_list: List[List[int]] = []
            
            for rollout_id, sample_info in finished_id_to_sample_info.items():
    # --------- 分组：基于 LCP(prev_context, curr_prompt) 的“连续扩展”判定 ---------

                merged_trace_idx = []           # List[List[int]]
                current_group = []              # 当前组内的 turn 索引
                prev_context = []               # 上一条 turn 的完整上下文 = prev_prompt + prev_response

                for turn_index, trace in enumerate(sample_info["trace_list"]):
                    curr_prompt = trace["prompt_ids"]
                    curr_resp   = trace["response_ids"]

                    if not current_group:
                        # 第一条，开启新组
                        current_group = [turn_index]
                        prev_context = curr_prompt + curr_resp
                        continue

                    # 仅比较：上一轮完整上下文 是否是 下一轮 prompt 的前缀
                    L = lcp_len(prev_context, curr_prompt)
                    if L == len(prev_context):
                        # 延续（连续扩展）
                        current_group.append(turn_index)
                        prev_context = curr_prompt + curr_resp
                    else:
                        # 断裂：收拢旧组，开新组
                        merged_trace_idx.append(current_group)
                        current_group = [turn_index]
                        prev_context = curr_prompt + curr_resp

                if current_group:
                    merged_trace_idx.append(current_group)

    # --------- 按组拼接：固定首个 prompt；新增 prompt 尾巴(mask=0) + 各轮响应(mask=1) ---------
                for group in merged_trace_idx:
                    # 基础样本：使用该组第一个 turn 的 prompt 作为整条样本的 prompt
                    first_prompt = sample_info["trace_list"][group[0]]["prompt_ids"]
                    first_resp   = sample_info["trace_list"][group[0]]["response_ids"]

                    prompt_ids    = first_prompt[:]                  # 固定为第一条的 prompt
                    response_ids  = first_resp[:]                    # 初始响应
                    response_mask = [1] * len(first_resp)            # 1 表示“模型输出（响应）”

                    prev_context = first_prompt + first_resp         # 该组内“上一轮完整上下文”

                    # 依次合入后续 turn：新增的 prompt 尾巴 → mask=0；当轮 response → mask=1
                    for turn_index in group[1:]:
                        trace = sample_info["trace_list"][turn_index]
                        curr_prompt = trace["prompt_ids"]
                        curr_resp   = trace["response_ids"]

                        # 计算 prev_context 与 curr_prompt 的 LCP
                        L = lcp_len(prev_context, curr_prompt)
                        # 新增的 prompt 尾巴（可能包含工具结果、role 分隔、模板 token 等）
                        new_prompt_tail = curr_prompt[L:]
                        if new_prompt_tail:
                            response_ids  += new_prompt_tail
                            response_mask += [0] * len(new_prompt_tail)   # 非输出，对齐上下文

                        # 当前 turn 的真实响应
                        if curr_resp:
                            response_ids  += curr_resp
                            response_mask += [1] * len(curr_resp)

                        # 更新“上一轮完整上下文”
                        prev_context = curr_prompt + curr_resp

                    # 奖励：整组共享
                    reward_list.append(sample_info["reward"])

                    # 过长的 prompt：标记丢弃（或截断为 max_prompt_length；保持你原有策略）
                    if len(prompt_ids) > max_prompt_length:
                        prompt_ids = prompt_ids[:max_prompt_length]
                        is_drop_list.append(True)
                    else:
                        is_drop_list.append(False)

                    # 响应超长：同步截断 response_ids 与 response_mask
                    if len(response_ids) > max_response_length:
                        response_ids  = response_ids[:max_response_length]
                        response_mask = response_mask[:max_response_length]
                        n_trunc_sample_because_of_response += 1

                    # 左填充 prompt；右填充 response / response_mask
                    one_input_ids, one_input_attention_mask = get_left_padded_ids_and_attention_mask(
                        prompt_ids, max_prompt_length, self.pad_token_id
                    )
                    one_response_ids, one_response_attention_mask = get_right_padded_ids_and_attention_mask(
                        response_ids, max_response_length, self.pad_token_id
                    )
                    one_response_mask, _ = get_right_padded_ids_and_attention_mask(
                        response_mask, max_response_length, 0
                    )

                    # 写入 batch 列表
                    input_ids_list.append(one_input_ids)
                    input_attention_mask_list.append(one_input_attention_mask)
                    response_ids_list.append(one_response_ids)
                    response_attention_mask_list.append(one_response_attention_mask)
                    response_mask_list.append(one_response_mask)
                    data_id_list.append(sample_info["data_id"])
                    rollout_id_list.append(rollout_id)
                    turn_index_list.append(group[0])     # 注意：这是变长结构，后面入 non_tensor_batch 时请用 dtype=object 或直接保留为 list
                    turn_count += 1
        elif self.trace_agg_mode == "use_last_trace":
            response_mask_list: List[List[int]] = []
            for rollout_id, sample_info in finished_id_to_sample_info.items():
                
                for turn_index, trace in enumerate(sample_info["trace_list"]):
                    if turn_index == 0:
                        prompt_ids = trace["prompt_ids"]
                        response_ids = trace["response_ids"]
                        last_prompt_ids = trace["prompt_ids"]
                        last_response_ids = trace["response_ids"]
                        response_mask = [1] * len(response_ids)
                    else:
                        curr_prompt_ids = trace["prompt_ids"]
                        curr_response_ids = trace["response_ids"]
                        tool_response_ids = curr_prompt_ids[len(last_prompt_ids)+len(last_response_ids):]
                        response_ids += tool_response_ids
                        response_mask += [0] * len(tool_response_ids)
                        response_ids += curr_response_ids
                        response_mask += [1] * len(curr_response_ids)
                        last_prompt_ids = curr_prompt_ids
                        last_response_ids = curr_response_ids
                    turn_count += 1
                        
                reward_list.append(sample_info["reward"])
                if len(prompt_ids) > max_prompt_length:
                        prompt_ids = prompt_ids[:max_prompt_length]
                        is_drop_list.append(True)
                else:
                    is_drop_list.append(False)
                
                if len(response_ids) > max_response_length:
                        response_ids  = response_ids[:max_response_length]
                        response_mask = response_mask[:max_response_length]
                        n_trunc_sample_because_of_response += 1
                # Pad prompts to the left and responses to the right
                one_input_ids, one_input_attention_mask = get_left_padded_ids_and_attention_mask(
                    prompt_ids, max_prompt_length, self.pad_token_id
                )
                one_response_ids, one_response_attention_mask = get_right_padded_ids_and_attention_mask(
                    response_ids, max_response_length, self.pad_token_id
                )
                one_response_mask, _ = get_right_padded_ids_and_attention_mask(
                    response_mask, max_response_length, 0
                )

                input_ids_list.append(one_input_ids)
                input_attention_mask_list.append(one_input_attention_mask)
                response_ids_list.append(one_response_ids)
                response_attention_mask_list.append(one_response_attention_mask)
                response_mask_list.append(one_response_mask)
                data_id_list.append(sample_info["data_id"])
                rollout_id_list.append(rollout_id)
                turn_index_list.append(0)

                
        else:
            raise ValueError(f"Unknown trace_agg_mode: {self.trace_agg_mode}")

        n_transition = len(input_ids_list)
        batch_input_ids = torch.LongTensor(input_ids_list).to(device)
        input_attention_mask = torch.LongTensor(input_attention_mask_list).to(device)
        batch_response_ids = torch.LongTensor(response_ids_list).to(device)
        response_attention_mask = torch.LongTensor(response_attention_mask_list).to(device)
        if self.trace_agg_mode == "trajectory" or self.trace_agg_mode == "use_last_trace":
            response_mask = torch.LongTensor(response_mask_list).to(device)
        else:
            response_mask = None

        # Concatenate prompts and responses to form the full sequence
        batch_seq = torch.cat([batch_input_ids, batch_response_ids], dim=-1)
        attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=-1)
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
        is_drop_mask = torch.BoolTensor(is_drop_list).to(device)
        scores = torch.tensor(reward_list, dtype=torch.bfloat16).to(device)

        # Create token-level scores by placing the final reward at the last token position
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
        # At the eos_mask_idx position of each sample, fill in the corresponding scores.
        # torch.arange(n_transition) generates [0,1,2,...,bsz-1] as indices for the batch dimension.
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores[torch.arange(n_transition), eos_mask_idx] = scores
        # Only take the last response_length part of the sequence to get the token-level scores for the model's response part.
        token_level_scores = token_level_scores[:, -max_response_length:]
        # Form the final batch using TensorDict
        batch = TensorDict(
            {
                "prompts": batch_input_ids,
                "responses": batch_response_ids,
                "input_ids": batch_seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "is_drop_mask": is_drop_mask,
                "token_level_scores": token_level_scores.contiguous(),
                **({"response_mask": response_mask} if self.trace_agg_mode == "trajectory" else {}),
            },
            batch_size=n_transition,
        )
        data_proto = DataProto(batch=batch)

        # Use appropriate metric prefix based on mode
        prefix = "training" if self.is_train else "val"
        data_metrics = {
            f"{prefix}/reward": np.mean(list(finished_id_to_final_reward.values())),
            f"{prefix}/n_rollouts": len(finished_id_to_final_reward),
            f"{prefix}/n_rollouts_w_trace": len(finished_id_to_sample_info),
            f"{prefix}/n_truncated_triplets": n_trunc_sample_because_of_response,
            f"{prefix}/n_triplets": n_transition,
        }
        data_metrics[f"{prefix}/acc"] = np.sum(list(finished_id_to_acc.values()))/len(finished_id_to_final_reward)
        print(f"[Daemon] {prefix.capitalize()}: computed {prefix}/acc={data_metrics[f'{prefix}/acc']:.4f} from {len(finished_id_to_acc)} rollouts")

        # Add non-tensor data for advantage calculation and logging
        data_proto.non_tensor_batch["data_id_list"] = np.array(data_id_list)  # type: ignore
        data_proto.non_tensor_batch["rollout_id_list"] = np.array(rollout_id_list)  # type: ignore
        data_proto.non_tensor_batch["turn_index_list"] = np.array(turn_index_list)  # type: ignore

        return data_proto, data_metrics

    def clear_data_and_server(self):
        """Resets the internal state of the daemon for the next run."""
        self.backend_llm_server_addresses = []
        self._completed_rollouts_v0.clear()
        self._task_id_to_original_sample.clear()
        self._total_tasks_queued = 0
        # For a true reset, the server's internal queues would also need clearing.
        # This implementation assumes that `set_up_data_and_server` is called
        # for each new run, effectively starting a fresh batch.

    def _fillna_reward(self, rollout: RolloutLegacy):
        if rollout.final_reward is None:
            if self.reward_fillna_value is not None:  # type: ignore
                final_reward = self.reward_fillna_value
            else:
                raise ValueError(f"Reward is None for rollout {rollout.rollout_id}, please check the reward function.")
        else:
            final_reward = rollout.final_reward
        return final_reward




if __name__ == "__main__":

    tokenizer_path = ""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tools_examples = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location", "unit"],
            },
        },
    ]
    toolcall_message_examples = [
        {"role": "user", "content": "What's the weather like in Paris?"},
        {"role": "assistant", "content": "", "tool_calls":[
            {
                "id": "fc_1234xyz",
                "type": "function",
                "function":{
                    "name": "get_weather",
                    "arguments": "{\"location\":\"Paris, France\"}"
                }
            }
        ]
        }
    ]
    toolcall_example_chat_template = tokenizer.apply_chat_template(
        toolcall_message_examples, tools=tools_examples,\
            add_generation_prompt=False, tokenize=False,
    )
    toolcall_example_chat_template_token_last2 = tokenizer.encode(toolcall_example_chat_template.strip())[-2:]
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    print(toolcall_example_chat_template_token_last2, eos_token_id, pad_token_id)

