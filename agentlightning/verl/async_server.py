# Copyright (c) Microsoft. All rights reserved.

# type: ignore

import logging
import os
import socket
import time
from contextlib import asynccontextmanager
from copy import deepcopy

import fastapi
import ray
import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.utils import random_uuid

from agentlightning.instrumentation.vllm import ChatCompletionResponsePatched, instrument_vllm

logger = logging.getLogger(__name__)


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _unwrap_ray_remote(cls):
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


@ray.remote(num_cpus=1)
class PatchedvLLMServer(_unwrap_ray_remote(AsyncvLLMServer)):

    def __init__(self, *args, **kwargs):
        instrument_vllm()
        super().__init__(*args, **kwargs)

        self.config = deepcopy(self.config)
        # self.config.rollout.multi_turn.tool_config_path = "/dev/null"

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print(f"FastAPI listen on {self.address}:{self.port}")
            self.server_ready.set()
            yield

            # There's no way to gracefully restart uvicorn server if port is already in use,
            # so we exit the process directly and let AsyncLLMServerManager restart it.
            print("FastAPI shutdown, maybe address already in use, exit process immediately.")
            os._exit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route("/v1/chat/completions", self.chat_completion, methods=["POST"])

        self.port = _get_free_port()
        # Configure uvicorn with connection limits to prevent "Maximum number of open connections reached" errors
        # limit_concurrency: Maximum number of concurrent connections per server instance
        # backlog: Maximum number of pending connections in TCP queue
        # timeout_keep_alive: Seconds to keep idle connections alive before closing
        config = uvicorn.Config(
            app,
            host=["::", "0.0.0.0"],
            port=self.port,
            log_level="warning",
            limit_concurrency=int(os.environ.get("UVICORN_LIMIT_CONCURRENCY", 200)),
            backlog=int(os.environ.get("UVICORN_BACKLOG", 2048)),
            timeout_keep_alive=int(os.environ.get("UVICORN_TIMEOUT_KEEP_ALIVE", 30)),
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    async def init_engine(self):
        """Init vLLM AsyncLLM engine with improved tool_parser configuration."""
        # Call parent's init_engine first
        await super().init_engine()
        
        # Apply improved tool_parser configuration logic
        # Only use tool_parser when tool_config_path is explicitly set and not "/dev/null"
        config = self.config.rollout
        tool_config = config.multi_turn.tool_config_path
        enable_auto_tools = tool_config is not None and tool_config != "/dev/null"
        tool_parser = config.multi_turn.format if enable_auto_tools else None
        
        # Recreate openai_serving_chat with improved configuration
        model_path = self.config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        model_config = self.engine.model_config
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            models,
            "assistant",
            request_logger=RequestLogger(max_log_len=4096),
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=enable_auto_tools,
            tool_parser=tool_parser,
        )

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        
    
        try:
            # Access tokenizer directly from engine (AsyncLLM has self.tokenizer attribute)
            tokenizer = self.engine.tokenizer.get_lora_tokenizer(None)
            messages = request_json.get("messages", [])
            
            # Apply chat template to messages
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_tokens = tokenizer.encode(prompt_text)
            prompt_len = len(prompt_tokens)
            
            # Calculate available max_tokens: same as generate() method
            available_max_tokens = self.max_model_len - prompt_len - 500
            
            if available_max_tokens <= 0:
                # Prompt is too long, return empty response without sending to vLLM
                print(f"[vLLM] ERROR: Prompt length ({prompt_len}) exceeds max_model_len ({self.max_model_len}). Returning empty response.")
                logger.error(
                    f"Prompt length ({prompt_len}) exceeds max_model_len ({self.max_model_len}). "
                    f"Returning empty response to avoid vLLM error."
                )
                
                # Create empty ChatCompletionResponse
                model_name = request_json.get("model", "model")
                empty_response = ChatCompletionResponse(
                    id=f"chatcmpl-{random_uuid()}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=model_name,
                    choices=[
                        ChatCompletionResponseChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=""),
                            finish_reason="length",
                        )
                    ],
                    usage=UsageInfo(
                        prompt_tokens=prompt_len,
                        completion_tokens=0,
                        total_tokens=prompt_len,
                    ),
                )
                return JSONResponse(content=empty_response.model_dump())
            
            # Adjust max_tokens in the JSON before creating request object
            request_json["max_tokens"] = available_max_tokens
            print(f"[vLLM] Adjusted max_tokens: {request_json['max_tokens']}")
        except Exception as e:
            print(f"[vLLM] ERROR in truncation: {e}")
            logger.error(f"Failed to estimate prompt length: {e}. Using conservative fallback.")
            import traceback
            traceback.print_exc()
            
            # Conservative fallback: limit max_tokens to 1/3 of max_model_len
            # This leaves room for potentially long prompts
            conservative_max_tokens = int(self.max_model_len / 3)
            original_max_tokens = request_json.get("max_tokens")
            
            if original_max_tokens is None:
                # If no max_tokens specified, set a conservative value
                request_json["max_tokens"] = conservative_max_tokens
                print(f"[vLLM] Fallback: Setting max_tokens to conservative {conservative_max_tokens}")
            elif original_max_tokens > conservative_max_tokens:
                # If requested max_tokens is too large, truncate it
                request_json["max_tokens"] = conservative_max_tokens
                print(f"[vLLM] Fallback truncation: {original_max_tokens} → {conservative_max_tokens}")
        
        # Now create the request object with adjusted max_tokens
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())
