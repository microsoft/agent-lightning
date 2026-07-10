# Copyright (c) Microsoft. All rights reserved.

from copy import deepcopy
from importlib import import_module
from typing import Any, Callable, cast

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from agentlightning.instrumentation.vllm import instrument_vllm

ray = import_module("ray")
AsyncvLLMServer = getattr(import_module("verl.workers.rollout.vllm_rollout.vllm_async_server"), "AsyncvLLMServer")
vllm_protocol = import_module("vllm.entrypoints.openai.protocol")
ChatCompletionRequest = getattr(vllm_protocol, "ChatCompletionRequest")
ErrorResponse = getattr(vllm_protocol, "ErrorResponse")


def _unwrap_ray_remote(cls: Any) -> Any:
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


@ray.remote(num_cpus=1)
class PatchedvLLMServer(_unwrap_ray_remote(AsyncvLLMServer)):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        instrument_vllm()
        base_init = cast(Callable[..., None], getattr(super(), "__init__"))
        base_init(*args, **kwargs)

        self_api = cast(Any, self)
        self_api.config = deepcopy(self_api.config)
        self_api.config.rollout.multi_turn.tool_config_path = "/dev/null"

    async def chat_completion(self, raw_request: Request) -> JSONResponse | StreamingResponse:
        """OpenAI-compatible HTTP endpoint.

        API reference: [OpenAI-compatible server documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            return JSONResponse(content=generator.model_dump())
