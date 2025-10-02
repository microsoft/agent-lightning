# Copyright (c) Microsoft. All rights reserved.

import asyncio
import multiprocessing
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import openai
import pytest
from opentelemetry import trace as trace_api

from agentlightning.litagent import LitAgent
from agentlightning.llm_proxy import LLMProxy
from agentlightning.reward import emit_reward
from agentlightning.runner import AgentRunnerV2
from agentlightning.store.base import LightningStore
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.tracer.base import BaseTracer
from agentlightning.types.core import LLM

from ..common.network import get_free_port
from ..common.tracer import clear_tracer_provider
from ..common.vllm import RemoteOpenAIServer


async def init_runner(
    agent: LitAgent[Any],
    *,
    resources: Optional[Dict[str, LLM]] = None,
) -> tuple[AgentRunnerV2[Any], InMemoryLightningStore]:
    store = InMemoryLightningStore()
    llm_resource = resources or {"llm": LLM(endpoint="http://localhost", model="dummy")}
    await store.update_resources("default", llm_resource)

    runner = AgentRunnerV2[Any](tracer=AgentOpsTracer(), poll_interval=0.01)
    runner.init(agent)
    runner.init_worker(worker_id=0, store=store)
    return runner, store


def teardown_runner(runner: AgentRunnerV2[Any]) -> None:
    runner.teardown_worker(worker_id=0)
    runner.teardown()


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    clear_tracer_provider()
    yield


@pytest.mark.asyncio
async def test_runner_integration_basic_rollout() -> None:
    class EchoAgent(LitAgent[str]):
        async def validation_rollout_async(self, task: str, *, resources: Dict[str, Any], rollout: Any) -> None:
            emit_reward(1.0)

    import logging

    from agentlightning.logging import configure_logger

    configure_logger(level=logging.DEBUG)

    agent = EchoAgent()
    runner, store = await init_runner(agent)
    try:
        await runner.step("hello integration")
    finally:
        teardown_runner(runner)

    rollouts = await store.query_rollouts()
    assert rollouts and rollouts[0].status == "succeeded"
    attempts = await store.query_attempts(rollouts[0].rollout_id)
    spans = await store.query_spans(rollouts[0].rollout_id, attempts[-1].attempt_id)
    print(store.__dict__)
    assert any(span.attributes.get("reward") == 1.0 for span in spans)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.getenv("OPENAI_BASE_URL") and os.getenv("OPENAI_API_KEY")),
    reason="OpenAI endpoint or key not configured",
)
async def test_runner_integration_with_openai() -> None:
    class OpenAIAgent(LitAgent[str]):
        async def validation_rollout_async(self, task: str, *, resources: Dict[str, LLM], rollout: Any) -> float:
            llm = resources["llm"]
            client = openai.AsyncOpenAI(base_url=llm.endpoint, api_key=llm.api_key)
            response = await client.chat.completions.create(
                model=llm.model,
                messages=[{"role": "user", "content": task}],
            )
            assert response.choices, "OpenAI response should contain choices"
            return 0.0

    base_url = os.environ["OPENAI_BASE_URL"]
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    agent = OpenAIAgent()
    resources = {"llm": LLM(endpoint=base_url, model=model, api_key=api_key)}
    runner, store = await init_runner(agent, resources=resources)
    try:
        await runner.step("Say hello in one word")
    finally:
        teardown_runner(runner)

    rollouts = await store.query_rollouts()
    assert rollouts and rollouts[0].status == "succeeded"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.getenv("OPENAI_BASE_URL") and os.getenv("OPENAI_API_KEY")),
    reason="OpenAI endpoint or key not configured",
)
async def test_runner_integration_with_litellm_proxy() -> None:
    litellm = pytest.importorskip("litellm")

    class LiteLLMAgent(LitAgent[str]):
        def validation_rollout(self, task: str, *, resources: Dict[str, LLM], rollout: Any) -> float:
            llm = resources["llm"]
            response = litellm.completion(
                model=llm.model,
                messages=[{"role": "user", "content": task}],
            )
            assert response.get("choices"), "litellm proxy should return choices"
            return 0.0

    agent = LiteLLMAgent()
    resources = {"llm": LLM(endpoint="http://dummy", model="openai/gpt-4o-mini")}
    runner, store = await init_runner(agent, resources=resources)
    try:
        await runner.step("Give me a short greeting")
    finally:
        teardown_runner(runner)

    rollouts = await store.query_rollouts()
    assert rollouts and rollouts[0].status == "succeeded"


@pytest.fixture(scope="module")
def server():
    vllm_port = get_free_port()
    with RemoteOpenAIServer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        vllm_serve_args=[
            "--gpu-memory-utilization",
            "0.7",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--port",
            str(vllm_port),
        ],
    ) as server:
        yield server


@pytest.mark.asyncio
async def test_runner_integration_with_spawned_litellm_proxy(server: RemoteOpenAIServer) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    proxy_store = InMemoryLightningStore()
    proxy = LLMProxy(
        port=get_free_port(),
        model_list=[
            {
                "model_name": "gpt-4o-arbitrary",
                "litellm_params": {
                    "model": "hosted_vllm/" + server.model,
                    "api_base": server.url_for("v1"),
                },
            }
        ],
        store=proxy_store,
    )

    process = multiprocessing.Process(target=proxy.start)
    process.start()

    class ProxyAgent(LitAgent[str]):
        async def validation_rollout_async(self, task: str, *, resources: Dict[str, LLM], rollout: Any) -> float:
            llm_resource = resources["llm"]
            client = openai.AsyncOpenAI(base_url=llm_resource.endpoint, api_key=llm_resource.api_key)
            response = await client.chat.completions.create(
                model=llm_resource.model,
                messages=[{"role": "user", "content": task}],
            )
            assert response.choices, "Proxy should return at least one choice"
            return 0.0

    try:
        agent = ProxyAgent()
        runner, store = await init_runner(agent)
        try:
            llm_resource = LLM(
                endpoint=f"http://{proxy.host}:{proxy.port}",
                model="gpt-4o-arbitrary",
                api_key="token-abc123",
            )
            await store.update_resources("proxy-resource", {"llm": llm_resource})
            await runner.step("Say hello to Agent Lightning")
        finally:
            teardown_runner(runner)

        rollouts = await store.query_rollouts()
        assert rollouts and rollouts[0].status == "succeeded"
    finally:
        process.terminate()
        await asyncio.to_thread(process.join, timeout=10)
