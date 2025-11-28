# Copyright (c) Microsoft. All rights reserved.

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, cast

import openai
import pytest

from agentlightning.litagent import LitAgent
from agentlightning.llm_proxy import LLMProxy
from agentlightning.reward import emit_reward
from agentlightning.runner import LitAgentRunner
from agentlightning.store.client_server import LightningStoreClient, LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.types import LLM, AttemptedRollout, NamedResources, Rollout

from ..common.network import get_free_port
from ..common.tracer import clear_tracer_provider
from ..common.vllm import VLLM_AVAILABLE, RemoteOpenAIServer


async def init_runner(
    agent: LitAgent[Any],
    *,
    resources: Optional[Dict[str, LLM]] = None,
) -> tuple[LitAgentRunner[Any], InMemoryLightningStore]:
    store = InMemoryLightningStore()
    llm_resource: NamedResources = resources or {"llm": LLM(endpoint="http://localhost", model="dummy")}  # type: ignore[assignment]
    await store.update_resources("default", llm_resource)

    runner = LitAgentRunner[Any](tracer=AgentOpsTracer(), poll_interval=0.01)
    runner.init(agent)
    runner.init_worker(worker_id=0, store=store)
    return runner, store


def teardown_runner(runner: LitAgentRunner[Any]) -> None:
    runner.teardown_worker(worker_id=0)
    runner.teardown()


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    # This must execute only once for this module.
    # Once agentops tracer is initialized, it cannot be reset,
    # otherwise it will never be rewired.
    clear_tracer_provider()
    yield


@pytest.mark.asyncio
async def test_runner_integration_basic_rollout() -> None:
    class EchoAgent(LitAgent[str]):
        async def validation_rollout_async(self, task: str, resources: Dict[str, Any], rollout: Any) -> None:
            emit_reward(1.0)

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
    assert any(span.attributes.get("agentlightning.reward.0.value") == 1.0 for span in spans)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.getenv("OPENAI_BASE_URL") and os.getenv("OPENAI_API_KEY")),
    reason="OpenAI endpoint or key not configured",
)
async def test_runner_integration_with_openai() -> None:
    class OpenAIAgent(LitAgent[str]):
        async def validation_rollout_async(self, task: str, resources: NamedResources, rollout: Rollout) -> float:
            llm = cast(LLM, resources["llm"])
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
        def validation_rollout(self, task: str, resources: NamedResources, rollout: Rollout) -> float:
            llm = cast(LLM, resources["llm"])
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


@pytest.fixture
def server():
    if not VLLM_AVAILABLE:
        pytest.skip("vLLM is not available")
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


class LLMProxyWithClearedTracerProvider(LLMProxy):
    """LLMProxy that clears the tracer provider before serving."""

    @asynccontextmanager
    async def _serve_context(self) -> AsyncGenerator[None, None]:
        # This will be run inside the LLM proxy's own process
        clear_tracer_provider()
        async with super()._serve_context():
            yield


@pytest.mark.asyncio
async def test_runner_integration_with_spawned_litellm_proxy(server: RemoteOpenAIServer) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    class ProxyAgent(LitAgent[str]):
        async def validation_rollout_async(self, task: str, resources: NamedResources, rollout: Rollout) -> float:
            attempted_rollout = cast(AttemptedRollout, rollout)
            llm_resource = cast(LLM, resources["llm"])
            client = openai.AsyncOpenAI(
                base_url=llm_resource.get_base_url(attempted_rollout.rollout_id, attempted_rollout.attempt.attempt_id),
                api_key="dummy",
            )
            response = await client.chat.completions.create(
                model=llm_resource.model,
                messages=[{"role": "user", "content": task}],
            )
            assert response.choices, "Proxy should return at least one choice"
            return 0.5

    agent = ProxyAgent()
    runner, store = await init_runner(agent)

    server_store = LightningStoreServer(store=store, host="127.0.0.1", port=get_free_port())
    await server_store.start()
    client_store = LightningStoreClient(server_store.endpoint)

    proxy = LLMProxyWithClearedTracerProvider(
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
        store=client_store,
    )

    await proxy.start()

    try:
        await runner.step("Say hello to Agent Lightning", resources={"llm": proxy.as_resource()})

        rollouts = await client_store.query_rollouts()
        assert rollouts and rollouts[0].status == "succeeded"

        spans = await client_store.query_spans(rollouts[0].rollout_id, "latest")
        assert len(spans) > 1
        first_spans = [span for span in spans if span.sequence_id == 1]
        assert len(first_spans) > 1
        assert any("llm.hosted_vllm.choices" in span.attributes for span in first_spans)
        assert any("llm.hosted_vllm.prompt_token_ids" in span.attributes for span in first_spans)
        assert any("gen_ai.prompt.0.content" in span.attributes for span in first_spans)

        second_spans = [span for span in spans if span.sequence_id == 2]
        assert len(second_spans) == 1
        assert second_spans[0].name == "openai.chat.completion"

        last_spans = [span for span in spans if span.sequence_id == max(span.sequence_id for span in spans)]
        assert len(last_spans) == 1
        assert last_spans[0].name == "agentlightning.annotation"
        assert (
            last_spans[0].attributes.get("agentlightning.reward.0.value") == 0.5
        ), f"Expected reward to be 0.5, found {last_spans[0].attributes}"
    finally:
        teardown_runner(runner)
        await proxy.stop()
        await client_store.close()
        await server_store.stop()
