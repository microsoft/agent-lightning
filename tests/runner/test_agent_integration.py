import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

import litellm
import openai
import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from agentlightning.litagent import LitAgent
from agentlightning.reward import emit_reward
from agentlightning.runner import AgentRunnerV2
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.base import BaseTracer
from agentlightning.types.core import LLM

trace_api.set_tracer_provider(TracerProvider())


class SimpleTracer(BaseTracer):
    def __init__(self) -> None:
        super().__init__()
        self._last_trace: list[Any] = []

    def init(self, *args: Any, **kwargs: Any) -> None:
        self._last_trace.clear()

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        self._last_trace.clear()

    def get_last_trace(self) -> list[Any]:
        return list(self._last_trace)

    @contextmanager
    def trace_context(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[InMemoryLightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> None:
        self._last_trace = []
        yield self._last_trace


async def init_runner(
    agent: LitAgent[Any],
    *,
    tracer: Optional[BaseTracer] = None,
    resources: Optional[Dict[str, LLM]] = None,
) -> tuple[AgentRunnerV2[Any], InMemoryLightningStore]:
    tracer = tracer or SimpleTracer()
    store = InMemoryLightningStore()
    llm_resource = resources or {"llm": LLM(endpoint="http://localhost", model="dummy")}
    await store.update_resources("default", llm_resource)

    runner = AgentRunnerV2[Any](tracer=tracer, poll_interval=0.01)
    runner.init(agent)
    runner.init_worker(worker_id=0, store=store)
    return runner, store


def teardown_runner(runner: AgentRunnerV2[Any]) -> None:
    runner.teardown_worker(worker_id=0)
    runner.teardown()


@pytest.mark.asyncio
async def test_runner_integration_basic_rollout() -> None:
    class EchoAgent(LitAgent[str]):
        async def validation_rollout_async(self, task: str, *, resources: Dict[str, Any], rollout: Any) -> list[Any]:
            return [emit_reward(1.0)]

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


def _has_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.getenv("OPENAI_BASE_URL") and os.getenv("OPENAI_API_KEY")),
    reason="LiteLLM proxy URL or GPU not available",
)
async def test_runner_integration_with_litellm_proxy() -> None:
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
