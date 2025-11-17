# Copyright (c) Microsoft. All rights reserved.

# pyright: reportPrivateUsage=false

from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, cast
from unittest.mock import AsyncMock, Mock

import pytest
from openai import AsyncOpenAI

import agentlightning.algorithm.apo.apo as apo_module
from agentlightning.adapter import TraceAdapter
from agentlightning.adapter.messages import TraceToMessages
from agentlightning.algorithm.apo.apo import APO, RolloutResultForAPO, VersionedPromptTemplate, batch_iter_over_dataset
from agentlightning.types import (
    Dataset,
    NamedResources,
)
from agentlightning.types import OtelResource as SpanResource
from agentlightning.types import (
    PromptTemplate,
    Rollout,
    Span,
    SpanContext,
    SpanNames,
    TraceStatus,
)


class DummyTraceMessagesAdapter(TraceToMessages):
    def __init__(self) -> None:
        super().__init__()
        self.seen_spans: Sequence[Span] | None = None

    def adapt(self, source: Sequence[Span], /) -> List[Dict[str, Any]]:  # type: ignore[override]
        self.seen_spans = list(source)
        return [dict(payload="converted")]


class WrongAdapter(TraceAdapter[List[int]]):
    def adapt(self, source: Sequence[Span], /) -> List[int]:
        return [len(source)]


class DummyStore:
    def __init__(self) -> None:
        self.update_resources_calls: List[Tuple[str, NamedResources]] = []
        self.enqueue_calls: List[Dict[str, Any]] = []
        self.wait_calls: List[Dict[str, Any]] = []
        self.wait_results_queue: List[List[Rollout]] = []
        self.query_spans_map: Dict[str, List[Span]] = {}
        self._counter = 0

    async def update_resources(self, resources_id: str, resources: NamedResources) -> Mock:
        self.update_resources_calls.append((resources_id, resources))
        update_mock = Mock()
        update_mock.resources_id = resources_id
        return update_mock

    async def enqueue_rollout(
        self,
        *,
        input: Dict[str, Any],
        mode: str,
        resources_id: Optional[str] = None,
    ) -> Mock:
        rollout_id = f"rollout-{self._counter}"
        self._counter += 1
        self.enqueue_calls.append(
            {"rollout_id": rollout_id, "input": input, "mode": mode, "resources_id": resources_id}
        )
        result = Mock()
        result.rollout_id = rollout_id
        return result

    async def wait_for_rollouts(self, rollout_ids: Sequence[str], timeout: float) -> List[Rollout]:
        self.wait_calls.append({"rollout_ids": tuple(rollout_ids), "timeout": timeout})
        if self.wait_results_queue:
            return self.wait_results_queue.pop(0)
        return []

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
        **_: Any,
    ) -> List[Span]:
        return list(self.query_spans_map.get(rollout_id, []))


def make_completion(content: str | None) -> Mock:
    """Create a mock OpenAI completion response."""
    message_mock = Mock()
    message_mock.content = content
    choice_mock = Mock()
    choice_mock.message = message_mock
    completion_mock = Mock()
    completion_mock.choices = [choice_mock]
    return completion_mock


def make_openai_client(create_mock: AsyncMock) -> Mock:
    """Create a mock AsyncOpenAI client with the given create method."""
    client = Mock(spec=AsyncOpenAI)
    completions = Mock()
    completions.create = create_mock
    chat = Mock()
    chat.completions = completions
    client.chat = chat
    return client


def make_reward_span(rollout_id: str, attempt_id: str, reward: float, sequence_id: int) -> Span:
    hex_id = f"{sequence_id:032x}"
    span_hex = f"{sequence_id:016x}"
    return Span(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=sequence_id,
        trace_id=hex_id,
        span_id=span_hex,
        parent_id=None,
        name=SpanNames.REWARD.value,
        status=TraceStatus(status_code="OK"),
        attributes={"reward": reward},
        events=[],
        links=[],
        start_time=None,
        end_time=None,
        context=SpanContext(trace_id=hex_id, span_id=span_hex, is_remote=False, trace_state={}),
        parent=None,
        resource=SpanResource(attributes={}, schema_url=""),
    )


def test_batch_iter_over_dataset_returns_full_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = [{"id": idx} for idx in range(3)]
    monkeypatch.setattr(apo_module.random, "shuffle", lambda seq: None)  # type: ignore

    iterator = batch_iter_over_dataset(cast(Dataset[Any], dataset), batch_size=5)

    first_batch = next(iterator)
    second_batch = next(iterator)

    assert len(first_batch) == len(dataset)
    assert len(second_batch) == len(dataset)
    assert {item["id"] for item in first_batch} == {0, 1, 2}


def test_batch_iter_over_dataset_cycles_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = [{"id": idx} for idx in range(4)]

    def fake_shuffle(seq: List[int]) -> None:
        seq.reverse()

    monkeypatch.setattr(apo_module.random, "shuffle", fake_shuffle)

    iterator = batch_iter_over_dataset(cast(Dataset[Any], dataset), batch_size=2)

    batch_one = next(iterator)
    batch_two = next(iterator)
    batch_three = next(iterator)

    assert len(batch_one) == 2
    assert len(batch_two) == 2
    assert {item["id"] for item in batch_three}.issubset({item["id"] for item in batch_one + batch_two})  # type: ignore


def test_apo_init_sets_configuration() -> None:
    client = Mock(spec=AsyncOpenAI)

    apo = APO[Any](
        client,
        gradient_model="g-model",
        apply_edit_model="a-model",
        diversity_temperature=0.7,
        gradient_batch_size=3,
        val_batch_size=5,
        beam_width=2,
        branch_factor=3,
        beam_rounds=4,
        rollout_batch_timeout=42.0,
        run_initial_validation=False,
    )

    assert apo.async_openai_client is client
    assert apo.gradient_model == "g-model"
    assert apo.apply_edit_model == "a-model"
    assert apo.diversity_temperature == 0.7
    assert apo.gradient_batch_size == 3
    assert apo.val_batch_size == 5
    assert apo.beam_width == 2
    assert apo.branch_factor == 3
    assert apo.beam_rounds == 4
    assert apo.rollout_batch_timeout == 42.0
    assert apo.run_initial_validation is False
    assert apo._history_best_prompt is None
    assert apo._history_best_score == float("-inf")


def test_get_seed_prompt_template_returns_prompt() -> None:
    client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](client)
    prompt = PromptTemplate(template="Seed: {x}", engine="f-string")
    resources: NamedResources = {
        "seed": prompt,
        "other": PromptTemplate(template="Other", engine="f-string"),
    }
    apo.set_initial_resources(resources)

    resource_name, seed_prompt = apo.get_seed_prompt_template()

    assert resource_name == "seed"
    assert seed_prompt is prompt


def test_get_seed_prompt_template_requires_resources() -> None:
    client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](client)

    with pytest.raises(ValueError):
        apo.get_seed_prompt_template()


def test_get_seed_prompt_template_requires_prompt_resource() -> None:
    client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](client)
    apo.set_initial_resources({})

    with pytest.raises(ValueError):
        apo.get_seed_prompt_template()


def test_get_adapter_returns_trace_messages_adapter() -> None:
    client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](client)
    adapter = DummyTraceMessagesAdapter()
    apo.set_adapter(adapter)

    assert apo.get_adapter() is adapter


def test_get_adapter_requires_trace_messages_adapter() -> None:
    client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](client)
    apo.set_adapter(WrongAdapter())

    with pytest.raises(ValueError):
        apo.get_adapter()


def test_get_best_prompt_requires_history() -> None:
    apo = APO[Any](Mock(spec=AsyncOpenAI))

    with pytest.raises(ValueError):
        apo.get_best_prompt()


def test_get_best_prompt_returns_prompt() -> None:
    apo = APO[Any](Mock(spec=AsyncOpenAI))
    prompt = PromptTemplate(template="Best", engine="f-string")
    apo._history_best_prompt = prompt

    assert apo.get_best_prompt() is prompt


@pytest.mark.asyncio
async def test_compute_textual_gradient_samples_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = AsyncMock(return_value=make_completion("critique"))
    client = make_openai_client(create_mock)
    apo = APO[Any](client, gradient_model="test-gradient-model", gradient_batch_size=2, diversity_temperature=0.8)
    versioned_prompt = apo._create_versioned_prompt(PromptTemplate(template="prompt", engine="f-string"))
    rollouts: List[RolloutResultForAPO] = [
        RolloutResultForAPO(status="succeeded", final_reward=float(i), spans=[], messages=[]) for i in range(3)
    ]

    sample_mock = Mock(return_value=rollouts[:2])
    monkeypatch.setattr(apo_module.random, "sample", sample_mock)
    monkeypatch.setattr(apo_module.random, "choice", lambda seq: seq[0])  # type: ignore

    result = await apo.compute_textual_gradient(versioned_prompt, rollouts)

    assert result == "critique"
    sample_mock.assert_called_once_with(rollouts, 2)
    # Verify OpenAI call was made with correct parameters
    create_mock.assert_awaited_once()
    call_kwargs = create_mock.await_args.kwargs  # type: ignore
    assert call_kwargs["model"] == "test-gradient-model"
    assert call_kwargs["temperature"] == 0.8
    assert len(call_kwargs["messages"]) == 1
    assert call_kwargs["messages"][0]["role"] == "user"
    assert call_kwargs["messages"][0]["content"].startswith("You optimize a prompt template.")


@pytest.mark.asyncio
async def test_compute_textual_gradient_uses_all_rollouts_when_insufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = AsyncMock(return_value=make_completion("critique"))
    client = make_openai_client(create_mock)
    apo = APO[Any](client, gradient_batch_size=3)
    versioned_prompt = apo._create_versioned_prompt(PromptTemplate(template="prompt", engine="f-string"))
    rollouts: List[RolloutResultForAPO] = [
        RolloutResultForAPO(status="succeeded", final_reward=1.0, spans=[], messages=[])
    ]

    sample_mock = Mock(side_effect=AssertionError("sample should not be called"))
    monkeypatch.setattr(apo_module.random, "sample", sample_mock)
    monkeypatch.setattr(apo_module.random, "choice", lambda seq: seq[0])  # type: ignore

    result = await apo.compute_textual_gradient(versioned_prompt, rollouts)

    assert result == "critique"


@pytest.mark.asyncio
async def test_textual_gradient_and_apply_edit_returns_new_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use two separate mocks for gradient and edit calls
    gradient_mock = AsyncMock(return_value=make_completion("critique text"))
    edit_mock = AsyncMock(return_value=make_completion("new prompt"))

    call_count = 0

    async def create_side_effect(*args: Any, **kwargs: Any) -> Mock:
        nonlocal call_count
        call_count += 1
        return gradient_mock.return_value if call_count == 1 else edit_mock.return_value

    create_mock = AsyncMock(side_effect=create_side_effect)
    client = make_openai_client(create_mock)
    apo = APO[Any](client, gradient_model="grad-model", apply_edit_model="edit-model", diversity_temperature=0.9)

    monkeypatch.setattr(apo_module.random, "choice", lambda seq: seq[0])  # type: ignore
    monkeypatch.setattr(apo_module.random, "sample", lambda population, k: list(population)[:k])  # type: ignore

    poml_calls: List[Dict[str, Any]] = []

    def poml_side_effect(template: Any, context: Dict[str, Any], format: str) -> Dict[str, Any]:
        poml_calls.append({"template": template, "context": context, "format": format})
        return {"messages": [{"role": "user", "content": "msg"}]}

    monkeypatch.setattr(apo_module.poml, "poml", poml_side_effect)

    versioned_prompt = apo._create_versioned_prompt(PromptTemplate(template="old prompt", engine="f-string"))
    rollouts: List[RolloutResultForAPO] = [
        RolloutResultForAPO(status="succeeded", final_reward=1.0, spans=[], messages=[])
    ]

    result = await apo.textual_gradient_and_apply_edit(versioned_prompt, rollouts)

    assert result == "new prompt"
    assert create_mock.await_count == 2

    # Verify gradient computation call
    first_call = create_mock.await_args_list[0].kwargs
    assert first_call["model"] == "grad-model"
    assert first_call["temperature"] == 0.9

    # Verify edit application call
    second_call = create_mock.await_args_list[1].kwargs
    assert second_call["model"] == "edit-model"
    assert second_call["temperature"] == 0.9

    # Verify critique was passed to edit context
    assert len(poml_calls) == 2
    assert poml_calls[1]["context"]["critique"] == "critique text"
    assert poml_calls[1]["context"]["prompt_template"] == "old prompt"


@pytest.mark.asyncio
async def test_textual_gradient_and_apply_edit_returns_original_if_no_critique(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock OpenAI to return None content
    create_mock = AsyncMock(return_value=make_completion(None))
    client = make_openai_client(create_mock)
    apo = APO[Any](client)

    monkeypatch.setattr(apo_module.random, "choice", lambda seq: seq[0])  # type: ignore
    monkeypatch.setattr(apo_module.random, "sample", lambda population, k: list(population)[:k])  # type: ignore

    versioned_prompt = apo._create_versioned_prompt(PromptTemplate(template="old prompt", engine="f-string"))
    rollouts: List[RolloutResultForAPO] = [
        RolloutResultForAPO(status="succeeded", final_reward=1.0, spans=[], messages=[])
    ]

    result = await apo.textual_gradient_and_apply_edit(versioned_prompt, rollouts)

    # Should return original prompt when gradient computation fails
    assert result == "old prompt"
    # Verify gradient computation was attempted
    create_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_rollout_results_adapts_spans() -> None:
    apo = APO[Any](Mock(spec=AsyncOpenAI))
    store = DummyStore()
    adapter = DummyTraceMessagesAdapter()
    apo.set_store(store)  # type: ignore
    apo.set_adapter(adapter)

    rollout = Rollout(
        rollout_id="r-1",
        input={"task": "value"},
        start_time=0.0,
        status="succeeded",
        mode="train",
    )
    span1 = make_reward_span("r-1", "attempt", 1.0, sequence_id=1)
    span2 = make_reward_span("r-1", "attempt", 2.0, sequence_id=2)
    store.query_spans_map["r-1"] = [span1, span2]

    results = await apo.get_rollout_results([rollout])

    assert len(results) == 1
    # Verify final reward is correctly extracted
    assert results[0]["final_reward"] == 2.0
    # Verify status is correctly mapped
    assert results[0]["status"] == "succeeded"
    # Verify adapter was called with correct spans
    assert adapter.seen_spans is not None
    assert len(adapter.seen_spans) == 2
    assert adapter.seen_spans[0] == span1
    assert adapter.seen_spans[1] == span2
    # Verify messages were converted
    assert results[0]["messages"] == [{"payload": "converted"}]
    # Verify spans were serialized
    assert len(results[0]["spans"]) == 2
    assert results[0]["spans"][0]["rollout_id"] == "r-1"
    assert results[0]["spans"][0]["name"] == SpanNames.REWARD.value
    assert results[0]["spans"][0]["attributes"]["reward"] == 1.0
    assert results[0]["spans"][1]["attributes"]["reward"] == 2.0


@pytest.mark.asyncio
async def test_evaluate_prompt_on_batch_runs_rollouts() -> None:
    client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](client, rollout_batch_timeout=100.0)
    store = DummyStore()
    adapter = DummyTraceMessagesAdapter()
    apo.set_store(store)  # type: ignore
    apo.set_adapter(adapter)

    dataset = [{"task": 1}, {"task": 2}]

    # Set up spans for rollouts
    store.query_spans_map["rollout-0"] = [make_reward_span("rollout-0", "attempt", 1.0, sequence_id=1)]
    store.query_spans_map["rollout-1"] = [make_reward_span("rollout-1", "attempt", 0.0, sequence_id=1)]

    store.wait_results_queue.append(
        [
            Rollout(
                rollout_id="rollout-0",
                input=dataset[0],
                start_time=0.0,
                status="succeeded",
                mode="train",
            ),
            Rollout(
                rollout_id="rollout-1",
                input=dataset[1],
                start_time=0.0,
                status="failed",
                mode="train",
            ),
        ]
    )

    prompt_template = PromptTemplate(template="test prompt", engine="f-string")
    versioned_prompt = apo._create_versioned_prompt(prompt_template)

    rollout_results, average = await apo.evaluate_prompt_on_batch(versioned_prompt, "seed", dataset, mode="train")

    # Verify results
    assert len(rollout_results) == 2
    assert rollout_results[0]["final_reward"] == 1.0
    assert rollout_results[0]["status"] == "succeeded"
    assert rollout_results[1]["final_reward"] == 0.0
    assert rollout_results[1]["status"] == "failed"
    assert average == pytest.approx(0.5)  # type: ignore

    # Verify resource was added with correct prompt
    assert len(store.update_resources_calls) == 1
    resources_id, resources_payload = store.update_resources_calls[0]
    assert resources_id == versioned_prompt.version
    assert "seed" in resources_payload
    added_resource = resources_payload["seed"]
    assert isinstance(added_resource, PromptTemplate)
    assert added_resource.template == "test prompt"
    assert added_resource.engine == "f-string"

    # Verify enqueue was called correctly
    assert len(store.enqueue_calls) == 2
    assert store.enqueue_calls[0]["input"] == dataset[0]
    assert store.enqueue_calls[0]["mode"] == "train"
    assert store.enqueue_calls[0]["resources_id"] == versioned_prompt.version
    assert store.enqueue_calls[1]["input"] == dataset[1]
    assert store.enqueue_calls[1]["mode"] == "train"
    assert store.enqueue_calls[1]["resources_id"] == versioned_prompt.version

    # Verify wait was called with correct rollout IDs
    assert len(store.wait_calls) == 1
    assert set(store.wait_calls[0]["rollout_ids"]) == {"rollout-0", "rollout-1"}
    assert store.wait_calls[0]["timeout"] == 0.0


def test_initialize_beam_sets_history(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](client, gradient_batch_size=2, val_batch_size=1)
    prompt = PromptTemplate(template="Seed", engine="f-string")
    apo.set_initial_resources({"seed": prompt})
    monkeypatch.setattr(apo_module.random, "shuffle", lambda seq: None)  # type: ignore

    train_dataset: Sequence[Dict[str, str]] = [{"x": "1"}, {"x": "2"}]
    val_dataset: Sequence[Dict[str, str]] = [{"y": "value"}]

    resource_name, seed_prompt, grad_iter, val_iter = apo._initialize_beam(train_dataset, val_dataset)  # type: ignore

    assert resource_name == "seed"
    assert seed_prompt is prompt
    assert apo._history_best_prompt is prompt
    assert apo._history_best_score == float("-inf")
    assert len(next(grad_iter)) == len(train_dataset)
    assert len(next(val_iter)) == len(val_dataset)


def test_initialize_beam_requires_train_dataset() -> None:
    apo = APO[Any](Mock(spec=AsyncOpenAI))
    apo.set_initial_resources({"seed": PromptTemplate(template="Seed", engine="f-string")})

    with pytest.raises(ValueError):
        apo._initialize_beam(None, [])  # type: ignore


def test_initialize_beam_requires_val_dataset() -> None:
    apo = APO[Any](Mock(spec=AsyncOpenAI))
    apo.set_initial_resources({"seed": PromptTemplate(template="Seed", engine="f-string")})

    with pytest.raises(ValueError):
        apo._initialize_beam([], None)  # type: ignore


def test_sample_parent_prompts_replicates_when_beam_too_small(monkeypatch: pytest.MonkeyPatch) -> None:
    apo = APO[Any](Mock(spec=AsyncOpenAI), beam_width=3)
    beam_prompt = apo._create_versioned_prompt(PromptTemplate(template="Seed", engine="f-string"))
    beam = [beam_prompt]
    monkeypatch.setattr(apo_module.random, "sample", lambda population, k: (_ for _ in ()).throw(AssertionError()))  # type: ignore

    sampled = apo._sample_parent_prompts(beam, round_num=0)

    assert len(sampled) == apo.beam_width
    assert all(index == 0 and prompt is beam_prompt for index, prompt in sampled)


def test_sample_parent_prompts_uses_random_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    apo = APO[Any](Mock(spec=AsyncOpenAI), beam_width=2)
    prompt_a = apo._create_versioned_prompt(PromptTemplate(template="A", engine="f-string"))
    prompt_b = apo._create_versioned_prompt(PromptTemplate(template="B", engine="f-string"))
    prompt_c = apo._create_versioned_prompt(PromptTemplate(template="C", engine="f-string"))

    monkeypatch.setattr(apo_module.random, "sample", lambda population, k: [0, 2])  # type: ignore

    sampled = apo._sample_parent_prompts([prompt_a, prompt_b, prompt_c], round_num=1)

    assert sampled == [(0, prompt_a), (2, prompt_c)]


@pytest.mark.asyncio
async def test_generate_candidate_prompts_creates_branch_factor_children() -> None:
    client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](client, branch_factor=2)
    store = DummyStore()
    adapter = DummyTraceMessagesAdapter()
    apo.set_store(store)  # type: ignore
    apo.set_adapter(adapter)

    parent_prompt = apo._create_versioned_prompt(PromptTemplate(template="Seed", engine="f-string"))
    grad_batches: Iterator[Sequence[Dict[str, Any]]] = iter(
        [
            [{"task": "a"}],
            [{"task": "b"}],
        ]
    )

    # Set up rollouts to complete immediately
    store.query_spans_map["rollout-0"] = [make_reward_span("rollout-0", "attempt", 0.5, sequence_id=1)]
    store.query_spans_map["rollout-1"] = [make_reward_span("rollout-1", "attempt", 0.6, sequence_id=1)]
    store.wait_results_queue.extend(
        [
            [Rollout(rollout_id="rollout-0", input={"task": "a"}, start_time=0.0, status="succeeded", mode="train")],
            [Rollout(rollout_id="rollout-1", input={"task": "b"}, start_time=0.0, status="succeeded", mode="train")],
        ]
    )

    counter = 0

    async def edit_side_effect(
        current_prompt: VersionedPromptTemplate,
        rollout: List[RolloutResultForAPO],
        **_: Any,
    ) -> str:
        nonlocal counter
        counter += 1
        return f"{current_prompt.prompt_template.template}-{counter}"

    apo.textual_gradient_and_apply_edit = AsyncMock(side_effect=edit_side_effect)

    candidates = await apo._generate_candidate_prompts([(0, parent_prompt)], "seed", grad_batches, round_num=0)

    # Verify correct number of candidates generated
    assert len(candidates) == apo.branch_factor
    assert {candidate.prompt_template.template for candidate in candidates} == {"Seed-1", "Seed-2"}
    assert all(candidate.prompt_template.engine == "f-string" for candidate in candidates)

    # Verify evaluate_prompt_on_batch was called for each candidate generation
    assert len(store.enqueue_calls) == 2
    assert store.enqueue_calls[0]["input"] == {"task": "a"}
    assert store.enqueue_calls[1]["input"] == {"task": "b"}
    assert all(call["mode"] == "train" for call in store.enqueue_calls)
    assert all(call["resources_id"] == parent_prompt.version for call in store.enqueue_calls)

    # Verify textual_gradient_and_apply_edit was called correct number of times
    assert apo.textual_gradient_and_apply_edit.await_count == 2


@pytest.mark.asyncio
async def test_generate_candidate_prompts_skips_failed_generations() -> None:
    """Test that None returns from textual_gradient_and_apply_edit are skipped."""
    client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](client, branch_factor=3)
    store = DummyStore()
    # Keep strong reference to prevent garbage collection since APO uses weakref
    apo._test_adapter = adapter = DummyTraceMessagesAdapter()  # type: ignore
    apo.set_store(store)  # type: ignore
    apo.set_adapter(adapter)

    parent_prompt = apo._create_versioned_prompt(PromptTemplate(template="Seed", engine="f-string"))
    grad_batches: Iterator[Sequence[Dict[str, Any]]] = iter([[{"task": f"t{i}"}] for i in range(3)])

    # Set up rollouts
    for i in range(3):
        store.query_spans_map[f"rollout-{i}"] = [make_reward_span(f"rollout-{i}", "attempt", 0.5, sequence_id=1)]
        store.wait_results_queue.append(
            [
                Rollout(
                    rollout_id=f"rollout-{i}", input={"task": f"t{i}"}, start_time=0.0, status="succeeded", mode="train"
                )
            ]
        )

    # Mock to return None for second call, valid prompts for others
    call_count = 0

    async def edit_side_effect(
        current_prompt: VersionedPromptTemplate,
        rollout: List[RolloutResultForAPO],
        **_: Any,
    ) -> Optional[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return None  # Simulate failure
        return f"{current_prompt.prompt_template.template}-{call_count}"

    apo.textual_gradient_and_apply_edit = AsyncMock(side_effect=edit_side_effect)

    candidates = await apo._generate_candidate_prompts([(0, parent_prompt)], "seed", grad_batches, round_num=0)

    # Should only have 2 candidates (one failed)
    assert len(candidates) == 2
    assert {candidate.prompt_template.template for candidate in candidates} == {"Seed-1", "Seed-3"}
    # Verify all three attempts were made
    assert apo.textual_gradient_and_apply_edit.await_count == 3


@pytest.mark.asyncio
async def test_evaluate_and_select_beam_sorts_by_score() -> None:
    apo = APO[Any](Mock(spec=AsyncOpenAI), beam_width=2)
    candidates = [
        apo._create_versioned_prompt(PromptTemplate(template="A", engine="f-string")),
        apo._create_versioned_prompt(PromptTemplate(template="B", engine="f-string")),
        apo._create_versioned_prompt(PromptTemplate(template="C", engine="f-string")),
    ]
    scores = {"A": 1.0, "B": 0.2, "C": 2.0}

    async def evaluate(
        prompt: VersionedPromptTemplate,
        resource_name: str,
        dataset: Sequence[Dict[str, Any]],
        mode: str,
        **_: Any,
    ) -> Any:
        return [], scores[prompt.prompt_template.template]

    apo.evaluate_prompt_on_batch = AsyncMock(side_effect=evaluate)  # type: ignore[assignment]

    val_iterator: Iterator[Sequence[Dict[str, Any]]] = iter([[{"task": "val"}]])

    selected = await apo._evaluate_and_select_beam(candidates, "seed", val_iterator, round_num=0)

    assert [prompt.prompt_template.template for prompt in selected] == ["C", "A"]


@pytest.mark.asyncio
async def test_evaluate_and_select_beam_raises_on_empty_candidates() -> None:
    """Test that ValueError is raised when no candidates remain after evaluation."""
    client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](client, beam_width=2)
    # Empty candidate list
    candidates: List[VersionedPromptTemplate] = []

    val_iterator: Iterator[Sequence[Dict[str, Any]]] = iter([[{"task": "val"}]])

    with pytest.raises(ValueError, match="No beam candidates any more"):
        await apo._evaluate_and_select_beam(candidates, "seed", val_iterator, round_num=0)


@pytest.mark.asyncio
async def test_update_best_prompt_updates_history() -> None:
    apo = APO[Any](Mock(spec=AsyncOpenAI))
    old_versioned = apo._create_versioned_prompt(PromptTemplate(template="Old", engine="f-string"))
    new_versioned = apo._create_versioned_prompt(PromptTemplate(template="New", engine="f-string"))
    apo._history_best_prompt = old_versioned.prompt_template
    apo._history_best_score = 0.5
    apo._history_best_version = old_versioned.version
    apo.evaluate_prompt_on_batch = AsyncMock(return_value=([], 1.2))  # type: ignore[assignment]

    await apo._update_best_prompt([new_versioned], "seed", [{"task": "val"}], round_num=0)  # type: ignore

    assert apo._history_best_prompt is new_versioned.prompt_template
    assert apo._history_best_score == pytest.approx(1.2)  # type: ignore
    assert apo._history_best_version == new_versioned.version


@pytest.mark.asyncio
async def test_update_best_prompt_keeps_history_when_not_improved() -> None:
    apo = APO[Any](Mock(spec=AsyncOpenAI))
    old_versioned = apo._create_versioned_prompt(PromptTemplate(template="Old", engine="f-string"))
    new_versioned = apo._create_versioned_prompt(PromptTemplate(template="New", engine="f-string"))
    apo._history_best_prompt = old_versioned.prompt_template
    apo._history_best_score = 2.0
    apo._history_best_version = old_versioned.version
    apo.evaluate_prompt_on_batch = AsyncMock(return_value=([], 1.5))  # type: ignore[assignment]

    await apo._update_best_prompt([new_versioned], "seed", [{"task": "val"}], round_num=0)  # type: ignore

    assert apo._history_best_prompt is old_versioned.prompt_template
    assert apo._history_best_score == pytest.approx(2.0)  # type: ignore
    assert apo._history_best_version == old_versioned.version


def test_apo_init_defaults_run_initial_validation_to_true() -> None:
    """Test that run_initial_validation defaults to True when not specified."""
    client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](client)

    assert apo.run_initial_validation is True


@pytest.mark.asyncio
async def test_run_performs_initial_validation_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that initial validation runs on seed prompt when run_initial_validation=True."""
    async_client = Mock(spec=AsyncOpenAI)
    apo = APO[Any](
        async_client,
        gradient_batch_size=1,
        val_batch_size=1,
        beam_width=1,
        branch_factor=1,
        beam_rounds=0,  # No optimization rounds, just initial validation
        run_initial_validation=True,
    )
    seed_prompt = PromptTemplate(template="Seed", engine="f-string")
    apo.set_initial_resources({"seed": seed_prompt})

    store = DummyStore()
    apo._test_adapter = adapter = DummyTraceMessagesAdapter()  # type: ignore
    apo.set_store(store)  # type: ignore
    apo.set_adapter(adapter)

    # Set up initial validation rollout
    store.query_spans_map["rollout-0"] = [make_reward_span("rollout-0", "attempt", 0.75, sequence_id=1)]
    store.wait_results_queue.append(
        [
            Rollout(
                rollout_id="rollout-0",
                input={"task": "val"},
                start_time=0.0,
                status="succeeded",
                mode="val",
            )
        ]
    )

    monkeypatch.setattr(apo_module.random, "shuffle", lambda seq: None)  # type: ignore

    val_dataset = [{"task": "val"}]
    await apo.run(train_dataset=[{"task": "train"}], val_dataset=val_dataset)  # type: ignore

    # Verify initial validation was performed
    assert apo._history_best_prompt is seed_prompt
    assert apo._history_best_score == pytest.approx(0.75)  # type: ignore

    # Verify a validation rollout was enqueued for initial validation
    val_calls = [c for c in store.enqueue_calls if c["mode"] == "val"]
    assert len(val_calls) == 1


@pytest.mark.asyncio
async def test_run_skips_initial_validation_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that initial validation is skipped when run_initial_validation=False."""
    create_mock = AsyncMock(side_effect=[make_completion("critique text"), make_completion("improved prompt")])
    async_client = make_openai_client(create_mock)

    apo = APO[Any](
        async_client,
        gradient_batch_size=1,
        val_batch_size=1,
        beam_width=1,
        branch_factor=1,
        beam_rounds=1,
        run_initial_validation=False,  # Disable initial validation
    )
    seed_prompt = PromptTemplate(template="Seed", engine="f-string")
    apo.set_initial_resources({"seed": seed_prompt})

    store = DummyStore()
    apo._test_adapter = adapter = DummyTraceMessagesAdapter()  # type: ignore
    apo.set_store(store)  # type: ignore
    apo.set_adapter(adapter)

    # Set up spans for rollouts (train + val for candidates + final val)
    rollout_rewards = [0.4, 0.5, 0.6, 1.1]
    for i, reward in enumerate(rollout_rewards):
        store.query_spans_map[f"rollout-{i}"] = [make_reward_span(f"rollout-{i}", "attempt", reward, sequence_id=1)]
        store.wait_results_queue.append(
            [
                Rollout(
                    rollout_id=f"rollout-{i}",
                    input={"task": f"data-{i}"},
                    start_time=0.0,
                    status="succeeded",
                    mode="train" if i == 0 else "val",
                )
            ]
        )

    monkeypatch.setattr(apo_module.random, "shuffle", lambda seq: None)  # type: ignore
    monkeypatch.setattr(apo_module.random, "sample", lambda population, k: list(population)[:k])  # type: ignore
    monkeypatch.setattr(apo_module.random, "choice", lambda seq: seq[0])  # type: ignore

    train_dataset = [{"task": "train"}]
    val_dataset = [{"task": "val"}]

    await apo.run(train_dataset=train_dataset, val_dataset=val_dataset)  # type: ignore

    # Verify best prompt was updated through normal optimization (not initial validation)
    best_prompt = apo.get_best_prompt()
    assert best_prompt.template == "improved prompt"

    # Count validation rollouts - should NOT include initial validation
    # Only candidate evaluation + final best prompt evaluation
    val_calls = [c for c in store.enqueue_calls if c["mode"] == "val"]
    # With run_initial_validation=False, we expect: 2 val calls (seed+new candidate) + 1 final val = 3 total
    assert len(val_calls) == 3


@pytest.mark.asyncio
async def test_run_updates_best_prompt_with_real_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration test for the full run method with minimal mocking."""
    create_mock = AsyncMock(side_effect=[make_completion("critique text"), make_completion("improved prompt")])
    async_client = make_openai_client(create_mock)

    apo = APO[Any](
        async_client,
        gradient_batch_size=1,
        val_batch_size=1,
        beam_width=1,
        branch_factor=1,
        beam_rounds=1,
        run_initial_validation=False,  # Skip initial validation for this test
    )
    seed_prompt = PromptTemplate(template="Seed", engine="f-string")
    apo.set_initial_resources({"seed": seed_prompt})

    store = DummyStore()
    # Keep strong reference to prevent garbage collection since APO uses weakref
    apo._test_adapter = adapter = DummyTraceMessagesAdapter()  # type: ignore
    apo.set_store(store)  # type: ignore
    apo.set_adapter(adapter)

    # Set up spans for all expected rollouts
    # For 1 round with beam_width=1, branch_factor=1, run_initial_validation=False, we expect:
    # 1. Training rollout for gradient computation
    # 2. Validation rollouts for candidate evaluation (seed + new candidate = 2)
    # 3. Final validation rollout on full dataset for best prompt
    rollout_rewards = [0.4, 0.5, 0.6, 1.1]
    for i, reward in enumerate(rollout_rewards):
        store.query_spans_map[f"rollout-{i}"] = [make_reward_span(f"rollout-{i}", "attempt", reward, sequence_id=1)]
        store.wait_results_queue.append(
            [
                Rollout(
                    rollout_id=f"rollout-{i}",
                    input={"task": f"data-{i}"},
                    start_time=0.0,
                    status="succeeded",
                    mode="train" if i == 0 else "val",
                )
            ]
        )

    monkeypatch.setattr(apo_module.random, "shuffle", lambda seq: None)  # type: ignore
    monkeypatch.setattr(apo_module.random, "sample", lambda population, k: list(population)[:k])  # type: ignore
    monkeypatch.setattr(apo_module.random, "choice", lambda seq: seq[0])  # type: ignore

    train_dataset = [{"task": "train"}]
    val_dataset = [{"task": "val"}]

    await apo.run(train_dataset=train_dataset, val_dataset=val_dataset)  # type: ignore

    # Verify best prompt was updated
    best_prompt = apo.get_best_prompt()
    assert best_prompt.template == "improved prompt"

    # Verify OpenAI was called twice (gradient + edit)
    assert create_mock.await_count == 2
    gradient_call = create_mock.await_args_list[0]
    assert gradient_call.kwargs["model"] == apo.gradient_model
    edit_call = create_mock.await_args_list[1]
    assert edit_call.kwargs["model"] == apo.apply_edit_model

    # Verify resources were updated (seed prompt + new candidate prompts)
    assert len(store.update_resources_calls) >= 2
    assert all(isinstance(entry[0], str) for entry in store.update_resources_calls)

    # Verify rollouts were enqueued (1 train + multiple val)
    assert len(store.enqueue_calls) >= 3
    train_calls = [c for c in store.enqueue_calls if c["mode"] == "train"]
    val_calls = [c for c in store.enqueue_calls if c["mode"] == "val"]
    assert len(train_calls) == 1
    assert len(val_calls) >= 2

    # Verify history was updated correctly
    assert apo._history_best_prompt is not None
    assert apo._history_best_score > 0
