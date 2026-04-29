# Copyright (c) Microsoft. All rights reserved.

"""Tests for the postprocess module adapters."""

from typing import Any, Dict, List, Optional, Sequence

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

from agentlightning.adapter.postprocess import (
    PropagateRewards,
    ToPromptCompletionAccumulations,
    ToPromptCompletionTriplets,
    ToTokensAccumulations,
    ToTokensTriplets,
    is_prefix,
)
from agentlightning.types.adapter import (
    AdaptingSequence,
    AdaptingSpan,
    AnnotatedChatCompletionCall,
    ChatCompletionCall,
    GeneralAnnotation,
    PromptCompletionAccumulation,
    PromptCompletionTriplet,
    TokenInput,
    TokenOutput,
    TokensAccumulation,
    TokensTriplet,
)
from agentlightning.types.tracer import Span

# ==============================================================================
# Helper functions to create mock objects
# ==============================================================================


def make_chat_completion(
    *,
    content: str = "Hello",
    role: str = "assistant",
    finish_reason: str = "stop",
    model: str = "gpt-4",
    completion_id: str = "chatcmpl-123",
    token_ids: Optional[Sequence[int]] = None,
    logprobs: Optional[List[float]] = None,
    provider_specific_fields: Optional[Dict[str, Any]] = None,
) -> ChatCompletion:
    """Create a mock ChatCompletion object."""
    choice_logprobs = None
    if logprobs is not None:
        choice_logprobs = ChoiceLogprobs(
            content=[
                ChatCompletionTokenLogprob(token=f"tok{i}", bytes=None, logprob=lp, top_logprobs=[])
                for i, lp in enumerate(logprobs)
            ],
            refusal=None,
        )

    message = ChatCompletionMessage(content=content, role=role, refusal=None)  # type: ignore
    choice = Choice(
        finish_reason=finish_reason,  # type: ignore
        index=0,
        message=message,
        logprobs=choice_logprobs,
    )

    # Add token_ids as an attribute if provided
    if token_ids is not None:
        choice.token_ids = token_ids  # type: ignore
    if provider_specific_fields is not None:
        choice.provider_specific_fields = provider_specific_fields  # type: ignore

    return ChatCompletion(
        id=completion_id,
        choices=[choice],
        created=1234567890,
        model=model,
        object="chat.completion",
    )


def make_completion_request(
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    model: str = "gpt-4",
    prompt_token_ids: Optional[Sequence[int]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a mock CompletionCreateParams-like dict."""
    if messages is None:
        messages = [{"role": "user", "content": "Hello"}]

    request: Dict[str, Any] = {
        "messages": messages,
        "model": model,
    }
    if prompt_token_ids is not None:
        request["prompt_token_ids"] = prompt_token_ids
    if tools is not None:
        request["tools"] = tools
    return request


def make_chat_completion_call(
    *,
    request: Optional[Dict[str, Any]] = None,
    response: Optional[ChatCompletion] = None,
    prompt_token_ids: Optional[Sequence[int]] = None,
    completion_token_ids: Optional[Sequence[int]] = None,
    logprobs: Optional[List[float]] = None,
) -> ChatCompletionCall:
    """Create a mock ChatCompletionCall."""
    if request is None:
        request = make_completion_request(prompt_token_ids=prompt_token_ids)
    if response is None:
        response = make_chat_completion(token_ids=completion_token_ids, logprobs=logprobs)
    # Use model_construct to bypass Pydantic validation which strips unknown fields like prompt_token_ids
    return ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})


def make_annotated_call(
    *,
    request: Optional[Dict[str, Any]] = None,
    response: Optional[ChatCompletion] = None,
    reward: Optional[float] = None,
    prompt_token_ids: Optional[Sequence[int]] = None,
    completion_token_ids: Optional[Sequence[int]] = None,
    logprobs: Optional[List[float]] = None,
) -> AnnotatedChatCompletionCall:
    """Create a mock AnnotatedChatCompletionCall with optional reward."""
    if request is None:
        request = make_completion_request(prompt_token_ids=prompt_token_ids)
    if response is None:
        response = make_chat_completion(token_ids=completion_token_ids, logprobs=logprobs)

    annotations: List[GeneralAnnotation] = []
    if reward is not None:
        annotations.append(GeneralAnnotation(primary_reward=reward))

    # Use model_construct to bypass Pydantic validation which strips unknown fields like prompt_token_ids
    return AnnotatedChatCompletionCall.model_construct(
        request=request,
        response=response,
        malformed_fields={},
        annotations=annotations,
    )


def make_adapting_span(data: Any, span_id: str = "span-1") -> AdaptingSpan:
    """Create an AdaptingSpan with the given data."""
    span = Span.from_attributes(
        rollout_id="rollout-1",
        attempt_id="attempt-1",
        sequence_id=0,
        trace_id="trace-1",
        span_id=span_id,
        parent_id=None,
        name="test-span",
        attributes={},
        start_time=0.0,
        end_time=1.0,
    )
    return AdaptingSpan.from_span(span, data=data)


def make_adapting_sequence(calls: Sequence[Any]) -> AdaptingSequence[AdaptingSpan]:
    """Create an AdaptingSequence from a list of calls."""
    spans = [make_adapting_span(call, span_id=f"span-{i}") for i, call in enumerate(calls)]
    return AdaptingSequence(spans)


# ==============================================================================
# Tests for is_prefix function
# ==============================================================================


def test_is_prefix_empty_shorter():
    """Empty sequence is a prefix of any sequence."""
    assert is_prefix([], [1, 2, 3]) is True
    assert is_prefix([], []) is True


def test_is_prefix_identical_sequences():
    """Identical sequences should be prefixes of each other."""
    assert is_prefix([1, 2, 3], [1, 2, 3]) is True


def test_is_prefix_true_case():
    """Shorter sequence is a prefix of longer sequence."""
    assert is_prefix([1, 2], [1, 2, 3, 4]) is True
    assert is_prefix(["a"], ["a", "b", "c"]) is True


def test_is_prefix_false_mismatch():
    """Sequences with mismatched elements are not prefixes."""
    assert is_prefix([1, 2, 3], [1, 2, 4]) is False
    assert is_prefix([1, 3], [1, 2, 3]) is False


def test_is_prefix_longer_than_sequence():
    """Longer sequence cannot be a prefix of shorter sequence."""
    assert is_prefix([1, 2, 3, 4], [1, 2, 3]) is False


def test_is_prefix_with_iterables():
    """is_prefix should work with any iterables."""
    assert is_prefix(iter([1, 2]), iter([1, 2, 3])) is True
    assert is_prefix(range(3), [0, 1, 2, 3, 4]) is True


# ==============================================================================
# Tests for ToTokensTriplets
# ==============================================================================


def test_to_tokens_triplets_basic():
    """Basic conversion from chat completion call to tokens triplet."""
    call = make_chat_completion_call(
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5, 6],
    )
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()
    triplets = adapter.adapt(source)

    assert len(triplets) == 1
    assert list(triplets[0].observation.token_ids) == [1, 2, 3]
    assert list(triplets[0].action.token_ids) == [4, 5, 6]
    assert triplets[0].done is True  # Last triplet should be done


def test_to_tokens_triplets_with_reward():
    """Triplet should include reward from annotations."""
    call = make_annotated_call(
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5],
        reward=0.75,
    )
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()
    triplets = adapter.adapt(source)

    assert triplets[0].reward == 0.75


def test_to_tokens_triplets_with_logprobs():
    """Triplet should include logprobs when available."""
    call = make_chat_completion_call(
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5],
        logprobs=[-0.1, -0.2],
    )
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()
    triplets = adapter.adapt(source)

    assert triplets[0].action.logprobs is not None
    assert list(triplets[0].action.logprobs) == [-0.1, -0.2]


def test_to_tokens_triplets_with_image_urls():
    """Triplet should extract image URLs from messages."""
    request = make_completion_request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img1.png"}},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img2.png"}},
                ],
            }
        ],
        prompt_token_ids=[1, 2, 3],
    )
    response = make_chat_completion(token_ids=[4, 5])
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()
    triplets = adapter.adapt(source)

    assert list(triplets[0].observation.image_urls) == [
        "http://example.com/img1.png",
        "http://example.com/img2.png",
    ]


def test_to_tokens_triplets_missing_prompt_token_ids():
    """Should raise error when prompt_token_ids is missing."""
    request = make_completion_request()  # No prompt_token_ids
    response = make_chat_completion(token_ids=[4, 5])
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()

    with pytest.raises(RuntimeError, match="failed to create any triplets"):
        adapter.adapt(source)


def test_to_tokens_triplets_missing_completion_token_ids():
    """Should raise error when completion token_ids is missing."""
    request = make_completion_request(prompt_token_ids=[1, 2, 3])
    response = make_chat_completion()  # No token_ids
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()

    with pytest.raises(RuntimeError, match="failed to create any triplets"):
        adapter.adapt(source)


def test_to_tokens_triplets_empty_prompt_token_ids():
    """Should raise error when prompt_token_ids is empty."""
    request = make_completion_request(prompt_token_ids=[])
    response = make_chat_completion(token_ids=[4, 5])
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()

    with pytest.raises(RuntimeError, match="failed to create any triplets"):
        adapter.adapt(source)


def test_to_tokens_triplets_empty_completion_token_ids():
    """Should raise error when completion token_ids is empty."""
    request = make_completion_request(prompt_token_ids=[1, 2, 3])
    response = make_chat_completion(token_ids=[])
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()

    with pytest.raises(RuntimeError, match="failed to create any triplets"):
        adapter.adapt(source)


def test_to_tokens_triplets_strict_mode():
    """In strict mode, should raise immediately on error."""
    request = make_completion_request()  # Missing prompt_token_ids
    response = make_chat_completion(token_ids=[4, 5])
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets(strict=True)

    with pytest.raises(ValueError, match="Prompt token ids not found"):
        adapter.adapt(source)


def test_to_tokens_triplets_multiple_calls():
    """Multiple calls should create multiple triplets."""
    call1 = make_chat_completion_call(prompt_token_ids=[1, 2], completion_token_ids=[3, 4])
    call2 = make_chat_completion_call(prompt_token_ids=[5, 6], completion_token_ids=[7, 8])
    source = make_adapting_sequence([call1, call2])

    adapter = ToTokensTriplets()
    triplets = adapter.adapt(source)

    assert len(triplets) == 2
    assert triplets[0].done is False
    assert triplets[1].done is True


def test_to_tokens_triplets_provider_specific_token_ids():
    """Should extract token_ids from provider_specific_fields."""
    request = make_completion_request(prompt_token_ids=[1, 2, 3])
    response = make_chat_completion(provider_specific_fields={"token_ids": [4, 5, 6]})
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()
    triplets = adapter.adapt(source)

    assert list(triplets[0].action.token_ids) == [4, 5, 6]


def test_to_tokens_triplets_skip_non_call_spans():
    """Should skip spans that don't contain chat completion calls."""
    call = make_chat_completion_call(prompt_token_ids=[1, 2], completion_token_ids=[3, 4])
    spans = [
        make_adapting_span("not a call", span_id="span-0"),
        make_adapting_span(call, span_id="span-1"),
        make_adapting_span(None, span_id="span-2"),
    ]
    source = AdaptingSequence(spans)

    adapter = ToTokensTriplets()
    triplets = adapter.adapt(source)

    assert len(triplets) == 1


# ==============================================================================
# Tests for ToTokensAccumulations
# ==============================================================================


def test_to_tokens_accumulations_single_triplet():
    """Single triplet should be converted to accumulation."""
    raw_call = make_chat_completion_call()
    triplet = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2, 3], image_urls=[]),
        action=TokenOutput(token_ids=[4, 5], logprobs=[-0.1, -0.2]),
        reward=0.5,
        done=True,
        raw_call=raw_call,
    )

    adapter = ToTokensAccumulations()
    accumulations = adapter.adapt([triplet])

    assert len(accumulations) == 1
    assert list(accumulations[0].token_ids) == [1, 2, 3, 4, 5]
    assert accumulations[0].response_mask == [0, 0, 0, 1, 1]
    assert accumulations[0].final_reward == 0.5


def test_to_tokens_accumulations_merge_sequential():
    """Sequential triplets with matching prefixes should be merged."""
    triplet1 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2], image_urls=[]),
        action=TokenOutput(token_ids=[3, 4], logprobs=None),
        reward=0.3,
        done=False,
        raw_call=make_chat_completion_call(),
    )
    triplet2 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2, 3, 4, 5], image_urls=[]),
        action=TokenOutput(token_ids=[6, 7], logprobs=None),
        reward=0.4,
        done=True,
        raw_call=make_chat_completion_call(),
    )

    adapter = ToTokensAccumulations()
    accumulations = adapter.adapt([triplet1, triplet2])

    assert len(accumulations) == 1
    # Final tokens: [1, 2, 3, 4] + [5] + [6, 7]
    assert list(accumulations[0].token_ids) == [1, 2, 3, 4, 5, 6, 7]
    # Response mask: [0, 0, 1, 1] + [0] + [1, 1]
    assert accumulations[0].response_mask == [0, 0, 1, 1, 0, 1, 1]
    # Rewards should be summed
    assert accumulations[0].final_reward == 0.7


def test_to_tokens_accumulations_no_merge_mismatch():
    """Triplets with mismatched prefixes should not merge."""
    triplet1 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2], image_urls=[]),
        action=TokenOutput(token_ids=[3, 4], logprobs=None),
        reward=0.3,
        done=False,
        raw_call=make_chat_completion_call(),
    )
    triplet2 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[5, 6, 7], image_urls=[]),  # Doesn't start with [1,2,3,4]
        action=TokenOutput(token_ids=[8, 9], logprobs=None),
        reward=0.4,
        done=True,
        raw_call=make_chat_completion_call(),
    )

    adapter = ToTokensAccumulations()
    accumulations = adapter.adapt([triplet1, triplet2])

    assert len(accumulations) == 2


def test_to_tokens_accumulations_image_url_mismatch():
    """Triplets with mismatched image URLs should not merge."""
    triplet1 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2], image_urls=["img1.png"]),
        action=TokenOutput(token_ids=[3, 4], logprobs=None),
        reward=None,
        done=False,
        raw_call=make_chat_completion_call(),
    )
    triplet2 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2, 3, 4, 5], image_urls=["img2.png"]),  # Different image
        action=TokenOutput(token_ids=[6, 7], logprobs=None),
        reward=None,
        done=True,
        raw_call=make_chat_completion_call(),
    )

    adapter = ToTokensAccumulations()
    accumulations = adapter.adapt([triplet1, triplet2])

    assert len(accumulations) == 2


def test_to_tokens_accumulations_image_url_extension():
    """Image URLs can be extended when merging."""
    triplet1 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2], image_urls=["img1.png"]),
        action=TokenOutput(token_ids=[3, 4], logprobs=None),
        reward=None,
        done=False,
        raw_call=make_chat_completion_call(),
    )
    triplet2 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2, 3, 4, 5], image_urls=["img1.png", "img2.png"]),
        action=TokenOutput(token_ids=[6, 7], logprobs=None),
        reward=None,
        done=True,
        raw_call=make_chat_completion_call(),
    )

    adapter = ToTokensAccumulations()
    accumulations = adapter.adapt([triplet1, triplet2])

    assert len(accumulations) == 1
    assert list(accumulations[0].image_urls) == ["img1.png", "img2.png"]


def test_to_tokens_accumulations_with_logprobs_merge():
    """Logprobs should be properly accumulated when merging."""
    triplet1 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2], image_urls=[]),
        action=TokenOutput(token_ids=[3, 4], logprobs=[-0.1, -0.2]),
        reward=None,
        done=False,
        raw_call=make_chat_completion_call(),
    )
    triplet2 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2, 3, 4, 5], image_urls=[]),
        action=TokenOutput(token_ids=[6, 7], logprobs=[-0.3, -0.4]),
        reward=None,
        done=True,
        raw_call=make_chat_completion_call(),
    )

    adapter = ToTokensAccumulations()
    accumulations = adapter.adapt([triplet1, triplet2])

    assert len(accumulations) == 1
    # Final tokens: [1, 2, 3, 4] + [5] + [6, 7] = 7 tokens
    assert list(accumulations[0].token_ids) == [1, 2, 3, 4, 5, 6, 7]
    assert accumulations[0].logprobs is not None
    # Logprobs should match token_ids length
    # [0, 0, -0.1, -0.2] from first triplet + [0] for observation extension + [-0.3, -0.4] for action
    expected_logprobs = [0.0, 0.0, -0.1, -0.2, 0.0, -0.3, -0.4]
    assert list(accumulations[0].logprobs) == expected_logprobs
    assert len(accumulations[0].logprobs) == len(accumulations[0].token_ids)


def test_to_tokens_accumulations_empty_input():
    """Empty input should return empty output."""
    adapter = ToTokensAccumulations()
    accumulations = adapter.adapt([])

    assert len(accumulations) == 0


def test_to_tokens_accumulations_none_reward_handling():
    """None rewards should be handled correctly."""
    triplet1 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2], image_urls=[]),
        action=TokenOutput(token_ids=[3, 4], logprobs=None),
        reward=None,
        done=False,
        raw_call=make_chat_completion_call(),
    )
    triplet2 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2, 3, 4], image_urls=[]),
        action=TokenOutput(token_ids=[5, 6], logprobs=None),
        reward=0.5,
        done=True,
        raw_call=make_chat_completion_call(),
    )

    adapter = ToTokensAccumulations()
    accumulations = adapter.adapt([triplet1, triplet2])

    assert len(accumulations) == 1
    assert accumulations[0].final_reward == 0.5


def test_to_tokens_accumulations_raw_calls_aggregated():
    """Raw calls from merged triplets should be aggregated."""
    raw_call1 = make_chat_completion_call(completion_token_ids=[1])
    raw_call2 = make_chat_completion_call(completion_token_ids=[2])
    triplet1 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2], image_urls=[]),
        action=TokenOutput(token_ids=[3, 4], logprobs=None),
        reward=None,
        done=False,
        raw_call=raw_call1,
    )
    triplet2 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2, 3, 4], image_urls=[]),
        action=TokenOutput(token_ids=[5, 6], logprobs=None),
        reward=None,
        done=True,
        raw_call=raw_call2,
    )

    adapter = ToTokensAccumulations()
    accumulations = adapter.adapt([triplet1, triplet2])

    assert len(accumulations[0].raw_calls) == 2
    # Verify they are the same objects (not copies)
    assert accumulations[0].raw_calls[0].response.choices[0].token_ids == [1]  # type: ignore
    assert accumulations[0].raw_calls[1].response.choices[0].token_ids == [2]  # type: ignore


# ==============================================================================
# Tests for ToPromptCompletionTriplets
# ==============================================================================


def test_to_prompt_completion_triplets_basic():
    """Basic conversion from chat completion call to prompt-completion triplet."""
    call = make_chat_completion_call()
    source = make_adapting_sequence([call])

    adapter = ToPromptCompletionTriplets()
    triplets = adapter.adapt(source)

    assert len(triplets) == 1
    assert triplets[0].observation == call.request
    assert triplets[0].action == call.response
    assert triplets[0].done is True


def test_to_prompt_completion_triplets_with_reward():
    """Triplet should include reward from annotations."""
    call = make_annotated_call(reward=0.9)
    source = make_adapting_sequence([call])

    adapter = ToPromptCompletionTriplets()
    triplets = adapter.adapt(source)

    assert triplets[0].reward == 0.9


def test_to_prompt_completion_triplets_multiple():
    """Multiple calls should create multiple triplets with correct done flags."""
    call1 = make_chat_completion_call()
    call2 = make_chat_completion_call()
    call3 = make_chat_completion_call()
    source = make_adapting_sequence([call1, call2, call3])

    adapter = ToPromptCompletionTriplets()
    triplets = adapter.adapt(source)

    assert len(triplets) == 3
    assert triplets[0].done is False
    assert triplets[1].done is False
    assert triplets[2].done is True


def test_to_prompt_completion_triplets_strict_mode():
    """In strict mode, errors should propagate immediately."""
    # Create a span with non-call data
    span = make_adapting_span("not a call")
    source = AdaptingSequence([span])

    adapter = ToPromptCompletionTriplets()

    # Should raise because no valid triplets can be created
    with pytest.raises(RuntimeError, match="failed to create any triplets"):
        adapter.adapt(source)


# ==============================================================================
# Tests for ToPromptCompletionAccumulations
# ==============================================================================


def test_to_prompt_completion_accumulations_single():
    """Single triplet converts to single accumulation."""
    request = make_completion_request(messages=[{"role": "user", "content": "Hi"}])
    response = make_chat_completion(content="Hello!")
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})

    triplet = PromptCompletionTriplet.model_construct(
        observation=request,
        action=response,
        reward=0.5,
        done=True,
        raw_call=call,
    )

    adapter = ToPromptCompletionAccumulations()
    accumulations = adapter.adapt([triplet])

    assert len(accumulations) == 1
    # Should have user message + assistant response
    assert len(accumulations[0].messages) == 2
    assert accumulations[0].messages[0]["role"] == "user"
    assert accumulations[0].messages[1]["role"] == "assistant"


def test_to_prompt_completion_accumulations_merge():
    """Sequential triplets with matching messages should merge."""
    request1 = make_completion_request(messages=[{"role": "user", "content": "Hi"}])
    response1 = make_chat_completion(content="Hello!")
    call1 = ChatCompletionCall.model_construct(request=request1, response=response1, malformed_fields={})

    # Second request includes the previous conversation
    request2 = make_completion_request(
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
    )
    response2 = make_chat_completion(content="I'm good!")
    call2 = ChatCompletionCall.model_construct(request=request2, response=response2, malformed_fields={})

    triplet1 = PromptCompletionTriplet.model_construct(
        observation=request1,
        action=response1,
        reward=0.3,
        done=False,
        raw_call=call1,
    )
    triplet2 = PromptCompletionTriplet.model_construct(
        observation=request2,
        action=response2,
        reward=0.4,
        done=True,
        raw_call=call2,
    )

    adapter = ToPromptCompletionAccumulations()
    accumulations = adapter.adapt([triplet1, triplet2])

    assert len(accumulations) == 1
    # Should have all 4 messages
    assert len(accumulations[0].messages) == 4
    assert accumulations[0].final_reward == 0.7


def test_to_prompt_completion_accumulations_no_merge_different_tools():
    """Triplets with different tools should not merge."""
    request1 = make_completion_request(tools=[{"type": "function", "function": {"name": "tool1"}}])
    response1 = make_chat_completion()
    call1 = ChatCompletionCall.model_construct(request=request1, response=response1, malformed_fields={})

    request2 = make_completion_request(tools=[{"type": "function", "function": {"name": "tool2"}}])
    response2 = make_chat_completion()
    call2 = ChatCompletionCall.model_construct(request=request2, response=response2, malformed_fields={})

    triplet1 = PromptCompletionTriplet.model_construct(
        observation=request1,
        action=response1,
        reward=None,
        done=False,
        raw_call=call1,
    )
    triplet2 = PromptCompletionTriplet.model_construct(
        observation=request2,
        action=response2,
        reward=None,
        done=True,
        raw_call=call2,
    )

    adapter = ToPromptCompletionAccumulations()
    accumulations = adapter.adapt([triplet1, triplet2])

    assert len(accumulations) == 2


def test_to_prompt_completion_accumulations_no_merge_message_mismatch():
    """Triplets with mismatched message history should not merge."""
    request1 = make_completion_request(messages=[{"role": "user", "content": "Hi"}])
    response1 = make_chat_completion(content="Hello!")
    call1 = ChatCompletionCall.model_construct(request=request1, response=response1, malformed_fields={})

    # Second request has completely different history
    request2 = make_completion_request(messages=[{"role": "user", "content": "Different question"}])
    response2 = make_chat_completion(content="Different answer")
    call2 = ChatCompletionCall.model_construct(request=request2, response=response2, malformed_fields={})

    triplet1 = PromptCompletionTriplet.model_construct(
        observation=request1,
        action=response1,
        reward=None,
        done=False,
        raw_call=call1,
    )
    triplet2 = PromptCompletionTriplet.model_construct(
        observation=request2,
        action=response2,
        reward=None,
        done=True,
        raw_call=call2,
    )

    adapter = ToPromptCompletionAccumulations()
    accumulations = adapter.adapt([triplet1, triplet2])

    assert len(accumulations) == 2


def test_to_prompt_completion_accumulations_preserves_tools():
    """Accumulation should preserve tool definitions."""
    tools: List[Dict[str, Any]] = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]
    request = make_completion_request(tools=tools)
    response = make_chat_completion()
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})

    triplet = PromptCompletionTriplet.model_construct(
        observation=request,
        action=response,
        reward=None,
        done=True,
        raw_call=call,
    )

    adapter = ToPromptCompletionAccumulations()
    accumulations = adapter.adapt([triplet])

    assert accumulations[0].tools == tools


def test_to_prompt_completion_accumulations_empty():
    """Empty input should return empty output."""
    adapter = ToPromptCompletionAccumulations()
    accumulations = adapter.adapt([])

    assert len(accumulations) == 0


# ==============================================================================
# Tests for PropagateRewards
# ==============================================================================


def test_propagate_rewards_forward_triplets():
    """Forward propagation fills None rewards with previous value."""
    triplets = [
        TokensTriplet.model_construct(
            observation=TokenInput(token_ids=[1], image_urls=[]),
            action=TokenOutput(token_ids=[2], logprobs=None),
            reward=0.5,
            done=False,
            raw_call=make_chat_completion_call(),
        ),
        TokensTriplet.model_construct(
            observation=TokenInput(token_ids=[1, 2], image_urls=[]),
            action=TokenOutput(token_ids=[3], logprobs=None),
            reward=None,
            done=False,
            raw_call=make_chat_completion_call(),
        ),
        TokensTriplet.model_construct(
            observation=TokenInput(token_ids=[1, 2, 3], image_urls=[]),
            action=TokenOutput(token_ids=[4], logprobs=None),
            reward=None,
            done=True,
            raw_call=make_chat_completion_call(),
        ),
    ]

    adapter = PropagateRewards(direction="forward")
    result = adapter.adapt(triplets)

    assert result[0].reward == 0.5
    assert result[1].reward == 0.5
    assert result[2].reward == 0.5


def test_propagate_rewards_backward_triplets():
    """Backward propagation fills None rewards with next value."""
    triplets = [
        TokensTriplet.model_construct(
            observation=TokenInput(token_ids=[1], image_urls=[]),
            action=TokenOutput(token_ids=[2], logprobs=None),
            reward=None,
            done=False,
            raw_call=make_chat_completion_call(),
        ),
        TokensTriplet.model_construct(
            observation=TokenInput(token_ids=[1, 2], image_urls=[]),
            action=TokenOutput(token_ids=[3], logprobs=None),
            reward=None,
            done=False,
            raw_call=make_chat_completion_call(),
        ),
        TokensTriplet.model_construct(
            observation=TokenInput(token_ids=[1, 2, 3], image_urls=[]),
            action=TokenOutput(token_ids=[4], logprobs=None),
            reward=0.8,
            done=True,
            raw_call=make_chat_completion_call(),
        ),
    ]

    adapter = PropagateRewards(direction="backward")
    result = adapter.adapt(triplets)

    assert result[0].reward == 0.8
    assert result[1].reward == 0.8
    assert result[2].reward == 0.8


def test_propagate_rewards_forward_accumulations():
    """Forward propagation works with accumulations (uses final_reward)."""
    accumulations = [
        TokensAccumulation(
            token_ids=[1, 2],
            image_urls=[],
            logprobs=None,
            response_mask=[0, 1],
            final_reward=0.3,
            raw_calls=[],
            diagnosis_info=None,
        ),
        TokensAccumulation(
            token_ids=[3, 4],
            image_urls=[],
            logprobs=None,
            response_mask=[0, 1],
            final_reward=None,
            raw_calls=[],
            diagnosis_info=None,
        ),
    ]

    adapter = PropagateRewards(direction="forward")
    result = adapter.adapt(accumulations)

    assert result[0].final_reward == 0.3
    assert result[1].final_reward == 0.3


def test_propagate_rewards_backward_accumulations():
    """Backward propagation works with accumulations."""
    accumulations = [
        TokensAccumulation(
            token_ids=[1, 2],
            image_urls=[],
            logprobs=None,
            response_mask=[0, 1],
            final_reward=None,
            raw_calls=[],
            diagnosis_info=None,
        ),
        TokensAccumulation(
            token_ids=[3, 4],
            image_urls=[],
            logprobs=None,
            response_mask=[0, 1],
            final_reward=0.7,
            raw_calls=[],
            diagnosis_info=None,
        ),
    ]

    adapter = PropagateRewards(direction="backward")
    result = adapter.adapt(accumulations)

    assert result[0].final_reward == 0.7
    assert result[1].final_reward == 0.7


def test_propagate_rewards_preserves_existing():
    """Propagation should not overwrite existing rewards."""
    triplets = [
        TokensTriplet.model_construct(
            observation=TokenInput(token_ids=[1], image_urls=[]),
            action=TokenOutput(token_ids=[2], logprobs=None),
            reward=0.1,
            done=False,
            raw_call=make_chat_completion_call(),
        ),
        TokensTriplet.model_construct(
            observation=TokenInput(token_ids=[1, 2], image_urls=[]),
            action=TokenOutput(token_ids=[3], logprobs=None),
            reward=0.5,
            done=False,
            raw_call=make_chat_completion_call(),
        ),
        TokensTriplet.model_construct(
            observation=TokenInput(token_ids=[1, 2, 3], image_urls=[]),
            action=TokenOutput(token_ids=[4], logprobs=None),
            reward=None,
            done=True,
            raw_call=make_chat_completion_call(),
        ),
    ]

    adapter = PropagateRewards(direction="forward")
    result = adapter.adapt(triplets)

    assert result[0].reward == 0.1
    assert result[1].reward == 0.5
    assert result[2].reward == 0.5  # Propagated from index 1


def test_propagate_rewards_empty_input():
    """Empty input should return empty output."""
    adapter = PropagateRewards(direction="forward")
    triplets: List[TokensTriplet] = []
    result = adapter.adapt(triplets)

    assert len(result) == 0


def test_propagate_rewards_all_none():
    """All None rewards should remain None."""
    triplets = [
        TokensTriplet.model_construct(
            observation=TokenInput(token_ids=[1], image_urls=[]),
            action=TokenOutput(token_ids=[2], logprobs=None),
            reward=None,
            done=False,
            raw_call=make_chat_completion_call(),
        ),
        TokensTriplet.model_construct(
            observation=TokenInput(token_ids=[1, 2], image_urls=[]),
            action=TokenOutput(token_ids=[3], logprobs=None),
            reward=None,
            done=True,
            raw_call=make_chat_completion_call(),
        ),
    ]

    adapter = PropagateRewards(direction="forward")
    result = adapter.adapt(triplets)

    assert result[0].reward is None
    assert result[1].reward is None


def test_propagate_rewards_prompt_completion_triplets():
    """Propagation works with PromptCompletionTriplet."""
    request = make_completion_request()
    response = make_chat_completion()
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})

    triplets = [
        PromptCompletionTriplet.model_construct(
            observation=request,
            action=response,
            reward=0.6,
            done=False,
            raw_call=call,
        ),
        PromptCompletionTriplet.model_construct(
            observation=request,
            action=response,
            reward=None,
            done=True,
            raw_call=call,
        ),
    ]

    adapter = PropagateRewards(direction="forward")
    result = adapter.adapt(triplets)

    assert result[0].reward == 0.6
    assert result[1].reward == 0.6


def test_propagate_rewards_prompt_completion_accumulations():
    """Propagation works with PromptCompletionAccumulation."""
    accumulations = [
        PromptCompletionAccumulation(
            messages=[{"role": "user", "content": "Hi"}],
            tools=None,
            final_reward=None,
            raw_calls=[],
        ),
        PromptCompletionAccumulation(
            messages=[{"role": "user", "content": "Bye"}],
            tools=None,
            final_reward=0.9,
            raw_calls=[],
        ),
    ]

    adapter = PropagateRewards(direction="backward")
    result = adapter.adapt(accumulations)

    assert result[0].final_reward == 0.9
    assert result[1].final_reward == 0.9


# ==============================================================================
# Tests for ToTokensAccumulations diagnosis feature
# ==============================================================================


def test_to_tokens_accumulations_diagnosis_disabled_by_default():
    """Diagnosis info should be None when diagnosis=False (default)."""
    triplet1 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[1, 2], image_urls=[]),
        action=TokenOutput(token_ids=[3, 4], logprobs=None),
        reward=None,
        done=False,
        raw_call=make_chat_completion_call(),
    )
    triplet2 = TokensTriplet.model_construct(
        observation=TokenInput(token_ids=[5, 6, 7], image_urls=[]),  # Mismatch
        action=TokenOutput(token_ids=[8, 9], logprobs=None),
        reward=None,
        done=True,
        raw_call=make_chat_completion_call(),
    )

    adapter = ToTokensAccumulations(diagnosis=False)
    accumulations = adapter.adapt([triplet1, triplet2])

    assert len(accumulations) == 2
    assert accumulations[0].diagnosis_info is None
    assert accumulations[1].diagnosis_info is None


# ==============================================================================
# Edge case tests
# ==============================================================================


def test_to_tokens_triplets_invalid_prompt_token_ids_type():
    """Should raise error when prompt_token_ids is not a list of ints."""
    request = make_completion_request()
    request["prompt_token_ids"] = "not a list"
    response = make_chat_completion(token_ids=[4, 5])
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()

    with pytest.raises(RuntimeError, match="failed to create any triplets"):
        adapter.adapt(source)


def test_to_tokens_triplets_invalid_completion_token_ids_type():
    """Should raise error when completion token_ids is not a list of ints."""
    request = make_completion_request(prompt_token_ids=[1, 2, 3])
    response = make_chat_completion(token_ids=["not", "ints"])  # type: ignore
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()

    with pytest.raises(RuntimeError, match="failed to create any triplets"):
        adapter.adapt(source)


def test_to_tokens_triplets_message_without_content():
    """Messages without content should be handled gracefully."""
    request = make_completion_request(
        messages=[{"role": "assistant"}],  # No content field
        prompt_token_ids=[1, 2, 3],
    )
    response = make_chat_completion(token_ids=[4, 5])
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()
    triplets = adapter.adapt(source)

    # Should succeed with empty image_urls
    assert len(triplets) == 1
    assert list(triplets[0].observation.image_urls) == []


def test_to_tokens_triplets_message_with_none_content():
    """Messages with None content should be handled gracefully."""
    request = make_completion_request(
        messages=[{"role": "assistant", "content": None}],
        prompt_token_ids=[1, 2, 3],
    )
    response = make_chat_completion(token_ids=[4, 5])
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()
    triplets = adapter.adapt(source)

    assert len(triplets) == 1
    assert list(triplets[0].observation.image_urls) == []


def test_to_tokens_triplets_mixed_content_types():
    """Messages with mixed content types should extract only image URLs."""
    request = make_completion_request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe these"},
                    {"type": "image_url", "image_url": {"url": "http://img1.png"}},
                    {"type": "text", "text": "Thanks"},
                ],
            }
        ],
        prompt_token_ids=[1, 2, 3],
    )
    response = make_chat_completion(token_ids=[4, 5])
    call = ChatCompletionCall.model_construct(request=request, response=response, malformed_fields={})
    source = make_adapting_sequence([call])

    adapter = ToTokensTriplets()
    triplets = adapter.adapt(source)

    assert list(triplets[0].observation.image_urls) == ["http://img1.png"]


def test_get_reward_from_non_annotated_call():
    """get_reward should return None for non-annotated calls."""
    call = make_chat_completion_call()
    adapter = ToTokensTriplets()

    reward = adapter.get_reward(call)

    assert reward is None


def test_get_reward_from_annotated_call_without_general_annotation():
    """get_reward should return None if no GeneralAnnotation exists."""
    call = make_annotated_call()  # No reward specified
    # Manually clear annotations
    call = AnnotatedChatCompletionCall(
        request=call.request,
        response=call.response,
        malformed_fields={},
        annotations=[],  # Empty annotations
    )

    adapter = ToTokensTriplets()
    reward = adapter.get_reward(call)

    assert reward is None


def test_to_triplets_sets_done_on_last():
    """to_triplets should set done=True only on the last triplet."""
    call1 = make_chat_completion_call(prompt_token_ids=[1], completion_token_ids=[2])
    call2 = make_chat_completion_call(prompt_token_ids=[3], completion_token_ids=[4])
    call3 = make_chat_completion_call(prompt_token_ids=[5], completion_token_ids=[6])
    source = make_adapting_sequence([call1, call2, call3])

    adapter = ToTokensTriplets()
    triplets = adapter.adapt(source)

    assert triplets[0].done is False
    assert triplets[1].done is False
    assert triplets[2].done is True
