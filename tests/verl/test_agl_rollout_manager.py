# Copyright (c) Microsoft. All rights reserved.

"""Tests for VERL rollout manager event conversion."""

from __future__ import annotations

import pytest

from agentlightning.schemas import Event, Rollout, RolloutConfig, RolloutLifecycleStatus, RolloutState
from agentlightning.verl.agl_rollout_manager import (
    AglRolloutManagerBase,
    EnqueuedRollout,
    _aligned_image_urls,
    _extract_image_urls_from_messages,
)


class _Manager(AglRolloutManagerBase):
    def __init__(self, triplet_events: list[Event]) -> None:
        self._triplet_events = triplet_events

    def _fetch_rollout_events(self, rollout_id: str) -> tuple[list[Event], list[Event]]:
        return self._triplet_events, self._triplet_events


class _ManagerWithViews(AglRolloutManagerBase):
    """Manager stub returning distinct raw and triplet event views."""

    def __init__(self, raw_events: list[Event], triplet_events: list[Event]) -> None:
        self._raw_events = raw_events
        self._triplet_view_events = triplet_events

    def _fetch_rollout_events(self, rollout_id: str) -> tuple[list[Event], list[Event]]:
        return self._raw_events, self._triplet_view_events


def _event(event_type: str, data: dict) -> Event:
    return Event(event_type=event_type, rollout_id="rollout-1", attempt_id="0", timestamp=0.0, data=data)


def _rollout() -> Rollout:
    return Rollout(
        rollout_id="rollout-1",
        input={"prompt": "hi"},
        config=RolloutConfig(),
        status=RolloutLifecycleStatus(
            state=RolloutState.SUCCEEDED,
            last_attempt_id="0",
            created_at=0.0,
            updated_at=0.0,
        ),
    )


def test_build_completed_rollout_skips_error_and_empty_model_requests() -> None:
    manager = _Manager(
        [
            _event(
                "model_request",
                {
                    "prompt_token_ids": [],
                    "response_token_ids": [],
                    "http_status": 400,
                    "status": "error",
                },
            ),
            _event(
                "model_request",
                {
                    "prompt_token_ids": [1],
                    "response_token_ids": [],
                    "http_status": 200,
                    "status": "ok",
                },
            ),
            _event(
                "model_request",
                {
                    "prompt_token_ids": [1],
                    "response_token_ids": [2],
                    "response_log_probs": [-0.1],
                    "http_status": 200,
                    "status": "ok",
                    "server": {"model": "test-model", "version": 3},
                },
            ),
            _event("reward", {"value": 1.0}),
        ]
    )

    completed = manager._build_completed_rollout(
        EnqueuedRollout(
            data_id="data-1",
            rollout_id="rollout-1",
            step=0,
            sample_idx_in_step=0,
            enqueue_time=0.0,
        ),
        _rollout(),
    )

    assert completed.final_reward == 1.0
    assert completed.triplets is not None
    assert len(completed.triplets) == 1
    assert completed.triplets[0].prompt["token_ids"] == [1]
    assert completed.triplets[0].response["token_ids"] == [2]
    assert completed.triplets[0].response["log_probs"] == [-0.1]
    assert completed.triplets[0].reward == 1.0
    assert completed.triplets[0].image_urls is None


_IMG = "data:image/jpeg;base64,QUJD"
_IMG2 = "data:image/png;base64,REVG"


def _raw_model_request_event(
    urls: list[str],
    prompt_token_ids: object,
    response_token_ids: list[int],
    status: str = "success",
    http_status: int = 200,
) -> Event:
    """Raw (untrimmed) model_request event: full request body + raw response payload."""
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image_url", "image_url": {"url": url}} for url in urls],
                {"type": "text", "text": "prompt"},
            ],
        }
    ]
    return _event(
        "model_request",
        {
            "request": {"messages": messages},
            "response": {
                "prompt_token_ids": prompt_token_ids,
                "choices": [{"token_ids": response_token_ids, "logprobs": None}],
            },
            "status": status,
            "http_status": http_status,
            "server": {},
        },
    )


def _triplet_model_request_event(prompt_token_ids: object, response_token_ids: list[int]) -> Event:
    """Trimmed (triplet-view) model_request event as stored by the server."""
    return _event(
        "model_request",
        {
            "prompt_token_ids": prompt_token_ids,
            "response_token_ids": response_token_ids,
            "response_log_probs": [-0.1] * len(response_token_ids),
            "http_status": 200,
            "status": "ok",
            "server": {},
        },
    )


def test_extract_image_urls_from_messages() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "file:///a.jpg"}},
                {"type": "text", "text": "x"},
                {"type": "image_url", "image_url": {"url": "https://h/b.jpg"}},
            ],
        }
    ]
    assert _extract_image_urls_from_messages(messages) == ["file:///a.jpg", "https://h/b.jpg"]

    # Content serialized as a JSON string (some clients do this).
    import json

    json_content = json.dumps([{"type": "image_url", "image_url": {"url": _IMG}}])
    assert _extract_image_urls_from_messages([{"content": json_content}]) == [_IMG]

    assert _extract_image_urls_from_messages([{"content": "plain text"}]) == []
    assert _extract_image_urls_from_messages("not-a-list") == []
    assert _extract_image_urls_from_messages([{"content": [{"type": "text", "text": "hi"}]}]) == []


def test_build_completed_rollout_aligns_image_urls_with_triplets() -> None:
    raw_events = [
        # Superseded retry (same prompt_token_ids, keep last) with an empty response.
        _raw_model_request_event([_IMG], [1, 2], []),
        _raw_model_request_event([_IMG], [1, 2], [3, 4]),
        # Filtered out: error status and http >= 400 are skipped like the triplet loop.
        _raw_model_request_event([_IMG], [5], [6], status="error"),
        _raw_model_request_event([_IMG], [50], [60], http_status=500),
        _raw_model_request_event([_IMG, _IMG2], [7, 8], [9]),
        _raw_model_request_event([], [10], [11]),
        _event("reward", {"value": 1.0}),
    ]
    triplet_events = [
        _triplet_model_request_event([1, 2], [3, 4]),
        _triplet_model_request_event([7, 8], [9]),
        _triplet_model_request_event([10], [11]),
        _event("reward", {"value": 1.0}),
    ]
    manager = _ManagerWithViews(raw_events, triplet_events)

    completed = manager._build_completed_rollout(
        EnqueuedRollout(
            data_id="data-1",
            rollout_id="rollout-1",
            step=0,
            sample_idx_in_step=0,
            enqueue_time=0.0,
        ),
        _rollout(),
    )

    assert completed.triplets is not None
    assert len(completed.triplets) == 3
    assert [triplet.image_urls for triplet in completed.triplets] == [[_IMG], [_IMG, _IMG2], None]


@pytest.mark.parametrize("prompt_token_ids", [None, [], [[1]], ["1"], [True]])
def test_build_completed_rollout_aligns_images_without_valid_prompt_token_ids(prompt_token_ids: object) -> None:
    trimmed_prompt_token_ids = [] if prompt_token_ids is None else prompt_token_ids
    raw_events = [
        _raw_model_request_event([_IMG], prompt_token_ids, [1]),
        _raw_model_request_event([_IMG2], prompt_token_ids, [2]),
    ]
    triplet_events = [
        _triplet_model_request_event(trimmed_prompt_token_ids, [1]),
        _triplet_model_request_event(trimmed_prompt_token_ids, [2]),
    ]
    manager = _ManagerWithViews(raw_events, triplet_events)

    completed = manager._build_completed_rollout(
        EnqueuedRollout(
            data_id="data-1",
            rollout_id="rollout-1",
            step=0,
            sample_idx_in_step=0,
            enqueue_time=0.0,
        ),
        _rollout(),
    )

    assert completed.triplets is not None
    assert [triplet.image_urls for triplet in completed.triplets] == [[_IMG], [_IMG2]]


def test_aligned_image_urls_count_mismatch_returns_none(capsys: pytest.CaptureFixture[str]) -> None:
    raw_events = [_raw_model_request_event([_IMG], [1, 2], [3, 4])]

    assert _aligned_image_urls(raw_events, 2) is None
    assert "cannot align raw model_request events" in capsys.readouterr().out


def test_aligned_image_urls_text_only_rollout_returns_none_without_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_events = [
        _raw_model_request_event([], [1, 2], [3, 4]),
        _raw_model_request_event([], [5], [6]),
    ]

    # Text-only rollouts keep the exact original behavior: no alignment, no warning.
    assert _aligned_image_urls(raw_events, 2) is None
    assert capsys.readouterr().out == ""
