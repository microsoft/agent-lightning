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
        _raw_model_request_event([_IMG2], [1, 2], [30, 40]),
        # Filtered out: error status and http >= 400 are skipped like the triplet loop.
        _raw_model_request_event([_IMG], [5], [6], status="error"),
        _raw_model_request_event([_IMG], [50], [60], http_status=500),
        _raw_model_request_event([_IMG, _IMG2], [7, 8], [9]),
        _raw_model_request_event([], [10], [11]),
        _event("reward", {"value": 1.0}),
    ]
    triplet_events = [
        _triplet_model_request_event([1, 2], [3, 4]),
        _triplet_model_request_event([1, 2], [30, 40]),
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
    assert len(completed.triplets) == 4
    assert [triplet.image_urls for triplet in completed.triplets] == [[_IMG], [_IMG2], [_IMG, _IMG2], None]


@pytest.mark.parametrize("prompt_token_ids", [None, [], [[1]], ["1"], [True], [1.0], "bad-ids", {"token": 1}])
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


@pytest.mark.parametrize("routed_experts", [False, True])
def test_distinct_images_survive_server_to_training_batch(routed_experts: bool) -> None:
    import base64
    import io
    from typing import cast

    import numpy as np
    import torch
    from fastapi.testclient import TestClient
    from PIL import Image

    from agentlightning.client import AgentLightningSyncClient
    from agentlightning.server.app import create_app
    from agentlightning.verl.rollout_adapter import RolloutAdapter

    class ImageProcessor:
        def __call__(self, text: object, images: list, return_tensors: str) -> dict:
            return {"pixel_values": torch.tensor([image.getpixel((0, 0)) for image in images])}

    urls = []
    for color in ("red", "blue"):
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
        urls.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())

    app = create_app(
        {
            "key": "test",
            "default_proxy": {"model_name": "test-model", "train": {"temperature": 1}, "val": {"temperature": 0}},
        }
    )
    with TestClient(app, headers={"Authorization": "Bearer test"}) as client:
        rollout_id = client.post("/api/rollouts", json=[{"input": {}}]).json()[0]["rollout_id"]
        try:
            client.patch(
                f"/api/rollouts/{rollout_id}", json={"status": {"state": "running", "last_attempt_id": "0"}}
            ).raise_for_status()
            for index, url in enumerate([urls[0], urls[0], urls[1]]):
                event = _raw_model_request_event([url], [1, 2], [index + 3])
                event.data["response"]["choices"][0]["logprobs"] = {"content": [{"logprob": -0.1}]}
                if routed_experts:
                    buffer = io.BytesIO()
                    np.save(buffer, np.array([index * 10, index * 10 + 1], dtype=np.uint8).reshape(2, 1, 1))
                    event.data["routed_experts"] = base64.b64encode(buffer.getvalue()).decode()
                response = client.post(
                    f"/api/rollouts/{rollout_id}/attempt/0/events",
                    json={"event_type": "model_request", "data": event.data},
                )
                assert response.status_code == 200
            client.post(
                f"/api/rollouts/{rollout_id}/attempt/0/events",
                json={"event_type": "reward", "data": {"value": 0.75}},
            ).raise_for_status()
            client.patch(
                f"/api/rollouts/{rollout_id}", json={"status": {"state": "succeeded", "last_attempt_id": "0"}}
            ).raise_for_status()
            manager = object.__new__(AglRolloutManagerBase)
            manager.client = cast(AgentLightningSyncClient, client)
            completed = manager._build_completed_rollout(
                EnqueuedRollout(data_id="sample", rollout_id=rollout_id, step=0, sample_idx_in_step=0, enqueue_time=0),
                manager._get_rollout(rollout_id),
            )
            assert completed.triplets is not None
            assert [triplet.image_urls for triplet in completed.triplets] == [[urls[0]], [urls[1]]]

            adapter = RolloutAdapter(
                max_prompt_length=4,
                max_response_length=2,
                device=torch.device("cpu"),
                pad_token_id=0,
                processor=ImageProcessor(),
                require_routed_experts=routed_experts,
            )
            batch, _ = adapter.get_train_data_batch([completed])
            assert batch.batch is not None
            assert batch.batch["responses"].tolist() == [[4, 0], [5, 0]]
            assert batch.batch["is_drop_mask"].tolist() == [False, False]
            assert batch.batch["token_level_scores"].sum(-1).tolist() == [0.75, 0.75]
            torch.testing.assert_close(batch.batch["rollout_log_probs"], torch.tensor([[-0.1, 0.0], [-0.1, 0.0]]))
            assert [row["pixel_values"].tolist() for row in batch.non_tensor_batch["multi_modal_inputs"]] == [
                [[255, 0, 0]],
                [[0, 0, 255]],
            ]
            if routed_experts:
                assert batch.batch["routed_experts"][:, :, 0, 0].tolist() == [
                    [0, 0, 10, 11, 0, 0],
                    [0, 0, 20, 21, 0, 0],
                ]
                raw_events = client.get(f"/api/rollouts/{rollout_id}/events").json()
                assert ["routed_experts" in event["data"] for event in raw_events[:3]] == [False, True, True]
        finally:
            client.delete(f"/api/rollouts/{rollout_id}").raise_for_status()
