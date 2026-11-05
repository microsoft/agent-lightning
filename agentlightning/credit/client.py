# Copyright (c) Microsoft. All rights reserved.

"""Explicit attempt-bound clients for idempotent generation, rewards and sealing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import httpx

from agentlightning.credit.types import (
    CreditCompletion,
    CreditConflictError,
    CreditEnvelope,
    CreditProtocolError,
    CreditReceipt,
    FinalizeCreditRequest,
    event_identifier,
    finite_float32,
    nonempty,
)
from agentlightning.schemas import Event


class _CreditManifest:
    """Shared transport-neutral protocol bookkeeping; never discovers ambient attempts."""

    def __init__(self, rollout_id: str, attempt_id: str, *, policy_version: int) -> None:
        self.rollout_id = nonempty(rollout_id)
        self.attempt_id = nonempty(attempt_id)
        self.policy_version = policy_version
        self._next_index = 0
        self._requests: dict[str, int] = {}
        self._pending: str | None = None
        self._actions: list[str] = []
        self._action_set: set[str] = set()
        self._rewards: set[str] = set()
        self._terminal: str | None = None

    @property
    def _events_path(self) -> str:
        return f"/api/rollouts/{self.rollout_id}/attempt/{self.attempt_id}/events"

    @property
    def _credit_path(self) -> str:
        return f"/api/rollouts/{self.rollout_id}/attempt/{self.attempt_id}/credit"

    @property
    def _proxy_path(self) -> str:
        return f"/proxy/rollout/{self.rollout_id}/attempt/{self.attempt_id}/mode/train/openai/v1/chat/completions"

    def _request_headers(self, request_id: str | None) -> tuple[str, dict[str, str]]:
        request_id = nonempty(request_id or uuid4().hex)
        if self._pending is not None and request_id != self._pending:
            raise CreditConflictError("retry the unresolved logical request ID before sampling another action")
        index = self._requests.setdefault(request_id, self._next_index)
        if index == self._next_index:
            self._pending = request_id
        return request_id, {"X-AGL-Request-ID": request_id, "X-AGL-Action-Index": str(index)}

    def _completion(self, response: httpx.Response, request_id: str) -> CreditCompletion:
        if "X-AGL-Event-ID" not in response.headers:
            response.raise_for_status()
            raise CreditProtocolError("model response is missing its captured original event identity")
        event_id = event_identifier(response.headers["X-AGL-Event-ID"])
        completion = CreditCompletion(
            body=response.json(),
            status_code=response.status_code,
            event_id=event_id,
            sequence=int(response.headers["X-AGL-Event-Sequence"]),
            action_index=self._requests[request_id],
            request_id=request_id,
            policy_version=self.policy_version,
        )
        self._next_index = max(self._next_index, completion.action_index + 1)
        if self._pending == request_id:
            self._pending = None
        choices = completion.body.get("choices", [])
        if (
            200 <= completion.status_code < 300
            and choices
            and choices[0].get("token_ids")
            and event_id not in self._action_set
        ):
            self._actions.append(event_id)
            self._action_set.add(event_id)
        return completion

    def _reward(
        self,
        value: float,
        *,
        reward_id: str,
        kind: str,
        source: str,
        target_event_id: str | None = None,
        evidence_through_event_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        reward_id = event_identifier(reward_id)
        finite_float32(value, "reward")
        if kind == "terminal" and self._terminal is not None and self._terminal != reward_id:
            raise CreditConflictError("one stable terminal reward ID is permitted per attempt")
        envelope = CreditEnvelope(
            event_id=reward_id,
            policy_version=self.policy_version,
            kind=kind,
            target_event_id=target_event_id,
            evidence_through_event_id=evidence_through_event_id,
        )
        data = {"value": value, "source": nonempty(source), "_agl_credit": envelope.model_dump(exclude_none=True)}
        if reason is not None:
            data["reason"] = reason
        return {"event_type": "reward", "data": data}

    def _reward_response(self, response: httpx.Response) -> Event:
        response.raise_for_status()
        event = Event.model_validate(response.json())
        envelope = CreditEnvelope.model_validate(event.data.get("_agl_credit"))
        self._rewards.add(envelope.event_id)
        if envelope.kind == "terminal":
            self._terminal = envelope.event_id
        return event

    def _manifest(self) -> dict[str, Any]:
        if self._pending is not None or self._terminal is None:
            raise CreditProtocolError("finalization requires captured actions and an explicit terminal reward")
        return FinalizeCreditRequest(
            action_event_ids=tuple(self._actions),
            reward_event_ids=tuple(self._rewards),
            terminal_reward_id=self._terminal,
            policy_version=self.policy_version,
        ).model_dump(mode="json")


class ProcessCreditClient(_CreditManifest):
    """Use a native synchronous HTTP client with an explicitly selected rollout/attempt.

    Caller-supplied request/reward UUIDs survive lost replies. Complete responses
    retain original model text/status; rewards attest their information cutoff,
    not merely the action they judge. No implicit grading or policy sampling occurs.
    """

    def __init__(self, client: httpx.Client, rollout_id: str, attempt_id: str, *, policy_version: int) -> None:
        super().__init__(rollout_id, attempt_id, policy_version=policy_version)
        self._client = client

    def complete(self, body: Mapping[str, Any], *, request_id: str | None = None) -> CreditCompletion:
        request_id, headers = self._request_headers(request_id)
        return self._completion(self._client.post(self._proxy_path, json=dict(body), headers=headers), request_id)

    def emit_process_reward(
        self,
        value: float,
        *,
        reward_id: str,
        target_event_id: str,
        evidence_through_event_id: str,
        source: str,
        reason: str | None = None,
    ) -> Event:
        payload = self._reward(
            value,
            reward_id=reward_id,
            kind="process",
            target_event_id=target_event_id,
            evidence_through_event_id=evidence_through_event_id,
            source=source,
            reason=reason,
        )
        return self._reward_response(self._client.post(self._events_path, json=payload))

    def emit_terminal_reward(self, value: float, *, reward_id: str, source: str, reason: str | None = None) -> Event:
        payload = self._reward(
            value,
            reward_id=reward_id,
            kind="terminal",
            source=source,
            reason=reason,
            evidence_through_event_id=self._actions[-1] if self._actions else None,
        )
        return self._reward_response(self._client.post(self._events_path, json=payload))

    def finalize(self) -> CreditReceipt:
        response = self._client.post(self._credit_path + "/finalize", json=self._manifest())
        response.raise_for_status()
        return CreditReceipt.model_validate(response.json())


class AsyncProcessCreditClient(_CreditManifest):
    """Async counterpart with exactly the same explicit identities and manifest semantics."""

    def __init__(self, client: httpx.AsyncClient, rollout_id: str, attempt_id: str, *, policy_version: int) -> None:
        super().__init__(rollout_id, attempt_id, policy_version=policy_version)
        self._client = client

    async def acomplete(self, body: Mapping[str, Any], *, request_id: str | None = None) -> CreditCompletion:
        request_id, headers = self._request_headers(request_id)
        return self._completion(await self._client.post(self._proxy_path, json=dict(body), headers=headers), request_id)

    async def aemit_process_reward(
        self,
        value: float,
        *,
        reward_id: str,
        target_event_id: str,
        evidence_through_event_id: str,
        source: str,
        reason: str | None = None,
    ) -> Event:
        payload = self._reward(
            value,
            reward_id=reward_id,
            kind="process",
            target_event_id=target_event_id,
            evidence_through_event_id=evidence_through_event_id,
            source=source,
            reason=reason,
        )
        return self._reward_response(await self._client.post(self._events_path, json=payload))

    async def aemit_terminal_reward(
        self, value: float, *, reward_id: str, source: str, reason: str | None = None
    ) -> Event:
        payload = self._reward(
            value,
            reward_id=reward_id,
            kind="terminal",
            source=source,
            reason=reason,
            evidence_through_event_id=self._actions[-1] if self._actions else None,
        )
        return self._reward_response(await self._client.post(self._events_path, json=payload))

    async def afinalize(self) -> CreditReceipt:
        response = await self._client.post(self._credit_path + "/finalize", json=self._manifest())
        response.raise_for_status()
        return CreditReceipt.model_validate(response.json())
