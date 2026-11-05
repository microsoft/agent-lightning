# Copyright (c) Microsoft. All rights reserved.

"""Single-event-loop attempt identity, idempotent delivery and atomic evidence seals."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from agentlightning.credit.types import (
    CreditCompletion,
    CreditConflictError,
    CreditEnvelope,
    CreditProtocolError,
    CreditReceipt,
    CreditRecord,
    CreditSnapshot,
    FinalizeCreditRequest,
    IncompletePolicyTraceError,
    ProcessCreditConfig,
    canonical_json,
    finite_number,
    nonempty,
    semantic_digest,
)
from agentlightning.schemas import Event


def successful_action(record: CreditRecord) -> bool:
    """HTTP failures and empty responses do not create sampled action indices."""
    data = record.data
    response = data.get("response", {})
    choices = response.get("choices", []) if isinstance(response, dict) else []
    tokens = choices[0].get("token_ids", []) if choices and isinstance(choices[0], dict) else []
    return (
        record.event_type == "model_request"
        and data.get("status") != "error"
        and data.get("http_status", 200) < 300
        and bool(tokens)
    )


def validate_snapshot(snapshot: CreditSnapshot) -> tuple[tuple[CreditRecord, ...], tuple[CreditRecord, ...]]:
    """Validate complete identity/reference membership independently of transport timestamps."""
    snapshot = CreditSnapshot.model_validate(snapshot.model_dump(mode="json"))
    identities = set()
    sequences = []
    actions = []
    rewards = []
    for record in snapshot.events:
        envelope = record.envelope
        if record.rollout_id != snapshot.rollout_id or record.attempt_id != snapshot.attempt_id:
            raise IncompletePolicyTraceError("foreign-attempt event in credit snapshot")
        if envelope.policy_version != snapshot.policy_version or envelope.sequence is None:
            raise IncompletePolicyTraceError("policy version or sequence missing from snapshot")
        if envelope.event_id in identities:
            raise IncompletePolicyTraceError("duplicate event identity in snapshot")
        identities.add(envelope.event_id)
        sequences.append(envelope.sequence)
        if record.event_type == "model_request":
            if envelope.kind is not None:
                raise IncompletePolicyTraceError("model request has reward identity")
            if successful_action(record):
                actions.append(record)
        else:
            if envelope.kind is None:
                raise IncompletePolicyTraceError("reward has no declared kind")
            rewards.append(record)
    if sequences != sorted(set(sequences)) or max(sequences, default=-1) != snapshot.watermark:
        raise IncompletePolicyTraceError("snapshot sequence order/watermark is inconsistent")
    action_ids = tuple(record.envelope.event_id for record in actions)
    reward_ids = tuple(sorted(record.envelope.event_id for record in rewards))
    if action_ids != snapshot.action_event_ids or reward_ids != snapshot.reward_event_ids:
        raise IncompletePolicyTraceError("manifest must contain every original action and reward exactly once")
    ordinals = [record.envelope.action_index for record in actions]
    if ordinals != sorted(set(ordinals)):
        raise IncompletePolicyTraceError("original logical action order is inconsistent")
    indices = {event_id: index for index, event_id in enumerate(action_ids)}
    terminals = []
    for record in rewards:
        envelope = record.envelope
        nonempty(record.data.get("source"))
        if envelope.kind == "terminal":
            terminals.append(envelope.event_id)
            if envelope.evidence_through_event_id != (action_ids[-1] if action_ids else None):
                raise IncompletePolicyTraceError("terminal evidence must reference the final successful action")
        elif envelope.target_event_id not in indices or envelope.evidence_through_event_id not in indices:
            raise IncompletePolicyTraceError("unresolved, foreign or failed policy action reference")
        elif indices[envelope.evidence_through_event_id] < indices[envelope.target_event_id]:
            raise IncompletePolicyTraceError("evidence cutoff precedes attribution target")
    if terminals != [snapshot.terminal_reward_id]:
        raise IncompletePolicyTraceError("exactly one explicit terminal reward is required")
    return tuple(actions), tuple(rewards)


@dataclass(frozen=True)
class RequestLease:
    """Original logical call identity and shared completion; duplicates are not owners."""

    request_id: str
    action_index: int
    request_digest: str
    event_id: str
    future: asyncio.Future[CreditCompletion]
    owner: bool = True


class AttemptCreditState:
    """Bounded attempt-local state. All mutation completes without awaiting.

    The server uses one event-loop thread, matching the native store concurrency
    contract. Completion futures coalesce retries; evidence seals freeze writes
    but do not claim durable recovery across process loss.
    """

    def __init__(
        self,
        rollout_id: str,
        attempt_id: str,
        *,
        policy_version: int,
        group_id: str,
        group_size: int,
        config: ProcessCreditConfig | None = None,
    ) -> None:
        self.rollout_id = nonempty(rollout_id)
        self.attempt_id = nonempty(attempt_id)
        self.group_id = nonempty(group_id)
        if type(policy_version) is not int or policy_version < 0 or type(group_size) is not int or group_size < 1:
            raise CreditProtocolError("invalid process policy version or scheduled group size")
        self.policy_version = policy_version
        self.group_size = group_size
        self.config = config or ProcessCreditConfig(enabled=True)
        self._requests: dict[str, RequestLease] = {}
        self._events: dict[str, Event] = {}
        self._reward_digests: dict[str, str] = {}
        self._operations: set[asyncio.Task[Any]] = set()
        self._inflight: str | None = None
        self._incomplete: str | None = None
        self._sealed: CreditSnapshot | None = None
        self._manifest: FinalizeCreditRequest | None = None
        self._bytes = 0
        self._action_count = 0
        self._reserved_ids: set[str] = set()

    @property
    def event_count(self) -> int:
        return len(self._events)

    def retain_operation(self, task: asyncio.Task[Any]) -> None:
        """Retain shared capture even if a waiting HTTP caller disconnects."""
        self._operations.add(task)
        task.add_done_callback(self._operations.discard)

    def close(self) -> None:
        """Deleting a rollout cancels its owned capture operations and drops state."""
        for operation in tuple(self._operations):
            operation.cancel()

    def begin_request(self, request_id: str, action_index: int, request_digest: str) -> RequestLease:
        request_id = nonempty(request_id)
        if previous := self._requests.get(request_id):
            if previous.action_index != action_index or previous.request_digest != request_digest:
                raise CreditConflictError("request identity reused with conflicting body/path/index")
            return replace(previous, owner=False)
        if self._sealed is not None or self._incomplete is not None:
            raise CreditConflictError("attempt is sealed or incomplete")
        if self._inflight is not None:
            raise CreditConflictError("concurrent distinct policy generations are unsupported")
        if type(action_index) is not int or action_index != len(self._requests):
            raise CreditConflictError("logical action index must be the next original request ordinal")
        if (
            self._action_count >= self.config.max_actions_per_rollout
            or self.event_count >= self.config.max_events_per_rollout
        ):
            raise CreditProtocolError("process action/event limit reached before generation")
        lease = RequestLease(
            request_id, action_index, request_digest, uuid4().hex, asyncio.get_running_loop().create_future()
        )
        self._requests[request_id] = lease
        self._reserved_ids.add(lease.event_id)
        self._inflight = request_id
        return lease

    def _append(self, event: Event) -> None:
        size = len(canonical_json(event.model_dump(exclude={"timestamp"})))
        if (
            self.event_count >= self.config.max_events_per_rollout
            or self._bytes + size > self.config.max_artifact_bytes
        ):
            raise CreditProtocolError("process evidence exceeds configured event/byte limit")
        self._events[event.data["_agl_credit"]["event_id"]] = event.model_copy(deep=True)
        self._bytes += size

    def complete_request(
        self, lease: RequestLease, *, event: Event, response_body: dict[str, Any], status_code: int
    ) -> CreditCompletion:
        if self._requests.get(lease.request_id) != lease or not lease.owner or self._inflight != lease.request_id:
            raise CreditConflictError("invalid or already completed policy request lease")
        if (
            event.rollout_id != self.rollout_id
            or event.attempt_id != self.attempt_id
            or event.event_type != "model_request"
        ):
            raise IncompletePolicyTraceError("captured response belongs to a foreign attempt")
        if event.data.get("model_version") != self.policy_version:
            raise IncompletePolicyTraceError("captured serving policy version differs from collection")
        envelope = CreditEnvelope(
            event_id=lease.event_id,
            policy_version=self.policy_version,
            sequence=self.event_count,
            request_id=lease.request_id,
            action_index=lease.action_index,
        )
        captured = event.model_copy(
            update={"data": {**copy.deepcopy(event.data), "_agl_credit": envelope.model_dump(exclude_none=True)}}
        )
        self._append(captured)
        self._action_count += int(successful_action(CreditRecord(**captured.model_dump(exclude={"timestamp"}))))
        completion = CreditCompletion(
            body=copy.deepcopy(response_body),
            status_code=status_code,
            event_id=lease.event_id,
            sequence=envelope.sequence,
            action_index=lease.action_index,
            request_id=lease.request_id,
            policy_version=self.policy_version,
        )
        self._inflight = None
        lease.future.set_result(completion)
        return completion

    def fail_request(self, lease: RequestLease, reason: str) -> None:
        """An ambiguous uncaptured sample makes the whole attempt untrainable."""
        self._incomplete = reason
        self._inflight = None
        if not lease.future.done():
            lease.future.set_exception(IncompletePolicyTraceError(reason))
            lease.future.exception()  # Owned failures are also observed when every waiter disconnected.

    def append_reward(self, payload: dict[str, Any]) -> Event:
        if set(payload) - {"value", "source", "reason", "_agl_credit"}:
            raise CreditProtocolError("unknown process reward fields")
        envelope = CreditEnvelope.model_validate(payload.get("_agl_credit"))
        if envelope.kind is None or envelope.policy_version != self.policy_version:
            raise CreditProtocolError("reward kind/policy version does not match attempt")
        value = payload.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise CreditProtocolError("reward must be a finite number")
        finite_number(value, "raw reward")
        normalized = {
            "value": float(value),
            "source": nonempty(payload.get("source")),
            "_agl_credit": envelope.model_dump(exclude_none=True, exclude={"sequence"}),
        }
        if payload.get("reason") is not None:
            if not isinstance(payload["reason"], str):
                raise CreditProtocolError("reward reason must be a string")
            normalized["reason"] = payload["reason"]
        digest = semantic_digest(normalized)
        if previous := self._reward_digests.get(envelope.event_id):
            if digest != previous:
                raise CreditConflictError("reward identity reused with conflicting semantic payload")
            return self.event(envelope.event_id)
        if self._sealed is not None:
            raise CreditConflictError("credit-affecting writes are sealed")
        if envelope.event_id in self._events or envelope.event_id in self._reserved_ids:
            raise CreditConflictError("reward identity collides with a captured action")
        normalized["_agl_credit"]["sequence"] = self.event_count
        event = Event(
            event_type="reward",
            rollout_id=self.rollout_id,
            attempt_id=self.attempt_id,
            timestamp=time.time(),
            data=normalized,
        )
        self._append(event)
        self._reward_digests[envelope.event_id] = digest
        return event.model_copy(deep=True)

    def event(self, event_id: str) -> Event:
        return self._events[event_id].model_copy(deep=True)

    def finalize(self, request: FinalizeCreditRequest) -> CreditReceipt:
        if self._sealed is not None:
            if request != self._manifest:
                raise CreditConflictError("finalization manifest differs from the sealed attempt")
            return self._receipt()
        if self._inflight is not None or self._incomplete is not None:
            raise IncompletePolicyTraceError("policy requests are in flight or uncaptured")
        if request.policy_version != self.policy_version:
            raise IncompletePolicyTraceError("finalization policy version differs from collection")
        payload = {
            "version": 1,
            "rollout_id": self.rollout_id,
            "attempt_id": self.attempt_id,
            "policy_version": self.policy_version,
            "group_id": self.group_id,
            "group_size": self.group_size,
            "action_event_ids": request.action_event_ids,
            "reward_event_ids": request.reward_event_ids,
            "terminal_reward_id": request.terminal_reward_id,
            "watermark": self.event_count - 1,
            "events": [event.model_dump(mode="json", exclude={"timestamp"}) for event in self._events.values()],
        }
        candidate = CreditSnapshot.model_validate({**payload, "snapshot_hash": semantic_digest(payload)})
        validate_snapshot(candidate)
        self._sealed = candidate
        self._manifest = request
        return self._receipt()

    def _receipt(self) -> CreditReceipt:
        snapshot = self.snapshot()
        return CreditReceipt(
            rollout_id=self.rollout_id,
            attempt_id=self.attempt_id,
            policy_version=self.policy_version,
            snapshot_hash=snapshot.snapshot_hash,
            watermark=snapshot.watermark,
            action_count=len(snapshot.action_event_ids),
            reward_count=len(snapshot.reward_event_ids),
        )

    def snapshot(self) -> CreditSnapshot:
        if self._sealed is None:
            raise IncompletePolicyTraceError("credit snapshot is pending finalization")
        return CreditSnapshot.model_validate(self._sealed.model_dump(mode="json"))
