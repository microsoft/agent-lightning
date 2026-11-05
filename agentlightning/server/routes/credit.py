# Copyright (c) Microsoft. All rights reserved.

"""Process-only evidence seal and immutable snapshot endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agentlightning.credit.state import AttemptCreditState
from agentlightning.credit.types import (
    CreditConflictError,
    CreditReceipt,
    CreditSnapshot,
    FinalizeCreditRequest,
    IncompletePolicyTraceError,
    ProcessCreditConfig,
    ProcessCreditContext,
)
from agentlightning.server.store import _credit_states, _rollouts

router = APIRouter(tags=["process-credit"])


def get_credit_state(rollout_id: str, attempt_id: str) -> AttemptCreditState | None:
    """Select explicit marked attempts; outcome requests never allocate credit state."""
    rollout = _rollouts.get(rollout_id)
    if rollout is None:
        raise HTTPException(status_code=404, detail=f"Rollout not found: {rollout_id}")
    context = rollout.metadata.model_dump().get("process_credit")
    if context is None:
        return None
    try:
        bound = ProcessCreditContext.model_validate(context)
        key = (rollout_id, attempt_id)
        if key not in _credit_states:
            _credit_states[key] = AttemptCreditState(
                rollout_id,
                attempt_id,
                policy_version=bound.policy_version,
                group_id=bound.group_id,
                group_size=bound.group_size,
                config=ProcessCreditConfig(enabled=True, **bound.limits.model_dump()),
            )
        state = _credit_states[key]
        if (state.policy_version, state.group_id, state.group_size, state.config.admission_limits) != (
            bound.policy_version,
            bound.group_id,
            bound.group_size,
            bound.limits,
        ):
            raise CreditConflictError("process admission binding changed after attempt creation")
    except ValueError as exc:
        raise credit_http_error(exc) from exc
    return _credit_states[key]


def credit_http_error(error: ValueError) -> HTTPException:
    """Conflicts/pending evidence cannot be mistaken for usable partial credit."""
    status = 409 if isinstance(error, CreditConflictError | IncompletePolicyTraceError) else 400
    return HTTPException(status_code=status, detail=str(error))


@router.post("/rollouts/{rollout_id}/attempt/{attempt_id}/credit/finalize", response_model=CreditReceipt)
async def finalize_credit(rollout_id: str, attempt_id: str, body: FinalizeCreditRequest) -> CreditReceipt:
    state = get_credit_state(rollout_id, attempt_id)
    if state is None:
        raise HTTPException(status_code=404, detail="rollout is not marked for process credit")
    try:
        return state.finalize(body)
    except ValueError as exc:
        raise credit_http_error(exc) from exc


@router.get("/rollouts/{rollout_id}/attempt/{attempt_id}/credit", response_model=CreditSnapshot)
async def query_credit(rollout_id: str, attempt_id: str) -> CreditSnapshot:
    state = get_credit_state(rollout_id, attempt_id)
    if state is None:
        raise HTTPException(status_code=404, detail="rollout is not marked for process credit")
    try:
        return state.snapshot()
    except ValueError as exc:
        raise credit_http_error(exc) from exc
