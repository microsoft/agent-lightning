"""Typed async HTTP client for the agl-lite API.

Shared by the K8s controller and algorithm. Wraps all agl-lite API endpoints
with typed methods using the same Pydantic schemas as the server.
"""

from __future__ import annotations

from typing import Any

import httpx

from agl_lite.schemas.api import (
    ArchiveBackend,
    ArchiveRequest,
    ArchiveResult,
    EnqueueBatchRequest,
    EnqueueRolloutRequest,
    PatchRolloutRequest,
    PostEventRequest,
    RegisterModelRequest,
)
from agl_lite.schemas.event import Event
from agl_lite.schemas.model_server import ModelServer
from agl_lite.schemas.resources import ResourcesUpdate
from agl_lite.schemas.rollout import Rollout, RolloutStatus


class AglLiteError(Exception):
    """Error from agl-lite API."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class AglLiteClient:
    """Async HTTP client for the agl-lite API.

    Args:
        base_url: agl-lite server URL (e.g., "http://localhost:8000").
        agl_key: API key for authentication. None = no auth.
        client: Optional pre-configured httpx.AsyncClient (for testing).
    """

    def __init__(
        self,
        base_url: str,
        agl_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        headers: dict[str, str] = {}
        if agl_key:
            headers["Authorization"] = f"Bearer {agl_key}"
        self._client = client or httpx.AsyncClient(base_url=self._base_url, headers=headers, timeout=30.0)
        self._owns_client = client is None

    async def close(self) -> None:
        """Close the underlying HTTP client (only if we created it)."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AglLiteClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # --- Internal helpers ---

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise AglLiteError(resp.status_code, detail)

    # --- Rollouts ---

    async def enqueue_rollouts(self, rollouts: list[EnqueueRolloutRequest]) -> list[Rollout]:
        """Enqueue one or more rollouts."""
        body = EnqueueBatchRequest(rollouts=rollouts)
        resp = await self._client.post("/api/rollouts", json=body.model_dump())
        self._raise_for_status(resp)
        return [Rollout.model_validate(r) for r in resp.json()]

    async def query_rollouts(
        self,
        *,
        ids: list[str] | None = None,
        status_in: list[RolloutStatus] | None = None,
        cancel_requested: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Rollout]:
        """Query rollouts with optional filters."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if ids:
            params["ids"] = ",".join(ids)
        if status_in:
            params["status"] = ",".join(s.value for s in status_in)
        if cancel_requested is not None:
            params["cancel_requested"] = cancel_requested
        resp = await self._client.get("/api/rollouts", params=params)
        self._raise_for_status(resp)
        return [Rollout.model_validate(r) for r in resp.json()]

    async def get_rollout(self, rollout_id: str) -> Rollout:
        """Get a single rollout by ID."""
        resp = await self._client.get(f"/api/rollouts/{rollout_id}")
        self._raise_for_status(resp)
        data = resp.json()
        return Rollout.model_validate(data["rollout"])

    async def patch_rollout(self, rollout_id: str, patch: PatchRolloutRequest) -> Rollout:
        """Partial update a rollout. Only fields set on `patch` are sent."""
        resp = await self._client.patch(
            f"/api/rollouts/{rollout_id}",
            json=patch.model_dump(exclude_unset=True),
        )
        self._raise_for_status(resp)
        return Rollout.model_validate(resp.json())

    async def cancel_rollout(self, rollout_id: str) -> Rollout:
        """Set cancel_requested flag on a rollout."""
        resp = await self._client.post(f"/api/rollouts/{rollout_id}/cancel")
        self._raise_for_status(resp)
        return Rollout.model_validate(resp.json())

    async def archive_rollouts(
        self,
        rollout_ids: list[str],
        backend: ArchiveBackend | None = None,
    ) -> ArchiveResult:
        """Archive and purge rollouts from hot store."""
        body = ArchiveRequest(rollout_ids=rollout_ids, backend=backend)
        resp = await self._client.post("/api/rollouts/archive", json=body.model_dump())
        self._raise_for_status(resp)
        return ArchiveResult.model_validate(resp.json())

    # --- Events ---

    async def get_events(
        self,
        rollout_id: str,
        *,
        attempt_id: str | None = None,
        event_type: str | None = None,
        format: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Event]:
        """Query events for a rollout.

        Args:
            format: Set to "triplet" to trim events for RL training
                    (model_request → prompt/response token_ids only,
                     reward → scalar value only).
        """
        params: dict[str, Any] = {"rollout_id": rollout_id, "limit": limit, "offset": offset}
        if attempt_id:
            params["attempt_id"] = attempt_id
        if event_type:
            params["event_type"] = event_type
        if format:
            params["format"] = format
        resp = await self._client.get("/api/events", params=params)
        self._raise_for_status(resp)
        return [Event.model_validate(e) for e in resp.json()]

    async def post_event(self, rollout_id: str, attempt_id: str, event: PostEventRequest) -> Event:
        """Post an event (reward, user-defined) to a rollout attempt."""
        resp = await self._client.post(
            f"/rollout/{rollout_id}/attempt/{attempt_id}/events",
            json=event.model_dump(),
        )
        self._raise_for_status(resp)
        return Event.model_validate(resp.json())

    # --- Models ---

    async def register_models(self, models: list[RegisterModelRequest]) -> list[ModelServer]:
        """Register model inference servers (upsert by model+endpoint)."""
        resp = await self._client.post("/api/models", json=[m.model_dump() for m in models])
        self._raise_for_status(resp)
        return [ModelServer.model_validate(m) for m in resp.json()]

    async def list_models(self) -> list[ModelServer]:
        """List all registered model servers."""
        resp = await self._client.get("/api/models")
        self._raise_for_status(resp)
        return [ModelServer.model_validate(m) for m in resp.json()]

    async def delete_model(self, model: str, endpoints: list[str] | None = None) -> None:
        """Delete servers for a model. If endpoints given, delete only those."""
        if endpoints:
            resp = await self._client.request("DELETE", f"/api/models/{model}", json={"endpoints": endpoints})
        else:
            resp = await self._client.delete(f"/api/models/{model}")
        self._raise_for_status(resp)

    async def delete_all_models(self) -> None:
        """Delete all model servers."""
        resp = await self._client.delete("/api/models")
        self._raise_for_status(resp)

    # --- Resources ---

    async def add_resources(self, resources: dict[str, Any]) -> ResourcesUpdate:
        """Add a new resource snapshot."""
        resp = await self._client.post("/api/resources", json=resources)
        self._raise_for_status(resp)
        return ResourcesUpdate.model_validate(resp.json())

    async def get_resources(self, resources_id: str) -> ResourcesUpdate:
        """Get a specific resource snapshot by ID."""
        resp = await self._client.get(f"/api/resources/{resources_id}")
        self._raise_for_status(resp)
        return ResourcesUpdate.model_validate(resp.json())

    async def get_latest_resources(self) -> ResourcesUpdate | None:
        """Get the latest resource snapshot. Returns None if none exist."""
        resp = await self._client.get("/api/resources/latest")
        self._raise_for_status(resp)
        data = resp.json()
        return ResourcesUpdate.model_validate(data) if data else None
