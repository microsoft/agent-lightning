"""agl-client CLI — API consumer commands for querying and managing agl-lite."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import typer

from agl_lite.client import AglLiteClient, AglLiteError

app = typer.Typer(name="agl-client", help="agl-lite API client — query and manage rollouts, events, models, resources.")

# --- Subcommand groups ---
rollouts_app = typer.Typer(help="Manage rollouts.")
events_app = typer.Typer(help="Query events.")
models_app = typer.Typer(help="Manage model servers.")
resources_app = typer.Typer(help="Manage resources.")

app.add_typer(rollouts_app, name="rollouts")
app.add_typer(events_app, name="events")
app.add_typer(models_app, name="models")
app.add_typer(resources_app, name="resources")


# --- Helpers ---


def _url() -> str:
    return os.environ.get("AGL_LITE_URL", "http://localhost:8080")


def _key() -> str | None:
    k = os.environ.get("AGL_KEY", "")
    return k or None


def _client() -> AglLiteClient:
    return AglLiteClient(base_url=_url(), agl_key=_key())


def _run(coro: Any) -> Any:
    """Run an async coroutine and handle errors."""
    async def _wrapper():
        client = _client()
        try:
            return await coro(client)
        except AglLiteError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1) from None
        finally:
            await client.close()
    return asyncio.run(_wrapper())


def _print_json(data: Any) -> None:
    """Pretty-print JSON to stdout."""
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif isinstance(data, list) and data and hasattr(data[0], "model_dump"):
        data = [d.model_dump() for d in data]
    typer.echo(json.dumps(data, indent=2, default=str))


# --- Rollouts ---


@rollouts_app.command("list")
def rollouts_list(
    status: str = typer.Option("", help="Filter by status (comma-separated, e.g. 'queuing,running')"),
    ids: str = typer.Option("", help="Filter by IDs (comma-separated)"),
    limit: int = typer.Option(100, help="Max results"),
    offset: int = typer.Option(0, help="Offset"),
) -> None:
    """List rollouts with optional filters."""
    from agl_lite.schemas.rollout import RolloutStatus

    async def _do(client: AglLiteClient):
        status_in = [RolloutStatus(s.strip()) for s in status.split(",") if s.strip()] or None
        id_list = [i.strip() for i in ids.split(",") if i.strip()] or None
        return await client.query_rollouts(ids=id_list, status_in=status_in, limit=limit, offset=offset)

    result = _run(_do)
    _print_json(result)


@rollouts_app.command("get")
def rollouts_get(
    rollout_id: str = typer.Argument(help="Rollout ID"),
) -> None:
    """Get a single rollout by ID."""
    async def _do(client: AglLiteClient):
        return await client.get_rollout(rollout_id)

    _print_json(_run(_do))


@rollouts_app.command("cancel")
def rollouts_cancel(
    rollout_id: str = typer.Argument(help="Rollout ID"),
) -> None:
    """Cancel a rollout."""
    async def _do(client: AglLiteClient):
        return await client.cancel_rollout(rollout_id)

    _print_json(_run(_do))


# --- Events ---


@events_app.command("list")
def events_list(
    rollout_id: str = typer.Option(..., help="Rollout ID (required)"),
    attempt_id: str = typer.Option("", help="Filter by attempt ID"),
    event_type: str = typer.Option("", help="Filter by event type"),
    limit: int = typer.Option(1000, help="Max results"),
    offset: int = typer.Option(0, help="Offset"),
) -> None:
    """List events for a rollout."""
    async def _do(client: AglLiteClient):
        return await client.get_events(
            rollout_id,
            attempt_id=attempt_id or None,
            event_type=event_type or None,
            limit=limit,
            offset=offset,
        )

    _print_json(_run(_do))


# --- Models ---


@models_app.command("list")
def models_list() -> None:
    """List all registered model servers."""
    async def _do(client: AglLiteClient):
        return await client.list_models()

    _print_json(_run(_do))


@models_app.command("register")
def models_register(
    model: str = typer.Option(..., help="Model name"),
    endpoint: str = typer.Option(..., help="Server endpoint URL"),
    version: int = typer.Option(0, help="Model version"),
    token: str = typer.Option("", help="Optional auth token for gateway → model server"),
) -> None:
    """Register a model inference server."""
    from agl_lite.schemas.api import RegisterModelRequest

    async def _do(client: AglLiteClient):
        req = RegisterModelRequest(model=model, endpoint=endpoint, version=version, token=token or None)
        return await client.register_models([req])

    _print_json(_run(_do))


@models_app.command("delete")
def models_delete(
    model: str = typer.Argument(help="Model name to delete"),
    endpoints: str = typer.Option("", help="Specific endpoints to remove (comma-separated). Empty = remove all."),
) -> None:
    """Delete model servers. Removes specific endpoints or entire pool."""
    async def _do(client: AglLiteClient):
        ep_list = [e.strip() for e in endpoints.split(",") if e.strip()] or None
        await client.delete_model(model, endpoints=ep_list)
        typer.echo(f"Deleted model servers for '{model}'")

    _run(_do)


@models_app.command("delete-all")
def models_delete_all() -> None:
    """Delete ALL model servers."""
    async def _do(client: AglLiteClient):
        await client.delete_all_models()
        typer.echo("Deleted all model servers")

    _run(_do)


# --- Resources ---


@resources_app.command("get")
def resources_get(
    resources_id: str = typer.Argument(help="Resources snapshot ID"),
) -> None:
    """Get a specific resources snapshot."""
    async def _do(client: AglLiteClient):
        return await client.get_resources(resources_id)

    _print_json(_run(_do))


@resources_app.command("latest")
def resources_latest() -> None:
    """Get the latest resources snapshot."""
    async def _do(client: AglLiteClient):
        result = await client.get_latest_resources()
        if result is None:
            typer.echo("No resources found")
            raise typer.Exit(code=0)
        return result

    _print_json(_run(_do))


@resources_app.command("add")
def resources_add(
    data: str = typer.Argument(help="JSON string or @filepath for resources data"),
) -> None:
    """Add a new resources snapshot. Pass JSON string or @path/to/file.json."""
    if data.startswith("@"):
        with open(data[1:]) as f:
            payload = json.load(f)
    else:
        payload = json.loads(data)

    async def _do(client: AglLiteClient):
        return await client.add_resources(payload)

    _print_json(_run(_do))


# --- Health ---


@app.command("health")
def health() -> None:
    """Check agl-lite service health."""
    import httpx

    try:
        resp = httpx.get(f"{_url()}/healthz", timeout=5)
        if resp.status_code == 200:
            typer.echo(f"OK ({_url()})")
        else:
            typer.echo(f"Unhealthy: {resp.status_code}", err=True)
            raise typer.Exit(code=1)
    except httpx.ConnectError:
        typer.echo(f"Cannot connect to {_url()}", err=True)
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
