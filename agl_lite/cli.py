"""agl-lite CLI — serve and controller entrypoints."""

from __future__ import annotations

import typer
import uvicorn

app = typer.Typer(name="agl-lite", help="Minimal agentic RL infrastructure.")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8080, help="Bind port"),
    gateway_config: str = typer.Option("", help="Path to gateway YAML config"),
) -> None:
    """Start the agl-lite HTTP service (store + gateway)."""
    import os

    # Set env vars so ServerSettings picks them up.
    if gateway_config:
        os.environ.setdefault("GATEWAY_CONFIG", gateway_config)

    from agl_lite.server.app import create_app
    from agl_lite.server.config import ServerSettings

    settings = ServerSettings(host=host, port=port, gateway_config=gateway_config)
    application = create_app(settings)
    uvicorn.run(application, host=host, port=port, workers=1)


@app.command()
def controller() -> None:
    """Start the K8s controller (reconcile loop)."""
    typer.echo("K8s controller not yet implemented (Phase 3)")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
