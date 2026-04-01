"""agl-lite CLI — serve and controller entrypoints."""

from __future__ import annotations

import typer
import uvicorn

app = typer.Typer(name="agl-lite", help="Minimal agentic RL infrastructure.")


@app.command()
def serve(
    host: str = typer.Option(..., help="Bind host"),
    port: int = typer.Option(..., help="Bind port"),
    gateway_config: str | None = typer.Option(None, help="Path to gateway YAML config"),
    hooks: str | None = typer.Option(None, help="Path to Python module with RolloutHooks subclass"),
    artifact_dir: str | None = typer.Option(None, help="Directory for artifact files"),
) -> None:
    """Start the agl-lite HTTP service (store + gateway)."""
    from agl_lite.server.app import create_app
    from agl_lite.server.config import ServerSettings

    settings = ServerSettings(gateway_config=gateway_config, hooks=hooks, artifact_dir=artifact_dir)
    application = create_app(settings)
    uvicorn.run(application, host=host, port=port, workers=1)


@app.command()
def controller(
    job_manifest_template: str = typer.Option(..., help="Path to Jinja2 job manifest template (AGL_JOB_MANIFEST_TEMPLATE)"),
) -> None:
    """Start the K8s controller (reconcile loop)."""
    import asyncio

    from agl_lite.client import AglLiteClient
    from agl_lite.controller.config import ControllerSettings
    from agl_lite.controller.reconciler import Reconciler

    settings = ControllerSettings(job_manifest_template=job_manifest_template)

    async def _run() -> None:
        api = AglLiteClient(base_url=settings.base_url, agl_key=settings.key or None)
        # kr8s client created here in production; for now use a placeholder.
        # The real kr8s adapter will be added in Phase 4 (E2E validation).
        try:
            from agl_lite.controller.kr8s_adapter import Kr8sClient

            k8s = Kr8sClient(namespace=settings.namespace)
        except ImportError:
            typer.echo("kr8s adapter not yet implemented — install kr8s and implement Kr8sClient")
            raise typer.Exit(code=1) from None

        reconciler = Reconciler(api=api, k8s=k8s, settings=settings)
        try:
            await reconciler.run()
        finally:
            await api.close()

    asyncio.run(_run())


@app.command("deploy")
def deploy_entrypoint(
    env_file: str = typer.Option(..., "--env-file", help="Path to .env deploy config file"),
    cleanup: bool = typer.Option(False, "--cleanup", help="Delete namespace and stop managed host service"),
) -> None:
    """Deploy controller/server in k8s/host/external modes from a .env config file."""
    from agl_lite.deploy import deploy_command

    deploy_command(env_file=env_file, cleanup=cleanup)


if __name__ == "__main__":
    app()
