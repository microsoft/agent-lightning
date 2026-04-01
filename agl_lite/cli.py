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
    hooks: str = typer.Option("", help="Path to Python module with RolloutHooks subclass"),
    artifact_dir: str = typer.Option("", help="Directory for artifact files (default: /tmp/agl-artifacts)"),
) -> None:
    """Start the agl-lite HTTP service (store + gateway)."""
    import os

    # Resolve settings from CLI args first, then env vars as fallback.
    if not gateway_config:
        gateway_config = os.environ.get("GATEWAY_CONFIG", "")
    if not hooks:
        hooks = os.environ.get("HOOKS", "")
    if not artifact_dir:
        artifact_dir = os.environ.get("ARTIFACT_DIR", "")

    from agl_lite.server.app import create_app
    from agl_lite.server.config import ServerSettings

    settings = ServerSettings(host=host, port=port, gateway_config=gateway_config, hooks=hooks, artifact_dir=artifact_dir)
    application = create_app(settings)
    uvicorn.run(application, host=host, port=port, workers=1)


@app.command()
def controller(
    agl_base_url: str = typer.Option("http://localhost:8000", help="agl-lite server URL"),
    namespace: str = typer.Option("default", help="K8s namespace for agent Jobs"),
    secret_name: str = typer.Option("agl-api-keys", help="K8s Secret name with API keys"),
    job_manifest_template: str = typer.Option(..., help="Path to Jinja2 job manifest template file"),
) -> None:
    """Start the K8s controller (reconcile loop)."""
    import asyncio
    import os

    from agl_lite.client import AglLiteClient
    from agl_lite.controller.config import ControllerSettings
    from agl_lite.controller.reconciler import Reconciler

    settings = ControllerSettings(
        lite_url=agl_base_url,
        key=os.environ.get("AGL_KEY", ""),
        namespace=namespace,
        secret_name=secret_name,
        job_manifest_template=job_manifest_template,
    )

    async def _run() -> None:
        api = AglLiteClient(base_url=settings.lite_url, agl_key=settings.key or None)
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
    config: str = typer.Option(..., "--config", help="Path to deploy YAML config file"),
    cleanup: bool = typer.Option(False, "--cleanup", help="Delete namespace and stop managed host service"),
) -> None:
    """Deploy controller/server in k8s/host/external modes from YAML config."""
    from agl_lite.deploy import deploy_command

    deploy_command(config=config, cleanup=cleanup)


if __name__ == "__main__":
    app()
