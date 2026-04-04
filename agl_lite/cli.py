"""agl-lite CLI — serve and controller entrypoints."""

from __future__ import annotations

import typer
import uvicorn

app = typer.Typer(name="agl-lite", help="Minimal agentic RL infrastructure.")


@app.command()
def serve(
    host: str = typer.Option(..., help="Bind host"),
    port: int = typer.Option(..., help="Bind port"),
    key: str = typer.Option("", envvar="AGL_KEY", help="Shared API key (empty = auth disabled)"),
    gateway_config: str | None = typer.Option(None, envvar="AGL_GATEWAY_CONFIG", help="Path to gateway YAML config"),
    hooks: str | None = typer.Option(None, envvar="AGL_HOOKS", help="Path to Python module with RolloutHooks subclass"),
    log_dir: str | None = typer.Option(None, envvar="AGL_LOG_DIR", help="Directory for log files and default archive (AGL_LOG_DIR)"),
    log_level: str = typer.Option("INFO", envvar="AGL_LOG_LEVEL", help="Log level: DEBUG / INFO / WARNING / ERROR"),
) -> None:
    """Start the agl-lite HTTP service (store + gateway)."""
    from agl_lite.logging_config import configure_logging
    from agl_lite.server.app import create_app
    from agl_lite.server.config import ServerSettings

    configure_logging(log_dir=log_dir, log_level=log_level, component="server")

    settings = ServerSettings(key=key, gateway_config=gateway_config, hooks=hooks, log_dir=log_dir)
    application = create_app(settings)
    uvicorn.run(application, host=host, port=port, workers=1)


@app.command()
def controller(
    base_url: str = typer.Option(..., envvar="AGL_BASE_URL", help="agl-lite server base URL"),
    namespace: str = typer.Option(..., envvar="AGL_NAMESPACE", help="K8s namespace for agent Jobs"),
    job_manifest_template: str = typer.Option(..., envvar="AGL_JOB_MANIFEST_TEMPLATE", help="Path to Jinja2 job manifest template"),
    key: str = typer.Option("", envvar="AGL_KEY", help="Shared API key (empty = auth disabled)"),
    poll_interval: int = typer.Option(10, envvar="AGL_POLL_INTERVAL", help="Seconds between reconcile cycles"),
    max_queue_time: int = typer.Option(3600, envvar="AGL_MAX_QUEUE_TIME", help="Max seconds a rollout stays in queuing"),
    ttl_after_finished: int = typer.Option(3600, envvar="AGL_TTL_AFTER_FINISHED", help="ttlSecondsAfterFinished on Jobs"),
) -> None:
    """Start the K8s controller (reconcile loop)."""
    import asyncio

    from agl_lite.client import AglLiteClient
    from agl_lite.controller.config import ControllerSettings
    from agl_lite.controller.reconciler import Reconciler

    settings = ControllerSettings(
        base_url=base_url,
        namespace=namespace,
        job_manifest_template=job_manifest_template,
        key=key,
        poll_interval=poll_interval,
        max_queue_time=max_queue_time,
        ttl_after_finished=ttl_after_finished,
    )

    async def _run() -> None:
        api = AglLiteClient(base_url=settings.base_url, agl_key=settings.key or None)
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
