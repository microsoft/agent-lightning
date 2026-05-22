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
    admin_key: str | None = typer.Option(
        None,
        envvar="AGL_ADMIN_KEY",
        help=(
            "Admin-only API key for /admin/gateway/* routes (used by the trainer for "
            "async-rollout pause/drain). Required whenever AGL_KEY is set."
        ),
    ),
    gateway_config: str | None = typer.Option(None, envvar="AGL_GATEWAY_CONFIG", help="Path to gateway YAML config"),
    hooks: str | None = typer.Option(None, envvar="AGL_HOOKS", help="Path to Python module with RolloutHooks subclass"),
    log_dir: str | None = typer.Option(
        None,
        envvar="AGL_LOG_DIR",
        help="Directory for log files and default archive (AGL_LOG_DIR)",
    ),
    log_level: str = typer.Option("INFO", envvar="AGL_LOG_LEVEL", help="Log level: DEBUG / INFO / WARNING / ERROR"),
) -> None:
    """Start the agl-lite HTTP service (store + gateway)."""
    from agl_lite.logging_config import configure_logging
    from agl_lite.server.app import create_app
    from agl_lite.server.config import ServerSettings

    configure_logging(log_dir=log_dir, log_level=log_level, component="server")

    settings = ServerSettings(
        key=key,
        admin_key=admin_key,
        gateway_config=gateway_config,
        hooks=hooks,
        log_dir=log_dir,
    )
    application = create_app(settings)
    # ``timeout_keep_alive`` controls how long uvicorn keeps an idle HTTP/1.1
    # keep-alive connection open before closing it. The library default (5s)
    # is too aggressive for long-running async-rollout poll loops: when the
    # trainer's httpx client holds an idle pooled socket for >5s and then
    # reuses it, uvicorn has already half-closed the connection and the next
    # request raises ``httpx.ReadError`` / ``RemoteProtocolError`` ("Server
    # disconnected without sending a response"). The client also retries
    # transient transport errors (see ``_RetryingTransport`` in client.py),
    # but giving the server a much longer keep-alive window eliminates the
    # race in the common case rather than relying on retries.
    uvicorn.run(
        application,
        host=host,
        port=port,
        workers=1,
        timeout_keep_alive=120,
    )


@app.command()
def controller(
    base_url: str = typer.Option(..., envvar="AGL_BASE_URL", help="agl-lite server base URL"),
    namespace: str = typer.Option(..., envvar="AGL_NAMESPACE", help="K8s namespace for agent Jobs"),
    runner_type: str = typer.Option(
        "k8s",
        envvar="AGL_RUNNER_TYPE",
        help="Runner backend: k8s | local",
    ),
    job_manifest_template: str | None = typer.Option(
        None,
        envvar="AGL_JOB_MANIFEST_TEMPLATE",
        help="Path to Jinja2 job manifest template (required for k8s runner)",
    ),
    local_pool_size: int | None = typer.Option(
        None,
        envvar="AGL_LOCAL_POOL_SIZE",
        help="Max concurrent local subprocesses (required when runner_type=local)",
    ),
    local_agent_class: str | None = typer.Option(
        None,
        envvar="AGL_LOCAL_AGENT_CLASS",
        help=("Python agent class path, e.g. examples.calc_x.agent:CalcAgent (required when runner_type=local)"),
    ),
    local_tick_interval: float = typer.Option(
        5.0,
        envvar="AGL_LOCAL_TICK_INTERVAL",
        help="Seconds between local runner reap / enforce / admit ticks",
    ),
    key: str = typer.Option("", envvar="AGL_KEY", help="Shared API key (empty = auth disabled)"),
    poll_interval: int = typer.Option(10, envvar="AGL_POLL_INTERVAL", help="Seconds between reconcile cycles"),
    max_queue_time: int = typer.Option(
        3600,
        envvar="AGL_MAX_QUEUE_TIME",
        help="Max seconds a rollout stays in queuing",
    ),
    ttl_after_finished: int = typer.Option(
        3600,
        envvar="AGL_TTL_AFTER_FINISHED",
        help="ttlSecondsAfterFinished on Jobs",
    ),
    max_pods_per_window: int = typer.Option(
        100,
        envvar="AGL_MAX_PODS_PER_WINDOW",
        help="Max agent Pods to create per rate-limit window",
    ),
    rate_limit_window_seconds: int = typer.Option(
        10,
        envvar="AGL_RATE_LIMIT_WINDOW_SECONDS",
        help="Pod creation rate-limit window in seconds",
    ),
) -> None:
    """Start the controller (reconcile loop)."""
    import asyncio
    import os

    from agl_lite.client import AglLiteClient
    from agl_lite.controller.config import ControllerSettings, RunnerType

    settings = ControllerSettings(
        base_url=base_url,
        namespace=namespace,
        runner_type=RunnerType(runner_type),
        job_manifest_template=job_manifest_template,
        local_pool_size=local_pool_size,
        local_agent_class=local_agent_class,
        local_tick_interval=local_tick_interval,
        key=key,
        poll_interval=poll_interval,
        max_queue_time=max_queue_time,
        ttl_after_finished=ttl_after_finished,
        max_pods_per_window=max_pods_per_window,
        rate_limit_window_seconds=rate_limit_window_seconds,
    )

    async def _run() -> None:
        api = AglLiteClient(base_url=settings.base_url, agl_key=settings.key or None)
        try:
            if settings.runner_type == RunnerType.K8S:
                if not settings.job_manifest_template:
                    raise typer.BadParameter("k8s runner requires --job-manifest-template / AGL_JOB_MANIFEST_TEMPLATE")
                try:
                    from agl_lite.controller.kr8s_adapter import Kr8sClient
                except ImportError:
                    typer.echo("kr8s adapter not yet implemented — install kr8s and implement Kr8sClient")
                    raise typer.Exit(code=1) from None
                from agl_lite.controller.reconciler import Reconciler

                k8s = Kr8sClient(namespace=settings.namespace)
                reconciler = Reconciler(api=api, k8s=k8s, settings=settings)
            else:
                from agl_lite.controller.local_reconciler import LocalReconciler

                reconciler = LocalReconciler(
                    api=api,
                    settings=settings,
                    base_env={**os.environ},
                )
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
