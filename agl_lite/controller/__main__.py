# Copyright (c) Microsoft. All rights reserved.

"""Hydra entrypoint for the agl-lite controller."""

from __future__ import annotations

import asyncio
import contextlib
import signal

import hydra
from omegaconf import DictConfig

from agl_lite.client import AglLiteAsyncClient


async def _run_controller(config: DictConfig) -> None:
    async with AglLiteAsyncClient(
        base_url=str(config.agl_server.url),
        key=str(config.agl_server.key or "") or None,
    ) as api:
        if config.runner_type == "k8s":
            try:
                from agl_lite.controller.k8s_reconciler import K8sReconciler
            except ImportError:
                raise RuntimeError("kr8s unavailable - install agl-lite[controller]") from None

            reconciler = K8sReconciler(api=api, config=config)
        elif config.runner_type == "local":
            from agl_lite.controller.local_reconciler import LocalReconciler

            reconciler = LocalReconciler(
                api=api,
                config=config,
            )
        else:
            raise ValueError(f"unknown runner_type: {config.runner_type}")

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, reconciler.stop)
        await reconciler.run()


@hydra.main(version_base=None, config_path="../config", config_name="controller")
def main(config: DictConfig) -> None:
    asyncio.run(_run_controller(config))


if __name__ == "__main__":
    main()