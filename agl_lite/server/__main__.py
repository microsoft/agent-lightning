"""Hydra entrypoint for the agl-lite server."""

from __future__ import annotations

import hydra
import uvicorn
from omegaconf import DictConfig

from agl_lite.server.app import create_app


@hydra.main(version_base=None, config_path="../config", config_name="server")
def main(config: DictConfig) -> None:
    application = create_app(config)
    uvicorn.run(
        application,
        host=str(config.host),
        port=int(config.port),
        workers=1,
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    main()