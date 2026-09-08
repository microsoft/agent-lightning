# Copyright (c) Microsoft. All rights reserved.

"""Serve a pinned VLABench OpenPI checkpoint with verifiable metadata.

Run this module in the pinned OpenPI uv environment, not the VLABench simulator
environment. It deliberately has no Agent Lightning dependency.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence, cast

from .actor_contract import (
    CHECKPOINT_MANIFEST_SHA256,
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    OPENPI_COMMIT,
    POLICY_CONFIG,
    checkpoint_manifest_digest,
)

METADATA_KEY = "shaper_actor"
SUPPORTED_OBSERVATION_SCHEMAS = ("reported_three_camera",)


def _revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument(
        "--observation-schema",
        choices=SUPPORTED_OBSERVATION_SCHEMAS,
        default="reported_three_camera",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt")
    args = parser.parse_args(argv)

    root = cast(Path, args.openpi_root).expanduser().resolve()
    revision = _revision(root)
    if revision != OPENPI_COMMIT:
        raise RuntimeError(f"OpenPI revision {revision} does not match pinned {OPENPI_COMMIT}.")
    actor_id = str(args.actor_id).strip()
    policy_config_name = str(args.policy_config).strip()
    policy_dir_argument = str(args.policy_dir).strip()
    observation_schema = str(args.observation_schema)
    if not actor_id or not policy_config_name or not policy_dir_argument:
        raise ValueError("actor-id, policy-config, and policy-dir must be non-empty.")
    if policy_config_name != POLICY_CONFIG:
        raise ValueError(
            "The bundled VLABench actor launcher requires the paper protocol config " f"{POLICY_CONFIG!r}."
        )
    policy_root = Path(policy_dir_argument).expanduser().resolve()
    manifest_digest = checkpoint_manifest_digest(policy_root)
    if manifest_digest != CHECKPOINT_MANIFEST_SHA256:
        raise RuntimeError(
            "Checkpoint manifest digest " f"{manifest_digest} does not match pinned {CHECKPOINT_MANIFEST_SHA256}."
        )
    policy_dir = str(policy_root)

    source_root = root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    openpi_policy_config = cast(
        Any,
        importlib.import_module("openpi.policies.policy_config"),
    )
    websocket_policy_server = cast(
        Any,
        importlib.import_module("openpi.serving.websocket_policy_server"),
    )
    openpi_config = cast(
        Any,
        importlib.import_module("openpi.training.config"),
    )

    train_config = openpi_config.get_config(policy_config_name)
    policy = openpi_policy_config.create_trained_policy(
        train_config,
        policy_dir,
        default_prompt=cast(str | None, args.default_prompt),
    )
    metadata: dict[str, Any] = dict(policy.metadata)
    metadata[METADATA_KEY] = {
        "protocol_version": 1,
        "actor_id": actor_id,
        "openpi_commit": OPENPI_COMMIT,
        "checkpoint_repository": CHECKPOINT_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "policy_config": policy_config_name,
        "observation_schema": observation_schema,
    }
    hostname = socket.gethostname()
    logging.info(
        "Serving SHAPER actor %s with %s from %s on %s:%d",
        actor_id,
        policy_config_name,
        policy_dir,
        hostname,
        int(args.port),
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=int(args.port),
        metadata=metadata,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
