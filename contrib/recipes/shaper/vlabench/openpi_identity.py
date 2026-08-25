# Copyright (c) Microsoft. All rights reserved.

"""Identity handshake for the frozen OpenPI actor used by VLABench."""

from __future__ import annotations

import importlib
from typing import Any, Mapping, cast

from .actor_contract import (
    CHECKPOINT_MANIFEST_SHA256,
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    OPENPI_COMMIT,
)

METADATA_KEY = "shaper_actor"
REPORTED_THREE_CAMERA = "reported_three_camera"
SUPPORTED_OBSERVATION_SCHEMAS = frozenset({REPORTED_THREE_CAMERA})


def read_server_metadata(host: str, port: int, timeout_seconds: float = 5.0) -> Mapping[str, Any]:
    """Read the first websocket message without issuing an actor inference."""

    websocket_client = cast(Any, importlib.import_module("websockets.sync.client"))
    msgpack_numpy = cast(Any, importlib.import_module("openpi_client.msgpack_numpy"))
    connection = websocket_client.connect(
        f"ws://{host}:{port}",
        compression=None,
        max_size=None,
        open_timeout=timeout_seconds,
        close_timeout=min(timeout_seconds, 5.0),
    )
    try:
        metadata: object = msgpack_numpy.unpackb(connection.recv(timeout=timeout_seconds))
    finally:
        connection.close()
    if not isinstance(metadata, Mapping):
        raise TypeError("OpenPI server metadata must be a mapping.")
    return cast(Mapping[str, Any], metadata)


def validate_server_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_actor_id: str,
    expected_policy_config: str,
    expected_observation_schema: str,
) -> list[str]:
    """Validate metadata emitted by the bundled pinned OpenPI launcher."""

    raw_identity = metadata.get(METADATA_KEY)
    if not isinstance(raw_identity, Mapping):
        return [
            "OpenPI server does not expose SHAPER actor identity metadata. Start it with "
            "contrib.recipes.shaper.vlabench.openpi_server."
        ]
    identity = cast(Mapping[str, Any], raw_identity)
    errors: list[str] = []
    expected = {
        "protocol_version": 1,
        "actor_id": expected_actor_id,
        "openpi_commit": OPENPI_COMMIT,
        "checkpoint_repository": CHECKPOINT_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "policy_config": expected_policy_config,
        "observation_schema": expected_observation_schema,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            errors.append(f"OpenPI actor metadata {key}={identity.get(key)!r}; expected {value!r}.")
    return errors


__all__ = [
    "METADATA_KEY",
    "REPORTED_THREE_CAMERA",
    "SUPPORTED_OBSERVATION_SCHEMAS",
    "read_server_metadata",
    "validate_server_metadata",
]
