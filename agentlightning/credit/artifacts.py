# Copyright (c) Microsoft. All rights reserved.

"""Bounded finite JSON audit artifacts with atomic writes and deterministic replay."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from agentlightning.credit.resolve import resolve_credit
from agentlightning.credit.types import (
    ALGORITHM_VERSION,
    CreditArtifact,
    CreditProtocolError,
    CreditSnapshot,
    ProcessCreditConfig,
    canonical_json,
    semantic_digest,
)


def make_credit_artifact(
    snapshot: CreditSnapshot,
    config: ProcessCreditConfig,
    *,
    projection: tuple[dict[str, Any], ...] = (),
    scheduled_groups: tuple[dict[str, Any], ...] = (),
) -> CreditArtifact:
    """Own validated evidence; paths/transport timestamps never become numerical state."""
    resolved = resolve_credit(snapshot, config)
    config = config.model_copy(update={"artifact_dir": None})
    payload = {
        "version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "snapshot": snapshot.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
        "resolved": resolved.model_dump(mode="json"),
        "projection": projection,
        "scheduled_groups": scheduled_groups,
    }
    return CreditArtifact.model_validate({**payload, "artifact_hash": semantic_digest(payload)})


def replay_credit(artifact: CreditArtifact):
    """Recompute returns without model calls or actor updates, rejecting modified evidence."""
    payload = artifact.model_dump(mode="json")
    if artifact.artifact_hash != semantic_digest(
        {key: value for key, value in payload.items() if key != "artifact_hash"}
    ):
        raise CreditProtocolError("credit artifact semantic checksum mismatch")
    checked = CreditArtifact.model_validate(payload)
    resolved = resolve_credit(checked.snapshot, checked.config)
    if resolved != checked.resolved:
        raise CreditProtocolError("recorded resolution differs from deterministic replay")
    return resolved


def write_credit_artifact(path: Path, artifact: CreditArtifact, *, max_bytes: int) -> str:
    """Validate before touching disk; atomic replacement never publishes a partial artifact."""
    replay_credit(artifact)
    payload = canonical_json(artifact.model_dump(mode="json"))
    if max_bytes <= 0 or len(payload) > max_bytes:
        raise CreditProtocolError("credit artifact exceeds configured size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return artifact.artifact_hash


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CreditProtocolError("duplicate JSON key in credit artifact")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise CreditProtocolError(f"nonfinite JSON credit artifact constant: {value}")


def read_credit_artifact(path: Path, *, max_bytes: int) -> CreditArtifact:
    """Bound input before parsing; no pickle, dynamic class path or executable state."""
    if max_bytes <= 0:
        raise CreditProtocolError("credit artifact size limit must be positive")
    with path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise CreditProtocolError("credit artifact exceeds configured size limit")
    try:
        primitive = json.loads(payload, object_pairs_hook=_unique_keys, parse_constant=_nonfinite)
    except (ValueError, RecursionError) as exc:
        raise CreditProtocolError("malformed credit artifact JSON") from exc
    artifact = CreditArtifact.model_validate(primitive)
    replay_credit(artifact)
    return artifact
