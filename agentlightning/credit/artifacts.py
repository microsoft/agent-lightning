# Copyright (c) Microsoft. All rights reserved.

"""Bounded finite JSON audit artifacts with atomic writes and deterministic replay."""

from __future__ import annotations

import json
import math
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
    artifact = CreditArtifact.model_validate({**payload, "artifact_hash": semantic_digest(payload)})
    replay_credit(artifact)
    return artifact


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
    _validate_projection(checked)
    return resolved


def _validate_projection(artifact: CreditArtifact) -> None:
    """Audit unpadded row provenance and fixed-scale math using primitive evidence."""
    from collections import defaultdict

    from agentlightning.credit.state import captured_action_geometry, validate_snapshot
    from agentlightning.credit.types import finite_float32, finite_number, nonempty

    if not artifact.scheduled_groups:
        if artifact.projection:
            raise CreditProtocolError("projection requires complete scheduled collection evidence")
        return
    resolved = artifact.resolved
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scheduled = {}
    fields = {"rollout_id", "group_id", "group_size", "policy_version", "episode_return", "snapshot_hash"}
    for item in artifact.scheduled_groups:
        if set(item) != fields:
            raise CreditProtocolError("invalid scheduled collection audit fields")
        rollout_id = nonempty(item["rollout_id"])
        group_id = nonempty(item["group_id"])
        if (
            rollout_id in scheduled
            or type(item["group_size"]) is not int
            or item["group_size"] < 1
            or type(item["policy_version"]) is not int
            or item["policy_version"] != resolved.policy_version
        ):
            raise CreditProtocolError("duplicate or incompatible scheduled rollout")
        digest = item["snapshot_hash"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise CreditProtocolError("invalid scheduled snapshot hash")
        finite_number(item["episode_return"], "scheduled discounted episode return")
        groups[group_id].append(item)
        scheduled[rollout_id] = item
    if any(any(item["group_size"] != len(group) for item in group) for group in groups.values()):
        raise CreditProtocolError("incomplete scheduled group in projection audit")
    own = scheduled.get(resolved.rollout_id)
    if own != {
        "rollout_id": resolved.rollout_id,
        "group_id": resolved.group_id,
        "group_size": resolved.group_size,
        "policy_version": resolved.policy_version,
        "episode_return": resolved.episode_return,
        "snapshot_hash": resolved.snapshot_hash,
    }:
        raise CreditProtocolError("scheduled evidence differs from this sealed rollout")
    peers = [item["episode_return"] for item in groups[resolved.group_id] if item["rollout_id"] != resolved.rollout_id]
    baseline = math.fsum(peers) / len(peers) if artifact.config.baseline == "task_loo" and peers else 0.0
    denominator = len(scheduled) * artifact.config.normalization_tokens
    if not math.isfinite(denominator):
        raise CreditProtocolError("scheduled K*H normalization must be finite")
    coefficients = tuple(
        finite_float32(discount * (value - baseline) / denominator, "replayed policy coefficient")
        for value, discount in zip(resolved.returns, resolved.discounts, strict=True)
    )
    actions, _ = validate_snapshot(artifact.snapshot)
    geometry = []
    for action in actions:
        source = captured_action_geometry(action)
        if source is None:
            raise CreditProtocolError("projected action lacks complete capture geometry")
        geometry.append(source)
    expected = [
        (action.envelope.event_id, index, offset)
        for index, action in enumerate(actions)
        for offset in range(len(geometry[index][1]))
    ]
    observed = []
    row_ids = set()
    row_fields = {
        "row_id",
        "prompt_token_ids",
        "response_token_ids",
        "origins",
        "response_mask",
        "coefficients",
        "returns",
    }
    for row in artifact.projection:
        if set(row) != row_fields or nonempty(row["row_id"]) in row_ids:
            raise CreditProtocolError("invalid or duplicate projection row")
        row_ids.add(row["row_id"])
        prompt, response = row["prompt_token_ids"], row["response_token_ids"]
        if (
            not isinstance(prompt, list | tuple)
            or not prompt
            or not isinstance(response, list | tuple)
            or not response
            or any(type(token) is not int or token < 0 for token in (*prompt, *response))
        ):
            raise CreditProtocolError("invalid projected token geometry")
        arrays = [row[key] for key in ("origins", "response_mask", "coefficients", "returns")]
        if any(not isinstance(value, list | tuple) or len(value) != len(response) for value in arrays):
            raise CreditProtocolError("unaligned projected provenance or coefficients")
        previous = None
        for position, (origin, mask, coefficient, value) in enumerate(zip(*arrays, strict=True)):
            finite_number(coefficient, "projected coefficient")
            finite_number(value, "projected return")
            if type(mask) is not int or mask not in (0, 1):
                raise CreditProtocolError("projected response mask must be binary integers")
            if origin is None:
                if mask or coefficient != 0 or value != 0:
                    raise CreditProtocolError("prompt/tool positions cannot receive policy credit")
                previous = None
                continue
            if not isinstance(origin, dict) or set(origin) != {"action_event_id", "action_index", "offset"}:
                raise CreditProtocolError("invalid original token provenance")
            index, offset = origin["action_index"], origin["offset"]
            if (
                type(index) is not int
                or not 0 <= index < len(actions)
                or type(offset) is not int
                or not 0 <= offset < len(geometry[index][1])
            ):
                raise CreditProtocolError("projected origin is outside its complete action")
            identity = (origin["action_event_id"], index, offset)
            source = geometry[index]
            if identity[0] != actions[index].envelope.event_id or response[position] != source[1][offset]:
                raise CreditProtocolError("projected token differs from original sampled action")
            if offset == 0:
                if tuple(prompt) + tuple(response[:position]) != source[0]:
                    raise CreditProtocolError("projected action context differs from original sampled prompt")
            elif previous != (identity[0], index, offset - 1):
                raise CreditProtocolError("projection split or interrupted an original action")
            if mask != 1 or coefficient != coefficients[index] or value != resolved.returns[index]:
                raise CreditProtocolError("projected coefficient/return differs from fixed-scale replay")
            observed.append(identity)
            previous = identity
    if observed != expected:
        raise CreditProtocolError("projection must retain every original action token in order exactly once")


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
