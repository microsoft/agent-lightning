# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json

import pytest

from agentlightning.credit.artifacts import (
    make_credit_artifact,
    read_credit_artifact,
    replay_credit,
    write_credit_artifact,
)
from agentlightning.credit.types import ProcessCreditConfig


@pytest.mark.asyncio
async def test_canonical_roundtrip_replay_and_tampering(sealed_snapshot, tmp_path):
    artifact = make_credit_artifact(await sealed_snapshot(), ProcessCreditConfig(enabled=True, gamma=0.5))
    path = tmp_path / "evidence.json"
    write_credit_artifact(path, artifact, max_bytes=100000)
    loaded = read_credit_artifact(path, max_bytes=100000)
    assert replay_credit(loaded) == artifact.resolved
    payload = json.loads(path.read_text())
    payload["resolved"]["returns"][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_credit_artifact(path, max_bytes=100000)


@pytest.mark.asyncio
async def test_artifact_size_limit_and_atomic_failure(sealed_snapshot, tmp_path, monkeypatch):
    artifact = make_credit_artifact(await sealed_snapshot(), ProcessCreditConfig(enabled=True))
    path = tmp_path / "evidence.json"
    path.write_text("original")
    with pytest.raises(ValueError, match="size"):
        write_credit_artifact(path, artifact, max_bytes=10)
    assert path.read_text() == "original"

    def interrupted(*args):
        raise OSError("simulated interruption")

    monkeypatch.setattr("agentlightning.credit.artifacts.os.replace", interrupted)
    with pytest.raises(OSError, match="interruption"):
        write_credit_artifact(path, artifact, max_bytes=100000)
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.asyncio
async def test_output_directory_does_not_change_semantic_artifact_identity(sealed_snapshot):
    snapshot = await sealed_snapshot()
    first = make_credit_artifact(snapshot, ProcessCreditConfig(enabled=True, artifact_dir="/tmp/first"))
    second = make_credit_artifact(snapshot, ProcessCreditConfig(enabled=True, artifact_dir="/tmp/second"))
    assert first.artifact_hash == second.artifact_hash
