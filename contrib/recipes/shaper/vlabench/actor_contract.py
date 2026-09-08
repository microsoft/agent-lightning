# Copyright (c) Microsoft. All rights reserved.

"""Dependency-free identity contract for the frozen VLABench actor."""

from __future__ import annotations

import hashlib
from pathlib import Path

OPENPI_REPOSITORY = "https://github.com/Shiduo-zh/openpi"
OPENPI_COMMIT = "4483d1da6332da44115fe530e4e6fdd89bd57b13"
POLICY_CONFIG = "pi0_ft_vlabench_primitive"

CHECKPOINT_REPOSITORY = "VLABench/pi0-primitive-10task"
CHECKPOINT_REVISION = "1ad73753a74d5cd97e67856664350f3f0baa21dc"
CHECKPOINT_MANIFEST_SHA256 = "39ef720bc93c4d3ccdd135ed6f4b803b8e7ae721cb8c068d9115fb55849d6203"
CHECKPOINT_MANIFEST_FILES = (
    "_CHECKPOINT_METADATA",
    "params/_METADATA",
    "params/_sharding",
    "params/manifest.ocdbt",
    "assets/vlabench/vlabench_ft_primitive/norm_stats.json",
)
CHECKPOINT_INFERENCE_PATTERNS = (
    "_CHECKPOINT_METADATA",
    "params/**",
    "assets/**",
)


def checkpoint_manifest_digest(root: Path) -> str:
    """Hash the released checkpoint's small identity-bearing manifests."""

    digest = hashlib.sha256()
    for relative_path in CHECKPOINT_MANIFEST_FILES:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint identity file: {path}")
        data = path.read_bytes()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


__all__ = [
    "CHECKPOINT_MANIFEST_FILES",
    "CHECKPOINT_MANIFEST_SHA256",
    "CHECKPOINT_INFERENCE_PATTERNS",
    "CHECKPOINT_REPOSITORY",
    "CHECKPOINT_REVISION",
    "OPENPI_COMMIT",
    "OPENPI_REPOSITORY",
    "POLICY_CONFIG",
    "checkpoint_manifest_digest",
]
