# Copyright (c) Microsoft. All rights reserved.

"""Download and verify the frozen VLABench actor checkpoint.

Run this helper from the Agent Lightning repository root in an environment
that has ``huggingface_hub`` installed. ``HF_TOKEN`` and ``HF_ENDPOINT`` are
honored by ``huggingface_hub`` without exposing credentials on the command
line.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from contrib.recipes.shaper.vlabench.actor_contract import (
    CHECKPOINT_INFERENCE_PATTERNS,
    CHECKPOINT_MANIFEST_SHA256,
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    checkpoint_manifest_digest,
)


def verify_checkpoint(path: Path) -> str:
    """Return the verified manifest digest for one local checkpoint."""

    resolved = path.expanduser().resolve()
    digest = checkpoint_manifest_digest(resolved)
    if digest != CHECKPOINT_MANIFEST_SHA256:
        raise RuntimeError(
            f"Checkpoint manifest digest {digest} does not match pinned " f"{CHECKPOINT_MANIFEST_SHA256}."
        )
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    """Download the immutable revision and verify its identity manifests."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not contact Hugging Face; verify an existing checkpoint directory.",
    )
    args = parser.parse_args(argv)
    destination = args.destination.expanduser().resolve()
    if int(args.max_workers) < 1:
        raise ValueError("max-workers must be positive.")

    if not args.verify_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub before downloading the actor checkpoint.") from exc
        snapshot_download(
            repo_id=CHECKPOINT_REPOSITORY,
            revision=CHECKPOINT_REVISION,
            local_dir=destination,
            max_workers=int(args.max_workers),
            allow_patterns=list(CHECKPOINT_INFERENCE_PATTERNS),
        )

    digest = verify_checkpoint(destination)
    print(f"checkpoint_repository={CHECKPOINT_REPOSITORY}")
    print(f"checkpoint_revision={CHECKPOINT_REVISION}")
    print(f"checkpoint_manifest_sha256={digest}")
    print(f"checkpoint_path={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
