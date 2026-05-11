#!/usr/bin/env python3
"""Build SWE-bench evaluation images for the bundled sample dataset.

When using minikube locally, run this against minikube's Docker daemon:

    eval "$(minikube -p minikube docker-env)"
    .venv/bin/python examples/swe_bench/build_images.py
    eval "$(minikube -p minikube docker-env -u)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import docker
from swebench.harness.docker_build import build_instance_images
from swebench.harness.test_spec.test_spec import make_test_spec

EXAMPLE_DIR = Path(__file__).resolve().parent


def load_dataset(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open() as dataset_file:
        for line in dataset_file:
            items.append(json.loads(line))
    if not items:
        raise ValueError(f"Dataset is empty: {path}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SWE-bench sample Docker images")
    parser.add_argument("--dataset", default=str(EXAMPLE_DIR / "swebench_samples.jsonl"))
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--env-image-tag", default="latest")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="build only the first N dataset instances")
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        dataset = dataset[: args.limit]
    expected_images = [
        make_test_spec(
            cast(Any, item),
            namespace=args.namespace,
            instance_image_tag=args.tag,
            env_image_tag=args.env_image_tag,
        ).instance_image_key
        for item in dataset
    ]

    print(f"Dataset: {dataset_path}")
    print(f"Instances: {len(dataset)}")
    print("Expected images:")
    for image in expected_images:
        print(f"  {image}")

    client = docker.from_env()
    successful, failed = build_instance_images(
        client,
        cast(Any, dataset),
        force_rebuild=args.force_rebuild,
        max_workers=args.max_workers,
        namespace=args.namespace,
        tag=args.tag,
        env_image_tag=args.env_image_tag,
    )
    if failed:
        failed_names = ", ".join(str(item) for item in failed)
        raise SystemExit(f"Failed to build SWE-bench images: {failed_names}")
    print(f"Built or found {len(successful)} SWE-bench instance images.")


if __name__ == "__main__":
    main()
