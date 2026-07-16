#!/usr/bin/env python3
"""Prepare SWE-smith images for Kubernetes rollouts.

The default job template runs ``{image_name}:openai`` so the agent starts
without a runtime ``pip install``. This script prepares those local tags from
the per-instance base images and only pulls a base image when it is missing
from the active Docker daemon.
"""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    EXAMPLE_DIR / "train_dataset.jsonl",
    EXAMPLE_DIR / "val_dataset.jsonl",
)
OPENAI_IMAGE_SUFFIX = "openai"
LOCAL_OPENAI_IMAGE_SUFFIXES = ("agl-openai",)
OPENAI_PACKAGE = "openai"


@dataclass(frozen=True)
class ImagePreparationResult:
    source_image: str
    target_image: str
    status: str
    error: str | None = None


def load_dataset(path: Path) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in path.open() if line.strip()]
    if not items:
        raise ValueError(f"Dataset is empty: {path}")
    return items


def load_datasets(paths: Sequence[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in paths:
        items.extend(load_dataset(path))
    if not items:
        raise ValueError("No SWE-smith instances loaded")
    return items


def distinct_images(dataset: list[dict[str, Any]], limit: int | None) -> list[str]:
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        dataset = dataset[:limit]
    images: list[str] = []
    for item in dataset:
        image = item.get("image_name")
        if not image:
            raise ValueError(f"Instance {item.get('instance_id')} missing 'image_name'")
        if image not in images:
            images.append(image)
    return images


def split_image_tag(image_name: str) -> tuple[str, str | None]:
    if not image_name:
        raise ValueError("image_name is required")
    if "@" in image_name:
        raise ValueError("digest-based image names are not supported")
    last_slash_index = image_name.rfind("/")
    last_colon_index = image_name.rfind(":")
    if last_colon_index > last_slash_index:
        return image_name[:last_colon_index], image_name[last_colon_index + 1 :]
    return image_name, None


def openai_image_name(image_name: str) -> str:
    _, tag = split_image_tag(image_name)
    if tag is not None:
        raise ValueError(
            f"SWE-smith image_name must be untagged because job-template-openai.yaml "
            f"uses '<image_name>:{OPENAI_IMAGE_SUFFIX}': {image_name}"
        )
    return f"{image_name}:{OPENAI_IMAGE_SUFFIX}"


def local_openai_aliases(image_name: str) -> list[str]:
    _, tag = split_image_tag(image_name)
    if tag is not None:
        return []
    return [f"{image_name}:{suffix}" for suffix in LOCAL_OPENAI_IMAGE_SUFFIXES]


def get_image(client: Any, image: str) -> Any | None:
    try:
        return client.images.get(image)
    except Exception:
        return None


def image_exists(client: Any, image: str) -> bool:
    return get_image(client, image) is not None


def tag_image(client: Any, source_image: str, target_image: str) -> None:
    image = get_image(client, source_image)
    if image is None:
        raise ValueError(f"source image is not loaded locally: {source_image}")
    repository, tag = split_image_tag(target_image)
    if tag is None:
        raise ValueError(f"target image must include a tag: {target_image}")
    image.tag(repository, tag=tag)


def openai_dockerfile(source_image: str) -> str:
    install_cmd = f"python -m pip install --no-cache-dir {shlex.quote(OPENAI_PACKAGE)}"
    return "".join(
        [
            f"FROM {source_image}\n",
            'LABEL agl-lite.swe-smith.openai-layer="1"\n',
            f"RUN bash -lc {shlex.quote(install_cmd)}\n",
        ]
    )


def build_openai_image(client: Any, source_image: str, target_image: str) -> None:
    dockerfile = openai_dockerfile(source_image)
    with TemporaryDirectory(prefix="swe-smith-openai-") as context_dir:
        dockerfile_path = Path(context_dir) / "Dockerfile"
        dockerfile_path.write_text(dockerfile)
        client.images.build(
            path=str(context_dir),
            tag=target_image,
            rm=True,
            forcerm=True,
            pull=False,
        )


def prepare_one(
    client: Any,
    source_image: str,
    *,
    pull_missing: bool,
    force_build: bool,
) -> ImagePreparationResult:
    try:
        target_image = openai_image_name(source_image)
        if not force_build and image_exists(client, target_image):
            return ImagePreparationResult(source_image, target_image, "exists")

        if not force_build:
            for alias in local_openai_aliases(source_image):
                if image_exists(client, alias):
                    tag_image(client, alias, target_image)
                    return ImagePreparationResult(source_image, target_image, f"retagged:{alias}")

        if image_exists(client, source_image):
            source_status = "base-local"
        elif pull_missing:
            client.images.pull(source_image)
            source_status = "base-pulled"
        else:
            return ImagePreparationResult(
                source_image,
                target_image,
                "failed",
                "base image is not loaded in the active Docker daemon; run with "
                "minikube docker-env active or pass --pull-missing",
            )

        status = "rebuilt" if force_build else "built"
        build_openai_image(client, source_image, target_image)
        return ImagePreparationResult(source_image, target_image, f"{source_status},{status}")
    except Exception as exc:
        target = f"{source_image}:{OPENAI_IMAGE_SUFFIX}"
        try:
            target = openai_image_name(source_image)
        except Exception:
            pass
        return ImagePreparationResult(source_image, target, "failed", str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SWE-smith per-repo Docker images")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="SWE-smith JSONL to prepare images for. May be passed multiple times. "
        "Defaults to train_dataset.jsonl and val_dataset.jsonl.",
    )
    parser.add_argument("--limit", type=int, default=None, help="only consider the first N instances")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--pull-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pull a base image only if it is not already loaded locally (default: true)",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="rebuild :openai images even when the target tag already exists",
    )
    return parser.parse_args()


def main() -> None:
    try:
        import docker
    except ImportError as exc:
        raise SystemExit("pull_images.py requires the Docker SDK for Python: pip install docker") from exc

    args = parse_args()
    dataset_paths = [Path(path) for path in (args.dataset or [str(path) for path in DEFAULT_DATASETS])]
    dataset = load_datasets(dataset_paths)
    images = distinct_images(dataset, args.limit)

    print("Datasets:")
    for path in dataset_paths:
        print(f"  {path}")
    print(f"Instances: {len(dataset)} | distinct images: {len(images)}")
    print("Preparing local :openai images for job-template-openai.yaml")
    print("Compatible local aliases:", ", ".join(f":{suffix}" for suffix in LOCAL_OPENAI_IMAGE_SUFFIXES))
    for image in images:
        print(f"  {image} -> {openai_image_name(image)}")

    client = docker.from_env()
    failed: list[ImagePreparationResult] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                prepare_one,
                client,
                image,
                pull_missing=args.pull_missing,
                force_build=args.force_build,
            ): image
            for image in images
        }
        for future in as_completed(futures):
            result = future.result()
            if result.error:
                print(f"  FAILED {result.source_image} -> {result.target_image}: {result.error}")
                failed.append(result)
            else:
                print(f"  {result.status.upper():18} {result.source_image} -> {result.target_image}")

    if failed:
        names = ", ".join(result.source_image for result in failed)
        raise SystemExit(f"Failed to prepare SWE-smith images: {names}")
    print(f"Prepared {len(images)} SWE-smith :openai images.")


if __name__ == "__main__":
    main()
