# Copyright (c) Microsoft. All rights reserved.

"""Check VLABench/OpenPI prerequisites without consuming an API request."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import socket
from pathlib import Path
from typing import Mapping, Sequence, cast

from ..cli import endpoint_socket
from .contracts import check_upstream_source
from .dataset import (
    TRACK_NAME,
    TRAIN_EPISODES,
    VALIDATION_EPISODES,
    load_reported_protocol_datasets,
    load_track,
    track_path,
)
from .openpi_identity import (
    REPORTED_THREE_CAMERA,
    SUPPORTED_OBSERVATION_SCHEMAS,
    read_server_metadata,
    validate_server_metadata,
)


def _configured_xml_paths(
    root: Path,
    *,
    track_name: str,
    specification: Mapping[str, Sequence[int]],
) -> tuple[list[Path], list[str]]:
    """Return model XMLs named by a deterministic episode specification."""

    errors: list[str] = []
    try:
        track = load_track(root, track_name)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        return [], [f"Cannot inspect VLABench task assets: {exc}"]

    relative_paths: set[str] = set()
    for task_name, indices in specification.items():
        episodes = track.get(task_name)
        if episodes is None:
            errors.append(f"VLABench asset preflight task {task_name!r} is absent from {track_name!r}.")
            continue
        for index in indices:
            if index < 0 or index >= len(episodes):
                errors.append(
                    f"VLABench asset preflight episode {task_name}/ep_{index:03d} is outside "
                    f"the track's {len(episodes)} episodes."
                )
                continue
            episode = cast(Mapping[str, object], episodes[index])
            task_value = episode.get("task")
            task = cast(Mapping[str, object], task_value) if isinstance(task_value, Mapping) else None
            components = task.get("components") if task is not None else None
            if not isinstance(components, list):
                errors.append(f"VLABench episode {task_name}/ep_{index:03d} has no component list.")
                continue
            for raw_component in cast(list[object], components):
                if not isinstance(raw_component, Mapping):
                    continue
                component = cast(Mapping[str, object], raw_component)
                xml_path = component.get("xml_path")
                if isinstance(xml_path, str) and xml_path.strip():
                    relative_paths.add(xml_path.strip())

    assets_root = (root / "assets").resolve()
    paths: list[Path] = []
    for relative in sorted(relative_paths):
        path = (assets_root / relative).resolve()
        if not path.is_relative_to(assets_root):
            errors.append(f"VLABench episode declares an asset outside the asset root: {relative!r}.")
            continue
        paths.append(path)
    return paths, errors


def check_vlabench_assets(
    root: Path,
    *,
    track_name: str = TRACK_NAME,
) -> list[str]:
    """Check real files required by the fixed 15/24 protocol."""

    required = [
        root / "assets" / "obj" / "meshes" / "table" / "table.xml",
        root / "assets" / "obj" / "assets" / "textures" / "wood0.png",
        root / "assets" / "scenes" / "default" / "empty.xml",
        root / "assets" / "scenes" / "default" / "studyroom" / "studyroom.xml",
    ]
    specification: dict[str, tuple[int, ...]] = {}
    for split in (TRAIN_EPISODES, VALIDATION_EPISODES):
        for task_name, indices in split.items():
            specification[task_name] = (*specification.get(task_name, ()), *indices)
    configured, errors = _configured_xml_paths(
        root,
        track_name=track_name,
        specification=specification,
    )
    required.extend(configured)
    missing = sorted({path for path in required if not path.is_file()})
    errors.extend(f"Missing VLABench asset payload: {path}" for path in missing[:20])
    if len(missing) > 20:
        errors.append(f"Missing {len(missing) - 20} additional VLABench asset files.")
    return errors


def _check_socket(label: str, host: str, port: int) -> str | None:
    try:
        with socket.create_connection((host, port), timeout=2.0):
            pass
    except OSError as exc:
        return f"{label} {host}:{port} is unreachable: {exc}"
    return None


def check_environment(
    *,
    root: Path,
    track_name: str = TRACK_NAME,
    host: str,
    port: int,
    require_vla: bool,
    expected_actor_id: str | None = None,
    expected_policy_config: str | None = None,
    expected_observation_schema: str = REPORTED_THREE_CAMERA,
    planner_endpoint: str | None = None,
    require_planner: bool = False,
) -> list[str]:
    """Return prerequisite errors; an empty list means the static checks pass."""

    errors: list[str] = []
    errors.extend(check_upstream_source(root))
    errors.extend(check_vlabench_assets(root, track_name=track_name))
    if platform.system() != "Linux":
        errors.append(f"VLABench rollout requires Linux; found {platform.system()} {platform.machine()}.")
    if not track_path(root, track_name).is_file():
        errors.append(f"Missing official track: {track_path(root, track_name)}")
    if importlib.util.find_spec("VLABench") is None:
        errors.append("Python package VLABench is not importable.")
    if importlib.util.find_spec("openpi_client") is None:
        errors.append("Python package openpi_client is not importable.")
    if require_vla:
        if expected_observation_schema not in SUPPORTED_OBSERVATION_SCHEMAS:
            errors.append(
                "VLABENCH_OBSERVATION_SCHEMA must be one of "
                + ", ".join(sorted(SUPPORTED_OBSERVATION_SCHEMAS))
                + f"; got {expected_observation_schema!r}."
            )
        if not expected_actor_id:
            errors.append("Set VLABENCH_ACTOR_ID to the identity declared by the OpenPI launcher.")
        if not expected_policy_config:
            errors.append("Set VLABENCH_OPENPI_POLICY_CONFIG to the pinned OpenPI policy config name.")
        if expected_actor_id and expected_policy_config:
            try:
                metadata = read_server_metadata(host, port)
            except Exception as exc:
                errors.append(f"OpenPI websocket endpoint {host}:{port} failed its metadata handshake: {exc}")
            else:
                errors.extend(
                    validate_server_metadata(
                        metadata,
                        expected_actor_id=expected_actor_id,
                        expected_policy_config=expected_policy_config,
                        expected_observation_schema=expected_observation_schema,
                    )
                )
    if planner_endpoint:
        planner_host, planner_port = endpoint_socket(planner_endpoint)
        if planner_host is None or planner_port is None:
            errors.append(f"SHAPER planner endpoint is not a valid HTTP(S) URL: {planner_endpoint!r}")
        else:
            endpoint_error = _check_socket("SHAPER planner endpoint", planner_host, planner_port)
            if endpoint_error:
                errors.append(endpoint_error)
    elif require_planner:
        errors.append("Set SHAPER_PLANNER_ENDPOINT to an OpenAI-compatible chat-completions base URL.")
    if track_path(root, track_name).is_file():
        try:
            train, validation = load_reported_protocol_datasets(root, track_name=track_name)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"Cannot load the fixed VLABench split: {exc}")
        else:
            if len(train) != 15 or len(validation) != 24:
                errors.append(f"Unexpected split sizes: train={len(train)} validation={len(validation)}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("VLABENCH_ROOT", ".")))
    parser.add_argument("--track", default=os.environ.get("VLABENCH_TRACK", TRACK_NAME))
    parser.add_argument("--vla-host", default=os.environ.get("VLABENCH_VLA_HOST", "127.0.0.1"))
    parser.add_argument("--vla-port", type=int, default=int(os.environ.get("VLABENCH_VLA_PORT", "8000")))
    parser.add_argument("--planner-endpoint", default=os.environ.get("SHAPER_PLANNER_ENDPOINT"))
    parser.add_argument("--actor-id", default=os.environ.get("VLABENCH_ACTOR_ID"))
    parser.add_argument(
        "--policy-config",
        default=os.environ.get("VLABENCH_OPENPI_POLICY_CONFIG"),
    )
    parser.add_argument(
        "--observation-schema",
        choices=sorted(SUPPORTED_OBSERVATION_SCHEMAS),
        default=os.environ.get("VLABENCH_OBSERVATION_SCHEMA", REPORTED_THREE_CAMERA),
    )
    parser.add_argument("--skip-vla-connect", action="store_true")
    parser.add_argument("--skip-planner-connect", action="store_true")
    args = parser.parse_args(argv)
    errors = check_environment(
        root=args.root.expanduser().resolve(),
        track_name=str(args.track),
        host=str(args.vla_host),
        port=int(args.vla_port),
        require_vla=not bool(args.skip_vla_connect),
        expected_actor_id=str(args.actor_id) if args.actor_id else None,
        expected_policy_config=str(args.policy_config) if args.policy_config else None,
        expected_observation_schema=str(args.observation_schema),
        planner_endpoint=str(args.planner_endpoint) if args.planner_endpoint else None,
        require_planner=not bool(args.skip_planner_connect),
    )
    if errors:
        for error in errors:
            print(f"[missing] {error}")
        return 2
    print("VLABench SHAPER prerequisites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
