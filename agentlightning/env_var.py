# Copyright (c) Microsoft. All rights reserved.

"""Environment variable managements."""

from __future__ import annotations

import os
from enum import Enum
from typing import overload

__all__ = [
    "LightningEnvVar",
    "resolve_bool_env_var",
    "resolve_int_env_var",
    "resolve_str_env_var",
]


class LightningEnvVar(Enum):
    """Environment variables for Agent Lightning."""

    AGL_EMITTER_DEBUG = "AGL_EMITTER_DEBUG"
    """Enable debug logging for the emitter."""

    AGL_MANAGED_STORE = "AGL_MANAGED_STORE"
    """If yes, the [`ExecutionStrategy`][agentlightning.ExecutionStrategy]
    constructs LightningStore wrappers automatically. When `False` the provided
    `store` is passed directly to the bundles, allowing callers to manage
    store wrappers manually."""

    AGL_CURRENT_ROLE = "AGL_CURRENT_ROLE"
    """Which side(s) to run in this process. Used in
    [`ClientServerExecutionStrategy`][agentlightning.ClientServerExecutionStrategy]."""

    AGL_SERVER_HOST = "AGL_SERVER_HOST"
    """Interface the [`LightningStoreServer`][agentlightning.LightningStoreServer]
    binds to when running the algorithm bundle locally."""

    AGL_SERVER_PORT = "AGL_SERVER_PORT"
    """Port the [`LightningStoreServer`][agentlightning.LightningStoreServer] listens to."""


_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_FALSY_VALUES = {"0", "false", "no", "off"}


@overload
def resolve_bool_env_var(env_var: LightningEnvVar, override: bool, fallback: bool) -> bool: ...


@overload
def resolve_bool_env_var(env_var: LightningEnvVar, *, fallback: bool) -> bool: ...


@overload
def resolve_bool_env_var(
    env_var: LightningEnvVar, override: bool | None = None, fallback: bool | None = None
) -> bool | None: ...


def resolve_bool_env_var(
    env_var: LightningEnvVar, override: bool | None = None, fallback: bool | None = None
) -> bool | None:
    """Resolve a boolean environment variable.

    Args:
        env_var: The environment variable to resolve.
        override: Optional override supplied by the caller.
        fallback: Default value if the environment variable is not set.
    """

    if override is not None:
        return override

    env_value = os.getenv(env_var.value)
    if env_value is None:
        return fallback

    normalized = env_value.strip().lower()
    if normalized in _TRUTHY_VALUES:
        return True
    if normalized in _FALSY_VALUES:
        return False

    raise ValueError(f"{env_var.value} must be one of {_TRUTHY_VALUES} or {_FALSY_VALUES}")


@overload
def resolve_int_env_var(env_var: LightningEnvVar, override: int, fallback: int) -> int: ...


@overload
def resolve_int_env_var(env_var: LightningEnvVar, *, fallback: int) -> int: ...


@overload
def resolve_int_env_var(
    env_var: LightningEnvVar, override: int | None = None, fallback: int | None = None
) -> int | None: ...


def resolve_int_env_var(
    env_var: LightningEnvVar, override: int | None = None, fallback: int | None = None
) -> int | None:
    """Resolve an integer environment variable.

    Args:
        env_var: The environment variable to resolve.
        override: Optional override supplied by the caller.
        fallback: Default value if the environment variable is not set.
    """
    if override is not None:
        return override

    env_value = os.getenv(env_var.value)
    if env_value is None:
        return fallback

    try:
        return int(env_value)
    except ValueError:
        raise ValueError(f"{env_var.value} must be an integer")


@overload
def resolve_str_env_var(env_var: LightningEnvVar, override: str, fallback: str) -> str: ...


@overload
def resolve_str_env_var(env_var: LightningEnvVar, *, fallback: str) -> str: ...


@overload
def resolve_str_env_var(
    env_var: LightningEnvVar, override: str | None = None, fallback: str | None = None
) -> str | None: ...


def resolve_str_env_var(
    env_var: LightningEnvVar, override: str | None = None, fallback: str | None = None
) -> str | None:
    """Resolve a string environment variable.

    Args:
        env_var: The environment variable to resolve.
        override: Optional override supplied by the caller.
        fallback: Default value if the environment variable is not set.
    """
    if override is not None:
        return override

    env_value = os.getenv(env_var.value)
    if env_value is None:
        return fallback

    return env_value
