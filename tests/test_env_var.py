# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import pytest

from agentlightning.env_var import (
    LightningEnvVar,
    resolve_bool_env_var,
    resolve_int_env_var,
    resolve_str_env_var,
)


def test_resolve_bool_env_var_override_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    env_name = LightningEnvVar.AGL_MANAGED_STORE.value
    monkeypatch.setenv(env_name, "0")

    assert resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, override=True, fallback=False) is True


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1", True),
        (" YES ", True),
        ("on", True),
        ("0", False),
        ("no", False),
        ("Off", False),
    ],
)
def test_resolve_bool_env_var_parses_truthy_and_falsy_values(
    monkeypatch: pytest.MonkeyPatch, raw_value: str, expected: bool
) -> None:
    env_name = LightningEnvVar.AGL_MANAGED_STORE.value
    monkeypatch.setenv(env_name, raw_value)

    assert resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, fallback=not expected) is expected


def test_resolve_bool_env_var_returns_fallback_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    env_name = LightningEnvVar.AGL_MANAGED_STORE.value
    monkeypatch.delenv(env_name, raising=False)

    assert resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, fallback=False) is False


def test_resolve_bool_env_var_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    env_name = LightningEnvVar.AGL_MANAGED_STORE.value
    monkeypatch.setenv(env_name, "maybe")

    with pytest.raises(ValueError):
        resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, fallback=False)


def test_resolve_int_env_var_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    env_name = LightningEnvVar.AGL_SERVER_PORT.value
    monkeypatch.setenv(env_name, "1234")

    assert resolve_int_env_var(LightningEnvVar.AGL_SERVER_PORT, fallback=4747) == 1234


def test_resolve_int_env_var_invalid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    env_name = LightningEnvVar.AGL_SERVER_PORT.value
    monkeypatch.setenv(env_name, "not-a-number")

    with pytest.raises(ValueError):
        resolve_int_env_var(LightningEnvVar.AGL_SERVER_PORT, fallback=4747)


def test_resolve_str_env_var_override_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    env_name = LightningEnvVar.AGL_CURRENT_ROLE.value
    monkeypatch.setenv(env_name, "client")

    assert resolve_str_env_var(LightningEnvVar.AGL_CURRENT_ROLE, override="server", fallback="both") == "server"

    monkeypatch.delenv(env_name, raising=False)
    assert resolve_str_env_var(LightningEnvVar.AGL_CURRENT_ROLE, fallback="both") == "both"
