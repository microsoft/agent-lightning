# Copyright (c) Microsoft. All rights reserved.

"""Test fixtures for VERL daemon unit tests.

``agentlightning.verl.daemon`` imports ``verl.DataProto`` at module load time.
These tests only exercise daemon bookkeeping helpers, so we stub the VERL
package when it is not installed.
"""

from __future__ import annotations

import sys
import types


def pytest_configure() -> None:
    if "verl" in sys.modules:
        return
    verl_stub = types.ModuleType("verl")

    class DataProto:  # noqa: D401 - minimal stub for import
        """Placeholder for verl.DataProto."""

    verl_stub.DataProto = DataProto
    sys.modules["verl"] = verl_stub
