# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""The vertexai module."""

import importlib

from .client import Client  # type: ignore[attr-defined]

_evals = None


def __getattr__(name):  # type: ignore[no-untyped-def]
    if name == "evals":
        global _evals
        if _evals is None:
            try:
                _evals = importlib.import_module(".evals", __package__)
            except ImportError as e:
                raise ImportError(
                    "The 'evals' module requires additional dependencies. "
                    "Please install them using pip install "
                    "google-cloud-aiplatform[evaluation]"
                ) from e
        return _evals
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "Client",
    "evals",
]
