# Copyright 2023 Google LLC
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
import sys

from google.cloud.aiplatform import version as aiplatform_version

__version__ = aiplatform_version.__version__

from google.cloud.aiplatform import init

_genai_client = None
_genai_types = None


def __getattr__(name):  # type: ignore[no-untyped-def]
    # Lazy importing the preview submodule
    # See https://peps.python.org/pep-0562/
    if name == "preview":
        # We need to import carefully to avoid `RecursionError`.
        # This won't work since it causes `RecursionError`:
        # `from vertexai import preview`
        # This won't work due to Copybara lacking a transform:
        # `import google.cloud.aiplatform.vertexai.preview as vertexai_preview`
        return importlib.import_module(".preview", __name__)

    if name == "Client":
        global _genai_client
        if _genai_client is None:
            _genai_client = importlib.import_module("._genai.client", __name__)
        return getattr(_genai_client, name)

    if name == "types":
        global _genai_types
        if _genai_types is None:
            _genai_types = importlib.import_module("._genai.types", __name__)
        if "vertexai.types" not in sys.modules:
            sys.modules["vertexai.types"] = _genai_types
        return _genai_types

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "init",
    "preview",
    "Client",
    "types",
]
