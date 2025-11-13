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

import asyncio
import importlib
from typing import Optional, Union, Any
from types import TracebackType

import google.auth
from google.cloud.aiplatform import version as aip_version
from google.genai import _api_client
from google.genai import _common
from google.genai import client as genai_client
from google.genai import types
from . import live


_GENAI_MODULES_TELEMETRY_HEADER = "vertex-genai-modules"


def _add_tracking_headers(headers: dict[str, str]) -> None:
    """Appends Vertex Gen AI modules tracking information to the request headers."""

    tracking_label = f"{_GENAI_MODULES_TELEMETRY_HEADER}/{aip_version.__version__}"

    user_agent = headers.get("user-agent", "")
    if tracking_label not in user_agent:
        headers["user-agent"] = f"{user_agent} {tracking_label}".strip()

    api_client = headers.get("x-goog-api-client", "")
    if tracking_label not in api_client:
        headers["x-goog-api-client"] = f"{api_client} {tracking_label}".strip()


_api_client.append_library_version_headers = _add_tracking_headers


class AsyncClient:
    """Async Gen AI Client for the Vertex SDK."""

    def __init__(self, api_client: genai_client.BaseApiClient):
        self._api_client = api_client
        self._live = live.AsyncLive(self._api_client)
        self._evals = None
        self._agent_engines = None
        self._prompt_optimizer = None
        self._prompts = None
        self._datasets = None

    @property
    @_common.experimental_warning(
        "The Vertex SDK GenAI live module is experimental, and may change in future "
        "versions."
    )
    def live(self) -> live.AsyncLive:
        return self._live

    @property
    def evals(self) -> Any:
        if self._evals is None:
            try:
                # We need to lazy load the evals module to avoid ImportError when
                # pandas/tqdm are not installed.
                self._evals = importlib.import_module(".evals", __package__)  # type: ignore[assignment]
            except ImportError as e:
                raise ImportError(
                    "The 'evals' module requires 'pandas' and 'tqdm'. "
                    "Please install them using pip install "
                    "google-cloud-aiplatform[evaluation]"
                ) from e
        return self._evals.AsyncEvals(self._api_client)  # type: ignore[attr-defined]

    @property
    @_common.experimental_warning(
        "The Vertex SDK GenAI prompt optimizer module is experimental, "
        "and may change in future versions."
    )
    def prompt_optimizer(self):
        if self._prompt_optimizer is None:
            self._prompt_optimizer = importlib.import_module(
                ".prompt_optimizer", __package__
            )
        return self._prompt_optimizer.AsyncPromptOptimizer(self._api_client)

    @property
    def agent_engines(self):
        if self._agent_engines is None:
            try:
                # We need to lazy load the agent_engines module to handle the
                # possibility of ImportError when dependencies are not installed.
                self._agent_engines = importlib.import_module(
                    ".agent_engines",
                    __package__,
                )
            except ImportError as e:
                raise ImportError(
                    "The 'agent_engines' module requires 'additional packages'. "
                    "Please install them using pip install "
                    "google-cloud-aiplatform[agent_engines]"
                ) from e
        return self._agent_engines.AsyncAgentEngines(self._api_client)

    @property
    def prompts(self):
        if self._prompts is None:
            self._prompts = importlib.import_module(
                ".prompts",
                __package__,
            )
        return self._prompts.AsyncPrompts(self._api_client)

    @property
    @_common.experimental_warning(
        "The Vertex SDK GenAI async datasets module is experimental, "
        "and may change in future versions."
    )
    def datasets(self):
        if self._datasets is None:
            self._datasets = importlib.import_module(
                ".datasets",
                __package__,
            )
        return self._datasets.AsyncDatasets(self._api_client)

    async def aclose(self) -> None:
        """Closes the async client explicitly.

        Example usage:

        from vertexai import Client

        async_client = vertexai.Client(
            project='my-project-id', location='us-central1'
        ).aio
        prompt_1 = await async_client.prompts.create(...)
        prompt_2 = await async_client.prompts.create(...)
        # Close the client to release resources.
        await async_client.aclose()
        """
        await self._api_client.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Exception],
        exc_value: Optional[Exception],
        traceback: Optional[TracebackType],
    ) -> None:
        await self.aclose()

    def __del__(self) -> None:
        try:
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:
            pass


class Client:
    """Gen AI Client for the Vertex SDK.

    Use this client to interact with Vertex-specific Gemini features.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        debug_config: Optional[genai_client.DebugConfig] = None,
        http_options: Optional[Union[types.HttpOptions, types.HttpOptionsDict]] = None,
    ):
        """Initializes the client.

        Args:
           api_key (str): The `API key
           <https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview#api-keys>`_
             to use for authentication. Applies to Vertex AI in express mode only.
           credentials (google.auth.credentials.Credentials): The credentials to use
             for authentication when calling the Vertex AI APIs. Credentials can be
             obtained from environment variables and default credentials. For more
             information, see `Set up Application Default Credentials
             <https://cloud.google.com/docs/authentication/provide-credentials-adc>`_.
           project (str): The `Google Cloud project ID
             <https://cloud.google.com/vertex-ai/docs/start/cloud-environment>`_ to
             use for quota. Can be obtained from environment variables (for example,
             ``GOOGLE_CLOUD_PROJECT``).
           location (str): The `location
             <https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations>`_
             to send API requests to (for example, ``us-central1``). Can be obtained
             from environment variables.
           debug_config (DebugConfig): Config settings that control network behavior
             of the client. This is typically used when running test code.
           http_options (Union[HttpOptions, HttpOptionsDict]): Http options to use
             for the client.
        """

        self._debug_config = debug_config or genai_client.DebugConfig()
        if isinstance(http_options, dict):
            http_options = types.HttpOptions(**http_options)

        self._api_client = genai_client.Client._get_api_client(
            vertexai=True,
            api_key=api_key,
            credentials=credentials,
            project=project,
            location=location,
            debug_config=self._debug_config,
            http_options=http_options,
        )
        self._aio = AsyncClient(self._api_client)
        self._evals = None
        self._prompt_optimizer = None
        self._agent_engines = None
        self._prompts = None
        self._datasets = None

    @property
    def evals(self) -> Any:
        if self._evals is None:
            try:
                # We need to lazy load the evals module to avoid ImportError when
                # pandas/tqdm are not installed.
                self._evals = importlib.import_module(".evals", __package__)  # type: ignore[assignment]
            except ImportError as e:
                raise ImportError(
                    "The 'evals' module requires additional dependencies. "
                    "Please install them using pip install "
                    "google-cloud-aiplatform[evaluation]"
                ) from e
        return self._evals.Evals(self._api_client)  # type: ignore[attr-defined]

    @property
    @_common.experimental_warning(
        "The Vertex SDK GenAI prompt optimizer module is experimental, and may change in future "
        "versions."
    )
    def prompt_optimizer(self):
        if self._prompt_optimizer is None:
            self._prompt_optimizer = importlib.import_module(
                ".prompt_optimizer", __package__
            )
        return self._prompt_optimizer.PromptOptimizer(self._api_client)

    @property
    def aio(self):
        return self._aio

    # This is only used for replay tests
    @staticmethod
    def _get_api_client(
        api_key: Optional[str] = None,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        debug_config: Optional[genai_client.DebugConfig] = None,
        http_options: Optional[genai_client.HttpOptions] = None,
    ) -> Optional[genai_client.BaseApiClient]:
        if debug_config and debug_config.client_mode in [
            "record",
            "replay",
            "auto",
        ]:
            return genai_client.ReplayApiClient(
                mode=debug_config.client_mode,  # type: ignore[arg-type]
                replay_id=debug_config.replay_id,  # type: ignore[arg-type]
                replays_directory=debug_config.replays_directory,
                vertexai=True,  # type: ignore[arg-type]
                api_key=api_key,
                credentials=credentials,
                project=project,
                location=location,
                http_options=http_options,
            )

    @property
    def agent_engines(self):
        if self._agent_engines is None:
            try:
                # We need to lazy load the agent_engines module to handle the
                # possibility of ImportError when dependencies are not installed.
                self._agent_engines = importlib.import_module(
                    ".agent_engines",
                    __package__,
                )
            except ImportError as e:
                raise ImportError(
                    "The 'agent_engines' module requires 'additional packages'. "
                    "Please install them using pip install "
                    "google-cloud-aiplatform[agent_engines]"
                ) from e
        return self._agent_engines.AgentEngines(self._api_client)

    @property
    def prompts(self):
        if self._prompts is None:
            # Lazy loading the prompts module
            self._prompts = importlib.import_module(
                ".prompts",
                __package__,
            )
        return self._prompts.Prompts(self._api_client)

    @property
    @_common.experimental_warning(
        "The Vertex SDK GenAI datasets module is experimental, "
        "and may change in future versions."
    )
    def datasets(self):
        if self._datasets is None:
            self._datasets = importlib.import_module(
                ".datasets",
                __package__,
            )
        return self._datasets.Datasets(self._api_client)
