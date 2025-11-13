# -*- coding: utf-8 -*-
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

import os
from typing import Any, Callable, Dict, List, Mapping, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    try:
        from a2a.server.request_handlers import RequestHandler
        from a2a.server.tasks import TaskStore
        from a2a.types import AgentCard, AgentSkill
        from a2a.server.agent_execution import AgentExecutor
        from a2a.server.context import ServerCallContext

        RequestHandler = RequestHandler
        TaskStore = TaskStore
        AgentCard = AgentCard
        AgentSkill = AgentSkill
        AgentExecutor = AgentExecutor
        ServerCallContext = ServerCallContext
    except (ImportError, AttributeError):
        RequestHandler = Any
        TaskStore = Any
        AgentCard = Any
        AgentSkill = Any
        AgentExecutor = Any
        ServerCallContext = Any

    try:
        from fastapi import Request

        Request = Request
    except (ImportError, AttributeError):
        Request = Any


def create_agent_card(
    agent_name: Optional[str] = None,
    description: Optional[str] = None,
    skills: Optional[List["AgentSkill"]] = None,
    agent_card: Optional[Dict[str, Any]] = None,
) -> "AgentCard":
    """Creates an AgentCard object.

    The function can be called in two ways:
    1. By providing the individual parameters: agent_name, description, and skills.
    2. By providing a single dictionary containing all the data.

    If a dictionary is provided, the other parameters are ignored.

    Args:
        agent_name (Optional[str]): The name of the agent.
        description (Optional[str]): A description of the agent.
        skills (Optional[List[AgentSkill]]): A list of AgentSkills.
        agent_card (Optional[Dict[str, Any]]): Agent Card as a dictionary.

    Returns:
        AgentCard: A fully constructed AgentCard object.

    Raises:
        ValueError: If neither a dictionary nor the required parameters are provided.
    """
    # pylint: disable=g-import-not-at-top
    from a2a.types import AgentCard, AgentCapabilities, TransportProtocol

    # Check if a dictionary was provided.
    if agent_card:
        return AgentCard(**agent_card)

    # If no dictionary, use the individual parameters.
    elif agent_name and description and skills:
        return AgentCard(
            name=agent_name,
            description=description,
            url="http://localhost:9999/",
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            # Agent Engine does not support streaming yet
            capabilities=AgentCapabilities(streaming=False),
            skills=skills,
            preferred_transport=TransportProtocol.http_json,  # Http Only.
            supports_authenticated_extended_card=True,
        )

    # Raise an error if insufficient data is provided.
    else:
        raise ValueError(
            "Please provide either an agent_card or all of the required "
            "parameters (agent_name, description, and skills)."
        )


def default_a2a_agent() -> "A2aAgent":
    """Creates a default A2aAgent instance."""
    # pylint: disable=g-import-not-at-top
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.types import AgentSkill
    from a2a.server.events import EventQueue
    from a2a.utils import new_agent_text_message

    skill = AgentSkill(
        id="hello_world",
        name="Returns hello world",
        description="just returns hello world",
        tags=["hello world"],
        examples=["hi", "hello world"],
    )
    agent_card = create_agent_card(
        agent_name="Hello World Agent",
        description="Just a hello world agent",
        skills=[skill],
    )

    class HelloWorldAgentExecutor(AgentExecutor):
        """Hello World Agent Executor."""

        def get_agent_response(self) -> str:
            return "Hello World"

        async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue,
        ) -> None:
            result = self.get_agent_response()
            await event_queue.enqueue_event(new_agent_text_message(result))

        async def cancel(
            self, context: RequestContext, event_queue: EventQueue
        ) -> None:
            raise Exception("cancel not supported")

    return A2aAgent(
        agent_card=agent_card,
        agent_executor_builder=HelloWorldAgentExecutor,
    )


class A2aAgent:
    """A class to initialize and set up an Agent-to-Agent application."""

    # TODO: Add instrumentation for the A2A agent.
    def __init__(
        self,
        *,
        agent_card: "AgentCard",
        task_store_builder: Callable[..., "TaskStore"] = None,
        task_store_kwargs: Optional[Mapping[str, Any]] = None,
        agent_executor_kwargs: Optional[Mapping[str, Any]] = None,
        agent_executor_builder: Optional[Callable[..., "AgentExecutor"]] = None,
        request_handler_kwargs: Optional[Mapping[str, Any]] = None,
        request_handler_builder: Optional[Callable[..., "RequestHandler"]] = None,
        extended_agent_card: "AgentCard" = None,
    ):
        """Initializes the A2A agent."""
        # pylint: disable=g-import-not-at-top
        from google.cloud.aiplatform import initializer
        from a2a.types import TransportProtocol

        if (
            agent_card.preferred_transport
            and agent_card.preferred_transport != TransportProtocol.http_json
        ):
            raise ValueError(
                "Only HTTP+JSON is supported for preferred transport on agent card "
            )
        if agent_card.capabilities and agent_card.capabilities.streaming:
            raise ValueError("Streaming is not supported by Agent Engine")

        self._tmpl_attrs: dict[str, Any] = {
            "project": initializer.global_config.project,
            "location": initializer.global_config.location,
            "agent_card": agent_card,
            "agent_executor": None,
            "agent_executor_kwargs": agent_executor_kwargs or {},
            "agent_executor_builder": agent_executor_builder,
            "task_store": None,
            "task_store_kwargs": task_store_kwargs or {},
            "task_store_builder": task_store_builder,
            "request_handler": None,
            "request_handler_kwargs": request_handler_kwargs or {},
            "request_handler_builder": request_handler_builder,
            "extended_agent_card": extended_agent_card,
        }
        self.agent_card = agent_card
        self.a2a_rest_adapter = None
        self.request_handler = None
        self.rest_handler = None
        self.task_store = None
        self.agent_executor = None

    def clone(self) -> "A2aAgent":
        """Clones the A2A agent."""
        import copy

        return A2aAgent(
            agent_card=copy.deepcopy(self.agent_card),
            task_store_builder=self._tmpl_attrs.get("task_store_builder"),
            task_store_kwargs=self._tmpl_attrs.get("task_store_kwargs"),
            agent_executor_kwargs=self._tmpl_attrs.get("agent_executor_kwargs"),
            agent_executor_builder=self._tmpl_attrs.get("agent_executor_builder"),
            request_handler_kwargs=self._tmpl_attrs.get("request_handler_kwargs"),
            request_handler_builder=self._tmpl_attrs.get("request_handler_builder"),
            extended_agent_card=self._tmpl_attrs.get("extended_agent_card"),
        )

    def set_up(self):
        """Sets up the A2A application."""
        # pylint: disable=g-import-not-at-top
        from a2a.server.apps.rest.rest_adapter import RESTAdapter
        from a2a.server.request_handlers.rest_handler import RESTHandler
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.tasks import InMemoryTaskStore

        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
        project = self._tmpl_attrs.get("project")
        os.environ["GOOGLE_CLOUD_PROJECT"] = project
        location = self._tmpl_attrs.get("location")
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        agent_engine_id = os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "test-agent-engine")
        version = "v1beta1"

        self.agent_card.url = f"https://{location}-aiplatform.googleapis.com/{version}/projects/{project}/locations/{location}/reasoningEngines/{agent_engine_id}/a2a"
        self._tmpl_attrs["agent_card"] = self.agent_card

        # Create the agent executor if a builder is provided.
        agent_executor_builder = self._tmpl_attrs.get("agent_executor_builder")
        if agent_executor_builder:
            self._tmpl_attrs["agent_executor"] = agent_executor_builder(
                **self._tmpl_attrs.get("agent_executor_kwargs")
            )
            self.agent_executor = self._tmpl_attrs.get("agent_executor")

        # Create the task store if a builder is provided.
        task_store_builder = self._tmpl_attrs.get("task_store_builder")
        if task_store_builder:
            self.task_store = task_store_builder(
                **self._tmpl_attrs.get("task_store_kwargs")
            )
        else:
            # Use the default task store if not provided. This could potentially
            # lead to unexpected behavior if the agent is running on
            # multiple instances.
            self.task_store = InMemoryTaskStore()

        self._tmpl_attrs["task_store"] = self.task_store

        # Create the request handler if a builder is provided.
        request_handler_builder = self._tmpl_attrs.get("request_handler_builder")
        if request_handler_builder:
            self.request_handler = request_handler_builder(
                **self._tmpl_attrs.get("request_handler_kwargs")
            )
        else:
            # Use the default request handler if not provided.
            self.request_handler = DefaultRequestHandler(
                agent_executor=self._tmpl_attrs.get("agent_executor"),
                task_store=self.task_store,
            )

        self._tmpl_attrs["request_handler"] = self.request_handler

        # a2a_rest_adapter is used to register the A2A API routes in the
        # Reasoning Engine API router.
        self.a2a_rest_adapter = RESTAdapter(
            agent_card=self.agent_card,
            http_handler=self._tmpl_attrs.get("request_handler"),
            extended_agent_card=self._tmpl_attrs.get("extended_agent_card"),
        )

        # rest_handler is used to handle the A2A API requests.
        self.rest_handler = RESTHandler(
            agent_card=self.agent_card,
            request_handler=self._tmpl_attrs.get("request_handler"),
        )

    async def on_message_send(
        self,
        request: "Request",
        context: "ServerCallContext",
    ) -> dict[str, Any]:
        return await self.rest_handler.on_message_send(request, context)

    async def on_cancel_task(
        self,
        request: "Request",
        context: "ServerCallContext",
    ) -> dict[str, Any]:
        return await self.rest_handler.on_cancel_task(request, context)

    async def on_get_task(
        self,
        request: "Request",
        context: "ServerCallContext",
    ) -> dict[str, Any]:
        return await self.rest_handler.on_get_task(request, context)

    async def handle_authenticated_agent_card(
        self,
        request: "Request",
        context: "ServerCallContext",
    ) -> dict[str, Any]:
        return await self.a2a_rest_adapter.handle_authenticated_agent_card(
            request, context
        )

    def register_operations(self) -> Dict[str, List[str]]:
        """Registers the operations of the A2A Agent."""
        routes = {
            "a2a_extension": [
                "on_message_send",
                "on_get_task",
                "on_cancel_task",
            ]
        }
        if self.agent_card.supports_authenticated_extended_card:
            routes["a2a_extension"].append("handle_authenticated_agent_card")
        return routes
