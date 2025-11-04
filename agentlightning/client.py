# Copyright (c) Microsoft. All rights reserved.

"""Utilities for interacting with legacy Agent Lightning servers.

This module contains compatibility shims that speak the deprecated HTTP
interface used by older Agent Lightning deployments. Modern code should prefer
the store-based APIs exposed by `agentlightning.store`, but keeping these
clients available makes it easier to migrate existing workflows incrementally.
"""

import asyncio
import logging
import time
import urllib.parse
import warnings
from typing import Any, Dict, List, Optional, Union

import aiohttp
import requests

from .types import NamedResources, ResourcesUpdate, RolloutLegacy, Task, TaskIfAny, TaskInput

logger = logging.getLogger(__name__)


class AgentLightningClient:
    """Client wrapper for the legacy version-aware Agent Lightning server.

    The client exposes synchronous and asynchronous helpers for polling tasks,
    retrieving resource bundles, and submitting rollouts. It also maintains a
    simple in-memory cache keyed by the server-provided resource identifier to
    avoid redundant network requests.

    !!! warning "Deprecated"
        [`AgentLightningClient`][agentlightning.client.AgentLightningClient] is part of
        the legacy client/server stack. New code should rely on the store-based APIs
        implemented in `agentlightning.store`.

    Attributes:
        endpoint: Base URL of the Agent Lightning server.
        poll_interval: Delay in seconds between polling attempts when no task is
            available.
        timeout: Timeout in seconds applied to HTTP requests.
        task_count: Number of tasks claimed during the lifetime of this client.
    """

    _next_task_uri = "/task"
    _resources_uri = "/resources"
    _latest_resources_uri = "/resources/latest"
    _report_rollout_uri = "/rollout"

    def __init__(self, endpoint: str, poll_interval: float = 5.0, timeout: float = 10.0):
        """Initialize the client.

        Args:
            endpoint: Root URL of the Agent Lightning server.
            poll_interval: Seconds to wait between polling attempts.
            timeout: Seconds before a request to the server is considered timed out.
        """
        warnings.warn(
            "AgentLightningClient is deprecated. Please use LightningStoreClient instead.", DeprecationWarning
        )
        self.endpoint = endpoint
        self.task_count = 0
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._resource_cache: Dict[str, ResourcesUpdate] = {}  # TODO: mechanism to evict cache
        self._default_headers = {"X-AgentLightning-Client": "true"}

    async def _request_json_async(self, url: str) -> Optional[Dict[str, Any]]:
        """Perform an asynchronous ``GET`` request and parse the JSON payload.

        Args:
            url: Fully qualified URL to query.

        Returns:
            Parsed JSON body as a dictionary if the request succeeds; otherwise ``None``.
        """
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(url, headers=self._default_headers) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                logger.debug(f"Async GET request failed for {url}: {e}")
                return None

    async def _post_json_async(self, url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform an asynchronous ``POST`` request with a JSON body.

        Args:
            url: Fully qualified URL that accepts the payload.
            payload: Dictionary that will be serialized and sent as JSON.

        Returns:
            Parsed JSON body as a dictionary if the request succeeds; otherwise ``None``.
        """
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload, headers=self._default_headers) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                logger.debug(f"Async POST request failed for {url}: {e}")
                return None

    async def poll_next_task_async(self) -> Optional[Task]:
        """Poll the server asynchronously until a task becomes available.

        Returns:
            The next [`Task`][agentlightning.Task] exposed by the server,
            or ``None`` if polling fails.
        """
        url = urllib.parse.urljoin(self.endpoint, self._next_task_uri)
        while True:
            response = await self._request_json_async(url)
            if response:
                task_if_any = TaskIfAny.model_validate(response)
                if task_if_any.is_available and task_if_any.task:
                    self.task_count += 1
                    logger.info(f"[Task {self.task_count} Received] ID: {task_if_any.task.rollout_id}")
                    return task_if_any.task
            logger.debug(f"No task available yet. Retrying in {self.poll_interval} seconds...")
            await asyncio.sleep(self.poll_interval)

    async def get_resources_by_id_async(self, resource_id: str) -> Optional[ResourcesUpdate]:
        """Fetch a specific resource bundle by identifier.

        Args:
            resource_id: Identifier sourced from the task metadata.

        Returns:
            Cached or freshly downloaded
            [`ResourcesUpdate`][agentlightning.ResourcesUpdate], or
            ``None`` when the server returns an error.
        """
        if resource_id in self._resource_cache:
            logger.debug(f"Found resources '{resource_id}' in cache.")
            return self._resource_cache[resource_id]

        url = urllib.parse.urljoin(self.endpoint, f"{self._resources_uri}/{resource_id}")
        response = await self._request_json_async(url)
        if response:
            resources_update = ResourcesUpdate.model_validate(response)
            self._resource_cache[resource_id] = resources_update
            logger.info(f"Fetched and cached resources for ID: {resource_id}")
            return resources_update
        return None

    async def get_latest_resources_async(self) -> Optional[ResourcesUpdate]:
        """Fetch the most recent resource bundle advertised by the server.

        Returns:
            [`ResourcesUpdate`][agentlightning.ResourcesUpdate] for the
            newest version, or ``None`` when unavailable.
        """
        url = urllib.parse.urljoin(self.endpoint, self._latest_resources_uri)
        response = await self._request_json_async(url)
        if response:
            resources_update = ResourcesUpdate.model_validate(response)
            # Cache this result as well
            self._resource_cache[resources_update.resources_id] = resources_update
            return resources_update
        return None

    async def post_rollout_async(self, rollout: RolloutLegacy) -> Optional[Dict[str, Any]]:
        """Submit a completed rollout back to the server.

        Args:
            rollout: Legacy rollout payload produced by the executor.

        Returns:
            Parsed JSON response returned by the server, or ``None`` when the request fails.
        """
        url = urllib.parse.urljoin(self.endpoint, self._report_rollout_uri)
        payload = rollout.model_dump(mode="json")
        return await self._post_json_async(url, payload)

    def _request_json(self, url: str) -> Optional[Dict[str, Any]]:
        """Perform a blocking ``GET`` request and parse the JSON payload.

        Args:
            url: Fully qualified URL to query.

        Returns:
            Parsed JSON body as a dictionary if the request succeeds; otherwise ``None``.
        """
        try:
            response = requests.get(url, timeout=self.timeout, headers=self._default_headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.debug(f"Sync GET request failed for {url}: {e}")
            return None

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform a blocking ``POST`` request with a JSON payload.

        Args:
            url: Fully qualified URL that accepts the payload.
            payload: Dictionary that will be serialized and sent as JSON.

        Returns:
            Parsed JSON body as a dictionary if the request succeeds; otherwise ``None``.
        """
        try:
            response = requests.post(url, json=payload, timeout=self.timeout, headers=self._default_headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.debug(f"Sync POST request failed for {url}: {e}")
            return None

    def poll_next_task(self) -> Optional[Task]:
        """Poll the server synchronously until a task becomes available.

        Returns:
            The next [`Task`][agentlightning.Task] available for execution, or
            ``None`` if polling fails.
        """
        url = urllib.parse.urljoin(self.endpoint, self._next_task_uri)
        while True:
            response = self._request_json(url)
            if response:
                task_if_any = TaskIfAny.model_validate(response)
                if task_if_any.is_available and task_if_any.task:
                    self.task_count += 1
                    logger.info(f"[Task {self.task_count} Received] ID: {task_if_any.task.rollout_id}")
                    return task_if_any.task
            logger.debug(f"No task available yet. Retrying in {self.poll_interval} seconds...")
            time.sleep(self.poll_interval)

    def get_resources_by_id(self, resource_id: str) -> Optional[ResourcesUpdate]:
        """Fetch a specific resource bundle by identifier.

        Args:
            resource_id: Identifier sourced from the task metadata.

        Returns:
            Cached or freshly downloaded
            [`ResourcesUpdate`][agentlightning.ResourcesUpdate], or
            ``None`` when the server returns an error.
        """
        if resource_id in self._resource_cache:
            logger.debug(f"Found resources '{resource_id}' in cache.")
            return self._resource_cache[resource_id]

        url = urllib.parse.urljoin(self.endpoint, f"{self._resources_uri}/{resource_id}")
        response = self._request_json(url)
        if response:
            resources_update = ResourcesUpdate.model_validate(response)
            self._resource_cache[resource_id] = resources_update
            logger.info(f"Fetched and cached resources for ID: {resource_id}")
            return resources_update
        return None

    def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """Fetch the most recent resource bundle advertised by the server.

        Returns:
            [`ResourcesUpdate`][agentlightning.ResourcesUpdate] for the
            newest version, or ``None`` when unavailable.
        """
        url = urllib.parse.urljoin(self.endpoint, self._latest_resources_uri)
        response = self._request_json(url)
        if response:
            resources_update = ResourcesUpdate.model_validate(response)
            self._resource_cache[resources_update.resources_id] = resources_update
            return resources_update
        return None

    def post_rollout(self, rollout: RolloutLegacy) -> Optional[Dict[str, Any]]:
        """Submit a completed rollout back to the server.

        Args:
            rollout: Legacy rollout payload produced by the executor.

        Returns:
            Parsed JSON response returned by the server, or ``None`` when the request fails.
        """
        url = urllib.parse.urljoin(self.endpoint, self._report_rollout_uri)
        payload = rollout.model_dump(mode="json")
        return self._post_json(url, payload)


class DevTaskLoader(AgentLightningClient):
    """In-memory task loader used for development and integration tests.

    The loader mimics the behavior of the legacy HTTP server by storing tasks and
    resources locally. Polling methods simply iterate over the provided collection,
    allowing rapid iteration without provisioning any external infrastructure.

    !!! warning "Deprecated"

        [`DevTaskLoader`][agentlightning.client.DevTaskLoader] is a compatibility shim.
        Prefer [`Trainer.dev`][agentlightning.Trainer.dev] for new code.
    """

    def __init__(
        self,
        tasks: Union[List[TaskInput], List[Task]],
        resources: Union[NamedResources, ResourcesUpdate],
        **kwargs: Any,
    ):
        """Initialize the loader with predefined tasks and resources.

        Args:
            tasks: Sequence of task inputs or preconstructed tasks that will be served in
                order.
            resources: Static resources returned for any `resources_id` query.
            **kwargs: Additional keyword arguments forwarded to the parent client.

        Raises:
            ValueError: If no tasks are provided or both [`Task`][agentlightning.Task]
                and [`TaskInput`][agentlightning.TaskInput] instances are mixed.
        """
        warnings.warn("DevTaskLoader is deprecated. Please use Trainer.dev instead.", DeprecationWarning)
        super().__init__(endpoint="local://", **kwargs)
        self._tasks = tasks.copy()
        if len(self._tasks) == 0:
            raise ValueError("DevTaskLoader requires at least one task to be provided.")

        # Check if tasks are mixture of TaskInput and Task
        if any(isinstance(task, Task) for task in self._tasks):
            if not all(isinstance(task, Task) for task in self._tasks):
                raise ValueError("All tasks must be either Task or TaskInput objects.")

        self._task_index = 0

        if isinstance(resources, ResourcesUpdate):
            self._resources_update = resources
        else:
            self._resources_update = ResourcesUpdate(
                resources_id="local", resources=resources, create_time=time.time(), update_time=time.time(), version=1
            )

        # Store rollouts posted back to the loader for easy debugging of local runs
        self._rollouts: List[RolloutLegacy] = []

    @property
    def rollouts(self) -> List[RolloutLegacy]:
        """Return the rollouts posted back to the loader during development runs."""
        return self._rollouts

    def poll_next_task(self) -> Optional[Task]:
        """Return the next task from the local queue.

        If [`TaskInput`][agentlightning.TaskInput] instances were provided,
        they are converted into [`Task`][agentlightning.Task] objects on the
        fly. Otherwise, the preconstructed tasks are returned in sequence.

        Returns:
            Next task to execute.
        """
        if self._task_index >= len(self._tasks):
            self._task_index = 0

        task_or_input = self._tasks[self._task_index]

        if isinstance(task_or_input, Task):
            task = task_or_input
        else:
            rollout_id = f"local_task_{self._task_index + 1:03d}"
            task = Task(
                rollout_id=rollout_id,
                input=task_or_input,
                resources_id=self._resources_update.resources_id,
                create_time=time.time(),
            )

        self._task_index += 1
        self.task_count += 1
        logger.info(f"[Task {self.task_count} Received] Task ID: {task.rollout_id}")
        return task

    def get_resources_by_id(self, resource_id: str) -> Optional[ResourcesUpdate]:
        logger.debug(f"DevTaskLoader checking resources for ID: {resource_id}")
        if resource_id != self._resources_update.resources_id:
            raise ValueError(
                f"Resource ID '{resource_id}' not found. Only '{self._resources_update.resources_id}' is available."
            )
        return self._resources_update

    def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        logger.debug("DevTaskLoader returning latest resources.")
        return self._resources_update

    def post_rollout(self, rollout: RolloutLegacy) -> Optional[Dict[str, Any]]:
        logger.debug(f"DevTaskLoader received rollout for task: {rollout.rollout_id}")
        self._rollouts.append(rollout)
        return {"status": "received", "rollout_id": rollout.rollout_id}

    async def poll_next_task_async(self) -> Optional[Task]:
        return self.poll_next_task()

    async def get_resources_by_id_async(self, resource_id: str) -> Optional[ResourcesUpdate]:
        return self.get_resources_by_id(resource_id)

    async def get_latest_resources_async(self) -> Optional[ResourcesUpdate]:
        return self.get_latest_resources()

    async def post_rollout_async(self, rollout: RolloutLegacy) -> Optional[Dict[str, Any]]:
        return self.post_rollout(rollout)

    def __repr__(self):
        return f"DevTaskLoader(num_tasks={len(self._tasks)}, resources={self._resources_update.resources})"
