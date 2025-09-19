# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import time
import urllib.parse
from typing import Any, Dict, Optional, List, Union

import aiohttp
import requests

from .types import Rollout, Task, TaskInput, TaskIfAny, ResourcesUpdate, NamedResources


logger = logging.getLogger(__name__)


class AgentLightningClient:
    """
    Client for interacting with a version-aware Agent Lightning Server.

    This client handles polling for tasks, fetching specific versions of resources
    (like model configurations), and posting completed rollouts back to the server.
    It provides both synchronous and asynchronous methods for these operations and
    includes a cache for resources.
    """

    _next_task_uri = "/task"
    _resources_uri = "/resources"
    _latest_resources_uri = "/resources/latest"
    _report_rollout_uri = "/rollout"

    def __init__(self, endpoint: str, poll_interval: float = 5.0, timeout: float = 10.0):
        """Initializes the AgentLightningClient.

        Args:
            endpoint: The root URL of the Agent Lightning server.
            poll_interval: The interval in seconds to wait between polling for new tasks.
            timeout: The timeout in seconds for HTTP requests.
        """
        self.endpoint = endpoint
        self.task_count = 0
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._resource_cache: Dict[str, ResourcesUpdate] = {}  # TODO: mechanism to evict cache
        self._default_headers = {"X-AgentLightning-Client": "true"}

    async def _request_json_async(self, url: str) -> Optional[Dict[str, Any]]:
        """Makes an async GET request to the specified URL and returns the JSON response.

        Args:
            url: The URL to request.

        Returns:
            The JSON response as a dictionary or None if the request fails.
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
        """Makes an async POST request with a JSON payload.

        Args:
            url: The URL to post to.
            payload: The dictionary data to send as JSON.

        Returns:
            The JSON response as a dictionary or None if the request fails.
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

    async def poll_next_task_async(self) -> Task:
        """Polls the server asynchronously for the next task until one is available.

        Returns:
            A Task object containing the task details.
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
        """Fetches a specific version of resources by its ID, using a cache.

        Args:
            resource_id: The ID of the resources to fetch, usually from a Task's metadata.

        Returns:
            A ResourcesUpdate object containing the versioned resources, or None if not found.
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
        """Fetches the latest available resources from the server.

        Returns:
            A ResourcesUpdate object containing the latest resources.
        """
        url = urllib.parse.urljoin(self.endpoint, self._latest_resources_uri)
        response = await self._request_json_async(url)
        if response:
            resources_update = ResourcesUpdate.model_validate(response)
            # Cache this result as well
            self._resource_cache[resources_update.resources_id] = resources_update
            return resources_update
        return None

    async def post_rollout_async(self, rollout: Rollout) -> Optional[Dict[str, Any]]:
        """Posts a completed rollout to the server asynchronously.

        Args:
            rollout: A Rollout object containing the results of a task.

        Returns:
            The server's JSON response as a dictionary.
        """
        url = urllib.parse.urljoin(self.endpoint, self._report_rollout_uri)
        payload = rollout.model_dump(mode="json")
        return await self._post_json_async(url, payload)

    def _request_json(self, url: str) -> Optional[Dict[str, Any]]:
        """Makes a sync GET request to the specified URL and returns the JSON response.

        Args:
            url: The URL to request.

        Returns:
            The JSON response as a dictionary or None if the request fails.
        """
        try:
            response = requests.get(url, timeout=self.timeout, headers=self._default_headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.debug(f"Sync GET request failed for {url}: {e}")
            return None

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Makes a sync POST request with a JSON payload.

        Args:
            url: The URL to post to.
            payload: The dictionary data to send as JSON.

        Returns:
            The JSON response as a dictionary or None if the request fails.
        """
        try:
            response = requests.post(url, json=payload, timeout=self.timeout, headers=self._default_headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.debug(f"Sync POST request failed for {url}: {e}")
            return None

    def poll_next_task(self) -> Task:
        """Polls the server synchronously for the next task until one is available.

        Returns:
            A Task object containing the task details, including the required `resources_id`.
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
        """Fetches a specific version of resources by its ID synchronously, using a cache.

        Args:
            resource_id: The ID of the resources to fetch, usually from a Task's metadata.

        Returns:
            A ResourcesUpdate object containing the versioned resources, or None if not found.
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
        """Fetches the latest available resources from the server synchronously.

        Returns:
            A ResourcesUpdate object containing the latest resources.
        """
        url = urllib.parse.urljoin(self.endpoint, self._latest_resources_uri)
        response = self._request_json(url)
        if response:
            resources_update = ResourcesUpdate.model_validate(response)
            self._resource_cache[resources_update.resources_id] = resources_update
            return resources_update
        return None

    def post_rollout(self, rollout: Rollout) -> Optional[Dict[str, Any]]:
        """Posts a completed rollout to the server synchronously.

        Args:
            rollout: A Rollout object containing the results of a task.

        Returns:
            The server's JSON response as a dictionary.
        """
        url = urllib.parse.urljoin(self.endpoint, self._report_rollout_uri)
        payload = rollout.model_dump(mode="json")
        return self._post_json(url, payload)


class DevTaskLoader(AgentLightningClient):
    """A local task manager for development that provides sample tasks and resources.

    This client mocks the server APIs by maintaining a local queue of tasks and resources
    within the same process. It's designed for development, testing, and scenarios where
    a full Agent Lightning server is not needed.

    The DevTaskLoader overrides the polling and resource fetching methods to return data
    from local collections instead of making HTTP requests to a remote server.
    """

    def __init__(
        self,
        tasks: Union[List[TaskInput], List[Task]],
        resources: Union[NamedResources, ResourcesUpdate],
        **kwargs: Any,
    ):
        """Initializes the DevTaskLoader with pre-defined tasks and resources.

        Args:
            tasks: Either a List of TaskInput objects or a List of Task objects.
            resources: Either NamedResources or ResourcesUpdate object.
            **kwargs: Additional arguments passed to the parent AgentLightningClient.
        """
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
            self._resources_update = ResourcesUpdate(resources_id="local", resources=resources)

        # Store rollouts posted back to the loader for easy debugging of local runs
        self._rollouts: List[Rollout] = []

    @property
    def rollouts(self) -> List[Rollout]:
        """Return rollouts that have been posted back to the loader."""
        return self._rollouts

    def poll_next_task(self) -> Task:
        """Returns the next task from the local queue.

        If tasks are TaskInput objects, assembles them into Task objects.
        If tasks are already Task objects, returns them directly.

        Returns:
            The next Task object from the local task list.
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

    def post_rollout(self, rollout: Rollout) -> Optional[Dict[str, Any]]:
        logger.debug(f"DevTaskLoader received rollout for task: {rollout.rollout_id}")
        self._rollouts.append(rollout)
        return {"status": "received", "rollout_id": rollout.rollout_id}

    async def poll_next_task_async(self) -> Task:
        return self.poll_next_task()

    async def get_resources_by_id_async(self, resource_id: str) -> Optional[ResourcesUpdate]:
        return self.get_resources_by_id(resource_id)

    async def get_latest_resources_async(self) -> Optional[ResourcesUpdate]:
        return self.get_latest_resources()

    async def post_rollout_async(self, rollout: Rollout) -> Optional[Dict[str, Any]]:
        return self.post_rollout(rollout)

    def __repr__(self):
        return f"DevTaskLoader(num_tasks={len(self._tasks)}, resources={self._resources_update.resources})"
