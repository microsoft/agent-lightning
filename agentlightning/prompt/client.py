import aiohttp
import asyncio
import logging
import requests
import time
import urllib.parse

from typing import Any, Dict, List, Optional

from .types import Task, TaskIfAny, Rollout, ResourcesUpdate, NamedResources


logger = logging.getLogger(__name__)


class AgentLightningClient:
    """
    Client for interacting with an Agent Lightning Server.

    This client handles polling for tasks, fetching resources (like model configurations),
    and posting completed rollouts back to the server. It provides both synchronous
    and asynchronous methods for these operations.
    """

    _next_task_uri = "/task"
    _resources_uri = "/resources"
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
                async with session.get(url) as resp:
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
                async with session.post(url, json=payload) as resp:
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

    async def poll_resources_async(self) -> NamedResources:
        """Polls the server asynchronously for the latest resources.

        The client agent is expected to use these resources (e.g., LLM sampling parameters)
        for its tasks.

        Returns:
            A dictionary of named resources.
        """
        url = urllib.parse.urljoin(self.endpoint, self._resources_uri)
        while True:
            response = await self._request_json_async(url)
            if response:
                resources_update = ResourcesUpdate.model_validate(response)
                logger.info(f"Resources received: {resources_update.resources}")
                return resources_update.resources
            logger.debug(f"No resources available yet. Retrying in {self.poll_interval} seconds...")
            await asyncio.sleep(self.poll_interval)

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
            response = requests.get(url, timeout=self.timeout)
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
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.debug(f"Sync POST request failed for {url}: {e}")
            return None

    def poll_next_task(self) -> Task:
        """Polls the server synchronously for the next task until one is available.

        Returns:
            A Task object containing the task details.
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

    def poll_resources(self) -> NamedResources:
        """Polls the server synchronously for the latest resources.

        The client agent is expected to use these resources (e.g., LLM sampling parameters)
        for its tasks.

        Returns:
            A dictionary of named resources.
        """
        url = urllib.parse.urljoin(self.endpoint, self._resources_uri)
        while True:
            response = self._request_json(url)
            if response:
                resources_update = ResourcesUpdate.model_validate(response)
                logger.info(f"Resources received: {resources_update.resources}")
                return resources_update.resources
            logger.debug(f"No resources available yet. Retrying in {self.poll_interval} seconds...")
            time.sleep(self.poll_interval)

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
