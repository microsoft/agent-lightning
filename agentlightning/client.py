import aiohttp
import asyncio
import logging
import requests
import time
import urllib.parse
from typing import List, TypedDict

from .trace import Transition


logger = logging.getLogger(__name__)


class TaskData(TypedDict):
    rollout_id: str
    is_train: bool


class SamplingParameters(TypedDict):
    model: str
    temperature: float


class VerlAgentClient:

    _next_task_uri = "next_data_sample"
    _sampling_parameters_uri = "train_information"
    _report_trajectory_uri = "report"

    def __init__(self, endpoint: str, poll_interval: float = 5.0, timeout: float = 10.0) -> None:
        """
        Initialize the VerlAgentClient with the given endpoint.

        :param endpoint: The root URL of the VeRL agent server.
        :param poll_interval: The interval in seconds to wait between polling for new tasks.
        :param timeout: The timeout in seconds for HTTP requests.
        """
        self.endpoint = endpoint
        self.task_count = 0
        self.poll_interval = poll_interval
        self.timeout = timeout

    @property
    def openai_endpoint(self) -> str:
        """The OpenAI endpoint for the VeRL agent server."""
        return urllib.parse.urljoin(self.endpoint, "v1")

    async def request_json_async(self, url: str) -> dict | None:
        """
        Make a GET request to the specified URL and return the JSON response.

        :param url: The URL to request.
        :return: The JSON response as a dictionary.
        """
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                logger.debug("GET request failed: %s", e)
                return None

    async def post_json_async(self, url: str, payload: dict) -> dict | None:
        """
        Make a POST request to the specified URL with the given payload and return the JSON response.

        :param url: The URL to post to.
        :param payload: The data to send in the POST request.
        :return: The JSON response as a dictionary.
        """
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                logger.debug("POST request failed: %s", e)
                return None

    async def poll_next_task_async(self) -> TaskData:
        """Poll the server for the next task data sample until it is available.

        Returns a task data dict which has the same format as the dataset sample.
        It has an extra `rollout_id` field, which is a unique identifier for the task,
        and an `is_train` field indicating whether the task is for training or evaluation.
        """
        url = urllib.parse.urljoin(self.endpoint, self._next_task_uri)
        while True:
            data = await self.request_json_async(url)
            if data and data.get("is_available"):
                task_data = data["data"]
                self.task_count += 1
                logger.info("[Task %d Received] %s", self.task_count, task_data)
                return task_data
            else:
                logger.debug("No task available yet. Retrying in 5 seconds...")
                await asyncio.sleep(self.poll_interval)

    async def poll_sampling_parameters_async(self) -> SamplingParameters:
        """Poll the server for sampling parameters until they are available.

        The client agent is expected to respect the designated sampling parameters
        when calling the LLMs, to maximize the power of the algorithms.
        """
        url = urllib.parse.urljoin(self.endpoint, self._sampling_parameters_uri)
        while True:
            data = await self.request_json_async(url)
            if data:
                logger.info("Sampling parameters received: %s", data)
                return data
            else:
                logger.debug(
                    "No sampling parameters available yet. Retrying in 5 seconds..."
                )
                await asyncio.sleep(self.poll_interval)

    async def post_trajectory_async(
        self, rollout_id: str, transitions: List[Transition]
    ) -> dict:
        url = urllib.parse.urljoin(self.endpoint, self._report_trajectory_uri)
        payload = self._to_acceptable_trajectory_payload(rollout_id, transitions)

        return await self.post_json_async(url, payload)

    def _to_acceptable_trajectory_payload(
        self, rollout_id: str, transitions: List[Transition]
    ) -> dict:
        """Convert a list of Transition objects to a payload dictionary."""
        return {
            "rollout_id": rollout_id,
            "reward": sum(t.reward for t in transitions if t.reward is not None),
            "trace_list": [
                {"prompt_ids": list(t.state), "response_ids": list(t.action)}
                for t in transitions
            ],
        }

    # Synchronous methods
    def request_json(self, url: str) -> dict | None:
        """
        Make a GET request to the specified URL and return the JSON response.

        :param url: The URL to request.
        :return: The JSON response as a dictionary.
        """
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug("GET request failed: %s", e)
            return None

    def post_json(self, url: str, payload: dict) -> dict | None:
        """
        Make a POST request to the specified URL with the given payload and return the JSON response.

        :param url: The URL to post to.
        :param payload: The data to send in the POST request.
        :return: The JSON response as a dictionary.
        """
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug("POST request failed: %s", e)
            return None

    def poll_next_task(self) -> TaskData:
        """Poll the server for the next task data sample until it is available.

        Returns a task data dict which has the same format as the dataset sample.
        It has an extra `rollout_id` field, which is a unique identifier for the task,
        and an `is_train` field indicating whether the task is for training or evaluation.
        """
        url = urllib.parse.urljoin(self.endpoint, self._next_task_uri)
        while True:
            data = self.request_json(url)
            if data and data.get("is_available"):
                task_data = data["data"]
                self.task_count += 1
                logger.info("[Task %d Received] %s", self.task_count, task_data)
                return task_data
            else:
                logger.debug("No task available yet. Retrying in %s seconds...", self.poll_interval)
                time.sleep(self.poll_interval)

    def poll_sampling_parameters(self) -> SamplingParameters:
        """Poll the server for sampling parameters until they are available.

        The client agent is expected to respect the designated sampling parameters
        when calling the LLMs, to maximize the power of the algorithms.
        """
        url = urllib.parse.urljoin(self.endpoint, self._sampling_parameters_uri)
        while True:
            data = self.request_json(url)
            if data:
                logger.info("Sampling parameters received: %s", data)
                return data
            else:
                logger.debug(
                    "No sampling parameters available yet. Retrying in %s seconds...",
                    self.poll_interval
                )
                time.sleep(self.poll_interval)

    def post_trajectory(
        self, rollout_id: str, transitions: List[Transition]
    ) -> dict:
        """Post a trajectory to the server synchronously.

        :param rollout_id: The unique identifier for the rollout.
        :param transitions: List of transitions in the trajectory.
        :return: The server response as a dictionary.
        """
        url = urllib.parse.urljoin(self.endpoint, self._report_trajectory_uri)
        payload = self._to_acceptable_trajectory_payload(rollout_id, transitions)
        return self.post_json(url, payload)
