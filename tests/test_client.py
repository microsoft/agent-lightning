# Copyright (c) Microsoft. All rights reserved.

# type: ignore

import asyncio
import time
from typing import Any, AsyncGenerator, Dict

import pytest
import pytest_asyncio
from httpx import AsyncClient

from agentlightning import (
    LLM,
    AgentLightningClient,
    AgentLightningServer,
    NamedResources,
    PromptTemplate,
    ResourcesUpdate,
    RolloutLegacy,
    Task,
    Triplet,
)
from agentlightning.client import DevTaskLoader


@pytest.fixture
def sample_resources() -> NamedResources:
    """Provides a sample NamedResources object for testing."""
    return {
        "main_llm": LLM(
            endpoint="http://localhost:8080/v1/chat/completions",
            model="gpt-4o",
            sampling_parameters={"temperature": 0.8, "max_tokens": 500},
        ),
        "system_prompt": PromptTemplate(template="You are a master of {domain}.", engine="f-string"),
    }


@pytest.fixture
def sample_task_input() -> Dict[str, Any]:
    """Provides a sample input for a task."""
    return {"prompt": "Tell me about the Roman Empire."}


@pytest_asyncio.fixture
async def server_setup() -> AsyncGenerator[Dict[str, Any], None]:
    """
    A pytest fixture to manage the lifecycle of the AgentLightningServer.
    It starts the server before a test runs and stops it afterward,
    providing the server instance and a test client to the test function.
    """
    server = AgentLightningServer(host="127.0.0.1", port=8008, task_timeout_seconds=2.0)
    await server.start()

    # httpx.AsyncClient is used for direct HTTP requests to the server endpoints
    # within the tests, which is more reliable than spinning up a full AgentLightningClient
    # for every single check.
    async with AsyncClient(base_url=server.endpoint) as http_client:
        yield {
            "server": server,
            "http_client": http_client,
            "endpoint": server.endpoint,
        }

    await server.stop()


@pytest.mark.asyncio
async def test_uri_and_semantic_correctness(server_setup: Dict[str, Any], sample_resources: NamedResources):
    """
    Ensures that client and server URIs match and that data models
    are serialized and deserialized correctly without semantic loss.
    """
    server: AgentLightningServer = server_setup["server"]
    endpoint: str = server_setup["endpoint"]

    # 1. Update resources on the server
    resources_id = await server.update_resources(sample_resources)
    assert isinstance(resources_id, str)

    # 2. Initialize a client pointing to the server
    client = AgentLightningClient(endpoint=endpoint)

    # 3. Fetch latest resources and verify integrity
    # This checks the /resources/latest URI and the ResourcesUpdate model
    latest_res_update = await client.get_latest_resources_async()
    assert latest_res_update is not None
    assert latest_res_update.resources_id == resources_id
    assert latest_res_update.resources["main_llm"].model == "gpt-4o"
    assert isinstance(latest_res_update.resources["system_prompt"], PromptTemplate)
    assert latest_res_update.resources["system_prompt"].engine == "f-string"

    # 4. Fetch resources by ID and verify integrity
    # This checks the /resources/{resource_id} URI
    specific_res_update = await client.get_resources_by_id_async(resources_id)
    assert specific_res_update is not None
    assert specific_res_update.model_dump() == latest_res_update.model_dump()


@pytest.mark.asyncio
async def test_full_lifecycle_async(
    server_setup: Dict[str, Any], sample_resources: NamedResources, sample_task_input: Dict[str, Any]
):
    """
    Tests the complete asynchronous workflow:
    1. Server queues a task.
    2. Async client polls for the task.
    3. Async client fetches resources for the task.
    4. Async client posts a completed rollout.
    5. Server retrieves the rollout.
    """
    server: AgentLightningServer = server_setup["server"]
    endpoint: str = server_setup["endpoint"]
    client = AgentLightningClient(endpoint=endpoint, poll_interval=0.1)

    # 1. Server updates resources and queues a task
    resources_id = await server.update_resources(sample_resources)
    rollout_id = await server.queue_task(sample=sample_task_input, mode="train", resources_id=resources_id)

    # 2. Client polls for the task
    task = await client.poll_next_task_async()
    assert task is not None
    assert task.rollout_id == rollout_id
    assert task.input == sample_task_input
    assert task.resources_id == resources_id  # Task is correctly associated with latest resources

    # 3. Client fetches resources
    res_update = await client.get_resources_by_id_async(task.resources_id)
    assert res_update is not None
    assert res_update.resources_id == resources_id

    # 4. Client posts a completed rollout
    rollout_payload = RolloutLegacy(
        rollout_id=rollout_id,
        final_reward=0.95,
        triplets=[Triplet(prompt="q", response="a", reward=1.0)],
        metadata={"client_version": "1.0"},
    )
    response = await client.post_rollout_async(rollout_payload)
    assert response is not None
    assert response.get("status") == "ok"

    # 5. Server retrieves the rollout and verifies its content
    completed_rollout = await server.get_completed_rollout(rollout_id)
    assert completed_rollout is not None
    assert completed_rollout.rollout_id == rollout_id
    assert completed_rollout.final_reward == 0.95
    assert completed_rollout.triplets[0].response == "a"
    assert completed_rollout.metadata == {"client_version": "1.0"}


@pytest.mark.asyncio
async def test_full_lifecycle_sync(
    server_setup: Dict[str, Any], sample_resources: NamedResources, sample_task_input: Dict[str, Any]
):
    """
    Tests the complete synchronous workflow in a separate thread to avoid
    blocking the asyncio event loop.
    """
    server: AgentLightningServer = server_setup["server"]
    endpoint: str = server_setup["endpoint"]
    client = AgentLightningClient(endpoint=endpoint, poll_interval=0.1)

    # 1. Server updates resources and queues a task
    resources_id = await server.update_resources(sample_resources)
    rollout_id = await server.queue_task(sample=sample_task_input, mode="val")

    # Define the synchronous client workflow
    def sync_client_workflow():
        # 2. Client polls for the task
        task = client.poll_next_task()
        assert task is not None
        assert task.rollout_id == rollout_id
        assert task.input == sample_task_input
        assert task.resources_id is None

        # 3. Client fetches resources
        res_update = client.get_latest_resources()
        assert res_update is not None
        assert res_update.resources_id == resources_id

        # 4. Client posts a completed rollout
        rollout_payload = RolloutLegacy(rollout_id=rollout_id, final_reward=0.88)
        response = client.post_rollout(rollout_payload)
        assert response is not None
        assert response.get("status") == "ok"

    # Run the sync workflow in a thread
    await asyncio.to_thread(sync_client_workflow)

    # 5. Server retrieves the rollout and verifies
    completed_rollout = await server.get_completed_rollout(rollout_id)
    assert completed_rollout is not None
    assert completed_rollout.final_reward == 0.88
    assert completed_rollout.rollout_id == rollout_id


@pytest.mark.asyncio
async def test_task_timeout_and_requeue(server_setup: Dict[str, Any]):
    """
    Tests that a task is correctly re-queued if a client claims it but does not
    complete it within the timeout period.
    """
    server: AgentLightningServer = server_setup["server"]
    http_client: AsyncClient = server_setup["http_client"]

    # 1. Queue a task
    rollout_id = await server.queue_task(sample={"data": "stale_test"})

    # 2. Client 1 gets the task
    response1 = await http_client.get("/task")
    assert response1.status_code == 200
    task1_data = response1.json()
    assert task1_data["is_available"] is True
    assert task1_data["task"]["rollout_id"] == rollout_id
    assert task1_data["task"]["num_claims"] == 1

    # 3. No more tasks are available immediately
    response2 = await http_client.get("/task")
    assert response2.json()["is_available"] is False

    # 4. Wait for the task to time out (server timeout is 2.0s)
    await asyncio.sleep(2.5)

    # 5. The timeout check is triggered by the next call to /task.
    # Client 2 should now receive the same task, but re-claimed.
    response3 = await http_client.get("/task")
    assert response3.status_code == 200
    task2_data = response3.json()
    assert task2_data["is_available"] is True
    assert task2_data["task"]["rollout_id"] == rollout_id
    assert task2_data["task"]["num_claims"] == 2  # The key assertion
    assert task2_data["task"]["last_claim_time"] > task1_data["task"]["last_claim_time"]


@pytest.mark.asyncio
async def test_error_handling_no_resources(server_setup: Dict[str, Any]):
    """
    Tests that the server correctly returns a 404 error when a client
    requests resources before any have been set.
    """
    http_client: AsyncClient = server_setup["http_client"]

    # Request latest resources when none exist
    response_latest = await http_client.get("/resources/latest")
    assert response_latest.status_code == 404
    assert "No resources have been set" in response_latest.text

    # Request a specific resource that doesn't exist
    response_specific = await http_client.get("/resources/non-existent-id")
    assert response_specific.status_code == 404
    assert "not found" in response_specific.text


@pytest.mark.asyncio
async def test_client_with_bad_endpoint():
    """
    Ensures the client handles connection errors gracefully when the
    server endpoint is incorrect.
    """
    # Point client to a non-existent server
    client = AgentLightningClient(endpoint="http://127.0.0.1:9999", timeout=0.5)

    # Async methods should return None after failing to connect
    task = await client.get_latest_resources_async()
    assert task is None

    # Sync methods should also return None
    sync_resources = client.get_latest_resources()
    assert sync_resources is None


def test_local_client_core_functionality(sample_resources: NamedResources):
    """Test core DevTaskLoader functionality: initialization, polling, resources, and rollouts."""
    # Test initialization with TaskInput objects
    task_inputs = ["input1", {"prompt": "input2"}]
    client = DevTaskLoader(tasks=task_inputs, resources=sample_resources)

    assert len(client._tasks) == 2
    assert client._resources_update.resources_id == "local"
    assert client._resources_update.resources == sample_resources
    assert client.task_count == 0

    # Test polling TaskInput objects
    task1 = client.poll_next_task()
    assert isinstance(task1, Task)
    assert task1.rollout_id == "local_task_001"
    assert task1.input == "input1"
    assert task1.resources_id == "local"
    assert client.task_count == 1

    task2 = client.poll_next_task()
    assert task2.rollout_id == "local_task_002"
    assert task2.input == {"prompt": "input2"}
    assert client.task_count == 2

    # Test initialization with Task objects and ResourcesUpdate
    resources_update = ResourcesUpdate(
        resources_id="version123",
        resources=sample_resources,
        create_time=time.time(),
        update_time=time.time(),
        version=1,
    )
    tasks = [Task(rollout_id="existing_task", input="existing_input", resources_id="version123")]
    client2 = DevTaskLoader(tasks=tasks, resources=resources_update)

    task3 = client2.poll_next_task()
    assert task3.rollout_id == "existing_task"
    assert task3.input == "existing_input"

    # Test resource retrieval
    resources = client2.get_resources_by_id("version123")
    assert resources is not None
    assert resources.resources_id == "version123"
    assert resources.resources == sample_resources

    latest = client2.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == "version123"

    # Test rollout posting
    rollout = RolloutLegacy(rollout_id="test_rollout", final_reward=0.9)
    result = client2.post_rollout(rollout)
    assert result is not None
    assert result["status"] == "received"
    assert result["rollout_id"] == "test_rollout"
    assert len(client2.rollouts) == 1
    assert client2.rollouts[0].rollout_id == "test_rollout"

    # Test repr
    repr_str = repr(client2)
    assert "DevTaskLoader" in repr_str
    assert "num_tasks=1" in repr_str


def test_local_client_error_handling(sample_resources: NamedResources):
    """Test DevTaskLoader error handling and edge cases."""
    # Empty tasks should raise error
    with pytest.raises(ValueError, match="DevTaskLoader requires at least one task"):
        DevTaskLoader(tasks=[], resources=sample_resources)

    # Mixed task types should raise error
    mixed_tasks = [Task(rollout_id="task1", input="input1"), "task_input2"]
    with pytest.raises(ValueError, match="All tasks must be either Task or TaskInput objects"):
        DevTaskLoader(tasks=mixed_tasks, resources=sample_resources)

    # Wrong resource ID should raise error
    resources_update = ResourcesUpdate(
        resources_id="version123",
        resources=sample_resources,
        create_time=time.time(),
        update_time=time.time(),
        version=1,
    )
    client = DevTaskLoader(tasks=["input1"], resources=resources_update)
    with pytest.raises(ValueError, match="Resource ID 'wrong_id' not found"):
        client.get_resources_by_id("wrong_id")

    # Polling beyond available tasks should cycle back
    single_task_client = DevTaskLoader(tasks=["only_task"], resources=sample_resources)
    task1 = single_task_client.poll_next_task()
    task2 = single_task_client.poll_next_task()
    assert task2.input == task1.input


@pytest.mark.asyncio
async def test_local_client_async_methods(sample_resources: NamedResources):
    """Test that DevTaskLoader async methods work correctly."""
    client = DevTaskLoader(tasks=["async_input"], resources=sample_resources)

    # Test all async methods delegate to sync versions
    task = await client.poll_next_task_async()
    assert task.input == "async_input"

    resources = await client.get_resources_by_id_async("local")
    assert resources is not None
    assert resources.resources_id == "local"

    latest = await client.get_latest_resources_async()
    assert latest is not None
    assert latest.resources_id == "local"

    rollout = RolloutLegacy(rollout_id="async_rollout", final_reward=0.8)
    result = await client.post_rollout_async(rollout)
    assert result is not None
    assert result["rollout_id"] == "async_rollout"
    assert len(client.rollouts) == 1
    assert client.rollouts[0].final_reward == 0.8
