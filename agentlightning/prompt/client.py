import logging

from typing import List
from agentlightning.trace import Transition
from agentlightning.client import VerlAgentClient
from agentlightning.trainer import Trainer


logger = logging.getLogger(__name__)


class AgentLightningClient(VerlAgentClient):
    """
    A client for interacting with the Agent Lightning server.

    This client is designed to work with the Agent Lightning server, allowing
    for task submission and resource management.
    """

    _next_task_uri = "task"
    _sampling_parameters_uri = (
        "resources"  # TODO: Currently I borrowed the sampling parameters place for general resources.
    )
    _report_trajectory_uri = "trajectory"

    def _to_acceptable_trajectory_payload(self, rollout_id: str, transitions: List[Transition]) -> dict:
        data = {"rollout_id": rollout_id, "transitions": [t.model_dump() for t in transitions]}
        print(data)
        return data


class TrainerV1(Trainer):

    def _init_verl_client(self, endpoint: str) -> VerlAgentClient:
        if self._verl_client is None:
            self._verl_client = AgentLightningClient(endpoint=endpoint)
        else:
            logger.warning("VerlAgentClient already initialized. Returning existing instance.")
        return self._verl_client
