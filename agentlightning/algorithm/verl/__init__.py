from hydra import initialize, compose
from omegaconf import OmegaConf

from agentlightning.verl.entrypoint import run_ppo
from agentlightning.client import AgentLightningClient
from agentlightning.algorithm.base import BaseAlgorithm
from agentlightning.types import Dataset


class VERL(BaseAlgorithm):
    def __init__(self, config: dict):
        super().__init__()

        # Compose the base config exactly like your decorator:
        with initialize(version_base=None, config_path="pkg://agentlightning/verl"):
            base_cfg = compose(config_name="config")

        # Merge your dict overrides
        override_conf = OmegaConf.create(config)
        self.config = OmegaConf.merge(base_cfg, override_conf)

    def run(self, train_dataset: Dataset, val_dataset: Dataset, dev_dataset: Dataset):
        run_ppo(self.config, train_dataset, val_dataset)

    def get_client(self) -> AgentLightningClient:
        port = self.config.agentlightning.port
        return AgentLightningClient(endpoint=f"http://localhost:{port}")
