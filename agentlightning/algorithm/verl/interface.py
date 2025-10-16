# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Optional

from hydra import compose, initialize
from omegaconf import OmegaConf

from agentlightning.algorithm.base import Algorithm
from agentlightning.client import AgentLightningClient
from agentlightning.types import Dataset
from agentlightning.verl.entrypoint import run_ppo  # type: ignore


class VERL(Algorithm):
    """Algorithm leveraging VERL as the backend framework.

    **Note on Customization:**

    At present, we recommend copying the source code from VERL and modifying it as needed to suit your requirements.
    Native support for customizing training logic will be provided in future releases.

    Args:
        config: The VERL configuration, matching what is typically provided when running VERL via the command line.
            This config will be merged with VERL's base configuration and processed by Hydra.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()

        # Compose the base config exactly like your decorator:
        with initialize(version_base=None, config_path="pkg://agentlightning/verl"):
            base_cfg = compose(config_name="config")

        # Merge your dict overrides
        override_conf = OmegaConf.create(config)
        self.config = OmegaConf.merge(base_cfg, override_conf)

    def run(
        self,
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
    ) -> None:
        try:
            store = self.get_store()
        except Exception:
            print("Store is not set. Assuming v0 execution mode.")
            run_ppo(
                self.config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                store=None,
                llm_proxy=None,
                adapter=None,
            )
        else:
            print("Store is set. Assuming v1 execution mode.")
            llm_proxy = self.get_llm_proxy()
            adapter = self.get_adapter()
            run_ppo(
                self.config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                store=store,
                llm_proxy=llm_proxy,
                adapter=adapter,
            )

    def get_client(self) -> AgentLightningClient:
        port = self.config.agentlightning.port
        return AgentLightningClient(endpoint=f"http://localhost:{port}")
