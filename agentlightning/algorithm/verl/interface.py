# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Optional

from hydra import compose, initialize
from omegaconf import OmegaConf

from agentlightning.algorithm.base import BaseAlgorithm
from agentlightning.client import AgentLightningClient
from agentlightning.types import Dataset
from agentlightning.verl.entrypoint import run_ppo  # type: ignore


class VERL(BaseAlgorithm):
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
        dev_dataset: Optional[Dataset[Any]] = None,
    ) -> None:
        if dev_dataset is not None:
            raise ValueError("dev_dataset is not supported for VERL.")

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
