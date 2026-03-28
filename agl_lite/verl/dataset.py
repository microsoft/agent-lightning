# Copyright (c) Microsoft. All rights reserved.

# type: ignore

import torch
from datasets import Dataset as HuggingFaceDataset
from omegaconf import DictConfig
from verl.utils.dataset.rl_dataset import RLHFDataset

from agentlightning.types import Dataset

__all__ = [
    "AgentDataset",
    "LoadedDataset",
]


class AgentDataset(RLHFDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filter_overlong_prompts = False

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        # Workaround for data proto. At least one tensor is needed.
        row_dict["fake_ids"] = torch.ones(1, dtype=torch.int)
        return row_dict


class LoadedDataset(AgentDataset):

    def __init__(self, dataset: Dataset):
        super().__init__([], None, DictConfig({}))  # type: ignore
        dataset_copy = [dataset[i] for i in range(len(dataset))]
        self.dataframe = HuggingFaceDataset.from_list(dataset_copy)

    def _read_files_and_tokenize(self):
        pass
