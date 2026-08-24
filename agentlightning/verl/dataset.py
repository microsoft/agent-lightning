# Copyright (c) Microsoft. All rights reserved.

# type: ignore

from collections.abc import Sequence
from typing import Any

import torch
from datasets import Dataset as HuggingFaceDataset
from verl.utils.dataset.rl_dataset import RLHFDataset

__all__ = [
    "LoadedDataset",
]


class LoadedDataset(RLHFDataset):
    """Dataset wrapper for pre-loaded in-memory data sequences.

    Bypasses RLHFDataset's file-based initialization and directly sets
    ``self.dataframe`` from the provided sequence.
    """

    def __init__(self, dataset: Sequence[Any]):
        # Skip file-based RLHFDataset initialization; only dataframe behavior is needed.
        dataset_copy = [dataset[i] for i in range(len(dataset))]
        self.dataframe = HuggingFaceDataset.from_list(dataset_copy)
        self.filter_overlong_prompts = False
        self.serialize_dataset = True  # Tell __getstate__ to serialize inline
        self.original_data_files = None  # Not file-backed

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        # Workaround for data proto. At least one tensor is needed.
        row_dict["fake_ids"] = torch.ones(1, dtype=torch.int)
        return row_dict

    def _read_files_and_tokenize(self):
        pass
