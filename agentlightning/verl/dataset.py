# Copyright (c) Microsoft. All rights reserved.

from importlib import import_module
from typing import Any, Callable, cast

from agentlightning.types import Dataset

__all__ = [
    "AgentDataset",
    "LoadedDataset",
]

torch = import_module("torch")
HuggingFaceDataset = getattr(import_module("datasets"), "Dataset")
DictConfig = getattr(import_module("omegaconf"), "DictConfig")
RLHFDataset = getattr(import_module("verl.utils.dataset.rl_dataset"), "RLHFDataset")


class AgentDataset(RLHFDataset):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        base_init = cast(Callable[..., None], getattr(super(), "__init__"))
        base_init(*args, **kwargs)

        self.filter_overlong_prompts = False

    def __getitem__(self, item: int) -> dict[str, Any]:
        row_dict = cast(dict[str, Any], self.dataframe[item])

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        # Workaround for data proto. At least one tensor is needed.
        row_dict["fake_ids"] = torch.ones(1, dtype=torch.int)
        return row_dict


class LoadedDataset(AgentDataset):

    def __init__(self, dataset: Dataset[Any]) -> None:
        super().__init__([], None, DictConfig({}))
        dataset_copy = [dataset[i] for i in range(len(dataset))]
        self.dataframe = HuggingFaceDataset.from_list(dataset_copy)

    def _read_files_and_tokenize(self) -> None:
        pass
