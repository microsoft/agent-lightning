# Copyright (c) Microsoft. All rights reserved.

import torch
from verl.utils.dataset.rl_dataset import RLHFDataset


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
