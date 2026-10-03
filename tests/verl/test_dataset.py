# Copyright (c) Microsoft. All rights reserved.

import torch

from agentlightning.verl.dataset import LoadedDataset


def test_loaded_dataset_handles_mixed_optional_extra_info() -> None:
    dataset = LoadedDataset(
        [
            {
                "data_id": "indexed",
                "extra_info": {"index": 7, "source": "fixture"},
                "payload": {"value": "kept"},
            },
            {"data_id": "missing", "payload": {"value": "also-kept"}},
        ]
    )

    indexed = dataset[0]
    missing = dataset[1]

    assert indexed["index"] == 7
    assert indexed["extra_info"] == {"index": 7, "source": "fixture"}
    assert indexed["payload"] == {"value": "kept"}
    assert missing["index"] == 0
    assert missing["extra_info"] is None
    assert missing["payload"] == {"value": "also-kept"}
    assert torch.equal(indexed["fake_ids"], torch.ones(1, dtype=torch.int))
    assert torch.equal(missing["fake_ids"], torch.ones(1, dtype=torch.int))


def test_loaded_dataset_defaults_null_missing_and_empty_extra_info() -> None:
    cases = [
        {"data_id": "null", "extra_info": None},
        {"data_id": "missing"},
        {"data_id": "empty", "extra_info": {}},
    ]

    for row in cases:
        assert LoadedDataset([row])[0]["index"] == 0
