"""Tests for async-rollout variable-size dataloader sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agl_lite.verl.sample_iterator import SampleIterator


def test_take_splits_and_buffers_mixed_batch_types() -> None:
    dataloader = [
        {
            "ids": torch.tensor([1, 2, 3]),
            "names": np.array(["a", "b", "c"], dtype=object),
            "meta": [{"i": 1}, {"i": 2}, {"i": 3}],
        }
    ]
    iterator = SampleIterator(dataloader)

    first, cross_epoch = iterator.take(2)
    assert cross_epoch is False
    assert first["ids"].tolist() == [1, 2]
    assert first["names"].tolist() == ["a", "b"]
    assert first["meta"] == [{"i": 1}, {"i": 2}]

    second, cross_epoch = iterator.take(1)
    assert cross_epoch is False
    assert second["ids"].tolist() == [3]
    assert second["names"].tolist() == ["c"]
    assert second["meta"] == [{"i": 3}]


def test_take_continues_into_next_epoch_when_no_samples_collected() -> None:
    dataloader = [{"x": torch.tensor([10, 11])}]
    iterator = SampleIterator(dataloader)

    first, cross_epoch = iterator.take(2)
    assert first["x"].tolist() == [10, 11]
    assert cross_epoch is False
    assert iterator.epoch == 0

    second, cross_epoch = iterator.take(1)
    assert second["x"].tolist() == [10]
    assert cross_epoch is True
    assert iterator.epoch == 1


def test_take_crosses_epoch_boundary_to_fill_request() -> None:
    dataloader = [{"x": torch.tensor([1, 2])}]
    iterator = SampleIterator(dataloader)

    batch, cross_epoch = iterator.take(3)
    assert batch["x"].tolist() == [1, 2, 1]
    assert cross_epoch is True
    assert iterator.epoch == 1
    assert iterator.consumed == 3


def test_take_crosses_epoch_boundary_and_buffers_tail() -> None:
    dataloader = [{"x": torch.tensor([1, 2, 3, 4])}]
    iterator = SampleIterator(dataloader)

    first, cross_epoch = iterator.take(6)
    assert first["x"].tolist() == [1, 2, 3, 4, 1, 2]
    assert cross_epoch is True
    assert iterator.epoch == 1

    second, cross_epoch = iterator.take(2)
    assert second["x"].tolist() == [3, 4]
    assert cross_epoch is False
    assert iterator.epoch == 1


def test_take_zero_does_not_advance() -> None:
    iterator = SampleIterator([{"x": torch.tensor([1])}])

    batch, cross_epoch = iterator.take(0)

    assert batch == {}
    assert cross_epoch is False
    assert iterator.epoch == 0
    assert iterator.consumed == 0


def test_take_rejects_negative_n() -> None:
    iterator = SampleIterator([{"x": torch.tensor([1])}])

    with pytest.raises(ValueError, match="n >= 0"):
        iterator.take(-1)
