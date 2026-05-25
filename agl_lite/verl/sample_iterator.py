"""Variable-size sample fetcher over a VERL DataLoader.

Sync RL `fit()` pulls one full `train_batch_size` batch per step:

    for batch_dict in self.train_dataloader: ...

Async `async_fit()` cannot do that — each step needs only
``async_train_batch_size - n_carry_over_dids`` new samples (variable). This
module provides the thin adapter:

    iterator = SampleIterator(dataloader)
    samples_dict, cross_epoch = iterator.take(n)   # async_fit passes n > 0

The returned ``samples_dict`` has exactly the same schema as one
``batch_dict`` yielded by the dataloader, just shorter. The trainer then calls
``DataProto.from_single_dict(samples_dict)`` and passes the gen_batch slice to
the bridge, identical to the sync path. Async rollout rejects carry-over-only
steps before calling this iterator.

``cross_epoch`` is True when an epoch boundary was crossed during this
``take`` call (used by the trainer to bump an epoch counter for logging).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch


class SampleIterator:
    """Pulls ``n`` samples at a time from a torch DataLoader.

    The dataloader is iterated lazily: a fresh iterator is created on first
    ``take``, batches are buffered as we go, and when the iterator is
    exhausted we restart it (next epoch). Each batch from the dataloader is a
    ``dict[str, Tensor | ndarray | list]`` where every value's leading
    dimension equals the dataloader's ``train_batch_size``.

    ``take(n)`` fills exactly ``n`` samples for non-empty dataloaders, crossing
    epoch boundaries if needed. The trainer uses ``cross_epoch`` as a metric
    signal that at least one epoch boundary was crossed during the call.

    The iterator is **not** thread-safe — drive it from a single trainer
    process.
    """

    def __init__(self, dataloader: Any) -> None:
        self._dataloader = dataloader
        self._iter: Iterator[dict[str, Any]] | None = None
        self._buf: dict[str, Any] | None = None
        self._epoch: int = 0
        self._consumed: int = 0

    @property
    def epoch(self) -> int:
        """Number of times the dataloader has been fully consumed."""
        return self._epoch

    @property
    def consumed(self) -> int:
        """Total number of samples returned by ``take`` so far."""
        return self._consumed

    def take(self, n: int) -> tuple[dict[str, Any], bool]:
        """Return up to ``n`` samples from the dataloader.

        Returns ``(samples_dict, cross_epoch)``. ``samples_dict`` has the
        same per-key schema as one dataloader batch, sliced/concatenated to
        exactly ``n`` returned samples for non-empty dataloaders. The call may
        cross epoch boundaries to fill ``n``. ``cross_epoch`` is True if
        this call hit at least one end-of-dataloader boundary.

        Edge cases:
          - ``n == 0``: returns ``({}, False)`` without touching the
            dataloader. This is a general utility behavior; async_fit rejects
            carry-over-only steps before it calls take().
          - Dataloader exhausted before or during the request: starts the next
            epoch transparently and continues filling. ``epoch`` is bumped and
            ``cross_epoch`` is set True. This keeps async steps able to top
            up to ``async_train_batch_size`` while preserving an epoch-boundary
            signal for metrics/logging.
        """
        if n < 0:
            raise ValueError(f"SampleIterator.take(n) requires n >= 0, got {n}")
        if n == 0:
            return {}, False

        if self._iter is None:
            self._iter = iter(self._dataloader)

        collected: list[dict[str, Any]] = []
        remaining = n
        cross_epoch = False

        if self._buf is not None:
            buf_len = _batch_len(self._buf)
            if buf_len <= remaining:
                collected.append(self._buf)
                remaining -= buf_len
                self._buf = None
            else:
                head, tail = _split_batch(self._buf, remaining)
                collected.append(head)
                self._buf = tail
                remaining = 0

        while remaining > 0:
            try:
                batch = next(self._iter)
            except StopIteration:
                self._iter = iter(self._dataloader)
                self._epoch += 1
                cross_epoch = True
                continue
            batch_len = _batch_len(batch)
            if batch_len <= remaining:
                collected.append(batch)
                remaining -= batch_len
            else:
                head, tail = _split_batch(batch, remaining)
                collected.append(head)
                self._buf = tail
                remaining = 0

        if not collected:
            return {}, cross_epoch
        merged = _concat_batches(collected)
        self._consumed += _batch_len(merged)
        return merged, cross_epoch

    def reset(self) -> None:
        """Forget current dataloader iterator state.

        Used by the trainer if a fresh re-seed of the sampler is needed. Does
        not reset ``epoch`` / ``consumed`` counters.
        """
        self._iter = None
        self._buf = None


def _batch_len(batch: dict[str, Any]) -> int:
    """Return the leading-dim length of a dataloader batch dict."""
    if not batch:
        return 0
    # All keys must share leading dim; sample any one.
    for v in batch.values():
        if hasattr(v, "__len__"):
            return len(v)
    raise ValueError("batch values must be sized (Tensor / ndarray / list)")


def _split_batch(batch: dict[str, Any], head_n: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a batch dict into (first head_n, remainder) along leading dim."""
    head: dict[str, Any] = {}
    tail: dict[str, Any] = {}
    for k, v in batch.items():
        head[k] = v[:head_n]
        tail[k] = v[head_n:]
    return head, tail


def _concat_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate a list of batch dicts along the leading dim.

    All dicts must share the same key set; per-key types are preserved
    (Tensor → cat, ndarray → concatenate, list → +). Mixed types per key are
    not supported and indicate a dataloader bug.
    """
    if len(batches) == 1:
        return batches[0]
    keys = batches[0].keys()
    out: dict[str, Any] = {}
    for k in keys:
        values = [b[k] for b in batches]
        sample = values[0]
        if isinstance(sample, torch.Tensor):
            out[k] = torch.cat(values, dim=0)
        elif isinstance(sample, np.ndarray):
            out[k] = np.concatenate(values, axis=0)
        elif isinstance(sample, list):
            merged: list[Any] = []
            for v in values:
                merged.extend(v)
            out[k] = merged
        else:
            raise TypeError(
                f"SampleIterator: unsupported batch value type for key {k!r}: {type(sample).__name__}. "
                "Expected torch.Tensor, numpy.ndarray, or list."
            )
    return out
