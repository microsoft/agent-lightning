# Copyright (c) Microsoft. All rights reserved.

from collections import OrderedDict
from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(OrderedDict[K, V]):
    """A simple LRU (Least Recently Used) cache implementation using OrderedDict.

    This cache has a fixed capacity. When the cache is full, adding a new item
    discards the least recently used item.

    Accessing an item (via `__getitem__` or `get`) moves it to the end, marking
    it as recently used.
    """

    def __init__(self, capacity: int, *args, **kwargs):
        self.capacity = capacity
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: K) -> V:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        if key in self:
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value
        return default
