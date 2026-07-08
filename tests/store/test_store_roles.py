# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from agentlightning.store.base import LightningStore
from agentlightning.store.client_server import LightningStoreClient, LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.store.threading import LightningStoreThreaded


def test_lightning_store_server_is_not_business_store() -> None:
    assert not issubclass(LightningStoreServer, LightningStore)


def test_store_adapters_are_business_stores() -> None:
    assert issubclass(LightningStoreClient, LightningStore)
    assert issubclass(LightningStoreThreaded, LightningStore)


def test_runtime_roles_match_design_intent() -> None:
    store = LightningStoreServer(InMemoryLightningStore(), host="127.0.0.1", port=55555)
    client = LightningStoreClient("http://127.0.0.1:55555")
    threaded = LightningStoreThreaded(InMemoryLightningStore())

    assert not isinstance(store, LightningStore)
    assert isinstance(client, LightningStore)
    assert isinstance(threaded, LightningStore)
