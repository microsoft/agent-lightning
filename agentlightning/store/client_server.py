from .base import LightningStore

class LightningStoreServerWrapper(LightningStore):

    def __init__(self, store: LightningStore):
        self.store = store

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


class LightningStoreClientWrapper(LightningStore):

    def __init__(self, server_address: str):
        ...
