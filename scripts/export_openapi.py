# Copyright (c) Microsoft. All rights reserved.

"""Generate OpenAPI specification for the LightningStore server.

Run this every time when you make changes to the LightningStore server.
"""

import asyncio
import json

from agentlightning.store.client_server import LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore


async def main():
    store = InMemoryLightningStore()
    server = LightningStoreServer(store, host="0.0.0.0", port=23333)
    await server.start()

    with open("docs/assets/store-openapi.json", "w") as f:
        json.dump(server.app.openapi(), f)  # type: ignore
    await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
