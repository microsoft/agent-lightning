# Copyright (c) Microsoft. All rights reserved.

"""Examples to serve an LLM proxy for a vLLM server or an OpenAI service.

Usage: run one of the following commands to start a server.

```bash
dotenv run python llm_proxy.py openai gpt-4.1-mini

python llm_proxy.py vllm Qwen/Qwen2.5-0.5B-Instruct
```

Use the following command to test the LLM proxy.

```bash
python llm_proxy.py test Qwen/Qwen2.5-0.5B-Instruct
```
"""

import argparse
import asyncio
import os

import aiohttp
from portpicker import pick_unused_port
from vllm_server import vllm_server

import agentlightning as agl


async def serve_llm_proxy_with_vllm(model_name: str, store_port: int = 43887):
    """Serve an LLM proxy for a vLLM server."""
    # Create a store to store the traces
    store = agl.InMemoryLightningStore()
    store_server = agl.LightningStoreServer(store, "127.0.0.1", store_port)
    await store_server.start()

    # Create a vLLM server
    vllm_port = pick_unused_port()
    with vllm_server(model_name, vllm_port) as vllm_endpoint:
        # Server is up.

        # Create an LLM proxy to guard the vLLM server and catch the traces
        llm_proxy = agl.LLMProxy(
            port=43886,
            model_list=[
                {
                    "model_name": model_name,
                    "litellm_params": {
                        "model": f"hosted_vllm/{model_name}",
                        "api_base": vllm_endpoint,
                    },
                }
            ],
            store=store_server,
        )

        try:
            await llm_proxy.start()

            # Wait forever
            await asyncio.sleep(float("inf"))

        finally:
            # Stop the LLM proxy and the store server
            await llm_proxy.stop()
            await store_server.stop()


async def serve_llm_proxy_with_openai(model_name: str, store_port: int = 43887):
    """Serve an LLM proxy for an OpenAI server."""
    # Create a store to store the traces
    store = agl.InMemoryLightningStore()
    store_server = agl.LightningStoreServer(store, "127.0.0.1", store_port)
    await store_server.start()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Create an LLM proxy to guard the OpenAI server and catch the traces
    llm_proxy = agl.LLMProxy(
        port=43886,
        model_list=[
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": "openai/" + model_name,
                    # Must have OpenAI API key set in the environment variable
                },
            }
        ],
        store=store_server,
        callbacks=["opentelemetry"],
    )

    try:
        await llm_proxy.start()
        # Wait forever
        await asyncio.sleep(float("inf"))
    finally:
        # Stop the LLM proxy and the store server
        await llm_proxy.stop()
        await store_server.stop()


async def test_llm_proxy(model_name: str, store_port: int = 43887):
    """Test the LLM proxy by sending a request to the proxy and checking the response.

    We do it via aiohttp here. This can also be done with OpenAI client.
    """
    # We first connect to the store server and start a rollout.
    store = agl.LightningStoreClient(f"http://localhost:{store_port}")
    rollout = await store.start_rollout(input={"origin": "test_llm_proxy"})

    # The chat completion URL is simply /v1/chat/completions under the namespace of current rollout and attempt.
    # This ensures the traces are properly put into the correct bucket.
    chat_completion_url = (
        f"http://localhost:43886/rollout/{rollout.rollout_id}/attempt/{rollout.attempt.attempt_id}/v1/chat/completions"
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(
            chat_completion_url,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello, what's your name?"}],
            },
        ) as response:
            response_body = await response.json()
            print(response_body)
            if "qwen" in model_name.lower():
                assert "qwen" in response_body["choices"][0]["message"]["content"].lower()
            else:
                assert "chatgpt" in response_body["choices"][0]["message"]["content"].lower()

    await store.close()


if __name__ == "__main__":
    agl.setup_logging()
    parser = argparse.ArgumentParser(description="LLM Proxy runner")
    parser.add_argument(
        "mode",
        choices=["vllm", "openai", "test"],
        help="Which function to run",
    )
    parser.add_argument("model", type=str, help="Model name to serve.")

    args = parser.parse_args()

    if args.mode == "vllm":
        asyncio.run(serve_llm_proxy_with_vllm(args.model))
    elif args.mode == "openai":
        asyncio.run(serve_llm_proxy_with_openai(args.model))
    elif args.mode == "test":
        asyncio.run(test_llm_proxy(args.model))
