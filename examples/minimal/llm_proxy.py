# Copyright (c) Microsoft. All rights reserved.

import argparse
import asyncio
import os

import aiohttp
from portpicker import pick_unused_port
from vllm_server import vllm_server

import agentlightning as agl


async def serve_llm_proxy_with_vllm(model_name: str):
    """Serve an LLM proxy for a vLLM server."""
    # Create a store to store the traces
    store = agl.InMemoryLightningStore()
    store_port = pick_unused_port()
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


async def serve_llm_proxy_with_openai(model_name: str):
    """Serve an LLM proxy for an OpenAI server."""
    # Create a store to store the traces
    store = agl.InMemoryLightningStore()
    store_port = pick_unused_port()
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
    )

    try:
        await llm_proxy.start()
        # Wait forever
        await asyncio.sleep(float("inf"))
    finally:
        # Stop the LLM proxy and the store server
        await llm_proxy.stop()
        await store_server.stop()


async def test_llm_proxy(model_name: str):
    """Test the LLM proxy by sending a request to the proxy and checking the response.

    We do it via aiohttp here. This can also be done with OpenAI client.
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:43886/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello, what's your name?"}],
            },
        ) as response:
            response_body = await response.json()
            print(response_body)
            assert "qwen" in response_body["choices"][0]["message"]["content"].lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Proxy runner")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["vllm", "openai", "test"],
        help="Which function to run",
    )
    parser.add_argument("--model-name", required=True, help="Model name to serve.")

    args = parser.parse_args()

    if args.mode == "vllm":
        asyncio.run(serve_llm_proxy_with_vllm(args.model_name))
    elif args.mode == "openai":
        asyncio.run(serve_llm_proxy_with_openai(args.model_name))
    elif args.mode == "test":
        asyncio.run(test_llm_proxy(args.model_name))
