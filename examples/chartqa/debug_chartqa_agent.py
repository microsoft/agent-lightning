# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, cast

import pandas as pd
from chartqa_agent import LitChartQAAgent

import agentlightning as agl

logger = logging.getLogger("chartqa_agent")


def create_llm_proxy_for_chartqa(vllm_endpoint: str, port: int = 8081) -> agl.LLMProxy:
    """Create LLMProxy configured for ChartQA with token ID capture."""
    store = agl.LightningStoreThreaded(agl.InMemoryLightningStore())

    llm_proxy = agl.LLMProxy(
        port=port,
        store=store,
        model_list=[
            {
                "model_name": "Qwen/Qwen2-VL-2B-Instruct",
                "litellm_params": {
                    "model": "hosted_vllm/Qwen/Qwen2-VL-2B-Instruct",
                    "api_base": vllm_endpoint,
                },
            }
        ],
        callbacks=["return_token_ids"],
        launch_mode="thread",
    )

    return llm_proxy


def debug_chartqa_agent():
    """Debug function to test agent with cloud APIs (default).

    Usage:
        python chartqa_agent.py

    Environment variables:
        MODEL: Model name (default: gpt-4o)
        OPENAI_API_BASE: API endpoint (default: https://api.openai.com/v1)
        OPENAI_API_KEY: API key for authentication
    """
    chartqa_dir = os.environ.get("CHARTQA_DATA_DIR", "data")
    test_data_path = os.path.join(chartqa_dir, "test_chartqa.parquet")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file {test_data_path} does not exist. Please run prepare_data.py first.")

    df = pd.read_parquet(test_data_path).head(10)  # type: ignore
    test_data = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore

    model = os.environ.get("MODEL", "gpt-4.1-mini")
    endpoint = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    logger.info(f"Debug data: {len(test_data)} samples, model: {model}, endpoint: {endpoint}")

    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=endpoint,
                model=model,
                sampling_parameters={"temperature": 0.0},
            )
        },
    )
    trainer.dev(LitChartQAAgent(use_base64_images=True), test_data)


def debug_chartqa_agent_with_llm_proxy():
    """Debug function to test agent with local vLLM server and LLMProxy.

    Usage:
        USE_LLM_PROXY=1 OPENAI_API_BASE=http://localhost:8088/v1 python chartqa_agent.py
    """
    chartqa_dir = os.environ.get("CHARTQA_DATA_DIR", "data")
    test_data_path = os.path.join(chartqa_dir, "test_chartqa.parquet")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file {test_data_path} does not exist. Please run prepare_data.py first.")

    df = pd.read_parquet(test_data_path).head(10)  # type: ignore
    test_data = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore

    vllm_endpoint = os.environ.get("OPENAI_API_BASE", "http://localhost:8088/v1")
    model = os.environ.get("MODEL", "Qwen/Qwen2-VL-2B-Instruct")

    store = agl.LightningStoreThreaded(agl.InMemoryLightningStore())

    llm_proxy = agl.LLMProxy(
        port=8089,
        store=store,
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": f"hosted_vllm/{model}",
                    "api_base": vllm_endpoint,
                },
            }
        ],
        callbacks=["return_token_ids"],
        launch_mode="thread",
    )

    trainer = agl.Trainer(
        n_workers=2,
        store=store,
        llm_proxy=llm_proxy,
        strategy={"name": "shm", "main_thread": "algorithm", "managed_store": False},
        initial_resources={
            "main_llm": agl.LLM(
                endpoint="http://localhost:8089/v1",
                model=model,
                sampling_parameters={"temperature": 0.0},
            )
        },
    )

    trainer.dev(LitChartQAAgent(), test_data)


if __name__ == "__main__":
    agl.setup_logging(apply_to=["chartqa_agent"])
    if os.environ.get("USE_LLM_PROXY", "").lower() in ("1", "true", "yes"):
        debug_chartqa_agent_with_llm_proxy()
    else:
        debug_chartqa_agent()
