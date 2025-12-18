# Copyright (c) Microsoft. All rights reserved.

"""Debugging helpers for the ChartQA agent.

Example usage for OpenAI API:

```bash
python debug_chartqa_agent.py
```

Example usage for self-hosted model.

```
vllm serve Qwen/Qwen2-VL-2B-Instruct \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --allowed-local-media-path $CHARTQA_DATA_DIR \
    --enable-prefix-caching \
    --port 8088
USE_LLM_PROXY=1 OPENAI_API_BASE=http://localhost:8088/v1 OPENAI_MODEL=Qwen/Qwen2-VL-2B-Instruct python debug_chartqa_agent.py
```

Ensure `CHARTQA_DATA_DIR` points to a directory with the prepared parquet file by running `python prepare_data.py` beforehand.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, cast

import env_var as chartqa_env_var
import pandas as pd
from chartqa_agent import ChartQAAgent

import agentlightning as agl

logger = logging.getLogger("chartqa_agent")


def create_llm_proxy_for_chartqa(vllm_endpoint: str, port: int = 8081) -> agl.LLMProxy:
    """Create an LLMProxy configured for ChartQA with token ID capture.

    Args:
        vllm_endpoint: Base URL for the hosted vLLM server.
        port: Local port where the proxy should listen.

    Returns:
        An [`LLMProxy`][agentlightning.LLMProxy] instance launched in a thread.
    """
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


def debug_chartqa_agent(use_llm_proxy: bool = False) -> None:
    """Debug the ChartQA agent against cloud APIs or a local vLLM proxy.

    Args:
        use_llm_proxy: When `True`, spin up an LLMProxy that points to a local vLLM endpoint.

    Raises:
        FileNotFoundError: If the prepared ChartQA parquet file is missing.
    """
    test_data_path = os.path.join(chartqa_env_var.CHARTQA_DATA_DIR, "test_chartqa.parquet")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file {test_data_path} does not exist. Please run prepare_data.py first.")

    df = pd.read_parquet(test_data_path).head(10)  # type: ignore
    test_data = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore

    model = chartqa_env_var.OPENAI_MODEL
    endpoint = chartqa_env_var.OPENAI_API_BASE
    logger.info(
        "Debug data: %s samples, model: %s, endpoint: %s, llm_proxy=%s",
        len(test_data),
        model,
        endpoint,
        use_llm_proxy,
    )

    llm_endpoint = endpoint
    trainer_kwargs: Dict[str, Any] = {}

    if use_llm_proxy:
        proxy_port = 8089
        llm_proxy = create_llm_proxy_for_chartqa(endpoint, port=proxy_port)
        trainer_kwargs["llm_proxy"] = llm_proxy
        trainer_kwargs["n_workers"] = 2
        llm_endpoint = f"http://localhost:{proxy_port}/v1"
        agent = ChartQAAgent()
    else:
        trainer_kwargs["n_workers"] = 1
        agent = ChartQAAgent(use_base64_images=True)

    trainer = agl.Trainer(
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=llm_endpoint,
                model=model,
                sampling_parameters={"temperature": 0.0},
            )
        },
        **trainer_kwargs,
    )

    trainer.dev(agent, test_data)


if __name__ == "__main__":
    agl.setup_logging(apply_to=["chartqa_agent"])
    debug_chartqa_agent(use_llm_proxy=chartqa_env_var.USE_LLM_PROXY)
