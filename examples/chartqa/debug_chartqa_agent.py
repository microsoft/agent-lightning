# Copyright (c) Microsoft. All rights reserved.

"""Debugging helpers for the ChartQA agent.

Example usage for OpenAI API:

```bash
python debug_chartqa_agent.py
```

Example usage for self-hosted model.

```
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --allowed-local-media-path $CHARTQA_DATA_DIR \
    --enable-prefix-caching \
    --port 8088
OPENAI_API_BASE=http://localhost:8088/v1 OPENAI_MODEL=Qwen/Qwen2.5-VL-3B-Instruct python debug_chartqa_agent.py
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


def is_local_endpoint(endpoint: str) -> bool:
    """Check if the endpoint is a local vLLM server."""
    return "localhost" in endpoint or "127.0.0.1" in endpoint


def debug_chartqa_agent() -> None:
    """Debug the ChartQA agent against cloud APIs or a local vLLM server.

    Automatically detects local vs cloud based on the OPENAI_API_BASE endpoint.
    For local vLLM, uses file:// paths. For cloud APIs, uses base64 encoding.

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
    api_key = chartqa_env_var.OPENAI_API_KEY
    use_local = is_local_endpoint(endpoint)

    logger.info(
        "Debug data: %s samples, model: %s, endpoint: %s, local=%s",
        len(test_data),
        model,
        endpoint,
        use_local,
    )

    # For local vLLM, use file:// paths; for cloud APIs, use base64 encoding
    agent = ChartQAAgent(use_base64_images=not use_local)

    trainer = agl.Trainer(
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                sampling_parameters={"temperature": 0.0},
            )
        },
        n_workers=1,
    )

    trainer.dev(agent, test_data)


if __name__ == "__main__":
    agl.setup_logging(apply_to=["chartqa_agent"])
    debug_chartqa_agent()
