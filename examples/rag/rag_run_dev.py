# Copyright (c) Microsoft. All rights reserved.

import os
from typing import Any, Dict, List

import pandas as pd
from rag_agent import RAGAgent  # Make sure to import your previously defined RAGAgent class

import agentlightning as agl

# 1. loading dataset
dataset_path = "dataset_tiny.parquet"
if os.path.exists(dataset_path):
    df: pd.DataFrame = pd.read_parquet(dataset_path)  # type: ignore
    data: List[Dict[str, Any]] = df.to_dict(orient="records")  # type: ignore
else:
    print(f"Warning: {dataset_path} not found. Using dummy data.")
    data: List[Dict[str, Any]] = [{"question": "What is the capital of France?", "answer": "Paris"}]

# 2. configuring resources (LLM)
# Note: You need to start a local service compatible with the OpenAI API (such as vLLM)
# For example: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
resources: dict[str, agl.ResourceUnion] = {
    "main_llm": agl.LLM(
        endpoint="http://localhost:8000/v1",  # Replace with your actual vLLM address
        model="Qwen/Qwen2.5-1.5B-Instruct",  # Replace with your actual loaded model name
        sampling_parameters={"temperature": 0.0},
    )
}

# 3. run Agent
trainer = agl.Trainer(initial_resources=resources)
trainer.dev(RAGAgent(), train_dataset=data)
