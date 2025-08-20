import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from agentlightning.server import AgentLightningServer
from agentlightning.types import NamedResources, PromptTemplate, LLM
from agentlightning import configure_logger

VLLM_ENDPOINT = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9999
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
# Define the path for the output file where results will be stored.
OUTPUT_PATH = Path(__file__).parent / "rollouts.jsonl"

configure_logger()

async def load_dataset() -> List[Dict[str, Any]]:
    """
    Loads the dataset for the tasks.
    It first tries to load from .parquet files in the 'data' directory.
    If that fails or the directory doesn't exist, it falls back to a default demo sample.
    """
    data_dir = Path(__file__).parent / "data"

    if not data_dir.exists():
        print("Data directory not found. Using default sample.")
        return [{"question": "What is 2 + 2?", "result": "4"}]

    try:
        samples = []
        for pf in sorted(data_dir.glob("*.parquet")):
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                q = row.get("question")
                r = row.get("result") 
                if q is None or r is None:
                    continue
                samples.append({"question": str(q), "result": str(r)})
        if samples:
            print(f"Loaded {len(samples)} samples from parquet files.")
            return samples
    except Exception as e:
        print(f"Failed to load from parquet files: {e}. Using default sample.")
        pass

    return [{"question": "What is 2 + 2?", "result": "4"}]


async def main(timeout_per_task: int = 30):
    # 1. Prepare the output file: ensure it's clean and empty before the run.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    OUTPUT_PATH.touch()
    print(f"Output will be saved to: {OUTPUT_PATH}")

    # 2. Initialize and start the AgentLightning server.
    server = AgentLightningServer(host=SERVER_HOST, port=SERVER_PORT)
    await server.start()
    print(f"[Server] Started and listening at {SERVER_URL}")

    # 3. Define and broadcast shared resources to all workers.
    # This is a core concept: the server dictates the configuration for all connecting workers.
    # Here, we instruct all workers to use our local vLLM instance.
    resources = {
            "main_llm": LLM(
                endpoint=VLLM_ENDPOINT,
                model=MODEL_NAME,
                sampling_parameters={"temperature": 0.0}, # Use 0 for deterministic results in calculation tasks.
            )
        }
    
    await server.update_resources(resources)
    print(f"[Server] Broadcasted resources: All workers will use LLM at {VLLM_ENDPOINT}")

    # 4. Load the dataset and queue all samples as tasks.
    samples = await load_dataset()
    print(f"[Server] Loaded {len(samples)} samples.")
    task_ids = []
    for s in samples:
        tid = await server.queue_task(sample=s, mode="train")
        task_ids.append(tid)
    print(f"[Server] Queued {len(task_ids)} tasks. Waiting for workers...")

    # 5. Poll for and collect the results (rollouts) for each task.
    rollouts = []
    for tid in task_ids:
        try:
            # This will block until the task is completed by a worker or the timeout is reached.
            rollout = await server.poll_completed_rollout(tid, timeout=timeout_per_task)
        except asyncio.TimeoutError:
            print(f"[Server] Warning: Timed out waiting for task {tid}.")
            rollout = None
        except Exception as e:
            print(f"[Server] Error waiting for task {tid}: {e}")
            rollout = None

        if rollout:
            # The rollout object is a Pydantic model; convert it to a dictionary for JSON serialization.
            rdict = rollout.model_dump()
            rollouts.append(rdict)
            with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rdict, ensure_ascii=False) + "\n")
            print(f"[Server] Received and saved rollout for task {tid}")
        else:
            print(f"[Server] No rollout received for task {tid}")

    # 6. Stop the server gracefully after all tasks have been processed.
    await server.stop()
    print("\n[Server] Stopped.")
    print(f"[Server] Collected {len(rollouts)} rollouts -> {OUTPUT_PATH}")


if __name__ == "__main__":
    # The n_workers argument here is for demonstration; in the server-client model,
    # the number of workers is determined by how many worker scripts you run.
    asyncio.run(main(timeout_per_task=30))
