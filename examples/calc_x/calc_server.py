import asyncio
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from agentlightning.server import AgentLightningServer
from agentlightning import configure_logger, Trainer
from examples.calc_x.calc_agent import CalcAgent

configure_logger()

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9999
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
DATA_PATH = Path(__file__).parent / "data.jsonl"
OUTPUT_PATH = Path(__file__).parent / "rollouts.jsonl"


def trainer_process_entry(server_url: str, n_workers: int = 1, max_tasks: int | None = None):
    configure_logger()
    agent = CalcAgent()
    trainer = Trainer(n_workers=n_workers, max_tasks=max_tasks)
    # This call will block in the child process and run worker loop connecting to server_url
    trainer.fit(agent, backend=server_url)


async def load_dataset() -> List[Dict[str, Any]]:
    """
    Load dataset from examples/calc_x/data/*.parquet if present,
    else from data.jsonl, else fallback to small demo list.
    """
    data_dir = Path(__file__).parent / "data"
    # 1) try parquet files
    if data_dir.exists():
        try:
            samples = []
            for pf in sorted(data_dir.glob("*.parquet")):
                df = pd.read_parquet(pf)
                for _, row in df.iterrows():
                    q = row.get("question") or row.get("prompt") or row.get("question_text")
                    r = row.get("result") or row.get("answer") or row.get("label")
                    if q is None or r is None:
                        continue
                    samples.append({"question": str(q), "result": str(r)})
            if samples:
                return samples
        except Exception:
            # if pandas/pyarrow not available or read fails, fall through to jsonl
            pass

    # 2) try data.jsonl (existing behavior)
    if DATA_PATH.exists():
        samples = []
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        if samples:
            return samples

    # 3) fallback small demo set
    return [
        {"question": "What is 2 + 2?", "result": "4"},
        {"question": "What is 3 * 5?", "result": "15"},
        {"question": "What is the square root of 16?", "result": "4"},
    ]



async def main(n_workers: int = 1, timeout_per_task: int = 30):
    # ensure output file exists / truncated
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    OUTPUT_PATH.touch()

    server = AgentLightningServer(host=SERVER_HOST, port=SERVER_PORT)
    await server.start()
    print(f"[server] started at {SERVER_URL}")

    samples = await load_dataset()
    print(f"[server] loaded {len(samples)} samples")

    # start agent workers in separate process
    proc = mp.Process(target=trainer_process_entry, args=(SERVER_URL, n_workers, None), daemon=False)
    proc.start()
    print(f"[server] started trainer process pid={proc.pid}")

    task_ids = []
    # queue tasks
    for s in samples:
        # queue_task expects a sample; make sure keys match CalcAgent expectations
        tid = await server.queue_task(sample=s, mode="train")
        task_ids.append(tid)
        print(f"[server] queued task {tid} -> {s.get('question')!r}")

    # collect completed rollouts for each task
    rollouts = []
    for tid in task_ids:
        try:
            rollout = await server.poll_completed_rollout(tid, timeout=timeout_per_task)
        except Exception as e:
            print(f"[server] timeout/warn waiting for task {tid}: {e}")
            rollout = None

        if rollout:
            # rollout is Pydantic model; convert to dict for JSON
            try:
                rdict = rollout.model_dump()
            except Exception:
                # fallback: attempt raw attributes
                rdict = rollout.__dict__
            rollouts.append(rdict)
            # append to file
            with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rdict, ensure_ascii=False) + "\n")
            print(f"[server] saved rollout for task {tid}")
        else:
            print(f"[server] no rollout for task {tid}")

    # optionally wait for trainer process to finish consuming (or terminate it)
    if proc.is_alive():
        print("[server] terminating trainer process")
        proc.terminate()
        proc.join(timeout=5)

    await server.stop()
    print("[server] stopped")
    print(f"[server] collected {len(rollouts)} rollouts -> {OUTPUT_PATH}")


if __name__ == "__main__":
    # run asyncio main
    asyncio.run(main(n_workers=1, timeout_per_task=30))