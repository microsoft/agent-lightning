# UltraRAG Search-R1 Example

## Overview
This example trains a Search-R1 style agent using the UltraRAG pipeline inside Agent Lightning. It reuses the `examples/search_r1` dataset and shows how to run end-to-end RL with Ray + vLLM.

## Included Files
| File/Directory | Description |
| --- | --- |
| `train.sh` | Launch RL training (Ray + vLLM) |
| `ultrarag_adapter.py` | UltraRAG-aware agent adapter |
| `search_r1_rl.yaml` | UltraRAG pipeline config for RL |
| `search_r1_rl_parameter.yaml` | UltraRAG parameter config |
| `requirements-ultrarag.txt` | Notes on installing deps via groups |

---

## Prepare Environment
From repo root:
```bash
uv pip install -e . --group torch-gpu-stable --group ultrarag
```
Data: expected under `examples/search_r1/data` (train/val parquet).
Base model: set `BASE_MODEL` (e.g., Llama-3.2-3B-Instruct).

---

## Run Training
1) Start Ray
```bash
bash scripts/restart_ray.sh
```
2) Run training
```bash
cd examples/ultrarag_search_r1
bash train.sh
```
Env overrides: `BASE_MODEL`, `DATA_DIR`, `RAY_ADDRESS`, `CUDA_VISIBLE_DEVICES`, `VLLM_PORT`, etc.

Optional sanity check (adapter import only):
```bash
cd examples/ultrarag_search_r1
python ultrarag_adapter.py
```

---

## Notes
- Validation runs before training and every `test_freq` steps (see `train.sh`).
- Checkpoints and validation results are written under `checkpoints/ultrarag_search_r1_checkpoints/`.
