# Spider Example

This example requires a single node with one GPU of at least 40GB memory.

1. Download Spider 1.0 dataset from [here](https://yale-lily.github.io/spider) and unzip it to the `data` folder.
2. Use `python spider_eval/convert_dataset.py` to convert the dataset to the parquet format.
3. Start ray: `bash ../../scripts/restart_ray.sh`. To use Wandb, you need to set the WANDB_API_KEY environment variable before starting ray.
4. Run the agent: `VERL_API_BASE=http://localhost:9999/ python sql_agent.py`. Use `python sql_agent.py --help` to see options like running multiple agents.
5. In another terminal, launch the training server: `bash train.sh`.

## Evaluation

Setting:

* 1 node with 4 80GB A100 GPUs.
* Model: Qwen/Qwen2.5-Coder-3B-Instruct
* Train write and rewrite agents only (`--litsqlagent.trained-agents write`). The check agent share the same model but the corresponding interaction is not trained.
* 10 parallel agent workers (`--trainer.n-workers 10`).
* Validation temperature = 0 (`--trainer.val-temperature 0`).
* Maximum 3 turns (1 write + 2 rewrites, `--litsqlagent.max-turns 3`).
* Truncate the database schema description and execution result to 512 characters (`--litsqlagent.table-info-truncate 2048 --litsqlagent.execution-truncate 2048`).
* RL algorithm is GRPO with learning rate 1e-6.
* Uses random 500 samples from the test set for validation.

Under the base setting, the performance on validation set boosted from 62.4% to 76.2% in 400 training steps.
The W&B report is available [here](https://api.wandb.ai/links/ultmaster/agnice3m).
