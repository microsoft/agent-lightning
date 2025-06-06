# Spider Example

This example requires a single node with one GPU of at least 40GB memory.

1. Download Spider 1.0 dataset from [here](https://yale-lily.github.io/spider) and unzip it to the `data` folder.
2. Use `python spider_eval/convert_dataset.py` to convert the dataset to the parquet format.
3. Start ray: `bash ../../scripts/restart_ray.sh`. To use Wandb, you need to set the WANDB_API_KEY environment variable before starting ray.
4. Run the agent: `VERL_API_BASE=http://localhost:9999/ python sql_agent.py`. Use `python sql_agent.py --help` to see options like running multiple agents.
5. In another terminal, launch the training server: `bash train.sh`.
