# Calc-X Example

This example requires a single node with one GPU of at least 40GB memory.

1. Download the data in parquet format from [here](https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view?usp=sharing) and unzip it to the `data` folder: `unzip calc-x-data.zip -d data`.
2. Start ray: `bash ../../scripts/restart_ray.sh`. To use Wandb, you need to set the WANDB_API_KEY environment variable before starting ray.
3. Run the agent: `python calc_agent.py`. It automatically launches 4 agent workers by default.
4. In another terminal, launch the training server: `bash train.sh`.

## Common Issues

1. The agent client will hang indefinitely if the environment is not properly configured. Check if uv and mcp are properly installed. Use `tests/test_mcp_calculator.py` to verify the installation.
