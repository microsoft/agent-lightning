# Copyright (c) Microsoft. All rights reserved.
import argparse
import os
import re
import subprocess
import time

from omegaconf import OmegaConf

from agentlightning import Trainer
from agentlightning.algorithm.verl import VERL
from contrib.agentlightning.contrib.algorithm.env_verl.daemon import EnvAgentModeDaemon
from contrib.agentlightning.contrib.algorithm.env_verl.trainer import EnvAgentLightningTrainer


def run_cmd(cmd):
    """Execute a shell command and print its output"""
    print(f"👉 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


def kill_process_on_port(port):
    result = subprocess.run(f"sudo lsof -t -i :{port}", shell=True, capture_output=True, text=True)
    pids = result.stdout.strip().split("\n")
    for pid in pids:
        if pid:
            print(f"🔪 Killing process {pid} on port {port}")
            subprocess.run(f"sudo kill -9 {pid}", shell=True)


def train_val_dataset(cfg):
    """Load training and validation datasets from parquet files."""
    from datasets import Dataset

    train_data = Dataset.from_parquet(cfg["data"]["train_files"])
    val_data = Dataset.from_parquet(cfg["data"]["val_files"])
    return train_data, val_data


def get_config(path):
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    if "variables" in cfg:
        del cfg["variables"]
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="scienceworld2")
    parser.add_argument("--algorithm", type=str, default="empo2_qwen_7b_instruct")
    parser.add_argument("--n_workers", type=int, default=16, help="Number of workers for training")
    parser.add_argument("--trial", type=int, default=0, help="Number of trials")
    parser.add_argument("--task_num", type=int, default=25, help="ScienceWorld Task number to inject as env var")
    parser.add_argument("--_background", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Kill any leftover processes from previous runs (excluding the current process)
    current_pid = os.getpid()
    run_cmd(f"pgrep -f train_env_agent.py | grep -vxF '{current_pid}' | xargs -r kill -9")
    run_cmd("pkill -9 -f server_bert.py")
    run_cmd("pkill -9 -f server_mem.py")
    kill_process_on_port(8000)
    kill_process_on_port(8001)

    # Restart Ray cluster cleanly
    kill_process_on_port(4747)
    run_cmd("pkill -f AgentLightning")
    run_cmd("ray stop")
    time.sleep(2)
    run_cmd("env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0")

    # set environment variable before loading configs
    os.environ["TRIAL"] = str(args.trial)
    if "scienceworld" in args.env:
        os.environ["TASK_NUM"] = str(args.task_num)

    # Load configs
    agent_config_path = f"config_env/{args.env}.yaml"
    agent_config = get_config(agent_config_path)

    env_prefix = re.sub(r"\d+$", "", args.env)
    trainer_config_path = f"config_verl/{env_prefix}/{args.algorithm}.yaml"

    if "gigpo" in args.algorithm:
        agent_config.log_env_obs = True
    rl_training_config = get_config(trainer_config_path)

    # Load datasets
    train_dataset, val_dataset = train_val_dataset(rl_training_config)

    # Initialize agent
    if "empo2" in args.algorithm:
        from contrib.agentlightning.contrib.agent.empo2_agent import EMPO2Agent, reset_memory

        os.makedirs("logs", exist_ok=True)

        subprocess.Popen(f"nohup python empo2_server/server_bert.py > logs/bert_{args.task_num}.log 2>&1 &", shell=True)
        subprocess.Popen(f"nohup python empo2_server/server_mem.py > logs/mem_{args.task_num}.log 2>&1 &", shell=True)

        NUM_MEMORY = 5
        time.sleep(1)
        reset_memory(NUM_MEMORY)

        agent = EMPO2Agent(agent_config)
    else:
        from contrib.agentlightning.contrib.agent.env_agent import EnvAgent

        agent = EnvAgent(agent_config)

    # Initialize trainer and start training
    trainer = Trainer(
        algorithm=VERL(
            config=rl_training_config,
            trainer_cls=EnvAgentLightningTrainer,
            daemon_cls=EnvAgentModeDaemon,
        ),
        n_workers=args.n_workers,
    )
    trainer.fit(agent, train_dataset, val_dataset=val_dataset)
