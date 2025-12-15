import argparse
import os

from omegaconf import OmegaConf

from agentlightning import Trainer
from agentlightning.algorithm.verl import VERL
from contrib.agentlightning.contrib.algorithm.simulation_verl.trainer import SimulationAgentLightningTrainer
from contrib.agentlightning.contrib.algorithm.simulation_verl.daemon import SimulationAgentModeDaemon
from contrib.recipes.simulation.utils import kill_process_on_port, run_cmd


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
    parser.add_argument("--env", type=str, default="scienceworld")
    parser.add_argument("--algorithm", type=str, default="grpo")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_workers", type=int, default=64, help="Number of workers for training")
    parser.add_argument("--trial", type=int, default=0, help="Number of trials")
    parser.add_argument("--task_num", type=int, default=25, help="ScienceWorld Task number to inject as env var")
    parser.add_argument("--_background", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Restart Ray cluster cleanly
    kill_process_on_port(4747)
    run_cmd("pkill -f AgentLightning")
    run_cmd("ray stop")
    run_cmd("env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0")

    # set environment variable before loading configs
    os.environ["TRIAL"] = str(args.trial)
    if args.env == "scienceworld":
        os.environ["TASK_NUM"] = str(args.task_num)

    # Load configs
    agent_config_path = f"contrib/recipes/simulation/env_config/{args.env}.yaml"
    if args.debug:
        trainer_config_path = f"contrib/recipes/simulation/run/{args.env}/debug/{args.algorithm}.yaml"
    else:
        trainer_config_path = f"contrib/recipes/simulation/run/{args.env}/{args.algorithm}.yaml"
    agent_config = get_config(agent_config_path)

    if "gigpo" in args.algorithm:
        agent_config.log_env_obs = True
    rl_training_config = get_config(trainer_config_path)

    # Load datasets
    train_dataset, val_dataset = train_val_dataset(rl_training_config)

    # Initialize agent
    from contrib.agentlightning.contrib.agent.simulation_agent import SimulationAgent

    agent = SimulationAgent(agent_config)

    # Initialize trainer and start training
    trainer = Trainer(
        algorithm=VERL(
            config=rl_training_config,
            trainer_cls=SimulationAgentLightningTrainer,
            daemon_cls=SimulationAgentModeDaemon,
        ),
        n_workers=args.n_workers,
    )
    trainer.fit(agent, train_dataset, val_dataset=val_dataset)
