import os
from typing import Optional

import alfworld
import yaml
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from envs.alfworld.base import AlfWorldEnv


def make_alfworld_env(env_name, task, config, render_mode: Optional[str] = None):
    os.environ["ALFWORLD_DATA"] = "examples/simulation/envs/alfworld/alfworld_source"

    alfworld_data_path = os.environ.get("ALFWORLD_DATA")

    with open(f"{alfworld_data_path}/base_config.yaml", "r") as f:
        alfworld_config = yaml.safe_load(f)

    AlfredTWEnv.collect_game_files = lambda self, verbose=False: None
    alfworld_env = AlfredTWEnv(alfworld_config, train_eval="train")
    alfworld_env.game_files = [task["game_file"]]
    alfworld_env.num_games = 1
    alfworld_env = alfworld_env.init_env(batch_size=1)

    env = AlfWorldEnv(alfworld_env, **config)

    return env
