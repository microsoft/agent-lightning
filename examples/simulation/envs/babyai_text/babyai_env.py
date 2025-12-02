from pathlib import Path
from typing import Optional

import gymnasium as gym
import minigrid

from envs.babyai_text import BabyAITextCleanLangWrapper, BabyAILLMAgentsWrapper

minigrid.register_minigrid_envs()

# see discussion starting here: https://github.com/Farama-Foundation/Minigrid/pull/381#issuecomment-1646800992
broken_bonus_envs = {
    "BabyAI-PutNextS5N2Carrying-v0",
    "BabyAI-PutNextS6N3Carrying-v0",
    "BabyAI-PutNextS7N4Carrying-v0",
    "BabyAI-KeyInBox-v0",
}


# get all babyai envs (except the broken ones)
BABYAI_ENVS = []
for env_spec in gym.envs.registry:
    id = env_spec
    if id.split("-")[0] == "BabyAI":
        if id not in broken_bonus_envs:
            BABYAI_ENVS.append(id)

BABYAI_ENVS += [
    "BabyAI-MixedTrainLocal-v0/goto",
    "BabyAI-MixedTrainLocal-v0/pickup",
    "BabyAI-MixedTrainLocal-v0/open",
    "BabyAI-MixedTrainLocal-v0/putnext",
    "BabyAI-MixedTrainLocal-v0/pick_up_seq_go_to",
]


def make_babyai_env(env_name, task, config, render_mode: Optional[str] = None):
    if task.startswith("BabyAI-MixedTrainLocal-v0/"):
        base_task, goal = task.split("/")
        while 1:
            env = gym.make(base_task, render_mode=render_mode, **config.babyai_kwargs)
            if env.unwrapped.action_kinds[0].replace(" ", "_") == goal:
                break

    env = BabyAITextCleanLangWrapper(env, **config.babyai_kwargs)
    env = BabyAILLMAgentsWrapper(env, **config)

    return env
