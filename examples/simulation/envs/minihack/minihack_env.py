from typing import Optional

import gym
import minihack  # NOQA: F401

from envs.nle import AutoMore, NLELanguageWrapper
from envs.wrappers import GymV21CompatibilityV0, NLETimeLimit
from envs.minihack.llm_agents_wrapper import MiniHackLLMAgentsWrapper

MINIHACK_ENVS = []
for env_spec in gym.envs.registry.all():
    id = env_spec.id
    if id.split("-")[0] == "MiniHack":
        MINIHACK_ENVS.append(id)


def make_minihack_env(env_name, task, config, render_mode: Optional[str] = None):
    minihack_kwargs = dict(config.minihack_kwargs)
    skip_more = minihack_kwargs.pop("skip_more", False)
    vlm = True if config.captioner.max_image_history > 0 else False
    env = gym.make(
        task,
        observation_keys=[
            "glyphs",
            "blstats",
            "tty_chars",
            "inv_letters",
            "inv_strs",
            "tty_cursor",
            "tty_colors",
        ],
        **minihack_kwargs,
    )
    if skip_more:
        env = AutoMore(env)
    env = NLELanguageWrapper(env, vlm=vlm)

    # wrap NLE with timeout
    env = NLETimeLimit(env)

    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)
    env = MiniHackLLMAgentsWrapper(env)

    return env
