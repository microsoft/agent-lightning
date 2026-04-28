from typing import Optional

from balrog.environments.textworld import global_textworld_context
from balrog.environments.wrappers import GymV21CompatibilityV0


def make_textworld_env(env_name, task, config, render_mode: Optional[str] = None):
    textworld_context = global_textworld_context(tasks=config.tasks.textworld_tasks, **config.textworld_kwargs)
    env = textworld_context(task, **config.env_kwargs)
    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env
