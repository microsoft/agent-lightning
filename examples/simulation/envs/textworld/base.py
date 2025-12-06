import glob
import importlib.resources
import os
from collections import defaultdict
from pathlib import Path

import gym
import textworld
import textworld.gym

workspace_dir = os.path.dirname(importlib.resources.files("balrog").__str__())


class TextWorldFactory:
    """
    A factory class for creating TextWorld environments.

    This class manages the creation of TextWorld environments for different tasks,
    cycling through available games for each task or allowing specific game selection.
    """

    _instance = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize(**kwargs)
        return cls._instance

    def initialize(self, textworld_games_path, tasks, max_episode_steps=40, **kwargs):
        self.max_steps = max_episode_steps

        textworld_games_path = os.path.join(workspace_dir, textworld_games_path)
        self.count = defaultdict(int)

        required_kwargs = ["objective", "description", "score", "max_score", "won"]
        for kwarg in required_kwargs:
            assert kwarg in kwargs and kwargs[kwarg]

        self.request_infos = textworld.EnvInfos(**kwargs)

        self.env_ids = defaultdict(list)
        for pattern in ["*.ulx", "*.z8"]:
            for entry in sorted(glob.glob(os.path.join(textworld_games_path, f"**/{pattern}"), recursive=True)):
                task = Path(entry).parent.name
                if task in tasks:
                    env_id = textworld.gym.register_game(entry, self.request_infos, max_episode_steps=max_episode_steps)
                    self.env_ids[task].append(env_id)

    def get_textworld_env(self, task, seed=None, **kwargs):
        """
        Create and return a TextWorld environment for the specified task.

        Args:
            task (str): The name of the task for which to create an environment.
            seed (int, optional): If provided, creates the environment for the
                                      specific game index. If None, cycles through
                                      available games.

        Returns:
            gym.Env: A TextWorld gym environment.

        Raises:
            KeyError: If the specified task is not found in the available tasks.
        """
        if task not in self.env_ids:
            raise KeyError(f"Task '{task}' not found. Available tasks are: {list(self.env_ids.keys())}")

        if seed is not None:
            env_id = self.env_ids[task][seed % len(self.env_ids[task])]
        else:
            self.count[task] += 1
            env_id = self.env_ids[task][self.count[task] % len(self.env_ids[task])]

        env = textworld.gym.make(env_id, **kwargs)
        env = TextWorldWrapper(env, max_steps=self.max_steps)
        return env

    def __call__(self, task, **kwargs):
        return self.get_textworld_env(task, **kwargs)


class AlwaysTrue:
    def __contains__(self, item):
        return True


class TextWorldWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps=40):
        super().__init__(env)
        self.language_action_space = AlwaysTrue()
        self.progression = 0.0
        self.max_steps = max_steps
        self.action_space = gym.spaces.Space()
        self.observation_space = gym.spaces.Space()

    @property
    def default_action(self):
        return "help"

    def get_text_action(self, action):
        return action

    def textworld_process_obsv(self, textworld_obsv):
        return {
            "text": textworld_obsv,
            "image": None,
        }

    def filter_objective(self, obs, info):
        objective = info["objective"]
        parts = obs.split(objective)
        if len(parts) == 1:
            return parts[0].strip()
        else:
            return parts[-1].strip()

    def reset(self):
        obs, info = self.env.reset()
        obs = self.filter_objective(obs, info)
        self.progression = 0.0

        return self.textworld_process_obsv(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.filter_objective(obs, info)

        if done:
            self.progression = max(info["score"] / info["max_score"], 1.0 if info["won"] else 0.0)

        return self.textworld_process_obsv(obs), reward, done, info

    def get_stats(self):
        return {"progression": self.progression}
