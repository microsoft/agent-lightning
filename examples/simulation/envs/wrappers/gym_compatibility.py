"""Compatibility wrappers for OpenAI gym V22 and V26."""

# pyright: reportGeneralTypeIssues=false, reportPrivateImportUsage=false
from __future__ import annotations

import sys
from typing import Any

import gymnasium
from gymnasium import error
from gymnasium.core import ActType, ObsType
from gymnasium.error import MissingArgument
from gymnasium.logger import warn
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
else:
    from typing_extensions import Protocol, runtime_checkable


try:
    import gym
    import gym.wrappers
except ImportError as e:
    GYM_IMPORT_ERROR = e
else:
    GYM_IMPORT_ERROR = None


@runtime_checkable
class LegacyV21Env(Protocol):
    """A protocol for OpenAI Gym v0.21 environment."""

    observation_space: gym.Space
    action_space: gym.Space

    def reset(self) -> Any:
        """Reset the environment and return the initial observation."""
        ...

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Run one timestep of the environment's dynamics."""
        ...

    def render(self, mode: str | None = "human") -> Any:
        """Render the environment."""
        ...

    def close(self):
        """Close the environment."""
        ...

    def seed(self, seed: int | None = None):
        """Set the seed for this env's random number generator(s)."""
        ...


class GymV21CompatibilityV0(gymnasium.Env[ObsType, ActType]):
    r"""A wrapper which can transform an environment from the old API to the new API.

    Old step API refers to step() method returning (observation, reward, done, info), and reset() only retuning the observation.
    New step API refers to step() method returning (observation, reward, terminated, truncated, info) and reset() returning (observation, info).
    (Refer to docs for details on the API change)

    Known limitations:
    - Environments that use `self.np_random` might not work as expected.
    """

    def __init__(
        self,
        env_id: str | None = None,
        make_kwargs: dict | None = None,
        env: gym.Env | None = None,
        render_mode: str | None = None,
    ):
        """A wrapper which converts old-style envs to valid modern envs.

        Some information may be lost in the conversion, so we recommend updating your environment.
        """
        if GYM_IMPORT_ERROR is not None:
            raise error.DependencyNotInstalled(
                f"{GYM_IMPORT_ERROR} (Hint: You need to install gym with `pip install gym` to use gym environments"
            )

        if make_kwargs is None:
            make_kwargs = {}

        if env is not None:
            gym_env = env
        elif env_id is not None:
            gym_env = gym.make(env_id, **make_kwargs)
        else:
            raise MissingArgument("Either env_id or env must be provided to create a legacy gym environment.")
        self.observation_space = _convert_space(gym_env.observation_space)
        self.action_space = _convert_space(gym_env.action_space)

        gym_env = _strip_default_wrappers(gym_env)

        self.metadata = getattr(gym_env, "metadata", {"render_modes": []})
        self.render_mode = render_mode
        self.reward_range = getattr(gym_env, "reward_range", None)
        self.spec = getattr(gym_env, "spec", None)

        self.gym_env: LegacyV21Env = gym_env

    def __getattr__(self, item: str):
        """Gets an attribute that only exists in the base environments."""
        return getattr(self.gym_env, item)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[ObsType, dict]:
        """Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        if seed is not None:
            self.gym_env.seed(seed)

        # Options are ignored - https://github.com/openai/gym/blob/c755d5c35a25ab118746e2ba885894ff66fb8c43/gym/core.py
        if options is not None:
            warn(f"Gym v21 environment do not accept options as a reset parameter, options={options}")

        obs, info = self.gym_env.reset(), {}

        if self.render_mode is not None:
            self.render()

        return obs, info

    def step(self, action: ActType) -> tuple[Any, float, bool, bool, dict]:
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        obs, reward, done, info = self.gym_env.step(action)

        if self.render_mode is not None:
            self.render()

        return convert_to_terminated_truncated_step_api((obs, reward, done, info))

    def render(self) -> Any:
        """Renders the environment.

        Returns:
            The rendering of the environment, depending on the render mode
        """
        return self.gym_env.render(mode=self.render_mode)

    def close(self):
        """Closes the environment."""
        self.gym_env.close()

    def __str__(self):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self.gym_env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)


def _strip_default_wrappers(env: gym.Env) -> gym.Env:
    """Strips builtin wrappers from the environment.

    Args:
        env: the environment to strip builtin wrappers from

    Returns:
        The environment without builtin wrappers
    """
    default_wrappers = ()
    if hasattr(gym.wrappers, "render_collection"):
        default_wrappers += (gym.wrappers.render_collection.RenderCollection,)
    if hasattr(gym.wrappers, "human_rendering"):
        default_wrappers += (gym.wrappers.human_rendering.HumanRendering,)
    while isinstance(env, default_wrappers):
        env = env.env
    return env


def _convert_space(space: gym.Space) -> gymnasium.Space:
    """Converts a gym space to a gymnasium space.

    Args:
        space: the space to convert

    Returns:
        The converted space
    """
    if isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gym.spaces.MultiBinary):
        return MultiBinary(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Tuple(spaces=tuple(map(_convert_space, space.spaces)))
    elif isinstance(space, gym.spaces.Dict):
        return Dict(spaces={k: _convert_space(v) for k, v in space.spaces.items()})
    elif isinstance(space, gym.spaces.Space):
        return Space()
    else:
        raise NotImplementedError(f"Cannot convert space of type {space}. Please upgrade your code to gymnasium.")
