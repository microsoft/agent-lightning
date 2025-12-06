import gymnasium as gym
from PIL import Image

BABYAI_ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]


class BabyAITextCleanLangWrapper(gym.Wrapper):
    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.language_action_space = BABYAI_ACTION_SPACE[:]
        self._mission = None
        self.progression = 0.0

    @property
    def max_steps(self):
        return self.env.unwrapped.max_steps

    @property
    def interleaving_token(self):
        return self._interleaving_token

    @property
    def default_action(self):
        return "go forward"

    def get_text_action(self, action):
        return self.language_action_space[action.value]

    def get_prompt(self, obs, infos):
        image = Image.fromarray(self.env.unwrapped.get_pov_render(tile_size=16)).convert("RGB")

        def _form_prompt(description):
            return "\n".join([d.replace("You see ", "") for d in description])

        prompt = _form_prompt(infos["descriptions"])
        return prompt, image

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        prompt, image = self.get_prompt(obs, info)
        self._mission = obs["mission"]
        self.progression = 0.0
        # Following the convention from NetHack Language Wrapper for specifying
        # short term vs long term context here. There is no equivalent long term
        # context like e.g. inventory in BabyAI-Text.
        obs["text"] = prompt
        obs["image"] = image
        return obs, info

    def step(self, action):
        action_int = self.language_action_space.index(action)
        obs, reward, terminated, truncated, infos = self.env.step(action_int)
        if reward > 0:
            self.progression = 1.0
        prompt, image = self.get_prompt(obs, infos)
        obs["text"] = prompt
        obs["image"] = image
        return obs, reward, terminated, truncated, infos

    def get_stats(self):
        # No special stats tracking implemented for now
        return {"mission": self._mission, "progression": self.progression}
