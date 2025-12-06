import gymnasium as gym
from captioners.extract_actions import extract_pure_action, extract_reasoning
from PIL import Image


class BabaIsAILLMAgentsWrapper(gym.Wrapper):
    def __init__(self, env, vlm=False, **kwargs):
        super().__init__(env)
        self.env = env
        self.binary_reward = kwargs.get("binary_reward", False)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.binary_reward:
            reward = 1.0 if reward > 0 else reward
        return obs, reward * 1.0, terminated, truncated, info

    def extract_action(self, llm_output, use_reasoning):
        if use_reasoning:
            reasoning, reasoning_valid = extract_reasoning(llm_output)
            action, action_valid = extract_pure_action(llm_output)
        else:
            reasoning = None
            reasoning_valid = None
            action = llm_output
            action_valid = True

        action_valid = action_valid and action in self.language_action_space
        is_valid = action_valid and (not use_reasoning or reasoning_valid)

        lower_pred_action = action.lower()
        lower_pred_action = lower_pred_action.replace("_", " ")

        action = lower_pred_action

        valid_action = action if action_valid else self.default_action

        total_but_occurrences = 0
        for word in ["However", "different", "but", "wait", "won't", "can't", "cannot", "another"]:
            total_but_occurrences += llm_output.lower().count(word.lower())
        metrics = {"behavior/valid_action_ratio": is_valid * 1.0, "behavior/backtrack_length": total_but_occurrences}

        return reasoning, valid_action, is_valid, metrics
