import gymnasium as gym
from captioners.extract_actions import extract_pure_action, extract_reasoning
from PIL import Image

POSSIBLE_ACTIONS = [
    "turn left",
    "turn to the left",
    "turn right",
    "turn to the right",
    "go forward",
    "move forward",
    "pick up",
    "pick it up",
    "drop",
    "toggle",
    "open",
    "turning left",
    "turning right",
    "moving forward",
    "picking up",
    "dropping",
    "toggling",
    "opening",
]


class BabyAILLMAgentsWrapper(gym.Wrapper):
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
        if lower_pred_action == "turnleft":
            lower_pred_action = "turn left"
        elif lower_pred_action == "turnright":
            lower_pred_action = "turn right"
        elif lower_pred_action == "goforward":
            lower_pred_action = "go forward"
        elif lower_pred_action == "pickup":
            lower_pred_action = "pick up"

        action = lower_pred_action

        valid_action = action if action_valid else self.default_action

        total_action_occurrences = 0
        for p_action in POSSIBLE_ACTIONS:
            total_action_occurrences += llm_output.lower().count(p_action.lower())

        valid_count = 1.0 if is_valid else 0.0

        total_but_occurrences = 0
        for word in ["However", "different", "but", "wait", "won't", "can't", "cannot", "another"]:
            total_but_occurrences += llm_output.lower().count(word.lower())

        metrics = {
            "behavior/valid_action_ratio": valid_count,
            "behavior/plan_length": total_action_occurrences,
            "behavior/backtrack_length": total_but_occurrences,
        }

        return reasoning, valid_action, is_valid, metrics
