import re
from abc import ABC
import logging

from captioners.extract_actions import extract_reasoning, extract_pure_action

logger = logging.getLogger("agent_frame")


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


class AlfWorldEnv(ABC):
    def __init__(
        self,
        env,
        **kwargs
    ):
        self.env = env
        self.max_steps = kwargs.get("alfworld_kwargs", {})["max_steps"]
        self.step_count = 0

        self.success = False

    def textworld_process_obsv(self, textworld_obsv):
        return {
            "text": textworld_obsv,
            "image": None,
        }
        
    def extract_action(self, llm_output, use_reasoning):
        llm_output = llm_output.lower()
        if use_reasoning:
            reasoning, reasoning_valid = extract_reasoning(llm_output)
            action, action_valid = extract_pure_action(llm_output)
        else:
            reasoning = None
            reasoning_valid = None
            action = llm_output
            action_valid = True
            
        # action_valid = action_valid and action in self.available_actions

        is_valid = action_valid and (not use_reasoning or reasoning_valid)
        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', llm_output):
            is_valid = False
        
        metrics = {
            "behavior/valid_action_ratio": is_valid * 1.0,
        }
        
        return reasoning, action, is_valid, metrics
    
    def step(self, action: str):
        (observation,), (reward,), (done,), info = self.env.step([action])
        observation = process_ob(observation)

        self.available_actions = info['admissible_commands'][0]
        self.available_actions_hint = "\n ".join(f"'{s}'" for s in info['admissible_commands'][0] if s != 'help')

        self.success = info['won'][0]

        if self.step_count >= self.max_steps:
            done = True
        self.step_count += 1
        return self.textworld_process_obsv(observation), reward, done, None, info

    def reset(self):
        obs, info = self.env.reset()
        self.available_actions = info['admissible_commands'][0]
        self.available_actions_hint = "\n ".join(f"'{s}'" for s in info['admissible_commands'][0] if s != 'help')

        obs = "\n".join(obs[0].split("\n\n")[1:])
        return self.textworld_process_obsv(obs), info

    def get_success_score(self):
        return 1.0 if self.success else 0.0

    def close(self):
        self.env.close()