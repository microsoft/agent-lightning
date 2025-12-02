import re
import json
from abc import ABC
import logging
from typing import Tuple

from captioners.extract_actions import extract_reasoning, extract_pure_action

from webshop.web_agent_site.envs import WebAgentTextEnv

logger = logging.getLogger("agent_frame")

class WebShopEnv(ABC):
    def __init__(
        self,
        env: WebAgentTextEnv,
    ):
        self.session_id = self.task.session_id
        self.session = {}
        self.env = env

    def textworld_process_obsv(self, textworld_obsv):
        return {
            "text": textworld_obsv,
            "image": None,
        }
        
    def extract_action(self, llm_output, use_reasoning):
        if use_reasoning:
            reasoning, reasoning_valid = extract_reasoning(llm_output)
            action, action_valid = extract_pure_action(llm_output)
        else:
            reasoning = None
            reasoning_valid = None
            action = llm_output
            action_valid = True

        action_valid = action_valid and action in self.available_actions
        is_valid = action_valid and (not use_reasoning or reasoning_valid)
        
        metrics = {
            "behavior/valid_action_ratio": is_valid * 1.0,
        }
        
        return reasoning, action, is_valid, metrics
    
    def step(self, action: str, is_valid):
        observation, reward, done, info = self.env.step(action=action)

        return observation, self.state
    
    def reset(self):
        self.env.reset(self.session_id)
        obs = self.env.observation
        return self.textworld_process_obsv(obs), None