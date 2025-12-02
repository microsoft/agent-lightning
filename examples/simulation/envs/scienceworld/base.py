import re
import json
from abc import ABC
import logging
import re
import numpy as np
from captioners.extract_actions import extract_reasoning, extract_pure_action

from scienceworld import ScienceWorldEnv

logger = logging.getLogger("agent_frame")

def jaccard_similarity(s1, s2):
    set1, set2 = set(s1.split()), set(s2.split())
    return len(set1 & set2) / len(set1 | set2)

class SciWorldEnv(ABC):
    def __init__(
        self,
        env: ScienceWorldEnv,
        **kwargs
    ):
        super().__init__()
        self.env = env
        self.step_count = 0

        self.success = False
        self.use_action_correction = kwargs["use_action_correction"]
    
    def set_max_steps(self, sub_task_name):
        self.sub_task_name = sub_task_name
        self.max_steps_dict = json.load(open("examples/simulation/task_data/scienceworld/split_sets/max_steps.json"))
        self.max_steps = self.max_steps_dict[sub_task_name]
    
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

        is_valid = action_valid and (not use_reasoning or reasoning_valid)
        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', llm_output):
            is_valid = False
        
        metrics = {
            "behavior/valid_action_ratio": is_valid * 1.0,
        }
        
        return reasoning, action, is_valid, metrics
    
    def step(self, action: str):
        if self.use_action_correction:
            validActions = self.env.get_valid_action_object_combinations_with_templates()
            if action not in validActions:
                sim_list = []
                for valid_action in validActions:
                    sim_list.append(jaccard_similarity(valid_action['action'], action))
                highest_idx = np.argmax(sim_list)
                action = validActions[highest_idx]['action']

        observation, reward, done, info = self.env.step(action)
        
        self.available_actions = info["valid"]
        
        valid_actions = self.env.get_possible_actions()
        valid_objs = self.env.get_possible_objects()
        self.available_actions_hint = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: focus on door"

        self.success = done and info["score"] > 70

        if self.step_count >= self.max_steps:
            done = True
        self.step_count += 1
        return self.textworld_process_obsv(observation), reward, done, None, info
    
    def reset(self):
        obs, info = self.env.reset()
        
        self.available_actions = info["valid"]

        valid_actions = self.env.get_possible_actions()
        valid_objs = self.env.get_possible_objects()
        self.available_actions_hint = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: focus on door"
        return self.textworld_process_obsv(obs), info

    def get_success_score(self):
        return 1.0 if self.success else 0.0

    def close(self):
        self.env.close()