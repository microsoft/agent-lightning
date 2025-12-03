from collections import defaultdict

import baba
import gym
import numpy as np
from baba.world_object import name_mapping
from PIL import Image

BABAISAI_ACTION_SPACE = [a.name for a in baba.grid.BabaIsYouEnv.Actions]


class BabaIsAIWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, add_ruleset=True, vlm=False):
        super().__init__(env)
        self.add_ruleset = add_ruleset
        self.language_action_space = BABAISAI_ACTION_SPACE[:]
        self.progression = 0.0
        self.target_plan = None

    @property
    def default_action(self):
        return BABAISAI_ACTION_SPACE[0]

    def get_text_action(self, action):
        return self.language_action_space[action.value]

    def get_ruleset(self):
        """
        Retrieve and format the ruleset for the current environment.

        This method extracts rules from the environment's grid ruleset,
        formats them into human-readable strings, and returns them as a
        single string with each rule on a new line.
        """
        rules = []
        for rule in self.env.grid._ruleset["_rule_"]:
            # all objects start with f, eg `fwall`, `fkey`...
            # are objects that can be manipulated, `wall` is used to indicate end of map
            if "object" not in rule:  # BabaIsAI bug fix
                continue
            name = rule["object"].removeprefix("f")
            named_property = name_mapping[rule["property"]]
            rules.append(f"{name} is {named_property}")

        return "\n".join(rules)

    def get_text_observation(self, obs):
        """
        Generate a text-based observation of the environment.

        This method creates a textual description of the environment,
        including the relative positions of various objects with respect
        to the player's position (represented by 'baba').
        """

        def find_objects(objects):
            obj = []
            for j in range(0, self.env.height):
                for i in range(0, self.env.width):
                    cell = self.env.grid.get(i, j)
                    if cell is not None and cell.type in objects:
                        if cell.type == "rule_object":
                            name = f"rule `{cell.name}`"
                        elif cell.type == "rule_is":
                            name = f"rule `{name_mapping[cell.name]}`"
                        elif cell.type == "rule_property":
                            name = f"rule `{name_mapping[cell.property]}`"
                        else:
                            name = cell.type
                        obj.append(((i, j), name))
            return obj

        def calculate_offsets(reference_position, positions):
            reference_position = np.asanyarray(reference_position)
            positions = np.asanyarray(positions)

            relative_positions = []
            for pos in positions:
                relative_positions.append(pos - reference_position)

            return relative_positions

        def form_description(relative_positions):
            def steps(v):
                return "steps" if v > 1 else "step"

            descriptions = []
            for pos in relative_positions:
                (x, y), name = pos
                name = name.removeprefix("f")

                x_direction = ""
                if x > 0:
                    x_direction = f"{x} {steps(x)} to the right"
                elif x < 0:
                    x_direction = f"{-x} {steps(x)} to the left"

                y_direction = ""
                if y > 0:
                    y_direction = f"{y} {steps(y)} down"
                elif y < 0:
                    y_direction = f"{-y} {steps(y)} up"

                description = ""
                if x_direction:
                    description = f"{name} {x_direction}"

                if y_direction:
                    if x_direction:
                        description += f" and {y_direction}"
                    else:
                        description = f"{name} {y_direction}"

                descriptions.append(description)

            return "\n".join(descriptions)

        you = None
        for rule in self.env.grid._ruleset["_rule_"]:
            if "property" not in rule:  # BabaIsAI bug fix
                continue
            named_property = name_mapping[rule["property"]]
            if named_property == "you":
                you = rule["object"]

        # TODO: we need to handle multilpe me)
        my_position = find_objects([you])
        if len(my_position) == 0:
            # We should reset the environment, as baba cannot legally move anymore
            return self.reset(), True
        my_position = my_position[0]
        other_positions = find_objects(
            [
                "fball",
                "fwall",
                "fdoor",
                "fkey",
                "rule_object",
                "rule_is",
                "rule_property",
            ]
        )
        offsets = calculate_offsets(my_position[0], [p[0] for p in other_positions])
        relative_positions = [(tuple(offset), pos[1]) for offset, pos in zip(offsets, other_positions)]
        text_observation = form_description(relative_positions)

        return text_observation, False

    def textworld_process_obsv(self, textworld_obsv):
        ruleset = self.get_ruleset()
        text_observation, reset = self.get_text_observation(textworld_obsv)
        prompt = "" if not reset else "[...] IS YOU rule was broken, environment reset"
        if self.add_ruleset:
            prompt += f"Active rules:\n{ruleset}\n\n"
        prompt += f"Objects on the map:\n{text_observation}"

        obs = defaultdict(lambda: None)

        obs["text"] = prompt
        image = Image.fromarray(self.env.render(mode="rgb_array")).convert("RGB")
        obs["image"] = image

        return obs

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.target_plan = self.env.target_plan
        self.progression = 0.0

        return self.textworld_process_obsv(obs)

    def step(self, action):
        action_int = self.language_action_space.index(action)
        obs, reward, done, info = self.env.step(action_int)

        if done:
            self.progression = 1.0 if reward > 0 else 0.0

        return self.textworld_process_obsv(obs), reward, done, info

    def get_stats(self):
        return {"target_plan": self.target_plan, "progression": self.progression}
