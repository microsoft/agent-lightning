from captioners.debugging import print_step

class EnvironmentManager:
    
    def __init__(self, env_name, config, env_fn, prompt_builder):
        self.config = config
        self.env = env_fn()  # Initialize the environment
        self.prompt_builder = prompt_builder
        self.env_name = env_name
        self.image = None  # Cached image for rendering
        self.obs_type = self.config.captioner.obs_type

        self.mission = None

    def get_mission(self, env_obs, info):
        if self.env_name == "babyai":
            self.mission = env_obs["mission"]
        if self.env_name == "minihack":
            if "corridor" in self.task.lower():
                self.mission = "Your goal is to explore the level and reach the stairs down"
            elif "quest" in self.task.lower():
                self.mission = "Your goal is to explore the level, fight monsters, and navigate rooms and mazes to ultimately reach the stairs down."
            elif "boxoban" in self.task.lower():
                self.mission = "You are playing Boxoban, a box-pushing game inspired by Sokoban. Your goal is to push the boulders onto the fountains on the map. You can push the boulders by walking into them, as long as there are no obstacles behind them."
            else:
                self.mission = "Your goal is to get as far as possible in the game."
        if self.env_name == "scienceworld":
            self.mission = info["taskDesc"].split("Task Description:\n")[-1]
        if self.env_name == "alfworld":
            self.mission = env_obs["text"].split("Your task is to: ")[-1]

    def get_instruction_prompt(self, info=None):
        env_map = {
            "minihack": ("envs.minihack", {"env": self.env, "mission": self.mission}),
            "babyai": ("envs.babyai_text", {"mission": self.mission}),
            # "textworld": ("envs.textworld", {"task": self.task_name}),
            "babaisai": ("envs.babaisai", {}),
            "scienceworld": ("envs.scienceworld", {"env": self.env, "mission": self.mission}),
        }

        if self.env_name not in env_map:
            raise ValueError(f"Unknown environment: {self.env_name}")

        module_name, kwargs = env_map[self.env_name]
        module = __import__(module_name, fromlist=["get_instruction_prompt"])
        get_prompt = getattr(module, "get_instruction_prompt")
        return get_prompt(**kwargs)
        
    def get_single_obs_template(self):
        if self.env_name == "scienceworld":
            from envs.scienceworld import get_single_obs_template

            return get_single_obs_template(self.mission)

        elif self.env_name == "alfworld":
            from envs.alfworld import get_single_obs_template

            return get_single_obs_template(self.mission)

    def get_obs(self):
        if self.obs_type == "chat":
            obs = self.prompt_builder.get_chat_prompt()
        elif self.obs_type == "single":
            obs = self.prompt_builder.get_single_prompt()
        return obs

    def get_success_score(self):
        return self.env.get_success_score()

    def step(self, llm_output, use_reasoning=True, use_success_rate=False):
        reasoning, executed_action, is_valid, metrics = self.env.extract_action(llm_output, use_reasoning)
        env_obs, reward, terminated, truncated, info = self.env.step(executed_action)

        if use_success_rate:
            reward = self.get_success_score()

        # debugging
        # print_step(reasoning, executed_action, env_obs, reward, terminated)

        self.image = env_obs.get("image", None)
            
        self.prompt_builder.update_step_count()
        self.prompt_builder.update_reasoning(reasoning)
        self.prompt_builder.update_action(executed_action)
        self.prompt_builder.update_observation(env_obs)
        if hasattr(self.env, "available_actions_hint"):
            self.prompt_builder.update_admissible_actions(self.env.available_actions_hint)
            
        info["metrics"] = metrics

        if self.obs_type == "chat":
            obs = self.prompt_builder.get_chat_prompt()
        elif self.obs_type == "single":
            obs = self.prompt_builder.get_single_prompt()

        pure_env_obs = self.prompt_builder.get_pure_env_obs(env_obs)
        
        return obs, pure_env_obs, executed_action, is_valid, reward, terminated, truncated, info

    def reset(self):
        # Reset both the environment and the captioner
        self.prompt_builder.reset()
        env_obs, info = self.env.reset()
        self.image = env_obs.get("image", None)

        self.get_mission(env_obs, info)

        if self.obs_type == "chat":
            inst_prompt = self.get_instruction_prompt(info)
            self.prompt_builder.update_instruction_prompt(inst_prompt)
        elif self.obs_type == "single":
            template_wo_his, template = self.get_single_obs_template()
            self.prompt_builder.update_single_obs_template(template_wo_his, template)
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")
        
        self.prompt_builder.update_observation(env_obs)
        if hasattr(self.env, "available_actions"):
            self.prompt_builder.update_admissible_actions(self.env.available_actions_hint)
            
        if self.obs_type == "chat":
            obs = self.prompt_builder.get_chat_prompt()
        elif self.obs_type == "single":
            obs = self.prompt_builder.get_single_prompt()

        pure_env_obs = self.prompt_builder.get_pure_env_obs(env_obs)

        return obs, pure_env_obs, info

    def render(self):
        return self.image

    def close(self):
        self.env.close()