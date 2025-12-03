def get_instruction_prompt(env, mission):
    instruction_prompt = ""

    return instruction_prompt


def get_single_obs_template(mission):
    template_wo_his = f"""
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {{current_observation}}
Your admissible actions of the current situation are: [{{admissible_actions}}].

""".strip()

    template = f"""
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {mission}
Prior to this step, you have already taken {{step_count}} step(s). Below are the most recent {{history_length}} observaitons and the corresponding actions you took: {{history}}
You are now at step {{current_step}} and your current observation is: {{current_observation}}
Your admissible actions of the current situation are: [{{admissible_actions}}].

""".strip()

    return template_wo_his, template
