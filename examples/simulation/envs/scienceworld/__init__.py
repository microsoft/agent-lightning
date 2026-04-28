import re

from envs.scienceworld.examples import example_prompts


def get_available_action_description():
    all_possible_actions = [
        "open OBJ",
        "go OBJ",
        "focus on OBJ",
        "move OBJ to OBJ",
        "connect OBJ to OBJ",
        "pour OBJ in OBJ",
        "mix OBJ",
        "pickup OBJ",
    ]
    available_action_desc = ", ".join(all_possible_actions)
    return available_action_desc


def get_instruction_prompt(env, mission):
    import json

    with open("examples/simulation/task_data/scienceworld/split_sets/taskname2id.json", "r") as f:
        taskname2id = json.load(f)
    available_action_desc = get_available_action_description()

    #! Need fix
    taskIdx = taskname2id[env.sub_task_name]
    example_1 = example_prompts[taskIdx][0]
    example_2 = example_prompts[taskIdx][1]

    inventory = env.env.inventory()

    pattern = r"focus on\s+(\b\w+\b(\s+\b\w+\b)*)"
    matches = re.findall(pattern, env.env.taskdescription())
    to_focus = [match[0].replace("the ", " ").strip() for match in matches]
    focus_items = ", ".join(to_focus)

    instruction_prompt = f"""
You have done a few science experiments successfully and below are the action history of your experiments with similar tasks.
Here is 2 examples:
{example_1}
{example_2}

Follow the report of the two example tasks shown to you previously, try to solve a similar new task.
{mission}

All your possible action formats are:
{available_action_desc}
If you enter an unfamiliar room for the first time, you can use the action 'look around' to discover the objects in it.

Items in your inventory:
{inventory}

Important! You can only use FOCUS actions on these items: {focus_items}.
You cannot FOCUS on any other things. Please only use FOCUS as required by the task description.
Also, please FOCUS more directly, try not to focus on the container.
""".strip()

    return instruction_prompt


def get_single_obs_template(mission):
    template_wo_his = f"""
You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your current task is: {mission}

Your current observation is: {{current_observation}}

Current available actions:
{{admissible_actions}}
""".strip()

    template = f"""
You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your current task is: {mission}

Prior to this step, you have already taken {{step_count}} step(s). Below are the most recent {{history_length}} observations and the corresponding actions you took: {{history}}
You are now at step {{current_step}} and your current observation is: {{current_observation}}

Current available actions:
{{admissible_actions}}
""".strip()

    return template_wo_his, template
