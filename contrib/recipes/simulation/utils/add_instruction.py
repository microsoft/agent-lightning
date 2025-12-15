import copy

from autogen_core.models import UserMessage

# 🧭 Instruction text definitions
COT_INSTRUCTION = """
Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an appropriate action for the current step and present it within <action> </action> tags.
""".strip()

NAIVE_INSTRUCTION = """
Please response with only one line with one sentence, following the possible action format shown above. No extra words are allowed.
""".strip()

TIP_INSTRUCTION = """
Thanks for your playing.
Now you have ended a trajectory and collect some meaningless or valuable information from the interactions with the environment.
Please summary the trajectory, and also summary what information you get from this trajectory, and how far this trajectory is from fully completing the task.
Please response with only one sentence with only one line, do not include any extra words.

Your response must strictly follow this format:
<tip> ___ </tip>
""".strip()

# 📦 Mapping for instruction text types
INSTRUCTION_MAP = {
    "cot": COT_INSTRUCTION,
    "naive": NAIVE_INSTRUCTION,
    "tip": TIP_INSTRUCTION,
}


def _get_instruction(type: str, env_name: str = None):
    """
    Return the appropriate instruction text based on the type and environment name.

    Args:
        type (str): The instruction type (e.g., "cot", "naive", "critic", "tip").
        env_name (str, optional): Environment name. If 'babaisai', returns a special cot instruction.

    Returns:
        str: The selected instruction text.

    Raises:
        ValueError: If the instruction type is not recognized.
    """
    if type in INSTRUCTION_MAP:
        return INSTRUCTION_MAP[type]
    else:
        raise ValueError(f"Unknown instruction type: {type}")


def add_chat_instruction(prompt, type: str, sep: str = "\n\n", env_name: str = None):
    """
    Add the selected instruction text to the last message in a chat prompt (list version).

    Args:
        prompt (list): The conversation history, each element containing `.content`.
        type (str): Instruction type (e.g., "cot", "naive", "critic", "tip").
        env_name (str, optional): Environment name for special handling.

    Returns:
        list: A new prompt list with the instruction appended to the last message.
    """
    if type == "tip":
        new_prompt = copy.deepcopy(prompt)
        tip_instruction = _get_instruction(type, env_name)
        new_prompt.append(UserMessage(source="user", content=tip_instruction))

        return new_prompt
    else:
        new_prompt = copy.deepcopy(prompt)
        instruction = _get_instruction(type, env_name)
        new_prompt[-1].content += sep + instruction

        return new_prompt


def add_single_instruction(prompt, type: str, sep: str = "\n\n", env_name: str = None):
    """
    Add the selected instruction text to a single prompt (string or list version).

    Args:
        prompt (str or list): A single string prompt or a conversation history list.
        type (str): Instruction type (e.g., "cot", "naive", "critic", "tip").
        env_name (str, optional): Environment name for special handling.

    Returns:
        str or list: Updated prompt with the instruction appended.

    Raises:
        TypeError: If the prompt type is not string or list.
    """
    instruction = _get_instruction(type, env_name)

    if isinstance(prompt, str):
        return prompt + sep + instruction
    elif isinstance(prompt, list):
        new_prompt = copy.deepcopy(prompt)
        new_prompt[-1].content += sep + instruction
        return new_prompt
    else:
        raise TypeError("Prompt must be a string or a list of strings")


def add_chat_tips(prompt, tips):
    new_prompt = copy.deepcopy(prompt)
    new_prompt[-1].content += f"\n\n<tip> {tips}\n</tip>\n\n"
    return new_prompt


def add_chat_all_tips(prompt, tip_list):
    new_prompt = copy.deepcopy(prompt)
    tips_iter = iter(tip_list)

    for item in new_prompt:
        if "User" in item.type:
            tip = next(tips_iter, None)
            if tip is None:
                break
            if not tip == "":
                item.content += f"\n\n<tip> {tip}\n</tip>\n\n"

    return new_prompt


def add_single_tips(prompt, tips):
    if isinstance(prompt, str):
        return prompt + f"\n\n<tip> {tips}\n</tip>\n\n"
    elif isinstance(prompt, list):
        new_prompt = copy.deepcopy(prompt)
        new_prompt[-1].content += f"\n\n<tip> {tips}\n</tip>\n\n"
        return new_prompt
    else:
        raise TypeError("Prompt must be a string or a list of strings")
