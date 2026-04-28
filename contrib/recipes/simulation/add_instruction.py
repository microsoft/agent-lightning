import copy

from autogen_core.models import UserMessage

# Instruction text definitions
COT_INSTRUCTION = """
Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an appropriate action for the current step and present it within <action> </action> tags.
""".strip()

NAIVE_INSTRUCTION = """
Please response with only one line with one sentence, following the possible action format shown above. No extra words are allowed.
""".strip()

# Mapping for instruction text types
INSTRUCTION_MAP = {
    "cot": COT_INSTRUCTION,
    "naive": NAIVE_INSTRUCTION,
}

def _get_instruction(type: str, env_name: str = None):
    """
    Retrieve an instruction string from INSTRUCTION_MAP based on the given type.

    Args:
        type (str): Instruction type key (e.g., "cot", "naive", "critic", "tip").
        env_name (str, optional): Currently unused. Reserved for future
            environment-specific instruction handling.

    Returns:
        str: The corresponding instruction text.

    Raises:
        ValueError: If the given instruction type is not found in INSTRUCTION_MAP.
    """
    if type in INSTRUCTION_MAP:
        return INSTRUCTION_MAP[type]
    else:
        raise ValueError(f"Unknown instruction type: {type}")


def add_chat_instruction(prompt, type: str, sep: str = "\n\n", env_name: str = None):
    """
    Append an instruction to the content of the last message in a chat-style prompt.

    This function does not modify the original prompt. Instead, it returns a
    deep-copied prompt list with the instruction appended.

    Args:
        prompt (list): A conversation history represented as a list of objects.
            Each object must have a `.content` attribute.
        type (str): Instruction type key (e.g., "cot", "naive", "critic", "tip").
        sep (str, optional): Separator inserted between the existing content
            and the instruction.
        env_name (str, optional): Currently unused. Reserved for future use.

    Returns:
        list: A new prompt list with the instruction appended to the last message.
    """
    new_prompt = copy.deepcopy(prompt)
    instruction = _get_instruction(type, env_name)
    new_prompt[-1].content += sep + instruction

    return new_prompt


def add_single_instruction(prompt, type: str, sep: str = "\n\n", env_name: str = None):
    """
    Append an instruction to a single prompt or a chat-style prompt.

    - If `prompt` is a string, the instruction is appended to the string.
    - If `prompt` is a list, the instruction is appended to the `.content`
      of the last message.

    Args:
        prompt (str or list): Either a single prompt string or a conversation
            history list whose elements have a `.content` attribute.
        type (str): Instruction type key (e.g., "cot", "naive", "critic", "tip").
        sep (str, optional): Separator inserted between the existing content
            and the instruction.
        env_name (str, optional): Currently unused. Reserved for future use.

    Returns:
        str or list: The updated prompt with the instruction appended.

    Raises:
        TypeError: If `prompt` is neither a string nor a list.
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