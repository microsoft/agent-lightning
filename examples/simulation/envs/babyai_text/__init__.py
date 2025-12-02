from .clean_lang_wrapper import BabyAITextCleanLangWrapper
from .llm_agents_wrapper import BabyAILLMAgentsWrapper

ACTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object one step in front of you",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object one step in front of you",
}

def get_available_action_description():
    available_action_desc = ",\n".join(f"{action}: {description}" for action, description in ACTIONS.items())
    return available_action_desc

def get_instruction_prompt(mission="BabyAI-MixedTrainLocal-v0",):
    available_action_desc = get_available_action_description()
    
    instruction_prompt = f"""
You are an agent playing a simple navigation game. Your goal is to {mission}. The following are the possible actions you can take in the game, followed by a short description of each action:

{available_action_desc}.

In a moment I will present you an observation.

Tips:
- Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.
- It doesn't make sense to repeat the same action over and over if the observation doesn't change.

PLAY!
""".strip()

    return instruction_prompt
