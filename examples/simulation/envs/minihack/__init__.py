from nle.language_wrapper.wrappers.nle_language_wrapper import NLELanguageWrapper

ACTIONS = {
    "north": "move north",
    "east": "move east",
    "south": "move south",
    "west": "move west",
    "northeast": "move northeast",
    "southeast": "move southeast",
    "southwest": "move southwest",
    "northwest": "move northwest",
    "far north": "move far north",
    "far east": "move far east",
    "far south": "move far south",
    "far west": "move far west",
    "far northeast": "move far northeast",
    "far southeast": "move far southeast",
    "far southwest": "move far southwest",
    "far northwest": "move far northwest",
    "up": "go up the stairs",
    "down": "go down the stairs",
    "wait": "rest one move while doing nothing",
    "more": "display more of the message",
    "apply": "apply (use) a tool",
    "close": "close an adjacent door",
    "open": "open an adjacent door",
    "eat": "eat something",
    "force": "force a lock",
    "kick": "kick an enemy or a locked door or chest",
    "loot": "loot a box on the floor",
    "pickup": "pick up things at the current location if there are any",
    "pray": "pray to the gods for help",
    "puton": "put on an accessory",
    "quaff": "quaff (drink) something",
    "search": "search for hidden doors and passages",
    "zap": "zap a wand",
}


def get_available_actions(env):
    available_actions = {}
    for action in env.actions:
        action_key = NLELanguageWrapper.all_nle_action_map[action][0]
        if action_key not in ACTIONS:
            continue
        available_actions[action_key] = ACTIONS[action_key]
    return available_actions


def get_available_action_description(env):
    available_actions = get_available_actions(env)
    available_action_desc = ",\n".join(f"{action}: {description}" for action, description in available_actions.items())
    return available_action_desc


def get_instruction_prompt(env, mission):
    available_action_desc = get_available_action_description(env)

    instruction_prompt = f"""
You are an agent playing MiniHack. The following are the possible actions you can take in the game, followed by a short description of each action:

{available_action_desc}.

In a moment I will present a history of actions and observations from the game.

Tip: there is no point in outputting the same action over and over if nothing changes.

{mission}

PLAY!
""".strip()

    return instruction_prompt
