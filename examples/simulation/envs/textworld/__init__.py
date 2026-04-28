from envs.textworld.base import TextWorldFactory

TEXTWORLD_FACTORY = None


def global_textworld_context(**kwargs) -> TextWorldFactory:
    global TEXTWORLD_FACTORY
    if TEXTWORLD_FACTORY is None:
        TEXTWORLD_FACTORY = TextWorldFactory(**kwargs)
    return TEXTWORLD_FACTORY


intruction_prompts = dict(
    treasure_hunter="""
    You are an agent playing TextWorld, a text-based adventure game where you are in a randomly generated
    maze and must find a specific object. You need to explore different rooms to find the target object.
    Here are the available commands: look: describe the current room. goal: print the goal of this game
    inventory: print the player’s inventory go <dir>: move the player north, east, south, or west. You can
    only go in the direction indicated with an exit or a door. open ...: open a door or a container. You need to
    open a closed door before you want to go through it. drop ...: drop an object on the floor take ...: take an
    object that is visible. Make sure the object is visible to take. put ... on ...: place an object on a supporter
    take ... from ...: take an object from a container or a supporter insert ... into ...: place an object into a
    container unlock ... with ...: unlock a door or a container with a key. You need to unlock a locked door
    with a matched key in your inventory before you want to open it.
    - The target object might be located in a closed or locked container. - The adjective is useful for
    determining whether the key is matched with the lock (e.g. non-euclidean keycard is matched with
    non-euclidean safe). Make sure it is matched to unlock! - The key required to unlock the door may be in
    another room or locked inside a container. - Take the key whenever you can. - After unlocking a locked
    door or container, it will remain closed. You will then need to open it.
    You have 40 steps to complete the task. Restarting is forbidden.
    """,
    the_cooking_game="""
    You are an agent playing TextWorld, a text-based adventure game where you navigate through different
    rooms, interact with objects, and solve puzzles. Your goal is to first find the recipe, find and prepare food
    according to the recipe, and finally prepare and eat the meal.
    Here are the available commands: look: describe the current room goal: print the goal of this game
    inventory: print player’s inventory go <dir>: move the player north, east, south or west. You can only go
    to directions indicated with an exit or a door. examine ...: examine something more closely eat ...: eat
    edible food open ...: open a door or a container. You need to open a closed door before you can go through
    it. drop ...: drop an object onto the floor take ...: take an object that is visible put ... on ...: place an object
    on a supporter take ... from ...: take an object from a container or a supporter insert ... into ...: place an
    object into a container lock ... with ...: lock a door or a container with a key unlock ... with ...: unlock a
    door or a container with a key cook ... with ...: cook cookable food with something providing heat slice ...
    with ...: slice cuttable food with something sharp chop ... with ...: chop cuttable food with something
    sharp dice ... with ...: dice cuttable food with something sharp prepare meal: combine ingredients from
    inventory into a meal. You can only prepare meals in the Kitchen.
    - You can examine the cookbook to see the recipe when it is visible. - The BBQ is for grilling things,
    the stove is for frying things, the oven is for roasting things. Cooking ingredients in the wrong way will
    lead to a failure of the game. - Once you have got processed ingredients and the appropriate cooking tool
    ready, cook all of them according to the recipe. - There are two conditions to correctly cook something
    (grill/fry/roast): a) the ingredient you want to cook is in your inventory and b) there is a suitable cooking
    tool in the room, and then use ‘cook . . . with . . . ’ command. - When you need to chop/slice/dice
    ingredients, you need to take the knife and the ingredient in your inventory and then ‘slice/chop/dice ...
    with knife’ - Make sure to first process the food (chop/slice/dice) before you try to cook them. - When
    you have all the ingredients (that got processed or cooked according to the menu), you can ‘prepare meal’
    in the kitchen and then ‘eat meal’ to win the game. - The ingredients should EXACTLY match the color
    in the recipe, but if the recipe doesn’t specify color, any color would be fine. When you ‘take ... with ...’,
    use the EXACT name you see. - You don’t need to examine the container/supporter (e.g. toolbox) when
    it says something like "there isn’t a thing on it"/"has nothing on it"
    You have 80 steps to complete the task. Restarting is forbidden.
    """,
    coin_collector="""
    You are an agent playing TextWorld, a text-based adventure game where you are in a randomly generated
    maze and must find the coin. You need to explore different rooms to find the target object.
    Here are the available commands: goal: print the goal of this game go <dir>: move the player north, east,
    south, or west. You can only go in the direction indicated with something like an exit or a door. take coin:
    2in the game by ‘take coin’ if you see the coin in the room
    The only action you can do is go <dir> to explore the maze and ‘take coin’ when you see the coin in the
    room.
    You have 25 steps to complete the task. Restarting is forbidden.
    """,
)


def get_instruction_prompt(task=None):
    instruction_prompt = intruction_prompts[task].strip()

    return instruction_prompt
