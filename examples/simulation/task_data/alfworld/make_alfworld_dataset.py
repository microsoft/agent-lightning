import json
import os
import random

import pandas as pd
from tqdm import tqdm

# Set ALFWORLD data path
os.environ["ALFWORLD_DATA"] = "examples/simulation/envs/alfworld/alfworld_source"
alfworld_data_path = os.environ.get("ALFWORLD_DATA")

train_data_path = f"{alfworld_data_path}/json_2.1.1/train"
test_data_path = f"{alfworld_data_path}/json_2.1.1/valid_seen"

output_dir = "examples/simulation/task_data/alfworld"
os.makedirs(output_dir, exist_ok=True)

task_types = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]


def collect_valid_game_files(data_path, split_name, sample_size=None):
    game_files = []

    print(f"\nProcessing {split_name} split...")
    for root, dirs, files in tqdm(list(os.walk(data_path, topdown=False))):
        if "traj_data.json" in files:
            json_path = os.path.join(root, "traj_data.json")
            game_file_path = os.path.join(root, "game.tw-pddl")

            if "movable" in root or "Sliced" in root:
                print("Movable & slice trajs not supported %s" % (root))
                continue

            # Get goal description
            with open(json_path, "r") as f:
                traj_data = json.load(f)

            # Check for any task_type constraints
            if not traj_data["task_type"] in task_types:
                print("Skipping task type")
                continue

            # Check if a game file exists
            if not os.path.exists(game_file_path):
                print(f"Skipping missing game! {game_file_path}")
                continue

            with open(game_file_path, "r") as f:
                gamedata = json.load(f)

            # Check if previously checked if solvable
            if "solvable" not in gamedata:
                print(f"-> Skipping missing solvable key! {game_file_path}")
                continue

            if not gamedata["solvable"]:
                print("Skipping known %s, unsolvable game!" % game_file_path)
                continue

            game_files.append(game_file_path)

    # Random sampling for test split
    if sample_size is not None and len(game_files) > sample_size:
        random.seed(42)
        game_files = random.sample(game_files, k=sample_size)

    # Save to parquet
    df = pd.DataFrame(game_files, columns=["game_file"])
    parquet_path = os.path.join(output_dir, f"{split_name}.parquet")
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    print(f"✅ Saved {split_name}.parquet with {len(df)} entries at {parquet_path}")


# Process both splits
collect_valid_game_files(train_data_path, "train")
collect_valid_game_files(test_data_path, "test")
