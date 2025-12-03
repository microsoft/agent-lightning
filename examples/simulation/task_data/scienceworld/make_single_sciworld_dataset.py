import json
import os

import pandas as pd
from scienceworld import ScienceWorldEnv

total_task_list = [
    "boil",
    "change-the-state-of-matter-of",
    "chemistry-mix",
    "chemistry-mix-paint-secondary-color",
    "chemistry-mix-paint-tertiary-color",
    "find-animal",
    "find-living-thing",
    "find-non-living-thing",
    "find-plant",
    "freeze",
    "grow-fruit",
    "grow-plant",
    "identify-life-stages-1",
    "identify-life-stages-2",
    "inclined-plane-determine-angle",
    "inclined-plane-friction-named-surfaces",
    "inclined-plane-friction-unnamed-surfaces",
    "lifespan-longest-lived",
    "lifespan-longest-lived-then-shortest-lived",
    "lifespan-shortest-lived",
    "measure-melting-point-known-substance",
    "measure-melting-point-unknown-substance",
    "melt",
    "mendelian-genetics-known-plant",
    "mendelian-genetics-unknown-plant",
    "power-component",
    "power-component-renewable-vs-nonrenewable-energy",
    "test-conductivity",
    "test-conductivity-of-unknown-substances",
    "use-thermometer",
]


def build_simplification_str(args):
    simplifications = list()
    if args["teleport"]:
        simplifications.append("teleportAction")
    if args["self_watering_plants"]:
        simplifications.append("selfWateringFlowerPots")
    if args["open_containers"]:
        simplifications.append("openContainers")
    if args["open_doors"]:
        simplifications.append("openDoors")
    if args["no_electrical"]:
        simplifications.append("noElectricalAction")
    return args["simplifications_preset"] or ",".join(simplifications)


def parse_args():
    from types import SimpleNamespace

    args = SimpleNamespace(
        jar_path=None,
        task_num=0,  # 7
        var_num=0,
        env_step_limit=100,
        num_episodes=5,
        seed=None,
        output_path_prefix="save-histories",
        max_episode_per_file=1000,
        simplifications_preset="easy",
        teleport=False,
        self_watering_plants=False,
        open_containers=True,
        open_doors=True,
        no_electrical=False,
    )
    params = vars(args)
    return params


env_args = parse_args()
env = ScienceWorldEnv("", serverPath=env_args["jar_path"], envStepLimit=env_args["env_step_limit"])
results = []

for taskidx, task_name in enumerate(total_task_list):
    print(f"\n=== Task {taskidx}: {task_name} ===")
    try:
        env.load(task_name, 0, build_simplification_str(env_args))
        train_list = env.get_variations_train()
        test_list = env.get_variations_test()
    except Exception as e:
        print(f"⚠️ Failed to load {task_name}: {e}")
        continue

    task_entry = {"task": task_name, "count": len(train_list), "variations": []}

    for var in train_list:
        try:
            env.load(task_name, var, build_simplification_str(env_args))
            obs, info = env.reset()
            task_desp = info["taskDesc"]
            print(f"[Variation {var}] {task_desp}")

            task_entry["variations"].append({"variation": var, "description": task_desp})

        except Exception as e:
            print(f"  ⚠️ Variation {var} failed: {e}")

    results.append(task_entry)

    # Save per-task parquet file
    output_dir = f"examples/simulation/task_data/scienceworld/single_data/{taskidx}"
    os.makedirs(output_dir, exist_ok=True)

    train_data = [[task_name, var] for var in train_list[:5]] * 64
    df = pd.DataFrame(train_data, columns=["sub_task_name", "variation_idx"])
    df.to_parquet(os.path.join(output_dir, "train.parquet"), engine="pyarrow", index=False)

    eval_data = [[task_name, var] for var in test_list[:20]]
    df = pd.DataFrame(eval_data, columns=["sub_task_name", "variation_idx"])
    df.to_parquet(os.path.join(output_dir, "test.parquet"), engine="pyarrow", index=False)

# Save summary JSON
output_path = "examples/simulation/task_data/scienceworld/single_data/scienceworld_tasks.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ All task info saved to: {output_path}")
