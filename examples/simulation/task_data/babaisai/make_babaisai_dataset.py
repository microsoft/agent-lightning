import pandas as pd
import random

# Define all supported tasks
supported_tasks = [
    "env/goto_win-distr_obj",
    "env/two_room-goto_win",
    "env/two_room-goto_win-distr_obj_rule",
    "env/two_room-maybe_break_stop-goto_win",
    "env/goto_win",
    "env/two_room-maybe_break_stop-goto_win-distr_obj",
    "env/goto_win-distr_obj-irrelevant_rule",
    "env/goto_win-distr_obj_rule",
    "env/goto_win-distr_rule",
    "env/two_room-break_stop-goto_win",
    "env/two_room-goto_win-distr_obj",
    "env/two_room-goto_win-distr_rule",
    "env/make_win-distr_obj_rule",
    "env/make_win-distr_obj",
    "env/make_win-distr_rule",
    "env/make_win",
    "env/make_win-distr_obj-irrelevant_rule",
    "env/two_room-goto_win-distr_obj-irrelevant_rule",
    "env/two_room-goto_win-distr_win_rule",
    "env/two_room-break_stop-goto_win-distr_obj_rule",
    "env/two_room-break_stop-goto_win-distr_obj",
    "env/two_room-break_stop-goto_win-distr_rule",
    "env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule",
    "env/two_room-maybe_break_stop-goto_win-distr_obj_rule",
    "env/two_room-maybe_break_stop-goto_win-distr_rule",
    "env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule",
    "env/two_room-make_win-distr_obj_rule",
    "env/two_room-make_win-distr_rule",
    "env/two_room-make_win",
    "env/two_room-make_win-distr_obj-irrelevant_rule",
    "env/two_room-make_win-distr_obj",
    "env/two_room-make_win-distr_win_rule",
    "env/two_room-break_stop-make_win-distr_obj_rule",
    "env/two_room-break_stop-make_win-distr_rule",
    "env/two_room-break_stop-make_win",
    "env/two_room-break_stop-make_win-distr_obj-irrelevant_rule",
    "env/two_room-break_stop-make_win-distr_obj",
    "env/two_room-make_you",
    "env/two_room-make_you-make_win",
    "env/two_room-make_wall_win"
]

# Function to create balanced dataset
def make_balanced_dataset(total_size, tasks):
    if total_size % len(tasks) != 0:
        raise ValueError(f"total_size={total_size} not divisible by number of tasks={len(tasks)}")
    
    per_task = total_size // len(tasks)
    gym_env_ids = []
    for task in tasks:
        gym_env_ids.extend([task] * per_task)
    
    random.shuffle(gym_env_ids)
    return pd.DataFrame(gym_env_ids, columns=["gym_env_id"])

# Save train dataset
train_data_size = 400  # Must be divisible by len(supported_tasks), which is 40
train_df = make_balanced_dataset(train_data_size, supported_tasks)
train_df.to_parquet("examples/simulation/task_data/babaisai/train.parquet", engine="pyarrow", index=False)
print("Parquet file saved: train.parquet")

# Save test dataset
test_data_size = 80  # Must also be divisible by 40
test_df = make_balanced_dataset(test_data_size, supported_tasks)
test_df.to_parquet("examples/simulation/task_data/babaisai/test.parquet", engine="pyarrow", index=False)
print("Parquet file saved: test.parquet")
