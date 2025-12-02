import pandas as pd
import random

# supported_tasks = [
#     "BabyAI-MixedTrainLocal-v0/goto",
#     "BabyAI-MixedTrainLocal-v0/pickup",
#     "BabyAI-MixedTrainLocal-v0/open",
#     "BabyAI-MixedTrainLocal-v0/pick_up_seq_go_to",
#     "BabyAI-MixedTrainLocal-v0/putnext"
# ]

task_name = "goto"
supported_tasks = [f"BabyAI-MixedTrainLocal-v0/{task_name}"]

def make_balanced_dataset(total_size, tasks):
    if total_size % len(tasks) != 0:
        raise ValueError(f"total_size={total_size} not divisible by number of tasks={len(tasks)}")
    
    per_task = total_size // len(tasks)
    gym_env_ids = []
    for task in tasks:
        gym_env_ids.extend([task] * per_task)
    
    random.shuffle(gym_env_ids)  # shuffle so tasks are mixed
    return pd.DataFrame(gym_env_ids, columns=["gym_env_id"])

# Train dataset
train_data_size = 300
train_df = make_balanced_dataset(train_data_size, supported_tasks)
train_df.to_parquet(f"examples/simulation/task_data/babyai_text/train_{task_name}.parquet",
                    engine="pyarrow", index=False)
print("Parquet file saved: train.parquet")

# Test dataset
test_data_size = 60
test_df = make_balanced_dataset(test_data_size, supported_tasks)
test_df.to_parquet(f"examples/simulation/task_data/babyai_text/test_{task_name}.parquet",
                   engine="pyarrow", index=False)
print("Parquet file saved: test.parquet")