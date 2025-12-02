import pandas as pd
import random

N = 100

supported_tasks = [
    "default"
]

gym_env_ids = [random.choice(supported_tasks) for _ in range(N)]

df = pd.DataFrame(gym_env_ids, columns=["gym_env_id"])
df.to_parquet("examples/simulation/task_data/crafter/train.parquet", engine="pyarrow", index=False)

print("Parquet file saved: train.parquet")