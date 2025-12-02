import json
import random
import pandas as pd

# unseen variants seen tasks

with open("examples/simulation/task_data/scienceworld/split_sets/variations_idx/seen_variants_seen_tasks.json", "r") as f:
    data = json.load(f)
with open("examples/simulation/task_data/scienceworld/split_sets/id2taskname.json", "r") as f:
    id2taskname = json.load(f)

train_list = [
    [id2taskname.get(str(pair[0]), None), pair[1], 30]
    for pair in data["train"]
]
df = pd.DataFrame(train_list, columns=["sub_task_name", "variation_idx", "max_steps"])
df.to_parquet("examples/simulation/task_data/scienceworld/multi_data/train.parquet", engine="pyarrow", index=False)

print("Parquet file saved: train.parquet")

groups = {}
for a,b in data["train"]:
    groups.setdefault(a, []).append([a,b])

# Sample up to 5 from each group
result = {k: random.sample(v, min(5, len(v))) for k,v in groups.items()}

test_id_list = []
for key in result:
    test_id_list.extend(result[key])

test_list = [
    [id2taskname.get(str(pair[0]), None), pair[1], 30]
    for pair in test_id_list
]

df = pd.DataFrame(test_list, columns=["sub_task_name", "variation_idx", "max_steps"])
df.to_parquet("examples/simulation/task_data/scienceworld/multi_data/test.parquet", engine="pyarrow", index=False)

print("Parquet file saved: test.parquet")