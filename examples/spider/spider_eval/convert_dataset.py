import os

import pandas as pd

data_dir = "data"
target_data_dir = "data"
columns = ["db_id", "question", "query"]

dev_path = os.path.join(data_dir, "dev.json")
dev_df = pd.read_json(dev_path)
print(dev_df)
dev_df[columns].to_parquet(os.path.join(target_data_dir, "dev.parquet"), index=False)

train_path = os.path.join(data_dir, "train_spider.json")
train_df = pd.read_json(train_path)
print(train_df)
train_df[columns].to_parquet(os.path.join(target_data_dir, "train_spider.parquet"), index=False)

test_path = os.path.join(data_dir, "test.json")
test_df = pd.read_json(test_path)
print(test_df)
test_df[columns].to_parquet(os.path.join(target_data_dir, "test.parquet"), index=False)

# Select 100 of test df as test_dev
test_dev_df = test_df.sample(n=100, random_state=42)
test_dev_df[columns].to_parquet(os.path.join(target_data_dir, "test_dev.parquet"), index=False)

# Select 500 of test df as test_dev
test_dev_df = test_df.sample(n=500, random_state=0)
test_dev_df[columns].to_parquet(os.path.join(target_data_dir, "test_dev_500.parquet"), index=False)
