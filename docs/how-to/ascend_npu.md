# agent-lightning x Ascend

We have added support for **Huawei Ascend NPUs** in **agent-lightning**, and provided an example of training a SQL agent based on the **Spider dataset**.

## Hardware Support

- Atlas 200T A2 Box16
- Atlas 900 A2 PODc
- Atlas 800T A3

At least **a single 40GB NPU** is required to run the Qwen2.5-Coder-1.5B-Instruct model.

## Environment Setup

### Basic Environment

- Python: 3.11.13
- CANN: 8.2.RC1
- torch: 2.7.1+cpu
- torch_npu: 2.7.1.dev20250724

> For basic environment preparation, please refer to this [document](https://gitcode.com/Ascend/pytorch).

### Configure Mirror Sources

Before installing dependencies, it is recommended to configure pip mirrors:

```
pip config set global.index-url http://repo.huaweicloud.com/repository/pypi/simple
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"

# Mirrors:
# http://repo.huaweicloud.com/repository/pypi/simple
# https://download.pytorch.org/whl/cpu/
# https://mirrors.huaweicloud.com/ascend/repos/pypi
```

### Install vLLM & vLLM-Ascend

```
pip install vllm==0.10.0 --trusted-host repo.huaweicloud.com
pip install vllm-Ascend==0.10.0rc1 --trusted-host repo.huaweicloud.com
```

### Install VERL

```
pip install verl==0.5.0
```

> ⚠️ Note: To ensure the VERL framework runs correctly on NPU, add the following two lines to the file `verl/utils/vllm_utils.py`:

```
from vllm_ascend.patch import platform
from vllm_ascend.patch import worker
```

### Install agent-lightning

```
pip install agentlightning==0.2.1
```

### Install Other Dependencies

```
pip install autogen-agentchat autogen-ext mcp
pip install langgraph "langchain[openai]" langchain-community langchain-text-splitters
pip install sqlparse nltk
```

## Model

We use the [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) model to train the SQL agent. Running requires at least **one 40GB NPU**.

## Dataset

We use the Spider 1.0 dataset, which contains about 8,000 samples, including natural language questions, database schemas, and corresponding standard SQL queries.

Training requires the following three Parquet files:

- `train_spider.parquet`
- `test_dev_500.parquet`
- `test_dev.parquet`

## Training Workflow

1. **Prepare the dataset**: Convert the Spider dataset into Parquet format and place it in the `data/` directory.

2. **Configure the environment**: Ensure vLLM-Ascend, VERL, and agent-lightning are correctly installed.

3. **Start training**: Run the following command to begin training the SQL agent:

   ```
   python train_sql_agent_npu.py qwen
   ```
