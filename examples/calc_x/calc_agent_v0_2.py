# Copyright (c) Microsoft. All rights reserved.

import re
from typing import Any, Tuple

from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from calc_agent import eval, get_agent
from datasets import Dataset as HuggingFaceDataset

from agentlightning import LLM, Trainer, rollout
from agentlightning.algorithm.verl import VERL
from agentlightning.types import Dataset

calculator_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-calculator"])


@rollout
async def calc_agent(task: Any, llm: LLM) -> Any:
    async with McpWorkbench(calculator_mcp_server) as workbench:
        calc_agent = get_agent(
            llm.model,
            llm.endpoint,
            llm.sampling_parameters.get("temperature", 0.7),
            workbench,
        )
        try:
            output_format = "Output the answer when you are ready. The answer should be surrounded by three sharps (`###`), in the form of ### ANSWER: <answer> ###."
            prompt = task["question"] + " " + output_format
            result = await calc_agent.run(task=prompt)
            # evaluate
            answer = re.search(r"###\s*ANSWER:\s*(.+?)(\s*###|$)", result.messages[-1].content)  # type: ignore
            if answer:
                answer = answer.group(1)
            else:
                answer = result.messages[-1].content  # type: ignore
        except Exception as e:
            print("Failure:", str(e))
            answer = "None"
        reward = await eval(answer, str(task["result"]))  # reward is tracked with the decorator  # type: ignore
        print("answer: {} ground_truth: {} reward: {}".format(answer, task["result"], reward))  # type: ignore


def verl_algorithm():
    rl_training_config = {
        "agentlightning": {
            "port": 9999,
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_files": "data/train.parquet",
            "val_files": "data/test_mini.parquet",
            "train_batch_size": 32,
            "max_prompt_length": 4096,
            "max_response_length": 2048,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 4,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.6,
            },
            "actor": {
                "ppo_mini_batch_size": 32,
                "ppo_micro_batch_size_per_gpu": 4,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 8,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": "Qwen/Qwen2.5-0.5B-Instruct",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "AgentLightningCI",
            "experiment_name": "train_verl_v0_2",
            "nnodes": 1,
            "save_freq": 3,
            "test_freq": 3,
            "total_epochs": 1,
            "total_training_steps": 3,
        },
    }

    return VERL(rl_training_config)


def train_val_dataset() -> Tuple[Dataset[Any], Dataset[Any]]:
    train_dataset = HuggingFaceDataset.from_parquet("data/train.parquet").to_list()  # type: ignore
    val_dataset = HuggingFaceDataset.from_parquet("data/test_mini.parquet").to_list()  # type: ignore

    return train_dataset, val_dataset  # type: ignore


def main():
    train_dataset, val_dataset = train_val_dataset()
    print("First 5 rows of train dataset:")
    print(train_dataset[:5])  # type: ignore
    print("First 5 rows of val dataset:")
    print(val_dataset[:5])  # type: ignore

    trainer = Trainer(algorithm=verl_algorithm(), n_workers=4)
    trainer.fit(calc_agent, train_dataset, val_dataset=val_dataset)  # type: ignore


if __name__ == "__main__":
    main()
