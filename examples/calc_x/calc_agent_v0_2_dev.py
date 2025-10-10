# Copyright (c) Microsoft. All rights reserved.

import os

from calc_agent_v0_2 import calc_agent, train_val_dataset

from agentlightning import LLM, Trainer


def main():
    train_dataset, val_dataset = train_val_dataset()

    trainer = Trainer(
        n_workers=4,
        initial_resources={
            "main_llm": LLM(
                endpoint=os.environ["OPENAI_API_BASE"],
                model="gpt-4.1-nano",
                sampling_parameters={"temperature": 0.7},
            )
        },
    )
    trainer.dev(calc_agent, train_dataset[:5], val_dataset=val_dataset[:5])  # type: ignore


if __name__ == "__main__":
    main()
