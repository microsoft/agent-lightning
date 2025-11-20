# Copyright (c) Microsoft. All rights reserved.

import pandas as pd
from aoai_finetune import AzureOpenAIFinetune
from capital_agent import capital_agent
from rich.console import Console

from agentlightning import TraceToMessages, Trainer, setup_logging

console = Console()


def main():
    setup_logging()
    finetune_algo = AzureOpenAIFinetune(
        base_deployment_name="gpt-4.1-mini",
        finetuned_deployment_name="gpt-4.1-mini-ft",
        base_model_name="gpt-4.1-mini-2025-04-14",
        finetune_every_n_rollouts=24,
        data_filter_ratio=0.6,
        n_iterations=2,
    )

    trainer = Trainer(n_runners=2, algorithm=finetune_algo, adapter=TraceToMessages())
    dataset = pd.read_csv("capital_samples.csv")  # type: ignore
    train_dataset = dataset.sample(frac=0.8, random_state=42)  # 80% for training  # type: ignore
    val_dataset = dataset.drop(train_dataset.index)  # Remaining 20% for validation  # type: ignore

    console.print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")  # type: ignore

    trainer.fit(
        capital_agent,
        train_dataset=train_dataset.to_dict(orient="records"),  # type: ignore
        val_dataset=val_dataset.to_dict(orient="records"),  # type: ignore
    )


if __name__ == "__main__":
    main()
