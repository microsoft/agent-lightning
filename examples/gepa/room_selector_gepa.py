# Copyright (c) Microsoft. All rights reserved.

"""GEPA prompt optimization for the room-booking agent with Azure OpenAI.

Usage::

    # Set environment variables (see .env.example) and authenticate:
    az login
    python room_selector_gepa.py
"""

import logging
import os
from typing import Tuple, cast

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from room_selector import (
    GRADER_DEPLOYMENT_NAME,
    RoomSelectionTask,
    load_room_tasks,
    prompt_template_baseline,
    room_selector,
)

from agentlightning import Trainer, setup_logging
from agentlightning.algorithm.gepa import GEPA, GEPAConfig
from agentlightning.types import Dataset

logger = logging.getLogger(__name__)


def load_train_val_dataset() -> Tuple[Dataset[RoomSelectionTask], Dataset[RoomSelectionTask]]:
    dataset_full = load_room_tasks()
    train_split = len(dataset_full) // 2
    dataset_train = [dataset_full[i] for i in range(train_split)]
    dataset_val = [dataset_full[i] for i in range(train_split, len(dataset_full))]
    return cast(Dataset[RoomSelectionTask], dataset_train), cast(Dataset[RoomSelectionTask], dataset_val)


def setup_gepa_logger(file_path: str = "gepa.log") -> None:
    """Dump a copy of all the logs produced by the GEPA algorithm to a file."""

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Process-%(process)d %(name)s)   %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger("agentlightning.algorithm.gepa").addHandler(file_handler)


def main() -> None:
    setup_logging()
    setup_gepa_logger()

    # Build the reflection model identifier in litellm's Azure format.
    # LiteLLM reads AZURE_API_BASE and AZURE_API_VERSION from the environment
    # for Azure-prefixed model strings.
    reflection_model = f"azure/{os.environ.get('AZURE_OPENAI_GRADER_DEPLOYMENT', GRADER_DEPLOYMENT_NAME)}"

    # Provide an Entra ID token provider so litellm can authenticate to Azure.
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    algo = GEPA(
        config=GEPAConfig(
            max_metric_calls=200,
            reflection_minibatch_size=8,
            candidate_selection_strategy="pareto",
            reflection_model=reflection_model,
            reflection_model_kwargs={"azure_ad_token_provider": token_provider},
            seed=42,
            display_progress_bar=True,
        ),
    )
    trainer = Trainer(
        algorithm=algo,
        n_runners=8,
        initial_resources={
            "prompt_template": prompt_template_baseline(),
        },
    )
    dataset_train, dataset_val = load_train_val_dataset()
    trainer.fit(agent=room_selector, train_dataset=dataset_train, val_dataset=dataset_val)

    best_prompt = algo.get_best_prompt()
    logger.info("Best prompt found:\n%s", best_prompt.template)
    print(f"\nBest prompt found:\n{best_prompt.template}")


if __name__ == "__main__":
    main()
