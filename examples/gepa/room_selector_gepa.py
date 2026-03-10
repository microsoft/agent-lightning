# Copyright (c) Microsoft. All rights reserved.

"""GEPA prompt optimization for the room-booking agent.

Supports Azure OpenAI (Entra ID or API key) and plain OpenAI. The backend is
selected via ``--provider`` or the ``LLM_PROVIDER`` env var — see
``llm_backend.py`` for details.

Usage::

    # Azure Entra ID (default):
    az login
    python room_selector_gepa.py

    # Azure API key:
    python room_selector_gepa.py --provider azure_key

    # OpenAI:
    python room_selector_gepa.py --provider openai

    # With W&B experiment tracking:
    python room_selector_gepa.py --wandb --wandb-project gepa-room-selector
"""

import argparse
import logging
from typing import Tuple, cast

from llm_backend import VALID_PROVIDERS, build_reflection_config, get_provider
from room_selector import (
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
    parser = argparse.ArgumentParser(description="GEPA prompt optimization for the room-booking agent")
    parser.add_argument(
        "--provider",
        type=str,
        choices=VALID_PROVIDERS,
        default=None,
        help="LLM backend (default: LLM_PROVIDER env var or azure_entra)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B experiment tracking")
    parser.add_argument("--wandb-project", type=str, default="gepa-room-selector", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    args = parser.parse_args()

    setup_logging()
    setup_gepa_logger()

    provider = get_provider(args.provider)
    reflection_model, reflection_model_kwargs = build_reflection_config(provider)
    logger.info("Using LLM provider: %s (reflection model: %s)", provider, reflection_model)

    wandb_init_kwargs = {"project": args.wandb_project}
    if args.wandb_name:
        wandb_init_kwargs["name"] = args.wandb_name

    algo = GEPA(
        config=GEPAConfig(
            max_metric_calls=250,
            reflection_minibatch_size=8,
            candidate_selection_strategy="pareto",
            reflection_model=reflection_model,
            reflection_model_kwargs=reflection_model_kwargs,
            seed=42,
            display_progress_bar=True,
            use_wandb=args.wandb,
            wandb_init_kwargs=wandb_init_kwargs,
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
