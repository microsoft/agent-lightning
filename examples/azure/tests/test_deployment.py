# Copyright (c) Microsoft. All rights reserved.

from aoai_finetune import AzureOpenAIFinetune

from agentlightning import setup_logging

finetune_algo = AzureOpenAIFinetune(
    base_deployment_name="gpt-4.1-mini",
    finetuned_deployment_name="gpt-4.1-mini-ft",
    base_model_name="gpt-4.1-mini-2025-04-14",
    finetune_every_n_rollouts=24,
    data_filter_ratio=0.6,
)

setup_logging()


def test_deployment():
    finetune_algo._deploy_model(  # pyright: ignore[reportPrivateUsage]
        model_name="gpt-4.1-mini-2025-04-14.ft-071a9d9c59ec4d088d1a3e56707d7361-aoai_ft_1",
        deployment_name="gpt-4.1-mini-ft",
        version="1",
    )


def test_wait_for_deployment_ready():
    finetune_algo._wait_for_deployment_ready("gpt-4.1-mini-ft", "1")  # pyright: ignore[reportPrivateUsage]


def test_delete_deployment():
    finetune_algo._delete_deployment("gpt-4.1-mini-ft_v01")  # pyright: ignore[reportPrivateUsage]
