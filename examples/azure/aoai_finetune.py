# Copyright (c) Microsoft. All rights reserved.

"""The Azure OpenAI fine-tuning algorithm implementation."""

import asyncio
import copy
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from openai import OpenAI

from agentlightning.adapter.messages import OpenAIMessages, TraceToMessages
from agentlightning.algorithm import Algorithm
from agentlightning.algorithm.utils import batch_iter_over_dataset
from agentlightning.reward import find_final_reward
from agentlightning.types import LLM, RolloutMode, TaskInput

logger = logging.getLogger("agentlightning.aoai")

ROLLOUT_IDLE_SLEEP_SECONDS = 5.0
FILE_STATUS_POLL_INTERVAL = 10
FINETUNE_JOB_POLL_INTERVAL = 60


class AzureOpenAIFinetune(Algorithm):
    """Coordinate iterative fine-tuning runs for an Azure OpenAI deployment.

    The algorithm batches rollouts, extracts the recorded traces, converts them into JSONL records
    that comply with Azure OpenAI fine-tuning, and optionally redeploys the resulting checkpoint so
    subsequent rollouts evaluate the newest model revision.
    """

    def __init__(
        self,
        base_deployment_name: str,
        finetuned_deployment_name: str,
        base_model_name: str,
        *,
        finetune_every_n_rollouts: int = 32,
        azure_openai_endpoint: Optional[str] = None,
        azure_openai_api_key: Optional[str] = None,
        azure_openai_api_version: Optional[str] = None,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
        resource_name: Optional[str] = None,
        seed: int = 42,
        n_iterations: int = 3,
        finetune_epochs: int = 1,
        finetune_batch_size: int = 2,
        finetune_learning_rate: float = 1.0,
        max_deployments: int = 2,
        data_filter_ratio: float = 0.5,
    ) -> None:
        """Create a fine-tuning workflow tied to an Azure OpenAI endpoint.

        Args:
            base_deployment_name: Deployment used as the base model for the first fine-tuning job.
            deployment_name: Deployment that should serve the fine-tuned weights after each round.
                Currently, this name is only used as a prefix for the actual deployment created after
                each fine-tuning job, because multiple versions cannot be assigned to the same deployment.
            base_model_name: On Azure, deployments are instantiated from base models
                (e.g., "gpt-4.1-mini" deployment is created from "gpt-4.1-mini-2025-04-14").
                This name is used to identify the latter name when launching fine-tuning jobs.
            finetune_every_n_rollouts: Number of rollouts grouped together before launching a job.
                We don't recommend setting this value too low as fine-tuning jobs have a minimum rows requirement.
            azure_openai_endpoint: Azure OpenAI endpoint (e.g. `https://{resource}.openai.azure.com`).
            azure_openai_api_key: API key with access to the Azure OpenAI resource.
            azure_openai_api_version: API version to use when talking to Azure OpenAI.
            subscription_id: Azure subscription that owns the OpenAI resource (used for deployment).
            resource_group: Resource group of the target Azure OpenAI resource.
            resource_name: Azure OpenAI resource name, usually the Azure OpenAI resource name.
            seed: Random seed forwarded to the fine-tuning job for reproducibility.
            n_iterations: Number of algorithm iterations (fine-tune → deploy → evaluate).
            finetune_epochs: Number of epochs per fine-tuning job (not the number of epochs to go through `train_dataset`).
            finetune_batch_size: Batch size to use for the fine-tuning job.
            finetune_learning_rate: Learning rate to use for the fine-tuning job.
            max_deployments: Maximum number of deployments to keep active; older ones are deleted.
                Use this to avoid hitting the capacity limit on Azure service.
            data_filter_ratio: Fraction of high-reward examples to keep when preparing JSONL data.
        """
        super().__init__()

        self.azure_openai_endpoint = azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        if not self.azure_openai_endpoint:
            raise ValueError("Azure OpenAI endpoint must be provided via parameter or AZURE_OPENAI_ENDPOINT env var")

        self.azure_openai_api_key = azure_openai_api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        if not self.azure_openai_api_key:
            raise ValueError("Azure OpenAI API key must be provided via parameter or AZURE_OPENAI_API_KEY env var")

        self.azure_openai_api_version = azure_openai_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "")
        if not self.azure_openai_api_version:
            raise ValueError(
                "Azure OpenAI API version must be provided via parameter or AZURE_OPENAI_API_VERSION env var"
            )

        self.subscription_id = subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID", "")
        if not self.subscription_id:
            raise ValueError("Azure subscription ID must be provided via parameter or AZURE_SUBSCRIPTION_ID env var")

        self.resource_group = resource_group or os.getenv("AZURE_RESOURCE_GROUP", "")
        if not self.resource_group:
            raise ValueError("Azure resource group must be provided via parameter or AZURE_RESOURCE_GROUP env var")

        self.resource_name = resource_name or os.getenv("AZURE_RESOURCE_NAME", "")
        if not self.resource_name:
            raise ValueError("Azure resource name must be provided via parameter or AZURE_RESOURCE_NAME env var")

        self.base_deployment_name = base_deployment_name
        self.finetuned_deployment_name = finetuned_deployment_name
        self.base_model_name = base_model_name

        self.finetune_every_n_rollouts = finetune_every_n_rollouts
        self.seed = seed
        self.n_iterations = n_iterations
        self.finetune_epochs = finetune_epochs
        self.finetune_batch_size = finetune_batch_size
        self.finetune_learning_rate = finetune_learning_rate
        self.max_deployments = max_deployments
        self.data_filter_ratio = data_filter_ratio

        self.openai_client = OpenAI(
            api_key=self.azure_openai_api_key,
            base_url=self.azure_openai_endpoint,
        )

        # Tracks the deployments created. They can be deleted later if needed.
        self._created_deployments: List[str] = []
        self._log_prefix: str = ""

    async def run(  # type: ignore
        self,
        train_dataset: Optional[List[TaskInput]] = None,
        val_dataset: Optional[List[TaskInput]] = None,
    ) -> None:
        """
        Run the training loop.

        Args:
            train_dataset: Optional training dataset
            val_dataset: Optional validation dataset
        """
        if train_dataset is None or val_dataset is None:
            raise ValueError("Both train_dataset and val_dataset must be provided")

        resources: LLM = LLM(endpoint=self.azure_openai_endpoint, model=self.base_deployment_name)
        store = self.get_store()

        # This tracks the model name used in training
        # It's different from the deployment name which used for inference
        training_model_name: str = self.base_model_name

        data_iterator = batch_iter_over_dataset(train_dataset, self.finetune_every_n_rollouts)
        for i_iteration in range(self.n_iterations):
            self._log_prefix = f"[AOAI FT {i_iteration + 1}/{self.n_iterations}] "
            # (1) Fetch the next batch of tasks to process
            tasks = next(data_iterator)
            self._log_info(f"[Stage 1] Starting fine-tuning iteration with {len(tasks)} tasks...")

            # (2) Update the current active LLM deployment address
            await store.add_resources({"main_llm": resources})
            self._log_info(f"[Stage 2] Using model deployment: {resources.model}")

            # (3) Spawn and wait for the rollouts to complete
            messages_group, reward_group = await self.batch_rollout_and_collect_data(tasks, "train")
            self._log_info(f"[Stage 3] Completed rollouts for {len(tasks)} tasks.")

            # (4) Filter the data based on rewards
            training_data = await self.prepare_data_for_training(messages_group, reward_group, "train")
            self._log_info(f"[Stage 4] Prepared {len(training_data)} training examples after filtering.")

            # (5) Perform fine-tuning
            self._log_info(f"[Stage 5] Starting fine-tuning for model {training_model_name}...")
            training_model_name = self.finetune(training_data, training_model_name, i_iteration)
            self._log_info(f"[Stage 5] Fine-tuning completed. Updated training model base name: {training_model_name}")

            # (6) Deploy the fine-tuned model
            self._log_info(f"[Stage 6] Deploying fine-tuned model...")
            resources = self.deploy_finetuned_model(training_model_name, i_iteration + 1)
            self._log_info(f"[Stage 6] Deployment completed. Updated resources to: {resources}")

            # (7) Evaluate on validation dataset
            self._log_info(f"[Stage 7] Evaluating on validation dataset...")
            _, val_reward_group = await self.batch_rollout_and_collect_data(val_dataset, "val")
            self._log_info(
                f"[Stage 7] Evaluation completed. Average reward: {sum(val_reward_group) / len(val_reward_group):.4f}"
            )

    async def batch_rollout_and_collect_data(
        self,
        tasks: Sequence[TaskInput],
        rollout_mode: RolloutMode = "train",
    ) -> Tuple[List[OpenAIMessages], List[float]]:
        """Launch rollouts for a batch of tasks and aggregate their traces.

        Each task is executed concurrently and the resulting spans are converted into OpenAI-style
        chat messages. Rewards from the traces are preserved so downstream filtering can prefer the
        highest quality examples.

        Args:
            tasks: Rollout payloads collected from the dataset.
            rollout_mode: Semantic label that differentiates training from validation passes.

        Returns:
            Tuple containing the flattened list of OpenAI messages and the aligned list of rewards.
        """
        if not tasks:
            return [], []

        results = await asyncio.gather(*(self.rollout_and_collect_data(task, mode=rollout_mode) for task in tasks))

        messages_group: List[OpenAIMessages] = []
        reward_group: List[float] = []

        for messages_list, reward in results:
            if not messages_list:
                continue
            messages_group.extend(messages_list)
            # Duplicate the reward for each message set produced by the rollout
            reward_group.extend([reward] * len(messages_list))

        return messages_group, reward_group

    async def rollout_and_collect_data(self, task: TaskInput, mode: RolloutMode) -> Tuple[List[OpenAIMessages], float]:
        """Execute a single rollout, returning OpenAI messages together with the final reward.

        The method waits for the rollout to enter a terminal state, retrieves the recorded spans,
        converts them into OpenAI chat messages using the configured trace adapter, and extracts the
        reward emitted by the runner.

        Args:
            task: Rollout payload to enqueue in the store.
            mode: Execution mode to annotate the rollout (`"train"`, `"val"` or `"test"`).

        Returns:
            A tuple containing the list of OpenAI messages reconstructed from the trace and the
            numeric reward associated with the rollout. Rewards default to `0.0` when not found.
        """
        store = self.get_store()
        rollout = await store.enqueue_rollout(input=task, mode=mode)
        rollout_id = rollout.rollout_id

        self._log_debug("Waiting for rollout %s to finish in mode=%s", rollout_id, mode)

        while True:
            completed = await store.wait_for_rollouts(rollout_ids=[rollout_id], timeout=0.0)
            if completed:
                finished = completed[0]
                if finished.status != "succeeded":
                    self._log_error(f"Rollout {rollout_id} finished with status {finished.status}. Skipping.")
                break
            await asyncio.sleep(ROLLOUT_IDLE_SLEEP_SECONDS)

        spans = await store.query_spans(rollout_id=rollout_id, attempt_id="latest")

        try:
            adapter = self.get_adapter()
        except ValueError:
            adapter = TraceToMessages()
            self.set_adapter(adapter)
        if not isinstance(adapter, TraceToMessages):
            raise RuntimeError(
                "The adapter is configured but not a TraceToMessages adapter. "
                "AzureOpenAIFinetune requires a TraceToMessages adapter. Please set that in Trainer."
            )

        messages_list = adapter.adapt(spans)
        if not messages_list:
            self._log_error(f"Rollout {rollout_id} produced no OpenAI messages for training.")

        # NOTE: Patch the messages list for AOAI requirements
        # This should ideally be merged into message adapter
        for messages in messages_list:
            for message in messages["messages"]:
                if "content" in message and message["content"] is None:
                    message.pop("content")

        reward = find_final_reward(spans)
        if reward is None:
            self._log_error(f"Rollout {rollout_id} produced no reward; defaulting to 0.0.")
            reward = 0.0

        self._log_info("Rollout %s produced %d message set(s) with reward %.3f", rollout_id, len(messages_list), reward)

        return messages_list, reward

    async def prepare_data_for_training(
        self,
        messages_group: List[OpenAIMessages],
        reward_group: List[float],
        split: RolloutMode,
    ) -> List[Dict[str, Any]]:
        """Combine rollouts and rewards into JSONL training payloads.

        Args:
            messages_group: Flattened list of OpenAI message dictionaries.
            reward_group: Rewards aligned with `messages_group` entries.
            split: Dataset split that produced the examples (e.g., `"train"` or `"val"`).

        Returns:
            JSON-serializable dictionaries ready to be written into a fine-tuning file.
        """
        if len(messages_group) != len(reward_group):
            raise ValueError("Mismatch between number of message entries and reward entries.")

        tagged_examples: List[Dict[str, Any]] = []
        for idx, (messages, reward) in enumerate(zip(messages_group, reward_group)):
            example: Dict[str, Any] = {
                "messages": messages["messages"],
                "metadata": {"split": split, "rollout_index": idx},
                "reward": reward,
                "reward_jitter": random.uniform(0, 1),
            }
            if messages.get("tools"):
                example["tools"] = messages["tools"]
            tagged_examples.append(example)

        self._log_info(
            "Collected %d candidate example(s) for split=%s before filtering (ratio=%.2f).",
            len(tagged_examples),
            split,
            self.data_filter_ratio,
        )

        filtered_examples = self._filter_training_data(tagged_examples)

        self._log_info("Keeping %d example(s) for fine-tuning after reward-based filtering.", len(filtered_examples))

        return filtered_examples

    def finetune(self, training_data: List[Dict[str, Any]], base_model: str, iteration_idx: int) -> str:
        """Launch a fine-tuning job on Azure OpenAI using the supplied dataset.

        Args:
            training_data: JSONL-ready records that describe the conversation transcripts.
            iteration_idx: Current iteration index.

        Returns:
            Identifier of the fine-tuned model produced by Azure OpenAI.
        """
        if not training_data:
            raise ValueError("Training data must not be empty before launching fine-tuning.")
        if not self.openai_client:
            raise RuntimeError("Azure OpenAI client is not initialized; cannot fine-tune.")

        next_iteration = iteration_idx + 1

        train_file_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", prefix=f"{base_model}_{iteration_idx:02d}_", suffix=".jsonl", delete=False
            ) as handle:
                for record in training_data:
                    handle.write(json.dumps(record) + "\n")
                train_file_path = handle.name

            self._log_info(
                "Prepared temporary training file %s with %d example(s).", train_file_path, len(training_data)
            )

            with open(train_file_path, "rb") as file_handle:
                training_response = self.openai_client.files.create(file=file_handle, purpose="fine-tune")
            train_file_id = training_response.id
            self._log_info("Uploaded training file to Azure OpenAI (file_id=%s).", train_file_id)

            self._wait_for_file_processed(train_file_id)

            job = self.openai_client.fine_tuning.jobs.create(
                training_file=train_file_id,
                model=base_model,
                seed=self.seed,
                method={
                    "type": "supervised",
                    "supervised": {
                        "hyperparameters": {
                            "batch_size": self.finetune_batch_size,
                            "learning_rate_multiplier": self.finetune_learning_rate,
                            "n_epochs": self.finetune_epochs,
                        }
                    },
                },
                # TODO: continuously adding suffix will make model names very long after a few iterations
                # investigate if we can just specify the fine-tuned model name directly
                suffix=f"v{next_iteration:02d}",
                # NOTE: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning
                # Other options are "GlobalStandard" and "Standard"
                extra_body={"trainingType": "GlobalStandard"},
            )
            job_id = job.id
            self._log_info("Fine-tuning job %s created for base model %s.", job_id, base_model)

            fine_tuned_model = self._wait_for_finetuning(job_id)
            if not fine_tuned_model:
                raise RuntimeError(f"Fine-tuning job {job_id} finished without producing a model id.")

            self._log_info("Fine-tuning job %s succeeded with new model id %s.", job_id, fine_tuned_model)

            return fine_tuned_model

        finally:
            if train_file_path and os.path.exists(train_file_path):
                try:
                    os.unlink(train_file_path)
                except OSError:
                    self._log_warning("Failed to remove temporary training file %s.", train_file_path)

    def deploy_finetuned_model(self, finetuned_model_id: str, iteration_idx: int) -> LLM:
        """Deploy the fine-tuned checkpoint and return an `LLM` resource descriptor.

        Args:
            finetuned_model_id: Identifier returned by the fine-tuning job.
            iteration_idx: Current iteration index.

        Returns:
            `LLM` resource pointing to either the Azure deployment or the direct model id.
        """
        if not finetuned_model_id:
            raise ValueError("finetuned_model_id must be a non-empty string.")

        while len(self._created_deployments) >= self.max_deployments:
            self._log_warning(
                "Maximum number of deployments reached (%d). Cleaning up old deployments.", self.max_deployments
            )
            oldest_deployment = self._created_deployments.pop(0)
            self._log_info("Deleting old deployment %s.", oldest_deployment)
            self._delete_deployment(oldest_deployment)

        if self.subscription_id and self.resource_group and self.resource_name:
            # version should be like this: str(iteration_idx)
            # Because of this issue: {"code":"ModelUpgradeNotSupported","message":"Model updates are not supported for finetuned model deployments."}
            # We need to concatenate the version to the model name
            # and version is always "1"
            deployment_name = f"{self.finetuned_deployment_name}_v{iteration_idx:02d}"
            self._deploy_model(finetuned_model_id, deployment_name, "1")
            self._wait_for_deployment_ready(deployment_name, "1")
            self._created_deployments.append(deployment_name)
            self._log_info(
                "Deployed fine-tuned model %s to deployment %s. We now have %d active deployments.",
                finetuned_model_id,
                deployment_name,
                len(self._created_deployments),
            )
        else:
            raise RuntimeError("Azure deployment parameters missing; using fine-tuned model id directly.")

        return LLM(endpoint=self.azure_openai_endpoint, model=deployment_name, api_key=self.azure_openai_api_key)

    def cleanup_deployments(self) -> None:
        """Delete all deployments created by this algorithm instance."""
        for deployment_name in self._created_deployments:
            self._log_info("Cleaning up deployment %s.", deployment_name)
            self._delete_deployment(deployment_name)
        self._created_deployments = []

    def _filter_training_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select the top-performing examples and strip reward metadata.

        Args:
            data: Candidate training examples carrying a temporary `reward` key.

        Returns:
            List of examples suitable for JSONL serialization (without the `reward` field).
        """
        if not data:
            return []

        if self.data_filter_ratio >= 1.0:
            selected = data
        else:
            sorted_data = sorted(data, key=lambda x: (x.get("reward", 0.0), x.get("reward_jitter", 0.0)), reverse=True)
            keep_count = max(1, int(len(sorted_data) * self.data_filter_ratio))
            selected = sorted_data[:keep_count]

        self._log_debug("Filtering kept %d/%d example(s).", len(selected), len(data))

        filtered: List[Dict[str, Any]] = []
        for entry in selected:
            entry_copy = copy.deepcopy(entry)
            entry_copy.pop("reward", None)
            entry_copy.pop("reward_jitter", None)
            entry_copy.pop("metadata", None)
            filtered.append(entry_copy)

        return filtered

    def _wait_for_file_processed(self, file_id: str, interval: int = FILE_STATUS_POLL_INTERVAL) -> None:
        """Poll the uploaded training file until Azure marks it as processed.

        Args:
            file_id: Identifier returned by `files.create`.
            interval: Number of seconds to wait between polling attempts.
        """
        self._log_info("Waiting for training file %s to reach the processed state.", file_id)
        while True:
            file_info = self.openai_client.files.retrieve(file_id)
            status = getattr(file_info, "status", None)
            self._log_debug("Training file %s status: %s", file_id, status)

            if status == "processed":
                return
            if status == "failed":
                raise RuntimeError(f"Azure OpenAI reported a failure while processing file {file_id}.")

            time.sleep(interval)

    def _wait_for_finetuning(self, job_id: str, interval: int = FINETUNE_JOB_POLL_INTERVAL) -> str:
        """Poll the fine-tuning job until a terminal status is reached.

        Args:
            job_id: Identifier of the fine-tuning job to monitor.
            interval: Number of seconds between polling attempts.

        Returns:
            The identifier of the fine-tuned model when successful.
            Otherwise, raise an exception.
        """

        self._log_info("Waiting for fine-tuning job %s to complete.", job_id)

        while True:
            job = self.openai_client.fine_tuning.jobs.retrieve(job_id)
            self._log_debug("Fine-tuning job %s status: %s", job_id, job.status)

            if job.status == "succeeded":
                if job.fine_tuned_model:
                    return job.fine_tuned_model
                else:
                    raise RuntimeError(f"Fine-tuning job {job_id} succeeded but no model id was returned: {job}")
            if job.status in {"failed", "cancelled"}:
                raise RuntimeError(f"Fine-tuning job {job_id} ended with status {job.status}.")

            time.sleep(interval)

    def _deploy_model(self, model_name: str, deployment_name: str, version: str) -> None:
        """Deploy the fine-tuned model using Azure's control plane REST API.

        Args:
            model_name: Fine-tuned (training) model identifier returned by Azure OpenAI.
            deployment_name: Name of the deployment to update.
            version: Version string to stamp on the deployment update.
        """
        token = self._get_azure_token()

        request_url = (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.CognitiveServices/accounts/{self.resource_name}"
            f"/deployments/{deployment_name}"
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Follows the setup in https://github.com/azure-ai-foundry/fine-tuning/blob/047fd230a77e327e75d4bc41403ee8e7bff4de9e/Demos/DistillingSarcasm/sarcasm.ipynb
        deploy_data = {
            "sku": {"name": "DeveloperTier", "capacity": 250},
            "properties": {
                "model": {
                    "format": "OpenAI",
                    "name": model_name,
                    "version": version,
                }
            },
        }

        self._log_info("Deploying model %s (version %s) to deployment %s.", model_name, version, deployment_name)

        response = requests.put(
            request_url,
            params={"api-version": "2025-06-01"},
            headers=headers,
            data=json.dumps(deploy_data),
            timeout=180,
        )

        if response.status_code < 400:
            self._log_info("Deployment %s updated successfully.", deployment_name)
        else:
            self._log_error("Deployment failed: %s %s", response.status_code, response.text)

    def _wait_for_deployment_ready(self, deployment_name: str, version: str, interval: int = 30) -> None:
        """Poll the deployment status until it is marked as ready.

        Args:
            deployment_name: Name of the deployment to monitor.
            interval: Number of seconds between polling attempts.
        """
        self._log_info("Waiting for deployment %s to become ready.", deployment_name)
        while True:
            request_url = (
                f"https://management.azure.com/subscriptions/{self.subscription_id}"
                f"/resourceGroups/{self.resource_group}"
                f"/providers/Microsoft.CognitiveServices/accounts/{self.resource_name}"
                f"/deployments/{deployment_name}"
            )

            token = self._get_azure_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

            response = requests.get(
                request_url,
                params={"api-version": "2025-06-01"},
                headers=headers,
                timeout=60,
            )

            if response.status_code >= 400:
                self._log_error(
                    "Failed to query deployment status. Retry later: %s, %s", response.status_code, response.text
                )
            else:
                deployment_info = response.json()
                properties = deployment_info.get("properties", {})
                model_info = properties.get("model", {})
                provisioning_state = properties.get("provisioningState")
                self._log_info(
                    "Waiting for deployment to be ready. Current provisioning state of %s: %s",
                    deployment_name,
                    provisioning_state,
                )

                if provisioning_state == "Succeeded":
                    version_found = model_info.get("version")
                    if version_found == version:
                        self._log_info("Deployment %s is ready with version %s.", deployment_name, version)
                        return
                    else:
                        self._log_warning(
                            "Deployment succeeded, but version mismatch: expected %s, got %s. Try again later.",
                            version,
                            version_found,
                        )
                elif provisioning_state == "Cancelled" or provisioning_state == "Failed":
                    raise RuntimeError(f"Deployment {deployment_name} failed with state {provisioning_state}.")
                else:
                    # Just wait and poll again
                    self._log_debug(
                        "Deployment %s not ready yet. Current state: %s", deployment_name, provisioning_state
                    )

            time.sleep(interval)

    def _delete_deployment(self, deployment_name: str) -> None:
        """Delete a specific deployment in Azure OpenAI.

        Args:
            deployment_name: Name of the deployment to delete.
        """
        token = self._get_azure_token()
        request_url = (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.CognitiveServices/accounts/{self.resource_name}"
            f"/deployments/{deployment_name}"
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        self._log_info("Deleting deployment %s...", deployment_name)

        response = requests.delete(
            request_url,
            params={"api-version": "2025-06-01"},
            headers=headers,
            timeout=60,
        )

        if response.status_code in (200, 202, 204):
            self._log_info("Deployment %s deleted successfully.", deployment_name)
        else:
            self._log_error(
                "Failed to delete deployment %s: %s %s",
                deployment_name,
                response.status_code,
                response.text,
            )

    def _get_azure_token(self) -> str:
        """Request an Azure management token via the Azure CLI.

        Returns:
            Bearer token that authorizes calls to the Azure management plane.
        """
        cmd = [
            "az",
            "account",
            "get-access-token",
            "--resource",
            "https://management.azure.com",
            "--query",
            "accessToken",
            "-o",
            "tsv",
        ]
        try:
            token = subprocess.check_output(cmd, text=True).strip()
        except subprocess.CalledProcessError as exc:
            raise ValueError("Azure CLI command failed. Could not fetch token from Azure CLI.") from exc
        if token:
            return token
        else:
            raise ValueError("Could not fetch token from Azure CLI.")

    # Logging helpers

    def _log_info(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.info(f"{self._log_prefix}{message}", *args, **kwargs)

    def _log_debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.debug(f"{self._log_prefix}{message}", *args, **kwargs)

    def _log_warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.warning(f"{self._log_prefix}{message}", *args, **kwargs)

    def _log_error(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.error(f"{self._log_prefix}{message}", *args, **kwargs)
