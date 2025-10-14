# Copyright (c) Microsoft. All rights reserved.

"""
APO with textual gradients that read rollout spans and outputs to modify the prompt.

- algo: beam search with span-aware textual gradients -> apply_edit via LLM
- rollout: same pattern as your example, but task is a dict (T_task)
"""

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Generic, Iterator, List, Optional, Sequence, Tuple, TypedDict, TypeVar, cast

import poml
from openai import AsyncOpenAI

from agentlightning.adapter.messages import TraceMessagesAdapter
from agentlightning.algorithm.base import BaseAlgorithm
from agentlightning.reward import find_final_reward
from agentlightning.types import Dataset, NamedResources, PromptTemplate, RolloutMode, RolloutStatus, RolloutV2

logger = logging.getLogger(__name__)

T_task = TypeVar("T_task", bound=Dict[str, Any])


class RolloutResultForAPO(TypedDict):
    """This must be all JSON serializable to be processable by POML."""

    status: RolloutStatus
    final_reward: Optional[float]
    spans: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]


GRADIENT_PROMPT_FILES = [
    Path(__file__).parent / "prompts" / "text_gradient_variant01.poml",
    Path(__file__).parent / "prompts" / "text_gradient_variant02.poml",
]

APPLY_EDIT_PROMPT_FILES = [
    Path(__file__).parent / "prompts" / "apply_edit_variant01.poml",
    Path(__file__).parent / "prompts" / "apply_edit_variant02.poml",
]


def batch_iter_over_dataset(dataset: Dataset[T_task], batch_size: int) -> Iterator[Sequence[T_task]]:
    """
    Create an infinite iterator that yields batches from the dataset.

    When batch_size >= dataset size, yields the entire shuffled dataset repeatedly.
    When batch_size < dataset size, yields batches of the specified size, reshuffling
    after each complete pass through the dataset.

    Args:
        dataset: The dataset to iterate over.
        batch_size: The desired batch size.

    Yields:
        Sequences of tasks from the dataset. Each task appears at most once per epoch.
    """
    if batch_size >= len(dataset):
        while True:
            dataset_copy = [dataset[i] for i in range(len(dataset))]
            random.shuffle(dataset_copy)
            yield dataset_copy

    else:
        current_batch: List[int] = []
        while True:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            for index in indices:
                if index in current_batch:
                    continue
                current_batch.append(index)
                if len(current_batch) == batch_size:
                    yield [dataset[index] for index in current_batch]
                    current_batch = []


class APO(BaseAlgorithm, Generic[T_task]):
    """Automatic Prompt Optimization (APO) algorithm using textual gradients and beam search.

    APO is an iterative prompt optimization algorithm that uses LLM-generated textual gradients
    to improve prompts through a beam search process. It evaluates prompts on rollouts,
    computes critiques based on the results, and applies edits to generate improved prompts.

    The algorithm operates in rounds, where each round:
    1. Samples parent prompts from the current beam
    2. Generates new prompts by computing textual gradients and applying edits
    3. Evaluates all candidates on a validation set
    4. Selects the top-k prompts for the next round

    Based on the ideas from:
    - ProTeGi: https://aclanthology.org/2023.emnlp-main.494.pdf
    - TextGrad: https://github.com/zou-group/textgrad
    """

    def __init__(
        self,
        async_openai_client: AsyncOpenAI,
        *,
        gradient_model: str = "gpt-5-mini",
        apply_edit_model: str = "gpt-4.1-mini",
        diversity_temperature: float = 1.0,
        gradient_batch_size: int = 4,
        val_batch_size: int = 16,
        beam_width: int = 4,
        branch_factor: int = 4,
        beam_rounds: int = 3,
        rollout_batch_timeout: float = 600.0,
        run_initial_validation: bool = True,
    ):
        """
        Initialize the APO algorithm with configuration parameters.

        Args:
            async_openai_client: AsyncOpenAI client for making LLM API calls.
            gradient_model: Model name for computing textual gradients (critiques).
            apply_edit_model: Model name for applying edits based on critiques.
            diversity_temperature: Temperature parameter for LLM calls to control diversity.
            gradient_batch_size: Number of rollout results to sample for gradient computation.
            val_batch_size: Number of validation examples to use for evaluation.
            beam_width: Number of top-scoring prompts to keep in the beam at each round.
            branch_factor: Number of new prompt candidates to generate from each parent prompt
                by applying textual gradient edits. This controls the expansion of the search tree.
            beam_rounds: Number of beam search rounds to perform.
            rollout_batch_timeout: Maximum time in seconds to wait for rollout batch completion.
            run_initial_validation: If True, runs validation on the seed prompt before starting
                optimization to establish a baseline score. Defaults to True.
        """
        self.async_openai_client = async_openai_client
        self.gradient_model = gradient_model
        self.apply_edit_model = apply_edit_model
        self.diversity_temperature = diversity_temperature
        self.gradient_batch_size = gradient_batch_size
        self.val_batch_size = val_batch_size
        self.beam_width = beam_width
        self.branch_factor = branch_factor
        self.beam_rounds = beam_rounds
        self.rollout_batch_timeout = rollout_batch_timeout
        self.run_initial_validation = run_initial_validation

        self._history_best_prompt: Optional[PromptTemplate] = None
        self._history_best_score: float = float("-inf")

    def get_seed_prompt_template(self) -> Tuple[str, PromptTemplate]:
        """
        Extract the initial prompt template from the algorithm's resources.

        Returns:
            A tuple of (resource_name, prompt_template) representing the seed prompt.

        Raises:
            ValueError: If initial_resources is not set or no PromptTemplate is found.
        """
        initial_resources = self.get_initial_resources()
        if initial_resources is None:
            raise ValueError(
                "initial_resources are not set for APO algorithm. "
                "Use algorithm.set_initial_resources() to set initial resources or set it in Trainer()"
            )
        for name, resource in initial_resources.items():
            if isinstance(resource, PromptTemplate):
                return name, resource
        raise ValueError("No prompt template resource found in initial_resources")

    def get_adapter(self) -> TraceMessagesAdapter:
        """
        Get the adapter for converting spans to messages.

        Returns:
            The TraceMessagesAdapter instance for this algorithm.

        Raises:
            ValueError: If the adapter is not a TraceMessagesAdapter.
        """
        adapter = super().get_adapter()
        if not isinstance(adapter, TraceMessagesAdapter):
            raise ValueError("Adapter must be a TraceMessagesAdapter for APO algorithm")
        return adapter

    def get_best_prompt(self) -> PromptTemplate:
        """
        Retrieve the best prompt discovered during optimization.

        Returns:
            The prompt template with the highest validation score found so far.

        Raises:
            ValueError: If no best prompt has been found yet (run() not called).
        """
        if self._history_best_prompt is None:
            raise ValueError("No best prompt found")
        return self._history_best_prompt

    async def compute_textual_gradient(
        self,
        current_prompt: str,
        rollout_results: List[RolloutResultForAPO],
    ) -> Optional[str]:
        """
        Compute a textual gradient (critique) for the current prompt based on rollout results.

        This method samples rollout results, sends them to an LLM along with the current prompt,
        and generates a critique describing how the prompt could be improved.

        Args:
            current_prompt: The prompt template to critique.
            rollout_results: List of rollout results containing spans, messages, and rewards.

        Returns:
            A textual critique generated by the LLM, or None if generation fails.
        """
        tg_template = random.choice(GRADIENT_PROMPT_FILES)

        if len(rollout_results) < self.gradient_batch_size:
            logger.warning(
                f"Only {len(rollout_results)} rollouts available, but {self.gradient_batch_size} are needed. Using all rollouts."
            )
            sampled_rollout_results = rollout_results
        else:
            sampled_rollout_results = random.sample(rollout_results, self.gradient_batch_size)

        logger.info(
            f"Gradient will be computed with {self.gradient_model} for {len(sampled_rollout_results)} rollouts with template: {tg_template}"
        )

        tg_msg = poml.poml(  # type: ignore
            tg_template,
            context={
                "experiments": sampled_rollout_results,
                "prompt_template": current_prompt,
            },
            format="openai_chat",
        )
        logger.debug(f"Gradient computed with {self.gradient_model} prompt: {tg_msg}")
        critique_response = await self.async_openai_client.chat.completions.create(
            model=self.gradient_model,
            messages=tg_msg["messages"],  # type: ignore
            temperature=self.diversity_temperature,
        )
        critique_text = critique_response.choices[0].message.content
        logger.debug(f"Gradient computed with {self.gradient_model} has result: {critique_text}")

        return critique_text

    async def textual_gradient_and_apply_edit(
        self,
        current_prompt: str,
        rollout: List[RolloutResultForAPO],
    ) -> Optional[str]:
        """
        Generate an improved prompt by computing a textual gradient and applying an edit.

        This is the main optimization step that:
        1. Computes a critique (textual gradient) based on rollout performance
        2. Uses another LLM to apply the critique and generate an improved prompt

        Args:
            current_prompt: The current prompt template to improve.
            rollout: List of rollout results to base the critique on.

        Returns:
            The improved prompt text, or the original prompt if gradient computation fails.
        """
        # 1) Critique
        critique_text = await self.compute_textual_gradient(current_prompt, rollout)
        if not critique_text:
            logger.error(f"Failed to compute critique for prompt.")
            return current_prompt

        # 2) Apply edit
        ae_template = random.choice(APPLY_EDIT_PROMPT_FILES)
        logger.info(f"Edit will be generated by {self.apply_edit_model} with template: {ae_template}")
        ae_msg = poml.poml(  # type: ignore
            ae_template,
            context={
                "prompt_template": current_prompt,
                "critique": critique_text,
            },
            format="openai_chat",
        )

        ae_response = await self.async_openai_client.chat.completions.create(
            model=self.apply_edit_model,
            messages=ae_msg["messages"],  # type: ignore
            temperature=self.diversity_temperature,
        )
        new_prompt = ae_response.choices[0].message.content
        if new_prompt:
            logger.info(f"Edit generated by {self.apply_edit_model}: {new_prompt}")
        return new_prompt

    async def get_rollout_results(self, rollout: List[RolloutV2]) -> List[RolloutResultForAPO]:
        """
        Convert completed rollouts to APO-compatible result format.

        Fetches spans for each rollout, adapts them to messages, and packages them
        with rewards and status information for gradient computation.

        Args:
            rollout: List of completed rollout metadata.

        Returns:
            List of rollout results formatted for APO processing.
        """
        rollout_results: List[RolloutResultForAPO] = []
        store = self.get_store()
        adapter = self.get_adapter()
        for r in rollout:
            spans = await store.query_spans(r.rollout_id)
            messages = adapter.adapt(spans)
            rollout_result = RolloutResultForAPO(
                status=r.status,
                final_reward=find_final_reward(spans),
                spans=[span.model_dump() for span in spans],
                messages=[m.model_dump() for m in messages],
            )
            logger.info(
                f"Rollout result for {r.rollout_id}: status {rollout_result['status']} "
                f"with final reward {rollout_result['final_reward']}. "
                f"{len(rollout_result['spans'])} spans and {len(rollout_result['messages'])} messages."
            )
            rollout_results.append(rollout_result)
        return rollout_results

    async def evaluate_prompt_on_batch(
        self,
        prompt: str,
        resource_name: str,
        dataset: Sequence[T_task],
        mode: RolloutMode,
    ) -> Tuple[List[RolloutResultForAPO], float]:
        """
        Evaluate a prompt on a batch of tasks by running rollouts and computing average reward.

        This method:
        1. Adds the prompt as a named resource to the store
        2. Enqueues rollouts for each task in the dataset
        3. Waits for rollouts to complete (with timeout)
        4. Computes and returns the average reward

        Args:
            prompt: The prompt template string to evaluate.
            resource_name: The name to register the prompt under in the store.
            dataset: Sequence of tasks to evaluate the prompt on.
            mode: Rollout mode ("train" or "val") for logging/tracking.

        Returns:
            A tuple of (rollout_results, average_reward) where rollout_results contains
            detailed information for each rollout and average_reward is the mean final reward.
        """
        store = self.get_store()
        logger.info(f'Evaluating prompt "{prompt[:50]}..." on {len(dataset)} tasks in {mode} mode')

        # Install prompt as named resource
        resources: NamedResources = {resource_name: PromptTemplate(template=prompt, engine="f-string")}
        await store.add_resources(resources)

        rollout_ids: List[str] = []
        for t in dataset:
            r = await store.enqueue_rollout(input=t, mode=mode)  # task can be any dict processable by the client
            rollout_ids.append(r.rollout_id)

        deadline = time.time() + self.rollout_batch_timeout
        finished: List[RolloutV2] = []
        while time.time() < deadline:
            finished = await store.wait_for_rollouts(rollout_ids=rollout_ids, timeout=0.0)
            if len(finished) >= len(rollout_ids):
                logger.info(f"All {len(rollout_ids)} rollouts finished within timeout.")
                break

        rollout_results = await self.get_rollout_results(finished)
        final_rewards = [rr["final_reward"] for rr in rollout_results]

        avg = float(sum([r or 0.0 for r in final_rewards]) / max(1, len(final_rewards)))

        logger.info(f"Evaluated {len(rollout_results)} rollouts. Rewards: {final_rewards}. Average reward: {avg}")
        return rollout_results, avg

    def _initialize_beam(
        self,
        train_dataset: Optional[Dataset[T_task]],
        val_dataset: Optional[Dataset[T_task]],
    ) -> Tuple[str, PromptTemplate, Iterator[Sequence[T_task]], Iterator[Sequence[T_task]]]:
        """
        Initialize the beam search with seed prompt and dataset iterators.

        Args:
            train_dataset: Dataset for computing gradients.
            val_dataset: Dataset for evaluating prompts.

        Returns:
            Tuple of (resource_name, seed_prompt, grad_iterator, val_iterator).

        Raises:
            ValueError: If either dataset is None.
        """
        resource_name, seed_prompt = self.get_seed_prompt_template()

        if train_dataset is None:
            raise ValueError("train_dataset is required for APO algorithm")
        if val_dataset is None:
            raise ValueError("val_dataset is required for APO algorithm")

        grad_dataset_iterator = batch_iter_over_dataset(train_dataset, self.gradient_batch_size)
        val_dataset_iterator = batch_iter_over_dataset(val_dataset, self.val_batch_size)

        # Initialize history tracking
        self._history_best_prompt = seed_prompt
        self._history_best_score = float("-inf")

        return resource_name, seed_prompt, grad_dataset_iterator, val_dataset_iterator

    def _sample_parent_prompts(self, beam: List[PromptTemplate], round_num: int) -> List[PromptTemplate]:
        """
        Sample parent prompts from the current beam for generating new candidates.

        If the beam has fewer prompts than beam_width, replicates existing prompts.
        Otherwise, randomly samples beam_width prompts.

        Args:
            beam: Current list of prompt templates in the beam.
            round_num: Current round number (for logging, 0-indexed).

        Returns:
            List of parent prompts to generate children from.
        """
        if len(beam) < self.beam_width:
            logger.warning(
                f"[Round {round_num + 1}] Beam width is currently {self.beam_width}, but only {len(beam)} prompts in beam. "
                "Replicating all prompts."
            )
            return [beam[i % len(beam)] for i in range(self.beam_width)]
        else:
            return random.sample(beam, self.beam_width)

    async def _generate_candidate_prompts(
        self,
        parent_prompts: List[PromptTemplate],
        resource_name: str,
        grad_dataset_iterator: Iterator[Sequence[T_task]],
        round_num: int,
    ) -> List[PromptTemplate]:
        """
        Generate new candidate prompts from parents using textual gradients.

        For each parent prompt, generates branch_factor new candidates by:
        1. Evaluating the parent on a training batch
        2. Computing textual gradient
        3. Applying edit to generate improved prompt

        Args:
            parent_prompts: List of parent prompts to generate children from.
            resource_name: Name to register prompts under in the store.
            grad_dataset_iterator: Iterator over training data batches.
            round_num: Current round number (for logging, 0-indexed).

        Returns:
            List of newly generated prompt templates.
        """
        logger.info(
            f"[Round {round_num + 1}] Applying {self.branch_factor} edits to each of "
            f"the {len(parent_prompts)} parents on training dataset"
        )

        candidates: List[PromptTemplate] = []
        for prompt in parent_prompts:
            for _ in range(self.branch_factor):
                grad_samples = next(grad_dataset_iterator)
                rollout_results, _ = await self.evaluate_prompt_on_batch(
                    prompt.template, resource_name, grad_samples, mode="train"
                )
                new_prompt = await self.textual_gradient_and_apply_edit(prompt.template, rollout_results)
                if not new_prompt:
                    logger.error(f"[Round {round_num + 1}] Failed to compute edit for prompt: {prompt.template}")
                    continue
                new_prompt_template = PromptTemplate(template=new_prompt, engine="f-string")
                logger.info(f"[Round {round_num + 1}] New prompt template: {new_prompt_template}")
                candidates.append(new_prompt_template)

        return candidates

    async def _evaluate_and_select_beam(
        self,
        candidates: List[PromptTemplate],
        resource_name: str,
        val_dataset_iterator: Iterator[Sequence[T_task]],
        round_num: int,
    ) -> List[PromptTemplate]:
        """
        Evaluate all candidate prompts on validation data and select top-k for the beam.

        Args:
            candidates: List of candidate prompts to evaluate.
            resource_name: Name to register prompts under in the store.
            val_dataset_iterator: Iterator over validation data batches.
            round_num: Current round number (for logging, 0-indexed).

        Returns:
            List of top beam_width prompts sorted by validation score (best first).

        Raises:
            ValueError: If no candidates remain after evaluation.
        """
        logger.info(f"[Round {round_num + 1}] Evaluating {len(candidates)} candidates on validation dataset")

        val_batch = next(val_dataset_iterator)
        scores: List[Tuple[PromptTemplate, float]] = []

        for idx, prompt in enumerate(candidates):
            _, score = await self.evaluate_prompt_on_batch(prompt.template, resource_name, val_batch, mode="val")
            scores.append((prompt, score))
            logger.info(f"[Round {round_num + 1}] Candidate {idx} score: {score:.3f}")

        # Sort by score (descending) and select top beam_width
        sorted_prompts = [p for p, _ in sorted(scores, key=lambda x: x[1], reverse=True)][: self.beam_width]
        logger.info(
            f"[Round {round_num + 1}] Top {len(sorted_prompts)} candidates on validation dataset: {sorted_prompts}"
        )

        if len(sorted_prompts) == 0:
            raise ValueError("No beam candidates any more")

        return sorted_prompts

    async def _update_best_prompt(
        self,
        beam: List[PromptTemplate],
        resource_name: str,
        val_dataset: Dataset[T_task],
        round_num: int,
    ) -> None:
        """
        Evaluate the best prompt in the beam on the full validation set and update history.

        Args:
            beam: Current beam of prompts (sorted, best first).
            resource_name: Name to register prompts under in the store.
            val_dataset: Full validation dataset.
            round_num: Current round number (for logging, 0-indexed).
        """
        best_prompt = beam[0]
        _, best_score = await self.evaluate_prompt_on_batch(
            best_prompt.template, resource_name, cast(Sequence[T_task], val_dataset), mode="val"
        )
        logger.info(f"[Round {round_num + 1}] Best prompt {best_prompt} has score: {best_score:.3f}")

        if best_score > self._history_best_score:
            logger.info(
                f"[Round {round_num + 1}] Best prompt updated. New best score: {best_score:.3f} (prev: {self._history_best_score:.3f})"
            )
            self._history_best_prompt = best_prompt
            self._history_best_score = best_score

    async def run(
        self,
        train_dataset: Optional[Dataset[T_task]] = None,
        val_dataset: Optional[Dataset[T_task]] = None,
    ) -> None:
        """
        Execute the APO algorithm to optimize prompts through beam search with textual gradients.

        The algorithm performs iterative prompt optimization over multiple rounds:
        - Each round: samples parent prompts, generates new candidates via textual gradients,
          evaluates all candidates on validation data, and keeps the top performers
        - Tracks the historically best prompt across all rounds
        - Uses different training data samples for each gradient computation to ensure diversity

        Args:
            train_dataset: Dataset of tasks for computing textual gradients. Required.
            val_dataset: Dataset of tasks for evaluating and selecting prompts. Required.

        Raises:
            ValueError: If train_dataset or val_dataset is None, or if resources are not set.
        """
        # Initialize beam search
        resource_name, seed_prompt, grad_iterator, val_iterator = self._initialize_beam(train_dataset, val_dataset)

        # Validation datasets are guaranteed to be non-None after initialization
        assert val_dataset is not None

        # Start with seed prompt in the beam
        beam: List[PromptTemplate] = [seed_prompt]

        # Optionally evaluate seed prompt on validation set to establish baseline
        if self.run_initial_validation:
            logger.info("Evaluating seed prompt on validation dataset before optimization...")
            _, seed_score = await self.evaluate_prompt_on_batch(
                seed_prompt.template, resource_name, cast(Sequence[T_task], val_dataset), mode="val"
            )
            logger.info(f"Seed prompt baseline score: {seed_score:.3f}")
            self._history_best_prompt = seed_prompt
            self._history_best_score = seed_score

        # Run beam search for specified number of rounds
        for rnd in range(self.beam_rounds):
            logger.info(f"[Round {rnd + 1}] Round {rnd + 1}/{self.beam_rounds}...")

            # Sample parent prompts from current beam
            parent_prompts = self._sample_parent_prompts(beam, rnd)

            # Generate new candidate prompts from parents
            new_candidates = await self._generate_candidate_prompts(parent_prompts, resource_name, grad_iterator, rnd)

            # Combine existing beam with new candidates
            all_candidates = [*beam, *new_candidates]

            # Evaluate and select top-k prompts for next beam
            beam = await self._evaluate_and_select_beam(all_candidates, resource_name, val_iterator, rnd)

            # Update historically best prompt if improved
            await self._update_best_prompt(beam, resource_name, val_dataset, rnd)
