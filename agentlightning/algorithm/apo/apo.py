# Copyright (c) Microsoft. All rights reserved.

"""
APO with textual gradients that read rollout spans and outputs to modify the prompt.

- algo: beam search with span-aware textual gradients -> apply_edit via LLM
- rollout: same pattern as your example, but task is a dict (T_task)
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Counter, Dict, Generic, Iterator, List, Optional, Sequence, Set, Tuple, TypedDict, TypeVar, cast

import poml
from openai import AsyncOpenAI

from agentlightning.adapter.messages import TraceToMessages
from agentlightning.algorithm.base import Algorithm
from agentlightning.algorithm.utils import batch_iter_over_dataset
from agentlightning.reward import find_final_reward
from agentlightning.types import Dataset, NamedResources, PromptTemplate, Rollout, RolloutMode, RolloutStatus

logger = logging.getLogger(__name__)

T_task = TypeVar("T_task")


class RolloutResultForAPO(TypedDict):
    """This must be all JSON serializable to be processable by POML."""

    status: RolloutStatus
    final_reward: Optional[float]
    spans: List[Dict[str, Any]]
    messages: List[Any]


@dataclass
class VersionedPromptTemplate:
    version: str
    prompt_template: PromptTemplate
    score: Optional[float] = None


GRADIENT_PROMPT_FILES = [
    Path(__file__).parent / "prompts" / "text_gradient_variant01.poml",
    Path(__file__).parent / "prompts" / "text_gradient_variant02.poml",
    Path(__file__).parent / "prompts" / "text_gradient_variant03.poml",
]

APPLY_EDIT_PROMPT_FILES = [
    Path(__file__).parent / "prompts" / "apply_edit_variant01.poml",
    Path(__file__).parent / "prompts" / "apply_edit_variant02.poml",
]


class APO(Algorithm, Generic[T_task]):
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

    - [ProTeGi](https://aclanthology.org/2023.emnlp-main.494.pdf)
    - [TextGrad](https://github.com/zou-group/textgrad)
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
        rollout_batch_timeout: float = 3600.0,
        run_initial_validation: bool = True,
        # Internal flags for debugging
        _poml_trace: bool = False,
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
        self._history_best_version: Optional[str] = None

        self._version_counter: int = 0

        self._poml_trace = _poml_trace

    def _create_versioned_prompt(
        self,
        prompt_template: PromptTemplate,
        *,
        score: Optional[float] = None,
    ) -> VersionedPromptTemplate:
        """
        Wrap a prompt template with a new monotonically increasing version identifier.
        """
        version = f"v{self._version_counter}"
        self._version_counter += 1
        return VersionedPromptTemplate(version=version, prompt_template=prompt_template, score=score)

    def _format_log_prefix(
        self,
        *,
        round_num: Optional[int] = None,
        beam_idx: Optional[int] = None,
        branch_idx: Optional[int] = None,
        prompt_version: Optional[str] = None,
    ) -> str:
        """
        Construct the standardized log prefix.
        """
        parts: List[str] = []
        if round_num is not None:
            parts.append(f"Round {round_num:02d}")
        if beam_idx is not None:
            parts.append(f"Beam {beam_idx:02d}")
        if branch_idx is not None:
            parts.append(f"Branch {branch_idx:02d}")
        if prompt_version is not None:
            parts.append(f"Prompt {prompt_version}")
        if not parts:
            return ""
        return f"[{' | '.join(parts)}]"

    def _log(self, level: int, message: str, *, prefix: Optional[str] = None) -> None:
        """
        Log a message with an optional standardized prefix.
        """
        effective_prefix = prefix
        if effective_prefix:
            logger.log(level, f"{effective_prefix} {message}")
        else:
            logger.log(level, message)

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

    def get_adapter(self) -> TraceToMessages:
        """
        Get the adapter for converting spans to messages.

        Returns:
            The TraceToMessages instance for this algorithm.

        Raises:
            ValueError: If the adapter is not a TraceToMessages.
        """
        adapter = super().get_adapter()
        if not isinstance(adapter, TraceToMessages):
            raise ValueError("Adapter must be a TraceToMessages for APO algorithm")
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
        current_prompt: VersionedPromptTemplate,
        rollout_results: List[RolloutResultForAPO],
        *,
        prefix: Optional[str] = None,
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
            self._log(
                logging.WARNING,
                f"Only {len(rollout_results)} rollouts available, but {self.gradient_batch_size} are needed. Using all rollouts.",
                prefix=prefix,
            )
            sampled_rollout_results = rollout_results
        else:
            sampled_rollout_results = random.sample(rollout_results, self.gradient_batch_size)

        self._log(
            logging.INFO,
            f"Gradient will be computed with {self.gradient_model} for {len(sampled_rollout_results)} rollouts with template: {tg_template.name}",
            prefix=prefix,
        )

        tg_msg = poml.poml(  # type: ignore
            tg_template,
            context={
                "experiments": sampled_rollout_results,
                "prompt_template": current_prompt.prompt_template.template,
            },
            format="openai_chat",
        )
        self._log(
            logging.DEBUG,
            f"Gradient computed with {self.gradient_model} prompt: {tg_msg}",
            prefix=prefix,
        )
        critique_response = await self.async_openai_client.chat.completions.create(
            model=self.gradient_model,
            messages=tg_msg["messages"],  # type: ignore
            temperature=self.diversity_temperature,
        )
        critique_text = critique_response.choices[0].message.content
        self._log(
            logging.INFO,
            f"Gradient computed with {self.gradient_model} has result: {critique_text}",
            prefix=prefix,
        )

        return critique_text

    async def textual_gradient_and_apply_edit(
        self,
        current_prompt: VersionedPromptTemplate,
        rollout: List[RolloutResultForAPO],
        *,
        prefix: Optional[str] = None,
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
        critique_text = await self.compute_textual_gradient(
            current_prompt,
            rollout,
            prefix=prefix,
        )
        if not critique_text:
            self._log(
                logging.ERROR,
                "Failed to compute critique for prompt.",
                prefix=prefix,
            )
            return current_prompt.prompt_template.template

        # 2) Apply edit
        ae_template = random.choice(APPLY_EDIT_PROMPT_FILES)
        self._log(
            logging.INFO,
            f"Edit will be generated by {self.apply_edit_model} with template: {ae_template.name}",
            prefix=prefix,
        )
        ae_msg = poml.poml(  # type: ignore
            ae_template,
            context={
                "prompt_template": current_prompt.prompt_template.template,
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
            self._log(
                logging.INFO,
                f"Edit generated by {self.apply_edit_model}: {new_prompt[:50]}...",
                prefix=prefix,
            )
        return new_prompt

    async def get_rollout_results(
        self,
        rollout: List[Rollout],
        *,
        prefix: Optional[str] = None,
    ) -> List[RolloutResultForAPO]:
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
                messages=messages,
            )
            self._log(
                logging.DEBUG,
                f"Rollout result for {r.rollout_id}: status {rollout_result['status']} with final reward {rollout_result['final_reward']}. "
                f"{len(rollout_result['spans'])} spans and {len(rollout_result['messages'])} messages.",
                prefix=prefix,
            )
            rollout_results.append(rollout_result)
        return rollout_results

    async def evaluate_prompt_on_batch(
        self,
        prompt: VersionedPromptTemplate,
        resource_name: str,
        dataset: Sequence[T_task],
        mode: RolloutMode,
        *,
        prefix: Optional[str] = None,
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
        preview = prompt.prompt_template.template[:50]
        self._log(
            logging.INFO,
            f'Evaluating prompt "{preview}..." on {len(dataset)} tasks in {mode} mode',
            prefix=prefix,
        )

        # Install prompt as named resource
        resources: NamedResources = {resource_name: prompt.prompt_template}
        resource_update = await store.update_resources(prompt.version, resources)

        rollout_ids: List[str] = []
        for t in dataset:
            r = await store.enqueue_rollout(input=t, mode=mode, resources_id=resource_update.resources_id)
            rollout_ids.append(r.rollout_id)

        deadline = time.time() + self.rollout_batch_timeout
        finished: List[Rollout] = []
        while time.time() < deadline:
            finished = await store.wait_for_rollouts(rollout_ids=rollout_ids, timeout=0.0)
            if len(finished) >= len(rollout_ids):
                self._log(
                    logging.INFO,
                    f"All {len(rollout_ids)} rollouts finished within timeout.",
                    prefix=prefix,
                )
                break
            else:
                self._log(
                    logging.DEBUG,
                    f"Only {len(finished)} rollouts finished within timeout. Waiting for remaining {len(rollout_ids) - len(finished)} rollouts.",
                    prefix=prefix,
                )
                # Sleep to avoid busy-waiting
                await asyncio.sleep(2.0)

        rollout_results = await self.get_rollout_results(
            finished,
            prefix=prefix,
        )
        final_rewards = [rr["final_reward"] for rr in rollout_results]

        avg = float(sum([r or 0.0 for r in final_rewards]) / max(1, len(final_rewards)))
        status_counter = Counter([rr["status"] for rr in rollout_results])

        self._log(
            logging.INFO,
            f"Evaluated {len(rollout_results)} rollouts. Statuses: {status_counter}. Rewards: {final_rewards}, average is {avg}",
            prefix=prefix,
        )
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

    def _sample_parent_prompts(
        self,
        beam: List[VersionedPromptTemplate],
        round_num: int,
    ) -> List[Tuple[int, VersionedPromptTemplate]]:
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
        display_round = round_num + 1
        if len(beam) < self.beam_width:
            prefix = self._format_log_prefix(round_num=display_round)
            self._log(
                logging.WARNING,
                f"Beam width is currently {self.beam_width}, but only {len(beam)} prompts in beam. Replicating all prompts.",
                prefix=prefix,
            )
            return [(i % len(beam), beam[i % len(beam)]) for i in range(self.beam_width)]

        selected_indices = random.sample(range(len(beam)), self.beam_width)
        return [(idx, beam[idx]) for idx in selected_indices]

    async def _generate_candidate_prompts(
        self,
        parent_prompts: List[Tuple[int, VersionedPromptTemplate]],
        resource_name: str,
        grad_dataset_iterator: Iterator[Sequence[T_task]],
        round_num: int,
    ) -> List[VersionedPromptTemplate]:
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
        display_round = round_num + 1
        round_prefix = self._format_log_prefix(round_num=display_round)
        self._log(
            logging.INFO,
            f"Applying {self.branch_factor} edits to each of the {len(parent_prompts)} parents based on "
            "gradients computed on training dataset",
            prefix=round_prefix,
        )

        parent_prompts_str = [
            f"{p.version}:{p.score:.3f}" if p.score is not None else p.version for _, p in parent_prompts
        ]
        self._log(
            logging.INFO,
            f"Parent prompts: {', '.join(parent_prompts_str)}",
            prefix=round_prefix,
        )

        candidates: List[VersionedPromptTemplate] = []
        used_beam_indices: Set[int] = set()
        for real_beam_idx, (beam_idx, prompt) in enumerate(parent_prompts):
            if beam_idx in used_beam_indices:
                beam_prefix = self._format_log_prefix(
                    round_num=display_round,
                    beam_idx=beam_idx + 1,
                    prompt_version=prompt.version,
                )
                self._log(
                    logging.WARNING,
                    "Duplicated beam index found. Might be caused by beam_width too high. "
                    + f"The real index of this beam is {real_beam_idx + 1}.",
                    prefix=beam_prefix,
                )
            else:
                used_beam_indices.add(beam_idx)
            for branch_idx in range(self.branch_factor):
                parent_prefix = self._format_log_prefix(
                    round_num=display_round,
                    beam_idx=beam_idx + 1,
                    branch_idx=branch_idx + 1,
                    prompt_version=prompt.version,
                )
                baseline_score = f"{prompt.score:.3f}" if prompt.score is not None else "N/A"
                self._log(
                    logging.INFO,
                    f"Use parent prompt {prompt.version} as a baseline to generate a new prompt. Baseline score: {baseline_score}",
                    prefix=parent_prefix,
                )
                grad_samples = next(grad_dataset_iterator)
                rollout_results, _ = await self.evaluate_prompt_on_batch(
                    prompt,
                    resource_name,
                    grad_samples,
                    mode="train",
                    prefix=parent_prefix,
                )
                new_prompt = await self.textual_gradient_and_apply_edit(
                    prompt,
                    rollout_results,
                    prefix=parent_prefix,
                )
                if not new_prompt:
                    self._log(
                        logging.ERROR,
                        f"Failed to compute edit for prompt: {prompt.prompt_template.template}",
                        prefix=parent_prefix,
                    )
                    continue
                new_prompt_template = PromptTemplate(template=new_prompt, engine="f-string")
                versioned_candidate = self._create_versioned_prompt(new_prompt_template)
                self._log(
                    logging.INFO,
                    f"New prompt template created from parent {prompt.version}: {versioned_candidate.version}",
                    prefix=parent_prefix,
                )
                candidate_prefix = self._format_log_prefix(
                    round_num=display_round, prompt_version=versioned_candidate.version
                )
                self._log(
                    logging.INFO,
                    f"New prompt template created from parent {prompt.version}:\n```\n{new_prompt}\n```",
                    prefix=candidate_prefix,
                )
                candidates.append(versioned_candidate)

        return candidates

    async def _evaluate_and_select_beam(
        self,
        candidates: List[VersionedPromptTemplate],
        resource_name: str,
        val_dataset_iterator: Iterator[Sequence[T_task]],
        round_num: int,
    ) -> List[VersionedPromptTemplate]:
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
        display_round = round_num + 1
        round_prefix = self._format_log_prefix(round_num=display_round)
        self._log(
            logging.INFO,
            f"Evaluating {len(candidates)} candidates on validation dataset",
            prefix=round_prefix,
        )

        val_batch = next(val_dataset_iterator)

        for prompt in candidates:
            candidate_prefix = self._format_log_prefix(
                round_num=display_round,
                prompt_version=prompt.version,
            )
            _, score = await self.evaluate_prompt_on_batch(
                prompt,
                resource_name,
                val_batch,
                mode="val",
                prefix=candidate_prefix,
            )
            prompt.score = score
            self._log(
                logging.INFO,
                f"Candidate score: {score:.3f}",
                prefix=candidate_prefix,
            )

        # Sort by score (descending) and select top beam_width
        sorted_prompts = [p for p in sorted(candidates, key=lambda x: cast(float, x.score), reverse=True)]
        selected_prompts = sorted_prompts[: self.beam_width]
        selected_versions = [
            f"{prompt.version}:{prompt.score:.3f}" if prompt.score is not None else prompt.version
            for prompt in selected_prompts
        ]
        self._log(
            logging.INFO,
            f"Top {len(selected_prompts)} candidates on validation dataset: {selected_versions}",
            prefix=round_prefix,
        )

        if len(selected_prompts) == 0:
            raise ValueError("No beam candidates any more")

        return selected_prompts

    async def _update_best_prompt(
        self,
        beam: List[VersionedPromptTemplate],
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
        display_round = round_num + 1
        best_prompt = beam[0]
        prefix = self._format_log_prefix(round_num=display_round, prompt_version=best_prompt.version)
        _, best_score = await self.evaluate_prompt_on_batch(
            best_prompt,
            resource_name,
            cast(Sequence[T_task], val_dataset),
            mode="val",
            prefix=prefix,
        )
        self._log(
            logging.INFO,
            f"Beam leader score: {best_score:.3f}",
            prefix=prefix,
        )

        if best_score > self._history_best_score:
            prev = self._history_best_score
            self._log(
                logging.INFO,
                f"Best prompt updated. New best score: {best_score:.3f} (prev: {prev:.3f})",
                prefix=prefix,
            )
            self._history_best_prompt = best_prompt.prompt_template
            self._history_best_score = best_score
            self._history_best_version = best_prompt.version
        else:
            self._log(
                logging.WARNING,
                f"Best prompt not updated. Current score: {best_score:.3f} vs. history best: {self._history_best_score:.3f})",
                prefix=prefix,
            )

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

        if self._poml_trace:
            poml.set_trace(trace_dir="pomltrace")

        # Validation datasets are guaranteed to be non-None after initialization
        assert val_dataset is not None

        # Start with seed prompt in the beam
        seed_versioned = self._create_versioned_prompt(seed_prompt)
        beam: List[VersionedPromptTemplate] = [seed_versioned]
        self._history_best_prompt = seed_prompt
        self._history_best_version = seed_versioned.version

        # Optionally evaluate seed prompt on validation set to establish baseline
        if self.run_initial_validation:
            seed_prefix = self._format_log_prefix(round_num=0, prompt_version=seed_versioned.version)
            self._log(
                logging.INFO,
                "Evaluating seed prompt on validation dataset before optimization...",
                prefix=seed_prefix,
            )
            _, seed_score = await self.evaluate_prompt_on_batch(
                seed_versioned,
                resource_name,
                cast(Sequence[T_task], val_dataset),
                mode="val",
                prefix=seed_prefix,
            )
            self._log(
                logging.INFO,
                f"Seed prompt baseline score: {seed_score:.3f}",
                prefix=seed_prefix,
            )
            self._history_best_prompt = seed_prompt
            self._history_best_score = seed_score
            self._history_best_version = seed_versioned.version

        # Run beam search for specified number of rounds
        for rnd in range(self.beam_rounds):
            display_round = rnd + 1
            round_prefix = self._format_log_prefix(round_num=display_round)
            self._log(
                logging.INFO,
                f"Round {display_round}/{self.beam_rounds}...",
                prefix=round_prefix,
            )

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
