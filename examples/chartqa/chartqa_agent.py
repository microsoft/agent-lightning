# Copyright (c) Microsoft. All rights reserved.

"""ChartQA agent with two-step pipeline.

This module implements a pipeline with minimal sequential steps
to reduce error propagation (exposure bias).

Two-step mode:
`extract_data` [model + image]: Read all relevant values from chart
`compute_answer` [model text-only]: Calculate final answer from extracted data

Model endpoint and parameters are configured automatically from resources during training.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, cast

import env_var as chartqa_env_var
import termcolor
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from multimodal_utils import encode_image_to_base64
from prompts import (
    COMPUTE_ANSWER_PROMPT,
    EXTRACT_DATA_PROMPT,
)

import agentlightning as agl

logger = logging.getLogger("chartqa_agent")


def evaluate_answer(predicted: str, ground_truth: str, raise_on_error: bool = False) -> float:
    """Evaluate answer accuracy.

    Returns:
        1.0 for exact match or numeric match within 2%
        0.5 for substring match
        0.0 otherwise
    """
    try:
        pred = predicted.lower().strip()
        gt = ground_truth.lower().strip()

        # Exact match
        if pred == gt:
            return 1.0

        # Try numeric comparison
        try:
            pred_num = float(pred.replace(",", ""))
            gt_num = float(gt.replace(",", ""))
            if abs(pred_num - gt_num) / max(abs(gt_num), 1e-9) < 0.02:
                return 1.0
        except (ValueError, AttributeError):
            pass

        # Partial credit for substring match
        if pred in gt or gt in pred:
            return 0.5

        return 0.0
    except Exception as e:
        if raise_on_error:
            raise
        logger.exception(f"Error evaluating answer: {e}")
        return 0.0


class ChartQAState(MessagesState):
    """State for the ChartQA agent."""

    question: str
    image_path: str
    # Extraction step
    extracted_data: str
    # Computation step
    calculation: str
    answer: str
    # Tracking
    num_turns: int
    num_model_calls: int
    messages: list[AnyMessage]


class ChartQAAgent(agl.LitAgent[Dict[str, Any]]):
    """ChartQA agent with reduced exposure bias.

    Uses a simplified 2-step pipeline to minimize error propagation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        endpoint: str = "http://localhost:8090/v1",
        max_turns: int = 1,
        debug: bool = False,
        temperature: float = 0.0,
        use_base64_images: bool = False,
    ):
        """Initialize the ChartQA agent.

        Args:
            model_name: Model name for the vision-language model.
            endpoint: API endpoint for the model.
            max_turns: Max tool call iterations (default 1).
            debug: Enable debug output.
            temperature: Sampling temperature.
            use_base64_images: Whether to encode images as base64.
        """
        self.model_name = model_name
        self.endpoint = endpoint
        self.max_turns = max_turns
        self.debug = debug
        self.temperature = temperature
        self.use_base64_images = use_base64_images

        self._model: BaseChatModel | None = None
        self._graph: CompiledStateGraph[ChartQAState] | None = None

    def _create_model(self) -> BaseChatModel:
        """Create the model instance."""
        return init_chat_model(
            self.model_name,
            model_provider="openai",
            openai_api_base=self.endpoint,
            openai_api_key=chartqa_env_var.OPENAI_API_KEY,
            temperature=self.temperature,
            max_retries=5,
            max_tokens=2048,
            timeout=1200,
        )

    def _ensure_model(self) -> BaseChatModel:
        """Ensure the model is created and cached."""
        if self._model is None:
            self._model = self._create_model()
        return self._model

    def update_llm_config(
        self,
        model_name: str | None = None,
        endpoint: str | None = None,
        temperature: float | None = None,
    ) -> None:
        """Update model configurations."""
        updated = False

        if model_name is not None and model_name != self.model_name:
            self.model_name = model_name
            updated = True
        if endpoint is not None and endpoint != self.endpoint:
            self.endpoint = endpoint
            updated = True
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            updated = True

        if updated:
            self._model = self._create_model()

    def invoke_with_image(self, prompt_text: str, image_path: str) -> str:
        """Invoke the model with an image."""
        if self.use_base64_images:
            image_url = encode_image_to_base64(image_path)
        else:
            if not image_path.startswith("file://"):
                image_path = f"file://{os.path.realpath(image_path)}"
            image_url = image_path

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        if self.debug:
            termcolor.cprint(f"[Model Input] {prompt_text[:200]}...", "cyan")

        try:
            result = self._ensure_model().invoke(messages)
            response = str(result.content) if hasattr(result, "content") else str(result)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        except Exception as e:
            logger.error(f"Failed to invoke model: {e}")
            response = "<answer>error</answer>"

        if self.debug:
            termcolor.cprint(f"[Model Output] {response[:300]}...", "magenta")

        return response

    def invoke_text_only(self, prompt_text: str) -> str:
        """Invoke the model with text only."""
        messages = [{"role": "user", "content": prompt_text}]

        if self.debug:
            termcolor.cprint(f"[Model Text Input] {prompt_text[:200]}...", "blue")

        try:
            result = self._ensure_model().invoke(messages)
            response = str(result.content) if hasattr(result, "content") else str(result)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        except Exception as e:
            logger.error(f"Failed to invoke model: {e}")
            response = "<answer>error</answer>"

        if self.debug:
            termcolor.cprint(f"[Model Text Output] {response[:300]}...", "green")

        return response

    def extract_content(self, text: str, tag: str) -> str:
        """Extract content between XML-style tags."""
        matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if not matches:
            return ""
        return "\n".join(match.strip() for match in matches)

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer to match evaluation format.

        Cleans up common issues:
        - Remove units (%, $, million, etc.)
        - Remove extra punctuation
        - Clean whitespace
        - Format numbers consistently
        """
        if not answer:
            return ""

        ans = answer.strip()

        # Remove common prefixes/suffixes
        prefixes = ["the answer is", "answer:", "approximately", "about", "around"]
        for prefix in prefixes:
            if ans.lower().startswith(prefix):
                ans = ans[len(prefix) :].strip()

        # Remove trailing punctuation
        ans = ans.rstrip(".,;:")

        # Remove units and symbols (but keep the number)
        # Pattern: number followed by unit
        unit_pattern = (
            r"^([-+]?\d*\.?\d+)\s*(%|percent|million|billion|thousand|USD|\$|€|£|dollars?|years?|km|miles?|kg|lbs?)\.?$"
        )
        match = re.match(unit_pattern, ans, re.IGNORECASE)
        if match:
            ans = match.group(1)

        # Remove leading $ or currency
        if ans.startswith("$"):
            ans = ans[1:].strip()

        # Remove trailing % if present
        if ans.endswith("%"):
            ans = ans[:-1].strip()

        # Clean up number formatting
        # Remove commas from numbers
        if re.match(r"^[-+]?\d{1,3}(,\d{3})*(\.\d+)?$", ans):
            ans = ans.replace(",", "")

        # Try to parse as number and format consistently
        try:
            num = float(ans)
            # If it's a whole number, format without decimals
            if num == int(num):
                ans = str(int(num))
            else:
                # Round to 2 decimal places, remove trailing zeros
                ans = f"{num:.2f}".rstrip("0").rstrip(".")
        except ValueError:
            # Not a number, keep as text
            # Clean up quotes if present
            ans = ans.strip("'\"")

        return ans

    # =========================================================================
    # Two-Step Pipeline Nodes
    # =========================================================================

    def extract_data(self, state: ChartQAState) -> ChartQAState:
        """Step 1 [model + image]: Extract all relevant data from chart."""
        prompt: Any = EXTRACT_DATA_PROMPT.invoke({"question": state["question"]})  # type: ignore[reportUnknownMemberType]
        prompt_text = "\n".join(msg.content for msg in prompt.messages)  # type: ignore

        result_text = self.invoke_with_image(prompt_text, state["image_path"])

        # Extract data JSON
        extracted_data = self.extract_content(result_text, "data")
        if not extracted_data:
            extracted_data = result_text

        if self.debug:
            termcolor.cprint(f"[Extract] Data: {extracted_data[:300]}", "yellow")

        return {  # type: ignore
            **state,
            "extracted_data": extracted_data,
            "num_model_calls": state.get("num_model_calls", 0) + 1,
            "messages": [HumanMessage(content=result_text)],
        }

    def compute_answer(self, state: ChartQAState) -> ChartQAState:
        """Step 2 [model text-only]: Compute answer from extracted data."""
        prompt: Any = COMPUTE_ANSWER_PROMPT.invoke(  # type: ignore[reportUnknownMemberType]
            {"question": state["question"], "extracted_data": state["extracted_data"]}
        )
        prompt_text = "\n".join(msg.content for msg in prompt.messages)  # type: ignore

        result_text = self.invoke_text_only(prompt_text)

        # Extract answer and calculation
        raw_answer = self.extract_content(result_text, "answer")
        calculation = self.extract_content(result_text, "think")

        if not raw_answer:
            # Fallback: try to find any number or short text
            raw_answer = result_text.strip().split("\n")[-1].strip()

        # Normalize answer for better evaluation matching
        answer = self.normalize_answer(raw_answer)

        if self.debug:
            termcolor.cprint(f"[Compute] Raw: {raw_answer} -> Normalized: {answer}", "yellow")

        return {  # type: ignore
            **state,
            "calculation": calculation,
            "answer": answer,
            "num_turns": 1,
            "num_model_calls": state.get("num_model_calls", 0) + 1,
            "messages": [*state.get("messages", []), HumanMessage(content=result_text)],
        }

    def graph(self) -> CompiledStateGraph[ChartQAState]:
        """Build the workflow graph.

        Two-step mode:
            START -> extract_data -> compute_answer -> END
        """
        if self._graph is not None:
            return self._graph

        builder = StateGraph(ChartQAState)

        # Two-step: extract then compute
        builder.add_node("extract_data", self.extract_data)  # type: ignore[reportUnknownMemberType]
        builder.add_node("compute_answer", self.compute_answer)  # type: ignore[reportUnknownMemberType]
        builder.add_edge(START, "extract_data")
        builder.add_edge("extract_data", "compute_answer")
        builder.add_edge("compute_answer", END)

        self._graph = builder.compile()  # type: ignore
        return self._graph

    def rollout(self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout) -> float | None:
        """AgentLightning wrapper for the ChartQA agent."""
        question = task["question"]
        rollout = cast(agl.AttemptedRollout, rollout)

        image_path = os.path.join(chartqa_env_var.CHARTQA_DATA_DIR, task["image_path"])
        ground_truth = task["answer"]

        if not os.path.exists(image_path):
            logger.error(f"Image {image_path} does not exist. Skipping.")
            return None

        # Update model configuration from resources
        if "main_llm" in resources:
            llm_resource = cast(agl.LLM, resources["main_llm"])
            llm_endpoint = llm_resource.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
            llm_temperature = llm_resource.sampling_parameters.get("temperature", 0.0)

            if llm_endpoint != self.endpoint or llm_temperature != self.temperature:
                self.endpoint = llm_endpoint
                self.temperature = llm_temperature
                self._model = self._create_model()

        try:
            handler = self.tracer.get_langchain_handler()
            result = self.graph().invoke(  # type: ignore[reportUnknownMemberType]
                {"question": question, "image_path": image_path},  # type: ignore[reportArgumentType]
                {"callbacks": [handler] if handler else [], "recursion_limit": 100},
            )
        except Exception as e:
            logger.error(f"[Rollout {rollout.rollout_id}] Error: {e}", exc_info=True)
            return 0.0

        predicted_answer = result["answer"]
        reward = evaluate_answer(predicted_answer, ground_truth, raise_on_error=False)

        return reward
