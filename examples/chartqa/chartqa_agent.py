# Copyright (c) Microsoft. All rights reserved.

"""ChartQA agent demonstrating LangGraph-based visual reasoning with refinement.

This module defines `ChartQAAgent` plus the supporting prompt utilities used by
`debug_chartqa_agent.py` and `train_chartqa_agent.py`.

1. `analyze_chart` observes and summarizes the chart.
2. `extract_data` calls a text-only LLM to extract the requested values.
3. `calculate_answer` runs calculations grounded in prior steps.
4. `check_answer` verifies reasoning quality.
5. `refine_answer` conditionally patches mistakes before responding.

Example usage can be found in `debug_chartqa_agent.py` and `train_chartqa_agent.py`.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Literal, cast

import env_var as chartqa_env_var
import termcolor
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from multimodal_utils import encode_image_to_base64
from prompts import (
    ANALYZE_CHART_PROMPT,
    CALCULATE_ANSWER_PROMPT,
    CHECK_ANSWER_PROMPT,
    EXTRACT_DATA_PROMPT,
    REFINE_ANSWER_PROMPT,
)

import agentlightning as agl

logger = logging.getLogger("chartqa_agent")


class ChartState(MessagesState):
    question: str
    image_path: str
    observation: str
    extracted_data: str
    calculation: str
    answer: str
    feedback: str
    num_turns: int
    messages: list[AnyMessage]


class ChartQAAgent(agl.LitAgent[Dict[str, Any]]):
    """LangGraph-powered ChartQA agent with multi-step reasoning and refinement.

    The implementation shares the same [`agl.LitAgent`][agentlightning.LitAgent] interface as
    the Calc-X sample agent but augments it with image handling and LangGraph state tracking.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_turns: int = 3,
        debug: bool = False,
        endpoint: str | None = None,
        temperature: float = 0.0,
        use_base64_images: bool = False,
    ):
        self.debug = debug
        self.max_turns = max_turns
        self.use_base64_images = use_base64_images
        self.model_name = model_name
        self.endpoint = endpoint
        self.temperature = temperature

        self._llm: BaseChatModel | None = None
        self._graph: CompiledStateGraph[ChartState] | None = None

    def _create_llm(self) -> BaseChatModel:
        if self.model_name is None:
            raise ValueError("model_name is required for creating LLM")
        return init_chat_model(
            self.model_name,
            model_provider="openai",
            openai_api_base=self.endpoint,
            openai_api_key=chartqa_env_var.OPENAI_API_KEY,
            temperature=self.temperature,
            max_retries=2,
            max_tokens=1024,
            timeout=300,
        )

    def update_llm_config(self, model_name: str, endpoint: str | None, temperature: float | None) -> None:
        """Update the LLM configuration. Re-create the LLM if the configuration is changed."""
        updated: bool = False
        if model_name != self.model_name:
            self.model_name = model_name
            updated = True
        if endpoint != self.endpoint:
            self.endpoint = endpoint
            updated = True
        if temperature != self.temperature:
            self.temperature = temperature
            updated = True
        if updated:
            self._llm = self._create_llm()

    def _ensure_llm(self) -> BaseChatModel:
        """Ensure the LLM is created and cached."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def invoke_prompt(self, prompt: Any) -> AnyMessage:
        """Invoke LLM with prompt."""
        if self.debug:
            for message in prompt.messages:
                termcolor.cprint(message.pretty_repr(), "blue")

        try:
            result = self._ensure_llm().invoke(prompt)
        except Exception as e:
            logger.error(f"Failed to invoke prompt: {e}")
            result = self._ensure_llm().invoke([HumanMessage(content="Please provide a reasonable answer.")])

        if self.debug:
            termcolor.cprint(result.pretty_repr(), "green")

        return result  # type: ignore

    def invoke_prompt_with_image(self, prompt_text: str, image_path: str) -> str:
        """Invoke vision-language model with image.

        Handles both local vLLM (file:// URLs) and cloud APIs (base64 encoding).
        Cloud APIs (OpenAI, Anthropic, Google, Azure, etc.) require base64 encoding.
        """
        # Determine image URL format based on endpoint
        if self.use_base64_images:
            # Cloud APIs require base64 encoding for local files
            image_url = encode_image_to_base64(image_path)
        else:
            # Local vLLM supports file:// URLs
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
            termcolor.cprint(f"[VLM Call] {prompt_text[:100]}...", "blue")

        try:
            result = self._ensure_llm().invoke(messages)
            response = result.content if hasattr(result, "content") else str(result)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to invoke VLM: {e}")
            response = "<observe>Unable to analyze chart</observe>"

        if self.debug:
            termcolor.cprint(f"[VLM Response] {response[:200]}...", "green")

        return response  # type: ignore

    def extract_content(self, text: str, tag: str) -> str:
        """Extract content between XML-style tags."""
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def analyze_chart(self, state: ChartState) -> ChartState:
        """Step 1: Observe and describe the chart."""
        prompt: Any = ANALYZE_CHART_PROMPT.invoke({"question": state["question"]})  # type: ignore
        prompt_text = prompt.messages[1].content

        result_text = self.invoke_prompt_with_image(prompt_text, state["image_path"])

        observation = self.extract_content(result_text, "observe")
        if not observation:
            observation = result_text

        return {  # type: ignore
            **state,
            "observation": observation,
            "num_turns": 1,
            "messages": [HumanMessage(content=result_text)],
        }

    def extract_data(self, state: ChartState) -> ChartState:
        """Step 2: Extract specific data values."""
        prompt: Any = EXTRACT_DATA_PROMPT.invoke(  # type: ignore
            {
                "observation": state["observation"],
                "question": state["question"],
            }
        )
        result = self.invoke_prompt(prompt)

        extracted_data = self.extract_content(result.content, "extract")  # type: ignore
        if not extracted_data:
            extracted_data = result.content  # type: ignore

        return {  # type: ignore
            **state,
            "extracted_data": extracted_data,  # type: ignore
            "messages": [*state.get("messages", []), result],
        }

    def calculate_answer(self, state: ChartState) -> ChartState:
        """Step 3: Calculate and provide answer."""
        prompt: Any = CALCULATE_ANSWER_PROMPT.invoke(  # type: ignore
            {
                "extracted_data": state["extracted_data"],
                "question": state["question"],
            }
        )
        result = self.invoke_prompt(prompt)

        calculation = self.extract_content(result.content, "calculate")  # type: ignore
        answer = self.extract_content(result.content, "answer")  # type: ignore
        if not answer:
            answer = cast(str, result.content)  # type: ignore

        return {  # type: ignore
            **state,
            "calculation": calculation,
            "answer": answer,
            "messages": [*state.get("messages", []), result],
        }

    def check_answer(self, state: ChartState) -> ChartState:
        """Step 4: Verify answer quality."""
        prompt: Any = CHECK_ANSWER_PROMPT.invoke(  # type: ignore
            {
                "observation": state["observation"],
                "extracted_data": state["extracted_data"],
                "question": state["question"],
                "answer": state["answer"],
                "calculation": state.get("calculation", "No calculation shown"),
            }
        )
        result = self.invoke_prompt(prompt)

        if self.debug:
            termcolor.cprint(f"[Check] {result.content}", "yellow")  # type: ignore

        return {  # type: ignore
            **state,
            "feedback": result.content,  # type: ignore
            "messages": [*state.get("messages", []), *prompt.messages, result],
        }

    def refine_answer(self, state: ChartState) -> ChartState:
        """Step 5: Refine answer based on feedback."""
        prompt: Any = REFINE_ANSWER_PROMPT.invoke(  # type: ignore
            {
                "observation": state["observation"],
                "extracted_data": state["extracted_data"],
                "question": state["question"],
                "answer": state["answer"],
                "calculation": state.get("calculation", ""),
                "feedback": state["feedback"],
            }
        )
        result = self.invoke_prompt(prompt)
        content: str = result.content  # type: ignore

        new_extracted = self.extract_content(content, "extract")
        extracted_data = new_extracted if new_extracted else state["extracted_data"]

        new_calculation = self.extract_content(content, "calculate")

        new_answer = self.extract_content(content, "answer")
        if not new_answer:
            new_answer = content

        return {  # type: ignore
            **state,
            "extracted_data": extracted_data,
            "calculation": new_calculation,
            "answer": new_answer,
            "num_turns": state.get("num_turns", 0) + 1,
            "messages": [*prompt.messages, result],
        }

    def should_continue(self, state: ChartState) -> Literal[END, "refine_answer"]:  # type: ignore
        """Determine if refinement is needed."""
        if state["messages"] and isinstance(
            state["messages"][-1], BaseMessage
        ):  # pyright: ignore[reportUnnecessaryIsInstance]
            last_message = state["messages"][-1]
            if "THE ANSWER IS CORRECT" in last_message.content:  # type: ignore
                if "THE ANSWER IS INCORRECT" in last_message.content:  # type: ignore
                    correct_index = last_message.content.rfind("THE ANSWER IS CORRECT")  # type: ignore
                    incorrect_index = last_message.content.rfind("THE ANSWER IS INCORRECT")  # type: ignore
                    if correct_index > incorrect_index:
                        return END
                else:
                    return END

        if state.get("num_turns", 0) >= self.max_turns:
            return END

        return "refine_answer"

    def graph(self) -> CompiledStateGraph[ChartState]:
        """Build the workflow graph with refinement loop."""
        # Check if the graph is already built
        if self._graph is not None:
            return self._graph

        builder = StateGraph(ChartState)
        builder.add_node(self.analyze_chart)  # type: ignore
        builder.add_node(self.extract_data)  # type: ignore
        builder.add_node(self.calculate_answer)  # type: ignore
        builder.add_node(self.check_answer)  # type: ignore
        builder.add_node(self.refine_answer)  # type: ignore

        builder.add_edge(START, "analyze_chart")
        builder.add_edge("analyze_chart", "extract_data")
        builder.add_edge("extract_data", "calculate_answer")
        builder.add_edge("calculate_answer", "check_answer")
        builder.add_conditional_edges(
            "check_answer",
            self.should_continue,  # type: ignore
        )
        builder.add_edge("refine_answer", "extract_data")

        self._graph = builder.compile()  # type: ignore
        return self._graph

    def rollout(self, task: Dict[str, Any], resources: agl.NamedResources, rollout: agl.Rollout) -> float | None:
        """AgentLightning wrapper for ChartQA agent."""

        question = task["question"]

        rollout = cast(agl.AttemptedRollout, rollout)
        llm = cast(agl.LLM, resources["main_llm"])

        image_path = os.path.join(chartqa_env_var.CHARTQA_DATA_DIR, task["image_path"])
        ground_truth = task["answer"]

        if not os.path.exists(image_path):
            logger.error(f"Image {image_path} does not exist. Skipping.")
            return None

        # The new rollout could have a different endpoint or temperature.
        # Update the LLM if necessary.
        self.update_llm_config(
            model_name=llm.model,
            endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
            temperature=llm.sampling_parameters.get("temperature", 0.0),
        )

        try:
            handler = self.tracer.get_langchain_handler()
            result = self.graph().invoke(  # type: ignore
                {"question": question, "image_path": image_path},  # type: ignore
                {"callbacks": [handler] if handler else [], "recursion_limit": 100},
            )
        except Exception as e:
            error_msg = f"[Rollout {rollout.rollout_id}] Error during agent invocation: {e}"
            logger.error(error_msg, exc_info=True)
            # Return 0.0 as reward to indicate failure
            return 0.0

        predicted_answer = result["answer"]
        reward = evaluate_answer(predicted_answer, ground_truth, raise_on_error=False)

        return reward


def evaluate_answer(predicted: str, ground_truth: str, raise_on_error: bool = False) -> float:
    """Evaluate answer accuracy."""
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
