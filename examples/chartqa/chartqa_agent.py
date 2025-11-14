# Copyright (c) Microsoft. All rights reserved.

"""ChartQA agent demonstrating multi-step visual reasoning with refinement loop.

This agent analyzes charts and answers questions using a multi-turn workflow:
1. analyze_chart: Observe and describe the chart
2. extract_data: Extract specific data values
3. calculate_answer: Perform calculations and provide answer
4. check_answer: Verify the answer quality
5. refine_answer: (conditional) Refine if errors detected

Architecture inspired by SQL agent's write→execute→check→rewrite pattern,
adapted for visual chart reasoning.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, cast

import pandas as pd
import termcolor
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

import agentlightning as agl

agl.configure_logger()

logger = agl.configure_logger(name=__name__)


# System Prompts

ANALYZE_CHART_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a visual reasoning expert analyzing charts and graphs.
Given a chart image and a question, first carefully observe and describe the chart.

Instructions:
- Identify the chart type (bar chart, line chart, pie chart, scatter plot, etc.)
- Note the axes labels and units (if applicable)
- Describe the data series or categories shown
- Observe key patterns, trends, or noteworthy values
- Pay attention to legends, titles, and annotations

## Output Format ##

Provide your observation inside <observe> and </observe> tags.

Example:
<observe>
Bar chart showing GDP of 5 countries. X-axis shows country names, Y-axis shows GDP in trillions of USD.
Data values: USA appears highest at around 25, China second at around 20, followed by India, UK, and France.
</observe>
""".strip(),
        ),
        ("user", "Question: {question}"),
    ]
)


EXTRACT_DATA_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
Based on your observation of the chart, extract the specific data values needed to answer the question.

Instructions:
- Extract only the data relevant to the question
- Be precise with values (read carefully from the chart)
- Include labels/categories with each value
- Use appropriate units

## Output Format ##

Provide extracted data inside <extract> and </extract> tags.
Format: Label1: Value1, Label2: Value2, ...

Example:
<extract>
USA: 25, China: 20, India: 15, UK: 10, France: 8
</extract>
""".strip(),
        ),
        (
            "user",
            """Observation: {observation}

Question: {question}

Please extract the relevant data values.""",
        ),
    ]
)


CALCULATE_ANSWER_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
Using the extracted data, perform any necessary calculations to answer the question.

Instructions:
- Show your calculation steps clearly
- Use correct mathematical operations
- Pay attention to the question (average, sum, difference, maximum, etc.)
- Provide a precise numerical answer if applicable
- Keep the answer concise (typically 1-10 words)

## Output Format ##

Show calculation inside <calculate> and </calculate> tags (if needed).
Provide final answer inside <answer> and </answer> tags.

Example:
<calculate>
Average = (25 + 20 + 15 + 10 + 8) / 5 = 78 / 5 = 15.6
</calculate>
<answer>
15.6
</answer>
""".strip(),
        ),
        (
            "user",
            """Extracted Data: {extracted_data}

Question: {question}

Please calculate and provide the answer.""",
        ),
    ]
)


CHECK_ANSWER_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a chart analysis expert with strong attention to detail.
Review the answer for potential mistakes.

Common mistakes to check:
- Incorrect data extraction from chart (misread values)
- Arithmetic errors in calculations
- Misunderstanding the question type (average vs. sum vs. difference)
- Wrong number of data points counted
- Incorrect units or scale interpretation
- Off-by-one errors

## Chart Information ##

Observation: {observation}
Extracted Data: {extracted_data}

## Output Format ##

If any mistakes are found, list each error clearly.
After listing mistakes (if any), conclude with **ONE** of the following exact phrases in all caps:
- If mistakes are found: `THE ANSWER IS INCORRECT.`
- If no mistakes are found: `THE ANSWER IS CORRECT.`

DO NOT write the corrected answer in this response. You only need to report mistakes.
""".strip(),
        ),
        (
            "user",
            """Question: {question}

Current Answer: {answer}

Calculation shown:
{calculation}

Please review this answer for correctness.""",
        ),
    ]
)


REFINE_ANSWER_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a chart analysis agent.
The previous answer had errors. Based on the feedback, provide a corrected answer.

Instructions:
- Re-examine the chart observation carefully
- Correct any data extraction errors by re-extracting if needed
- Fix calculation mistakes
- Address all points mentioned in the feedback

## Chart Observation ##

{observation}

## Output Format ##

If you need to re-extract data, provide it inside <extract> and </extract> tags.
Show corrected calculation inside <calculate> and </calculate> tags.
Provide corrected answer inside <answer> and </answer> tags.
""".strip(),
        ),
        (
            "user",
            """Question: {question}

## Previous Attempt ##

Extracted Data: {extracted_data}
Calculation: {calculation}
Answer: {answer}

## Feedback ##

{feedback}

Please provide the corrected answer.""",
        ),
    ]
)


# State Management

class ChartState(MessagesState):
    """State for chart reasoning workflow."""

    question: str
    image_path: str
    observation: str
    extracted_data: str
    calculation: str
    answer: str
    feedback: str
    num_turns: int
    messages: list[AnyMessage]


# Agent Class

class ChartQAAgent:
    """Chart QA agent with multi-step reasoning and refinement loop."""

    def __init__(
        self,
        max_turns: int = 3,
        debug: bool = False,
        endpoint: str | None = None,
        verl_replacement: Dict[str, Any] | None = None,
    ):
        self.debug = debug
        self.max_turns = max_turns

        if verl_replacement is not None:
            self.model_name: str = verl_replacement["model"]  # type: ignore
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
                temperature=verl_replacement["temperature"],
                max_retries=0,
                max_tokens=1024,
            )
        else:
            self.model_name: str = os.environ.get("MODEL", "Qwen/Qwen2-VL-2B-Instruct")
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint or os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
                temperature=0.0,
                max_retries=1,
                max_tokens=1024,
            )

    def invoke_prompt(self, prompt: Any) -> AnyMessage:
        """Invoke LLM with prompt (with debug printing)."""
        if self.debug:
            for message in prompt.messages:
                termcolor.cprint(message.pretty_repr(), "blue")

        try:
            result = self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Failed to invoke prompt: {e}")
            # Fallback to create a random response
            result = self.llm.invoke([HumanMessage(content="Please provide a reasonable answer.")])

        if self.debug:
            termcolor.cprint(result.pretty_repr(), "green")

        return result  # type: ignore

    def invoke_prompt_with_image(self, prompt_text: str, image_path: str) -> str:
        """Invoke vision-language model with image."""
        # Construct multimodal message
        if not image_path.startswith("file://"):
            image_path = f"file://{os.path.abspath(image_path)}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_path}},
                ],
            }
        ]

        if self.debug:
            termcolor.cprint(f"[VLM Call] {prompt_text[:100]}...", "blue")

        try:
            result = self.llm.invoke(messages)
            response = result.content if hasattr(result, "content") else str(result)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to invoke VLM: {e}")
            response = "<observe>Unable to analyze chart</observe>"

        if self.debug:
            termcolor.cprint(f"[VLM Response] {response[:200]}...", "green")

        return response  # type: ignore

    def extract_content(self, text: str, tag: str) -> str:
        """Extract content between XML-style tags."""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def analyze_chart(self, state: ChartState) -> ChartState:
        """Step 1: Observe and describe the chart."""
        prompt = ANALYZE_CHART_PROMPT.invoke({"question": state["question"]})
        prompt_text = prompt.messages[1].content  # Get the user message text

        # Call VLM with image
        result_text = self.invoke_prompt_with_image(prompt_text, state["image_path"])

        # Extract observation
        observation = self.extract_content(result_text, "observe")
        if not observation:
            # Fallback if tags not used
            observation = result_text

        return {  # type: ignore
            **state,
            "observation": observation,
            "num_turns": 1,
            "messages": [HumanMessage(content=result_text)],
        }

    def extract_data(self, state: ChartState) -> ChartState:
        """Step 2: Extract specific data values."""
        prompt = EXTRACT_DATA_PROMPT.invoke(
            {
                "observation": state["observation"],
                "question": state["question"],
            }
        )
        result = self.invoke_prompt(prompt)

        # Extract data
        extracted_data = self.extract_content(result.content, "extract")  # type: ignore
        if not extracted_data:
            extracted_data = result.content  # type: ignore

        return {  # type: ignore
            **state,
            "extracted_data": extracted_data,
            "messages": [*state.get("messages", []), result],
        }

    def calculate_answer(self, state: ChartState) -> ChartState:
        """Step 3: Calculate and provide answer."""
        prompt = CALCULATE_ANSWER_PROMPT.invoke(
            {
                "extracted_data": state["extracted_data"],
                "question": state["question"],
            }
        )
        result = self.invoke_prompt(prompt)

        # Extract calculation and answer
        calculation = self.extract_content(result.content, "calculate")  # type: ignore
        answer = self.extract_content(result.content, "answer")  # type: ignore

        if not answer:
            # Fallback if answer tag not found
            answer = result.content  # type: ignore

        return {  # type: ignore
            **state,
            "calculation": calculation,
            "answer": answer,
            "messages": [*state.get("messages", []), result],
        }

    def check_answer(self, state: ChartState) -> ChartState:
        """Step 4: Verify answer quality."""
        prompt = CHECK_ANSWER_PROMPT.invoke(
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
        prompt = REFINE_ANSWER_PROMPT.invoke(
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

        # Re-extract data if provided
        new_extracted = self.extract_content(result.content, "extract")  # type: ignore
        if new_extracted:
            extracted_data = new_extracted
        else:
            extracted_data = state["extracted_data"]

        # New calculation
        new_calculation = self.extract_content(result.content, "calculate")  # type: ignore

        # New answer
        new_answer = self.extract_content(result.content, "answer")  # type: ignore
        if not new_answer:
            new_answer = result.content  # type: ignore

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
        if state["messages"] and isinstance(state["messages"][-1], BaseMessage):
            last_message = state["messages"][-1]
            if "THE ANSWER IS CORRECT" in last_message.content:  # type: ignore
                if "THE ANSWER IS INCORRECT" in last_message.content:  # type: ignore
                    # Both found, check which is last
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

        # Conditional: continue or refine
        builder.add_conditional_edges(
            "check_answer",
            self.should_continue,  # type: ignore
        )

        # Refinement loop
        builder.add_edge("refine_answer", "extract_data")

        return builder.compile()  # type: ignore


def evaluate_answer(predicted: str, ground_truth: str, raise_on_error: bool = False) -> float:
    """Evaluate answer accuracy."""
    try:
        # Normalize strings
        pred = predicted.lower().strip()
        gt = ground_truth.lower().strip()

        # Exact match
        if pred == gt:
            return 1.0

        # Try numeric comparison
        try:
            pred_num = float(pred.replace(",", ""))
            gt_num = float(gt.replace(",", ""))
            # Allow small relative error
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
        else:
            logger.exception(f"Error evaluating answer: {e}")
            return 0.0


class LitChartQAAgent(agl.LitAgent[Dict[str, Any]]):
    """AgentLightning wrapper for ChartQA agent."""

    def __init__(
        self,
        trained_agents: Optional[str] = r"analyze|extract|calculate",
        val_temperature: Optional[float] = None,
        max_turns: int = 3,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.chartqa_dir = os.environ.get("CHARTQA_DATA_DIR", "data")
        self.max_turns = max_turns

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        """Execute agent rollout on a ChartQA task."""
        question = task["question"]
        start_time = time.time()
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        # Construct full image path
        image_path = os.path.join(self.chartqa_dir, task["image_path"])
        ground_truth = task["answer"]

        if not os.path.exists(image_path):
            logger.error(f"Image {image_path} does not exist. Skipping.")
            return None

        rollout_id = rollout.rollout_id

        logger.info(f"[Rollout {rollout_id}] Question: {question}")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")
        logger.info(f"[Rollout {rollout_id}] Image: {image_path}")

        # Create agent
        agent = ChartQAAgent(
            max_turns=self.max_turns,
            debug=False,
            endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),  # type: ignore
            verl_replacement=(
                {"model": llm.model, **llm.sampling_parameters}
                if rollout.mode == "train"
                else {
                    "model": llm.model,
                    "temperature": (
                        self.val_temperature
                        if self.val_temperature is not None
                        else llm.sampling_parameters.get("temperature", 0.0)
                    ),
                }
            ),
        ).graph()

        try:
            # Run agent
            handler = self.tracer.get_langchain_handler()
            result = agent.invoke(  # type: ignore
                {"question": question, "image_path": image_path},  # type: ignore
                {"callbacks": [handler] if handler else [], "recursion_limit": 100},
            )
        except Exception as e:
            logger.exception(f"[Rollout {rollout_id}] Error during agent invocation: {e}")
            return None

        predicted_answer = result["answer"]
        logger.info(f"[Rollout {rollout_id}] Predicted Answer: {predicted_answer}")

        end_time_rollout = time.time()

        # Evaluate
        reward = evaluate_answer(predicted_answer, ground_truth, raise_on_error=False)
        logger.info(f"[Rollout {rollout_id}] Reward: {reward}")

        end_time_eval = time.time()

        logger.info(f"[Rollout {rollout_id}] Time taken for rollout: {end_time_rollout - start_time:.2f} seconds")
        logger.info(f"[Rollout {rollout_id}] Time taken for evaluation: {end_time_eval - end_time_rollout:.2f} seconds")

        return reward


def debug_chartqa_agent():
    """Debug function to test agent with sample data."""
    chartqa_dir = os.environ.get("CHARTQA_DATA_DIR", "data")
    test_data_path = os.path.join(chartqa_dir, "test_chartqa.parquet")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file {test_data_path} does not exist. Please run prepare_data.py first.")

    df = pd.read_parquet(test_data_path).head(5)  # type: ignore
    test_data = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore
    print("Debug data:", test_data[0])

    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
                model=os.environ.get("MODEL", "Qwen/Qwen2-VL-2B-Instruct"),
                sampling_parameters={"temperature": 0.0},
            )
        },
    )
    trainer.dev(LitChartQAAgent(), test_data)


if __name__ == "__main__":
    debug_chartqa_agent()
