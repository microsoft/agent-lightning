"""ChartQA agent demonstrating multi-step visual reasoning with refinement loop.

This agent analyzes charts and answers questions using a multi-turn workflow:
1. analyze_chart: Observe and describe the chart
2. extract_data: Extract specific data values
3. calculate_answer: Perform calculations and provide answer
4. check_answer: Verify the answer quality
5. refine_answer: (conditional) Refine if errors detected
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional, cast

import pandas as pd
import termcolor
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from prompts import (
    ANALYZE_CHART_PROMPT,
    CALCULATE_ANSWER_PROMPT,
    CHECK_ANSWER_PROMPT,
    EXTRACT_DATA_PROMPT,
    REFINE_ANSWER_PROMPT,
)

import agentlightning as agl
from agentlightning.adapter.multimodal import encode_image_to_base64

agl.setup_logging(apply_to=[__name__])

logger = logging.getLogger(__name__)


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


class ChartQAAgent:
    """Chart QA agent with multi-step reasoning and refinement loop."""

    def __init__(
        self,
        max_turns: int = 3,
        debug: bool = False,
        endpoint: str | None = None,
        verl_replacement: Dict[str, Any] | None = None,
        use_base64_images: bool = False,
    ):
        self.debug = debug
        self.max_turns = max_turns
        self.use_base64_images = use_base64_images
        if verl_replacement is not None:
            self.model_name: str = verl_replacement["model"]  # type: ignore
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
                temperature=verl_replacement["temperature"],
                max_retries=2,
                max_tokens=1024,
                timeout=300,
            )
        else:
            self.model_name = os.environ.get("MODEL", "Qwen/Qwen2-VL-2B-Instruct")
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint or os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
                temperature=0.0,
                max_retries=1,
                max_tokens=512,
            )

    def invoke_prompt(self, prompt: Any) -> AnyMessage:
        """Invoke LLM with prompt."""
        if self.debug:
            for message in prompt.messages:
                termcolor.cprint(message.pretty_repr(), "blue")

        try:
            result = self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Failed to invoke prompt: {e}")
            result = self.llm.invoke([HumanMessage(content="Please provide a reasonable answer.")])

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
            "extracted_data": extracted_data,
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
            answer = result.content  # type: ignore

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
        if state["messages"] and isinstance(state["messages"][-1], BaseMessage):
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

        return builder.compile()  # type: ignore


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


class LitChartQAAgent(agl.LitAgent[Dict[str, Any]]):
    """AgentLightning wrapper for ChartQA agent."""

    def __init__(
        self,
        trained_agents: Optional[str] = r"analyze|extract|calculate",
        val_temperature: Optional[float] = None,
        max_turns: int = 3,
        use_base64_images: bool = False,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.max_turns = max_turns
        self.use_base64_images = use_base64_images
        default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.chartqa_dir = os.environ.get("CHARTQA_DATA_DIR", default_data_dir)

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        """Execute agent rollout on a ChartQA task."""
        question = task["question"]
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        image_path = os.path.join(self.chartqa_dir, task["image_path"])
        ground_truth = task["answer"]

        if not os.path.exists(image_path):
            logger.error(f"Image {image_path} does not exist. Skipping.")
            return None

        rollout_id = rollout.rollout_id

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
            use_base64_images=self.use_base64_images,
        ).graph()

        try:
            handler = self.tracer.get_langchain_handler()
            result = agent.invoke(  # type: ignore
                {"question": question, "image_path": image_path},  # type: ignore
                {"callbacks": [handler] if handler else [], "recursion_limit": 100},
            )
        except Exception as e:
            import traceback

            error_msg = f"[Rollout {rollout_id}] Error during agent invocation: {e}\n{traceback.format_exc()}"
            logger.exception(error_msg)
            return None

        predicted_answer = result["answer"]
        reward = evaluate_answer(predicted_answer, ground_truth, raise_on_error=False)

        return reward


def create_llm_proxy_for_chartqa(vllm_endpoint: str, port: int = 8081) -> agl.LLMProxy:
    """Create LLMProxy configured for ChartQA with token ID capture."""
    store = agl.LightningStoreThreaded(agl.InMemoryLightningStore())

    llm_proxy = agl.LLMProxy(
        port=port,
        store=store,
        model_list=[
            {
                "model_name": "Qwen/Qwen2-VL-2B-Instruct",
                "litellm_params": {
                    "model": "hosted_vllm/Qwen/Qwen2-VL-2B-Instruct",
                    "api_base": vllm_endpoint,
                },
            }
        ],
        callbacks=["return_token_ids"],
        launch_mode="thread",
    )

    return llm_proxy


def debug_chartqa_agent():
    """Debug function to test agent with cloud APIs (default).

    Usage:
        python chartqa_agent.py

    Environment variables:
        MODEL: Model name (default: gpt-4o)
        OPENAI_API_BASE: API endpoint (default: https://api.openai.com/v1)
        OPENAI_API_KEY: API key for authentication
    """
    chartqa_dir = os.environ.get("CHARTQA_DATA_DIR", "data")
    test_data_path = os.path.join(chartqa_dir, "test_chartqa.parquet")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file {test_data_path} does not exist. Please run prepare_data.py first.")

    df = pd.read_parquet(test_data_path).head(10)  # type: ignore
    test_data = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore

    model = os.environ.get("MODEL", "gpt-4o")
    endpoint = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    logger.info(f"Debug data: {len(test_data)} samples, model: {model}, endpoint: {endpoint}")

    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=endpoint,
                model=model,
                sampling_parameters={"temperature": 0.0},
            )
        },
    )
    trainer.dev(LitChartQAAgent(use_base64_images=True), test_data)


def debug_chartqa_agent_with_llm_proxy():
    """Debug function to test agent with local vLLM server and LLMProxy.

    Usage:
        USE_LLM_PROXY=1 OPENAI_API_BASE=http://localhost:8088/v1 python chartqa_agent.py
    """
    import asyncio

    import nest_asyncio

    nest_asyncio.apply()

    chartqa_dir = os.environ.get("CHARTQA_DATA_DIR", "data")
    test_data_path = os.path.join(chartqa_dir, "test_chartqa.parquet")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file {test_data_path} does not exist. Please run prepare_data.py first.")

    df = pd.read_parquet(test_data_path).head(10)  # type: ignore
    test_data = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore

    vllm_endpoint = os.environ.get("OPENAI_API_BASE", "http://localhost:8088/v1")
    model = os.environ.get("MODEL", "Qwen/Qwen2-VL-2B-Instruct")

    store = agl.LightningStoreThreaded(agl.InMemoryLightningStore())

    llm_proxy = agl.LLMProxy(
        port=8089,
        store=store,
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": f"hosted_vllm/{model}",
                    "api_base": vllm_endpoint,
                },
            }
        ],
        callbacks=["return_token_ids"],
        launch_mode="thread",
    )

    try:
        logger.info("Starting LLMProxy...")
        asyncio.run(llm_proxy.start())

        logger.info("Waiting for LLMProxy to be ready...")

        trainer = agl.Trainer(
            n_workers=2,
            store=store,
            llm_proxy=llm_proxy,
            strategy={"name": "shm", "main_thread": "algorithm", "managed_store": False},
            initial_resources={
                "main_llm": agl.LLM(
                    endpoint="http://localhost:8089/v1",
                    model=model,
                    sampling_parameters={"temperature": 0.0},
                )
            },
        )

        trainer.dev(LitChartQAAgent(), test_data)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during debug session: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down LLMProxy...")
        try:
            asyncio.run(llm_proxy.stop())
            logger.info("LLMProxy stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping LLMProxy: {e}")
        logger.info("Debug session completed")


if __name__ == "__main__":
    if os.environ.get("USE_LLM_PROXY", "").lower() in ("1", "true", "yes"):
        debug_chartqa_agent_with_llm_proxy()
    else:
        debug_chartqa_agent()
