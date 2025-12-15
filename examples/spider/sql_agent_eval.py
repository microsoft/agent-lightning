# Copyright (c) Microsoft. All rights reserved.

"""Sample code that demonstrates an SQL agent using LangGraph and LangChain,
trainable with Agent-lightning.

Adapted from https://python.langchain.com/docs/tutorials/sql_qa/
as well as https://langchain-ai.github.io/langgraph/tutorials/sql-agent/
"""


from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import threading
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional, cast

import pandas as pd
import termcolor
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from spider_eval.exec_eval import eval_exec_match

import agentlightning as agl

# agl.setup_logging(apply_to=[__name__])

logger = logging.getLogger(__name__)

# Default knobs for convenience; change here to adjust global defaults.
DEFAULT_MODE = "eval"  # "debug" or "eval"
DEFAULT_NUM_SAMPLES = -1  # use -1 to mean "all samples"
DEFAULT_OUTPUT_PATH = "outputs/qwen3-4b-dev.jsonl"  # e.g., "outputs/qwen3-4b-dev.jsonl"
DEFAULT_USE_TEST_SPLIT = False
DEFAULT_CONCURRENCY = 4


def _load_env_from_file(env_path: str) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ if not already set."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Skip export statements
                if key.lower().startswith("export "):
                    key = key[7:].strip()
                # Environment variables already set take precedence
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as e:  # pragma: no cover - defensive load
        logger.warning("Failed to load env file %s: %s", env_path, e)


# Load project-level .env so OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_MODEL are available.
_project_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
_load_env_from_file(_project_env)


class TokenCounterHandler(BaseCallbackHandler):
    """Lightweight callback to aggregate token usage per LLM run."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):  # type: ignore[override]
        usage = None
        # langchain_openai returns token_usage or usage inside llm_output
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
        if not usage:
            return
        self.input_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        self.output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        self.total_tokens += usage.get("total_tokens", self.input_tokens + self.output_tokens)


WRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
     Given an input question, create a syntactically correct {dialect} query to run to help find the answer.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
GENERATED QUERY
```
""".strip(),
        ),
        ("user", "Question: {input}"),
    ]
)


CHECK_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Explicit query execution failures
- Clearly unreasoable query execution results

## Table Schema ##

{table_info}

## Output Format ##

If any mistakes from the list above are found, list each error clearly.
After listing mistakes (if any), conclude with **ONE** of the following exact phrases in all caps and without surrounding quotes:
- If mistakes are found: `THE QUERY IS INCORRECT.`
- If no mistakes are found: `THE QUERY IS CORRECT.`

DO NOT write the corrected query in the response. You only need to report the mistakes.
""".strip(),
        ),
        (
            "user",
            """Question: {input}

Query:

```{dialect}
{query}
```

Execution result:

```
{execution}
```""",
        ),
    ]
)


REWRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
Rewrite the previous {dialect} query to fix errors based on the provided feedback.
The goal is to answer the original question.
Make sure to address all points in the feedback.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
REWRITTEN QUERY
```
""".strip(),
        ),
        (
            "user",
            """Question: {input}

## Previous query ##

```{dialect}
{query}
```

## Previous execution result ##

```
{execution}
```

## Feedback ##

{feedback}

Please rewrite the query to address the feedback.""",
        ),
    ]
)


class State(MessagesState):
    question: str
    query: str
    execution: str
    answer: str
    feedback: str
    num_turns: int
    messages: list[AnyMessage]


class SQLAgent:

    def __init__(
        self,
        db: str,
        max_turns: int = 5,
        debug: bool = False,
        db_schema: str | None = None,
        endpoint: str | None = None,
        verl_replacement: Dict[str, Any] | None = None,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
    ):
        self.db = SQLDatabase.from_uri(db)  # type: ignore
        self.db_schema = db_schema
        self.debug = debug
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate
        if verl_replacement is not None:
            self.model_name: str = verl_replacement["model"]  # type: ignore
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=verl_replacement["temperature"],
                max_retries=0,
                max_tokens=2048,
                model_kwargs={"extra_body": {"enable_thinking": False}},
                extra_headers={"X-DashScope-Enable-Thinking": "false"},  # 使用headers方式传递参数
            )
        else:
            self.model_name: str = os.environ.get("MODEL", "gpt-4.1-mini")
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint or os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"],
                temperature=0,
                max_retries=1,
                max_tokens=2048,
                model_kwargs={"extra_body": {"enable_thinking": False}},
                extra_headers={"X-DashScope-Enable-Thinking": "false"},  # 使用headers方式传递参数
            )

    def get_table_info(self) -> str:
        """Get the table information in a human-readable format."""
        try:
            table_info = self.db.get_table_info()
            if len(table_info) > self.table_info_truncate:
                table_info = table_info[: self.table_info_truncate] + "\n... (truncated)"
            return table_info
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            if self.db_schema:
                if len(self.db_schema) > self.table_info_truncate:
                    return self.db_schema[: self.table_info_truncate] + "\n... (truncated)"
                return self.db_schema
            return "No schema available."

    def invoke_prompt(self, prompt: Any) -> AnyMessage:
        if self.debug:
            for message in prompt.messages:
                termcolor.cprint(message.pretty_repr(), "blue")

        try:
            result = self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Failed to invoke prompt: {e}")
            # FIXME: fallback to create a random trajectory
            result = self.llm.invoke([HumanMessage(content="Please create a random SQL query as an example.")])

        if self.debug:
            termcolor.cprint(result.pretty_repr(), "green")

        return result  # type: ignore

    def truncate_execution(self, execution: str) -> str:
        """Truncate the execution result to a reasonable length."""
        if len(execution) > self.execution_truncate:
            return execution[: self.execution_truncate] + "\n... (truncated)"
        return execution

    def parse_query(self, message: AnyMessage) -> str | None:
        result: str | None = None
        for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):  # type: ignore
            result = match.group(1).strip()  # type: ignore
        return result  # type: ignore

    def write_query(self, state: State) -> State:
        """Generate SQL query to fetch information."""
        prompt: Any = WRITE_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)  # type: ignore

        query = self.parse_query(result) or result.content  # type: ignore

        return {  # type: ignore
            **state,
            "query": query,  # type: ignore
            "num_turns": 1,
            "messages": [*prompt.messages, result],
        }

    def execute_query(self, state: State) -> State:
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        execution_result = execute_query_tool.invoke(state["query"])  # type: ignore
        if not isinstance(execution_result, str):
            # Convert to string if it's not already
            execution_result = str(execution_result)
        if self.debug:
            termcolor.cprint(execution_result, "yellow")
        return {**state, "execution": execution_result}

    def check_query(self, state: State) -> State:
        """Check the SQL query for correctness."""
        prompt: Any = CHECK_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execution(state["execution"]),
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)  # type: ignore

        res = {  # type: ignore
            **state,
            "feedback": result.content,  # type: ignore
            "messages": [*state.get("messages", []), *prompt.messages, result],
        }
        return res  # type: ignore

    def rewrite_query(self, state: State) -> State:
        """Rewrite SQL query if necessary."""
        prompt: Any = REWRITE_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execution(state["execution"]),
                "feedback": state["feedback"],
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)  # type: ignore

        rewritten_query = self.parse_query(result)  # type: ignore

        return {
            **state,
            "query": rewritten_query or state["query"],
            "num_turns": state.get("num_turns", 0) + 1,
            "messages": [*prompt.messages, result],  # clear previous prompts
        }

    def should_continue(self, state: State) -> Literal[END, "rewrite_query"]:  # type: ignore
        """Determine if the agent should continue based on the result."""
        if state["messages"] and isinstance(state["messages"][-1], BaseMessage):  # type: ignore
            last_message = state["messages"][-1]
            if "THE QUERY IS CORRECT" in last_message.content:  # type: ignore
                if "THE QUERY IS INCORRECT" in last_message.content:  # type: ignore
                    # Both correct and incorrect messages found
                    # See which is the last one
                    correct_index = last_message.content.rfind("THE QUERY IS CORRECT")  # type: ignore
                    incorrect_index = last_message.content.rfind("THE QUERY IS INCORRECT")  # type: ignore
                    if correct_index > incorrect_index:
                        return END
                else:
                    return END

        if state.get("num_turns", 0) >= self.max_turns:
            return END

        return "rewrite_query"

    def graph(self) -> CompiledStateGraph[State]:
        builder = StateGraph(State)
        builder.add_node(self.write_query)  # type: ignore
        builder.add_node(self.execute_query)  # type: ignore
        builder.add_node(self.check_query)  # type: ignore
        builder.add_node(self.rewrite_query)  # type: ignore

        builder.add_edge(START, "write_query")
        builder.add_edge("write_query", "execute_query")
        builder.add_edge("execute_query", "check_query")
        builder.add_conditional_edges(
            "check_query",
            self.should_continue,  # type: ignore
        )
        builder.add_edge("rewrite_query", "execute_query")

        return builder.compile()  # type: ignore


def evaluate_query(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
    # TODO(yuge): Maybe we can evaluate intermediate queries and assign more precise rewards.

    # included in the original evaluation script
    # query = query.replace("value", "1")

    try:
        database = os.path.abspath(database)
        if not os.path.exists(database):
            raise FileNotFoundError(f"Database file {database} does not exist.")

        # Parameters following the default setting
        exec_score = eval_exec_match(
            db=database,
            p_str=query,
            g_str=ground_truth,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
        if exec_score == 1:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        if raise_on_error:
            raise
        else:
            logger.exception(f"Error evaluating query: {e}")
            return 0.0


class LitSQLAgent(agl.LitAgent[Dict[str, Any]]):

    def __init__(
        self,
        trained_agents: Optional[str] = r"write",
        val_temperature: Optional[float] = None,
        max_turns: int = 3,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.spider_dir = os.environ.get("VERL_SPIDER_DATA_DIR", "data")
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        question = task["question"]
        start_time = time.time()
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        if rollout.mode == "train":
            original_db_path = os.path.join(self.spider_dir, "database", task["db_id"], task["db_id"] + ".sqlite")
        else:
            original_db_path = os.path.join(self.spider_dir, "test_database", task["db_id"], task["db_id"] + ".sqlite")
        ground_truth = task["query"]

        if not os.path.exists(original_db_path):
            logger.error(f"Database {original_db_path} does not exist. Skipping.")
            return None

        schema_path = os.path.join(os.path.dirname(original_db_path), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema = f.read()
        else:
            logger.error("Schema file not found: %s", schema_path)
            schema = "No schema available."

        rollout_id = rollout.rollout_id

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)
            logger.info(f"[Rollout {rollout_id}] Question: {question}")
            logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")

            # Run the agent
            agent = SQLAgent(
                "sqlite:///" + db_path,
                max_turns=self.max_turns,
                table_info_truncate=self.table_info_truncate,
                execution_truncate=self.execution_truncate,
                debug=False,
                db_schema=schema,
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
                # Required to make the langchain tracing work
                handler = self.tracer.get_langchain_handler()
                result = agent.invoke(  # type: ignore
                    {"question": question},  # type: ignore
                    {"callbacks": [handler] if handler else [], "recursion_limit": 100},
                )
            except Exception as e:
                logger.exception(f"[Rollout {rollout_id}] Error during agent invocation: {e}")
                return

            logger.info(f"[Rollout {rollout_id}] Generated Query: {result['query']}")

        end_time_rollout = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)

            reward = evaluate_query(result["query"], ground_truth, db_path, raise_on_error=False)
            logger.info("[Rollout %s] Reward: %s", rollout_id, reward)

        end_time_eval = time.time()

        logger.info("[Rollout %s] Time taken for rollout: %.2f seconds", rollout_id, end_time_rollout - start_time)
        logger.info(
            "[Rollout %s] Time taken for evaluation: %.2f seconds", rollout_id, end_time_eval - end_time_rollout
        )

        return reward


def debug_sql_agent():
    spider_dev_data_path = os.path.join(os.environ.get("VERL_SPIDER_DATA_DIR", "data"), "dev.parquet")
    if not os.path.exists(spider_dev_data_path):
        raise FileNotFoundError(f"Spider dev data file {spider_dev_data_path} does not exist.")
    df = pd.read_parquet(spider_dev_data_path).head(10)  # type: ignore
    df = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore
    print("Debug data:", df)

    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=os.environ["OPENAI_API_BASE"],
                model=os.environ.get("OPENAI_MODEL", "qwen3-4b"),
                sampling_parameters={"temperature": 0.7},
            )
        },
    )
    trainer.dev(LitSQLAgent(), df)


def evaluate_sql_agent(
    num_samples: int | None = None,
    output_path: str | None = None,
    use_test_split: bool = False,
    concurrency: int = 1,
) -> None:
    """Run the agent on Spider and compute execution accuracy; optionally save predictions.

    Args:
        num_samples: number of samples to run (None/-1 means all).
        output_path: optional JSONL path to save predictions.
        use_test_split: whether to read from test_database instead of database.
        concurrency: number of parallel workers for evaluation.
    """
    spider_dir = os.environ.get("VERL_SPIDER_DATA_DIR", "data")
    split = "dev"
    parquet_path = os.path.join(spider_dir, f"{split}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Spider data file {parquet_path} does not exist.")

    df = pd.read_parquet(parquet_path)  # type: ignore
    if num_samples is not None and num_samples > 0:
        df = df.head(num_samples)
    df_records = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore

    # Respect OPENAI_MODEL if provided; fallback to MODEL.
    if "OPENAI_MODEL" in os.environ and "MODEL" not in os.environ:
        os.environ["MODEL"] = os.environ["OPENAI_MODEL"]

    rows: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    token_totals = {"input": 0, "output": 0, "total": 0}
    concurrency = max(1, concurrency)

    lock = threading.Lock()
    throttle_lock = threading.Lock()
    last_call_ts = [0.0]
    min_interval = 0.5  # seconds between starting any two requests to respect rate limits

    def process_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
        db_id: str = row["db_id"]
        question: str = row["question"]
        gold_query: str = row["query"]

        split_folder = "test_database" if use_test_split else "database"
        original_db_path = os.path.join(spider_dir, split_folder, db_id, f"{db_id}.sqlite")
        if not os.path.exists(original_db_path):
            logger.error("Database %s does not exist. Skipping sample.", original_db_path)
            return None

        schema_path = os.path.join(os.path.dirname(original_db_path), "schema.sql")
        schema = "No schema available."
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema = f.read()

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)
            agent = SQLAgent(
                "sqlite:///" + db_path,
                max_turns=3,
                db_schema=schema,
                execution_truncate=2048,
                table_info_truncate=2048,
            ).graph()

            token_handler = TokenCounterHandler()
            try:
                # Global throttle: ensure at least min_interval between requests.
                with throttle_lock:
                    now = time.time()
                    wait = min_interval - (now - last_call_ts[0])
                    if wait > 0:
                        time.sleep(wait)
                    last_call_ts[0] = time.time()

                result = agent.invoke(  # type: ignore
                    {"question": question},
                    {"recursion_limit": 100, "callbacks": [token_handler]},
                )
                pred_query = result["query"]
            except Exception as e:  # pragma: no cover - defensive logging
                logger.exception("Failed to generate query for %s: %s", db_id, e)
                pred_query = ""

        reward = evaluate_query(pred_query, gold_query, original_db_path, raise_on_error=False) if pred_query else 0.0

        return {
            "db_id": db_id,
            "question": question,
            "gold": gold_query,
            "pred": pred_query,
            "reward": reward,
            "tokens": {
                "input": token_handler.input_tokens,
                "output": token_handler.output_tokens,
                "total": token_handler.total_tokens
                or (token_handler.input_tokens + token_handler.output_tokens),
            },
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process_row, row) for row in df_records]
        for fut in as_completed(futures):
            res = fut.result()
            if not res:
                continue
            rows.append(
                {
                    "db_id": res["db_id"],
                    "question": res["question"],
                    "gold": res["gold"],
                    "pred": res["pred"],
                    "reward": res["reward"],
                }
            )
            correct += int(res["reward"] == 1.0)
            total += 1
            with lock:
                token_totals["input"] += res["tokens"]["input"]
                token_totals["output"] += res["tokens"]["output"]
                token_totals["total"] += res["tokens"]["total"]
            logger.info("[Eval] db_id=%s reward=%.1f pred=%s", res["db_id"], res["reward"], res["pred"])

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Saved predictions to %s", output_path)

    if total == 0:
        logger.warning("No samples evaluated.")
    else:
        acc = correct / total
        logger.info("Execution accuracy: %d/%d = %.3f", correct, total, acc)
        logger.info(
            "Token usage (approx): input=%d output=%d total=%d",
            token_totals["input"],
            token_totals["output"],
            token_totals["total"],
        )


if __name__ == "__main__":
    # Show rollout progress and rewards on stdout when running this script directly.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run or evaluate the Spider SQL agent.")
    parser.add_argument(
        "--mode",
        choices=["debug", "eval"],
        default=DEFAULT_MODE,
        help="debug: small sample run; eval: compute exec accuracy.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of samples to run. Use -1 for all.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Optional path to save predictions as JSONL when mode=eval.",
    )
    parser.add_argument(
        "--use-test-split",
        action="store_true",
        default=DEFAULT_USE_TEST_SPLIT,
        help="Use Spider test_database instead of database when evaluating.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of parallel workers for evaluation (default: 1).",
    )
    args = parser.parse_args()

    if args.mode == "eval":
        evaluate_sql_agent(
            num_samples=None if args.num_samples and args.num_samples < 0 else args.num_samples,
            output_path=args.output,
            use_test_split=args.use_test_split,
            concurrency=args.concurrency,
        )
    else:
        debug_sql_agent()
