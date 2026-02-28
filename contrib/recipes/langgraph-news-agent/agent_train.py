# Copyright (c) Microsoft. All rights reserved.

import os
import re
from datetime import datetime, timedelta
from typing import Annotated, List, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from newsapi import NewsApiClient
from openai import AsyncOpenAI

import agentlightning as agl

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")


# ===== Data Types =====
class AgentState(TypedDict):
    """Class for taking input of user input"""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class NewsTask(TypedDict):
    """Task definition for the news agent"""

    query: str  # The user query
    expected_keywords: List[str]  # Keywords we expect in a good response (for grading)


# ===== Tools =====
newsapi = NewsApiClient(NEWSAPI_KEY)


@tool
def get_news(query: str):
    """Tool for calling the newsapi"""
    today = datetime.now()
    thirty_days_ago = today - timedelta(days=30)

    from_date = thirty_days_ago.strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    all_articles = newsapi.get_everything(q=query, from_param=from_date, to=to_date, language="en", sort_by="relevancy")
    return all_articles


@tool
def sum_of_numbers(a: int, b: int):
    """Tool for summing two numbers"""
    return a + b


tools = [get_news, sum_of_numbers]


# ===== Grader Function (Critical for RL) =====
def grade_response(final_response: str, task: NewsTask) -> float:
    """Grade how well the agent performed (0.0 to 1.0)"""
    score = 0.0

    # FIX: Handle both string and list responses
    if isinstance(final_response, list):
        final_response = " ".join(str(item) for item in final_response)
    elif not isinstance(final_response, str):
        final_response = str(final_response)

    if task.get("expected_keywords"):
        keywords_found = sum(1 for keyword in task["expected_keywords"] if keyword.lower() in final_response.lower())
        score = keywords_found / len(task["expected_keywords"])

    if len(final_response) > 50:
        score += 0.2

    return min(score, 1.0)


# ===== Agent Logic (wrapped with @agl.rollout) =====
@agl.rollout
def news_agent_rollout(task: NewsTask, prompt_template: agl.PromptTemplate) -> float:
    """
    This is the main agent function that Agent Lightning will optimize.

    Args:
        task: The task containing the user query
        prompt_template: The prompt that will be optimized by APO

    Returns:
        float: Reward score (0.0 to 1.0)
    """

    # Create LLM with tools bound
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=1.0, max_tokens=500, timeout=None, max_retries=2, api_key=GOOGLE_API_KEY
    ).bind_tools(tools)

    # ===== FIX 1: BULLETPROOF PLACEHOLDER HANDLING =====
    # Auto-detect ALL placeholders in template
    placeholders = re.findall(r"\{([^}]+)\}", prompt_template.template)

    # Prepare safe variables dict
    safe_vars = {"query": task["query"]}

    # Fill any unknown placeholders with empty string
    for placeholder in placeholders:
        if placeholder not in safe_vars:
            safe_vars[placeholder] = ""

    # Format safely - no KeyError possible!
    system_prompt = prompt_template.template.format(**safe_vars)

    def query_node(state: AgentState) -> AgentState:
        system_message = SystemMessage(content=system_prompt)
        response = llm_model.invoke([system_message] + state["messages"])
        return {"messages": response}

    def should_continue(state: AgentState) -> Literal["continue", "exit"]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "continue"
        return "exit"

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("user_query_node", query_node)
    tool_node = ToolNode(tools)
    graph.add_node("use_tool_node", tool_node)

    graph.add_edge(START, "user_query_node")
    graph.add_conditional_edges("user_query_node", should_continue, {"continue": "use_tool_node", "exit": END})
    graph.add_edge("use_tool_node", "user_query_node")

    compiled_graph = graph.compile()

    # Run the agent
    input_state = {"messages": [("user", task["query"])]}
    final_state = None

    for state in compiled_graph.stream(input_state, stream_mode="values"):
        final_state = state

    # Extract final response
    final_response = ""
    if final_state and "messages" in final_state:
        last_message = final_state["messages"][-1]
        if hasattr(last_message, "content"):
            final_response = last_message.content
        else:
            final_response = str(last_message)

    # Grade the response and return reward
    reward = grade_response(final_response, task)

    return reward


# ===== Initial Prompt Template =====
def get_baseline_prompt_template() -> agl.PromptTemplate:
    """
    This is the initial prompt that APO will try to improve.
    """
    return agl.PromptTemplate(
        template="""You are a helpful assistant who answers user queries.

The user asked: {query}

Please provide a clear, accurate, and helpful response. Use the tools available to you when necessary.""",
        engine="f-string",
    )


# ===== Training Script =====
def train_agent():
    """
    Main training function using Agent Lightning's APO algorithm
    """

    # 1. Create training and validation datasets
    train_dataset = [
        NewsTask(
            query="Tell me news about Bitcoin", expected_keywords=["Bitcoin", "cryptocurrency", "price", "market"]
        ),
        NewsTask(
            query="What's happening with AI technology?",
            expected_keywords=["AI", "artificial intelligence", "technology"],
        ),
        NewsTask(
            query="Tell me about climate change news",
            expected_keywords=["climate", "environment", "carbon", "temperature"],
        ),
    ]

    val_dataset = [
        NewsTask(query="Latest news about Tesla", expected_keywords=["Tesla", "electric", "vehicle", "Elon"]),
        NewsTask(query="What's the sum of 25 and 37?", expected_keywords=["62", "sum", "25", "37"]),
    ]

    # 2. Initialize the APO algorithm
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # ===== FIX 2: LIMIT ITERATIONS =====
    algo = agl.APO(
        openai_client,
        gradient_batch_size=2,
        beam_width=2,
        beam_rounds=2,  # ONLY DO 2 ROUNDS (not infinite)
    )

    # 3. Create the Trainer
    trainer = agl.Trainer(
        algorithm=algo,
        n_runners=4,
        initial_resources={"prompt_template": get_baseline_prompt_template()},
        adapter=agl.TraceToMessages(),
    )

    # 4. Start training!
    print("Starting Agent Lightning training...")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Will run for {algo.beam_rounds} rounds with beam_width={algo.beam_width}")

    trainer.fit(agent=news_agent_rollout, train_dataset=train_dataset, val_dataset=val_dataset)

    # ===== FIX 3: PROPERLY EXTRACT THE BEST PROMPT =====
    print("\n" + "=" * 60)
    print(" Training complete!")
    print("=" * 60)

    try:
        # CORRECT METHOD: Use the APO's get_best_prompt() method
        # This accesses trainer.algorithm._history_best_prompt internally
        best_prompt_obj = trainer.algorithm.get_best_prompt()
        best_prompt_text = best_prompt_obj.template

        # Get the best score from the algorithm's internal tracking
        best_score = trainer.algorithm._history_best_score
        best_version = trainer.algorithm._history_best_version

        print(f"Best validation score: {best_score}")
        print(f"Best prompt version: {best_version}")
        print("\n OPTIMIZED PROMPT:")
        print("=" * 60)
        print(best_prompt_text)
        print("=" * 60)

        # Save to file
        with open("optimized_prompt.txt", "w") as f:
            f.write(best_prompt_text)

        print("\n Saved optimized prompt to: optimized_prompt.txt")

    except ValueError as e:
        # get_best_prompt() raises ValueError if no best prompt found
        print(f"\n No best prompt found: {e}")
        print("This may happen if training didn't complete successfully")
        print("Check AgentOps session replays for details")
    except Exception as e:
        print(f"\n Error extracting prompt: {e}")
        print("Check AgentOps session replays for successful prompts")

    print("\nTraining complete!")


if __name__ == "__main__":
    train_agent()
