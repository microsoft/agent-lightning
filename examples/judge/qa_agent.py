# Copyright (c) Microsoft. All rights reserved.

import argparse
import asyncio
import json
from typing import List, Optional, TypedDict, cast

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from rich.console import Console

import agentlightning as agl

console = Console()


class JudgeResponse(BaseModel):
    """Structured output from LLM judge."""

    accuracy: float = Field(description="Factual correctness (0-1)")
    completeness: float = Field(description="Covers all aspects (0-1)")
    clarity: float = Field(description="Easy to understand (0-1)")
    reasoning: str = Field(description="Justification for scores (2-3 sentences)")
    overall_score: float = Field(description="Weighted average (0-1)")


class QATask(TypedDict):
    """A question-answering task."""

    id: str
    question: str
    reference_answer: str
    domain: str


def baseline_prompt() -> agl.PromptTemplate:
    """Baseline prompt template that APO will optimize."""
    return agl.PromptTemplate(
        template="Answer this question: {question}",
        engine="f-string",
    )


async def llm_judge(question: str, answer: Optional[str], reference: str) -> float:
    """Evaluates answer quality using GPT-4-mini as judge.

    NOTE: This implementation uses a GENERIC rubric (Accuracy, Completeness, Clarity)
    for all tasks to keep the example simple and executable.

    For production systems with diverse domains, consider using DOMAIN-SPECIFIC rubrics:
    - Geography: Prioritize factual_accuracy (0.9) + conciseness (0.1)
    - Biology: Prioritize completeness (0.5) + scientific_accuracy (0.3) + clarity (0.2)
    - Code: Prioritize correctness (0.6) + efficiency (0.25) + readability (0.15)

    See the Recipe docs section "Generic vs Domain-Specific Rubrics" for implementation
    patterns and when to use each approach.

    Args:
        question: The question being answered
        answer: The agent's answer (can be None if generation failed)
        reference: Reference answer for comparison

    Returns:
        Overall score (0-1) weighted across criteria
    """
    if answer is None:
        console.print("[red]Error: No answer generated[/red]")
        return 0.0

    client = OpenAI()

    judge_prompt = f"""You are an expert evaluator for question-answering systems.

EVALUATION CRITERIA:
1. Accuracy (0-1): Does the answer contain factually correct information matching the reference?
   - 0.0: Completely wrong or contradicts reference
   - 0.5: Partially correct, missing key facts
   - 1.0: Fully accurate, aligns with reference

2. Completeness (0-1): Does it address all parts of the question?
   - 0.0: Ignores major aspects
   - 0.5: Covers main point but misses details
   - 1.0: Comprehensive coverage

3. Clarity (0-1): Is it well-structured and easy to understand?
   - 0.0: Confusing or incoherent
   - 0.5: Understandable but awkward
   - 1.0: Clear and well-organized

QUESTION: {question}

STUDENT ANSWER: {answer}

REFERENCE ANSWER: {reference}

Provide scores for each criterion and explain your reasoning in 2-3 sentences."""

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": judge_prompt}],
            response_format=JudgeResponse,
            temperature=0.0,  # Deterministic evaluation
        )

        judge_result = response.choices[0].message.parsed
        if judge_result is None:
            console.print("[red]Judge returned no result[/red]")
            return 0.0

        console.print(f"[yellow]=== Judge Evaluation ===[/yellow]")
        console.print(f"Accuracy: {judge_result.accuracy:.2f}")
        console.print(f"Completeness: {judge_result.completeness:.2f}")
        console.print(f"Clarity: {judge_result.clarity:.2f}")
        console.print(f"Reasoning: {judge_result.reasoning}")
        console.print(f"Overall: {judge_result.overall_score:.2f}")

        # Weight criteria: accuracy is most important
        overall = (
            0.5 * judge_result.accuracy + 0.3 * judge_result.completeness + 0.2 * judge_result.clarity
        )

        return overall

    except Exception as e:
        console.print(f"[red]Judge error: {e}[/red]")
        return 0.0


@agl.rollout
async def qa_agent(task: QATask, prompt_template: agl.PromptTemplate) -> float:
    """A question-answering agent evaluated by LLM judge.

    This agent demonstrates the light executor + heavy judge pattern:
    - Executor: GPT-4o-mini generates answers (cheap)
    - Judge: GPT-4o-mini evaluates quality (more expensive)

    APO optimizes the prompt_template parameter to improve answer quality.

    Args:
        task: QA task with question and reference answer
        prompt_template: Template to format the question (optimized by APO)

    Returns:
        Reward signal from judge (0-1)
    """
    client = OpenAI()

    # Format user message using prompt template
    user_message = prompt_template.format(question=task["question"])

    console.print(f"[green]=== Task ===[/green]")
    console.print(f"Question: {task['question']}")
    console.print(f"[blue]=== Prompt ===[/blue]")
    console.print(user_message)

    try:
        # Generate answer with lightweight model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=300,
        )

        answer = response.choices[0].message.content

        console.print(f"[cyan]=== Answer ===[/cyan]")
        console.print(answer)

    except Exception as e:
        console.print(f"[red]Generation error: {e}[/red]")
        answer = None

    # Evaluate with judge
    reward = await llm_judge(task["question"], answer, task["reference_answer"])

    console.print(f"[magenta]=== Reward ===[/magenta]")
    console.print(f"{reward:.2f}\n")

    return reward


def load_qa_tasks(filepath: str) -> agl.Dataset[QATask]:
    """Load QA tasks from JSONL file."""
    tasks: List[QATask] = []
    try:
        with open(filepath) as f:
            for line in f:
                task_data = json.loads(line)
                tasks.append(
                    QATask(
                        id=task_data["id"],
                        question=task_data["question"],
                        reference_answer=task_data["reference_answer"],
                        domain=task_data.get("domain", "general"),
                    )
                )
        console.print(f"[green]Loaded {len(tasks)} tasks from {filepath}[/green]")
    except FileNotFoundError:
        console.print(f"[red]Error: {filepath} not found. Run with --generate-data first.[/red]")
        raise

    return cast(agl.Dataset[QATask], tasks)


async def run_baseline(limit: int = 50):
    """Run baseline evaluation without optimization."""
    console.print("[bold green]=== Baseline Evaluation ===[/bold green]\n")

    runner = agl.LitAgentRunner(agl.AgentOpsTracer())
    store = agl.InMemoryLightningStore()
    prompt_template = baseline_prompt()

    # Load validation set for baseline test
    tasks = load_qa_tasks("qa_val.jsonl")[:limit]

    total_reward = 0.0
    with runner.run_context(agent=qa_agent, store=store):
        for i, task in enumerate(tasks):
            console.print(f"[bold]Task {i+1}/{len(tasks)}[/bold]")
            rollout = await runner.step(task, resources={"prompt_template": prompt_template})
            spans = await store.query_spans(rollout.rollout_id)
            reward = agl.find_final_reward(spans)
            total_reward += reward if reward is not None else 0.0

    avg_reward = total_reward / len(tasks)
    console.print(f"\n[bold green]=== Baseline Results ===[/bold green]")
    console.print(f"Average reward: {avg_reward:.2f}")
    console.print(f"Tasks evaluated: {len(tasks)}")


async def run_training(iterations: int = 10):
    """Run APO training to optimize prompts."""
    console.print("[bold green]=== APO Training ===[/bold green]\n")

    openai_client = AsyncOpenAI()

    # Load datasets
    train_tasks = load_qa_tasks("qa_train.jsonl")
    val_tasks = load_qa_tasks("qa_val.jsonl")

    # Configure APO
    algo = agl.APO(
        openai_client,
        val_batch_size=10,
        gradient_batch_size=4,
        beam_width=2,
        branch_factor=2,
        beam_rounds=iterations,
    )

    # Create trainer
    trainer = agl.Trainer(
        algorithm=algo,
        n_runners=8,
        initial_resources={"prompt_template": baseline_prompt()},
        adapter=agl.TraceToMessages(),
    )

    # Run training
    trainer.fit(agent=qa_agent, train_dataset=train_tasks, val_dataset=val_tasks)

    console.print(f"\n[bold green]=== Training Complete ===[/bold green]")
    console.print("Check trainer logs for best prompt and results")


async def run_eval(prompt_file: str):
    """Evaluate with optimized prompt."""
    console.print("[bold green]=== Evaluation with Optimized Prompt ===[/bold green]\n")

    # Load optimized prompt
    with open(prompt_file) as f:
        prompt_text = f.read()
    optimized_prompt = agl.PromptTemplate(template=prompt_text, engine="f-string")

    runner = agl.LitAgentRunner(agl.AgentOpsTracer())
    store = agl.InMemoryLightningStore()

    tasks = load_qa_tasks("qa_val.jsonl")

    total_reward = 0.0
    with runner.run_context(agent=qa_agent, store=store):
        for i, task in enumerate(tasks):
            console.print(f"[bold]Task {i+1}/{len(tasks)}[/bold]")
            rollout = await runner.step(task, resources={"prompt_template": optimized_prompt})
            spans = await store.query_spans(rollout.rollout_id)
            reward = agl.find_final_reward(spans)
            total_reward += reward if reward is not None else 0.0

    avg_reward = total_reward / len(tasks)
    console.print(f"\n[bold green]=== Evaluation Results ===[/bold green]")
    console.print(f"Average reward: {avg_reward:.2f}")


async def run_debug(task_id: int):
    """Debug a single task with trace visualization."""
    console.print(f"[bold green]=== Debug Task {task_id} ===[/bold green]\n")

    runner = agl.LitAgentRunner(agl.AgentOpsTracer())
    store = agl.InMemoryLightningStore()
    prompt_template = baseline_prompt()

    tasks = load_qa_tasks("qa_val.jsonl")
    task = tasks[task_id]

    with runner.run_context(agent=qa_agent, store=store):
        rollout = await runner.step(task, resources={"prompt_template": prompt_template})

        # Get spans and convert to messages
        spans = await store.query_spans(rollout.rollout_id)
        adapter = agl.TraceToMessages()
        messages = adapter.adapt(spans)

        console.print(f"\n[bold purple]=== Trace Messages ===[/bold purple]")
        for i, msg in enumerate(messages):
            console.print(f"Message {i}: {json.dumps(msg, indent=2)}")

        reward = agl.find_final_reward(spans)
        console.print(f"\n[bold purple]=== Final Reward ===[/bold purple]")
        console.print(reward)


def generate_sample_data():
    """Generate sample QA dataset for demo purposes."""
    console.print("[bold green]=== Generating Sample Data ===[/bold green]\n")

    train_data = [
        {
            "id": "qa_001",
            "question": "What is the capital of France?",
            "reference_answer": "Paris is the capital and largest city of France.",
            "domain": "geography",
        },
        {
            "id": "qa_002",
            "question": "What is photosynthesis?",
            "reference_answer": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.",
            "domain": "biology",
        },
        {
            "id": "qa_003",
            "question": "Who wrote 'Hamlet'?",
            "reference_answer": "William Shakespeare wrote 'Hamlet' around 1600. It is one of his most famous tragedies.",
            "domain": "literature",
        },
        {
            "id": "qa_004",
            "question": "What is Newton's first law of motion?",
            "reference_answer": "Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and direction unless acted upon by an unbalanced force.",
            "domain": "physics",
        },
        {
            "id": "qa_005",
            "question": "What is the difference between a compiler and an interpreter?",
            "reference_answer": "A compiler translates entire source code into machine code before execution, while an interpreter executes code line by line, translating and running it simultaneously.",
            "domain": "computer_science",
        },
    ]

    val_data = [
        {
            "id": "qa_val_001",
            "question": "What is the process of mitosis?",
            "reference_answer": "Mitosis is the process by which a single cell divides into two identical daughter cells, each containing the same number of chromosomes as the parent cell.",
            "domain": "biology",
        },
        {
            "id": "qa_val_002",
            "question": "Who painted the Mona Lisa?",
            "reference_answer": "Leonardo da Vinci painted the Mona Lisa during the Renaissance period, around 1503-1519.",
            "domain": "art",
        },
    ]

    # Write train data
    with open("qa_train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    # Write val data
    with open("qa_val.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    console.print(f"Generated {len(train_data)} training examples -> qa_train.jsonl")
    console.print(f"Generated {len(val_data)} validation examples -> qa_val.jsonl")


async def main():
    parser = argparse.ArgumentParser(description="QA Agent with LLM Judge")
    parser.add_argument(
        "--mode",
        choices=["baseline", "train", "eval", "debug", "generate-data"],
        default="baseline",
        help="Execution mode",
    )
    parser.add_argument("--limit", type=int, default=50, help="Limit number of tasks for baseline")
    parser.add_argument("--iterations", type=int, default=10, help="Number of APO iterations")
    parser.add_argument("--prompt", type=str, default="outputs/best_prompt.txt", help="Path to optimized prompt")
    parser.add_argument("--task", type=int, default=0, help="Task ID for debug mode")

    args = parser.parse_args()

    if args.mode == "generate-data":
        generate_sample_data()
    elif args.mode == "baseline":
        await run_baseline(args.limit)
    elif args.mode == "train":
        await run_training(args.iterations)
    elif args.mode == "eval":
        await run_eval(args.prompt)
    elif args.mode == "debug":
        await run_debug(args.task)


if __name__ == "__main__":
    asyncio.run(main())
