# Copyright (c) Microsoft. All rights reserved.

"""CrewAI-based 20 Questions agents tailored for Agent-lightning demos.

This module wires up the player, answerer, and auxiliary tooling that power the
20 Questions examples under ``examples/tinker``.

It mirrors the high-level game loop used in the original Tinker Cookbook example,
but is far more complicated in that it uses an advanced agent orchestration framework (CrewAI)
and incorporates a simulated web search tool, as well as interactions between multiple agents.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, cast

from crewai import LLM as CrewLLM
from crewai import Agent as CrewAgent
from crewai import BaseLLM
from crewai.flow import Flow, listen, router, start
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class AnswererResponse(BaseModel):
    """Response schema for the answerer in 20 Questions game."""

    # Keep this short; do NOT ask for chain-of-thought
    brief_reason: Optional[str] = Field(description="1-2 sentences justification (optional, high level only).")
    yes_or_no: Literal["yes", "no", "n/a"] = Field(
        description="Whether the correct answer to the player's question is yes, no, or not applicable."
    )
    correct: bool = Field(
        description="Whether the player has correctly guessed the entity, and the game should end now."
    )


PLAYER_QUERY_TEMPLATE = """You are playing 20 Questions as the **Player**.
Ask one high-information **yes/no** question that most reduces the remaining possibility space.
If you think you have figured out the secret entity, ask a direct guess in the form: **"Is it <entity>?"**

THIS IS TURN #{turn_index} OF 20. You have {remaining_turns} turns left. The quicker you make a correct guess, the higher your score.

## Important assumptions
- Your answer BELONGS TO the category of "{category}".
- The answer is **straightforward, familiar, and commonly known**. They can be at most 3 words long (and only one word long in a majority of cases).
- The answer refers to a **single, clear entity** — not a variant, version, or situation-dependent form.

## What you have: Game history (Q/A pairs):

{history}

## Strategy guidelines (concise, practical)
- **Start broad, then narrow**: prioritize sub-category-level splits first (within {category}), then mid-level properties, then identifiers.
- **Binary partitioning**: prefer questions that split the remaining set near the middle.
- **Property over identity**: ask about features/roles/usages before naming brands, species, or specific titles.
- **Contradiction guard**: if past answers imply a contradiction, ask a short reset/sanity-check question to reconcile.
- **n/a handling**: if the last answer was **n/a**, pivot wording to a clearer, factual property.
- **Endgame**: if you have only one turn left, make a direct guess.

## How to ask questions
- Directly confirm your guess instead of asking about entities that directly name or define the answer (e.g., "Is it a type of pizza?" if "Pizza" is an option).
- Avoid questions that depend on subjective or situational conditions (e.g., "Would most people consider it artistic?").
- You are encouraged to use the search tool to check factual details or implications behind your potential question. This helps ensure your reasoning is grounded, accurate, and avoids irrelevant or trivial inquiries. However, you can only use the search tool at most once for each question; you must not use the search tool consecutively without asking a question in between.

## Output format (critical)

- Output **only** one yes/no question on a single line.
- No preamble, no numbering, no quotes, no meta commentary.
- Keep it concise, under 50 words.
- If guessing: use the form **Is it <entity>?**

Now produce your single best next question."""


ANSWERER_QUERY_TEMPLATE = """You are the **Answerer** in 20 Questions. Answer yes/no questions truthfully about the secret entity; mark correct if guessed exactly.

Your secret entity is: "{answer}". It belongs to the category of "{category}".

## The player's current question

{next_question}

## Rules

- Respond only with a structured yes/no evaluation about the entity.
- Be concise, objective, and consistent with previous answers.
- Never reveal the entity unless the player guessed correctly.
- If you don't know the answer, for example, the information is never publicly known, or the question is irrelevant to the entity's nature, answer **"n/a"**.

### Primary-sense rule (important)
- Answer based on the entity's **primary, literal identity**, not metaphorical associations or what it “can represent.”
  (Example: a famous building is **not** a "symbol" just because people call it a symbol of love.)
- Use the multi-meaning rule **only** when the entity's **name itself** has multiple mainstream senses (e.g., “football” the sport vs. the ball). Otherwise, stick to the primary sense.

### Handling unknown or irrelevant questions
- If the question asks about something **not publicly known**, **not factual**, **ambiguous**, or **irrelevant**, respond **"n/a"**.
- Use **"n/a"** only when a yes/no would be **misleading or nonsensical**.
- Examples:
  - "Does it have parents?" -> *n/a* (not meaningful for a place or object)
  - "Is it alive?" -> valid for all entities (answer yes/no if possible)
  - "Is it an animal?" -> *n/a* if the entity is a person, as this can be ambiguous.
  - "Does it post on social media?" -> *n/a* unless the entity is a living or fictional character known for doing so.
  - "Is the chair branded by a famous manufacturer?" -> *n/a* for a general object like "chair".

### Handling ambiguous entities
If the secret entity truly has multiple common meanings:
- Answer **"yes"** if the question is true for **any** major, well-recognized meaning.
- Answer **"no"** only if it's false for **all** reasonable interpretations.
- Do **not** stretch to metaphors or loose associations.

### Handling direct guesses
If the player's question is a direct guess ("Is it ...?"):
- Set **correct = true** if the guess is a close match in meaning to the secret entity (e.g., “Is it cell phone?” ≈ “Smartphone”).
- Otherwise, set **correct = false**.
"""


SEARCH_PROMPT_TEMPLATE = """You are simulating a web search.

Query: "{search_query}"

Return a concise, factual summary (2-4 sentences) of the most relevant information you would find online.
Avoid speculation, filler, or references to being an AI. Just give the facts."""


class SearchToolInput(BaseModel):
    """Schema for search tool input."""

    search_query: str = Field(
        ...,
        description="A short, factual query describing what to search for (e.g., 'capital of France', 'biography of Ada Lovelace').",
    )


class SearchTool(BaseTool):
    """A mock web search tool powered by an LLM.

    This class mimics a real search engine call by using a lightweight LLM model.
    It can later be replaced by a real API (like Serper or Bing) without changing its interface.
    """

    model: BaseLLM
    name: str = "search"
    description: str = "Search the web. Provide a concise, factual summary of what is known about the given topic."
    num_called: int = 0

    def _run(self, search_query: str) -> str:
        """Perform a mocked search request using an LLM."""
        self.num_called += 1
        # Safety: ensure input is not too long or empty
        search_query = search_query.strip()
        if not search_query:
            return "No query provided."
        if len(search_query) > 500:
            search_query = search_query[:500] + "..."

        # Use a lightweight CrewAgent to simulate a factual web summary
        agent = CrewAgent(
            role="Search engine summarizer",
            goal=(
                "Given a user's search query, return a concise, factual summary "
                "as if retrieved from reliable sources. "
                "Act like a real search engine summarizer. "
                "Never disclose that you are a simulator of a search engine."
            ),
            backstory=(
                "You simulate a web search engine, producing factual, neutral summaries. "
                "Do not fabricate sources or URLs. Focus on core, verifiable facts."
            ),
            llm=self.model,
        )

        prompt = SEARCH_PROMPT_TEMPLATE.format(search_query=search_query)
        result = agent.kickoff(prompt)
        return result.raw.strip()


class Turn(BaseModel):
    """Represents a single turn in a 20 Questions game.

    Attributes:
        question: The question asked by the player.
        response: The answerer's "yes" or "no" or "n/a" response.
    """

    question: str
    response: Literal["yes", "no", "n/a"]


class TwentyQuestionsGameState(BaseModel):
    """State of a 20 Questions game session.

    Attributes:
        answer: The secret entity the player is trying to guess.
        category: The category of the secret entity.
        correct: Whether the player has guessed correctly.
        num_tool_calls: Number of search tool calls made during the game.
        next_question: The current question being processed.
        turn_index: Current turn number (1-20).
        interactions: History of question-answer turns.
    """

    answer: str = ""
    category: str = ""
    correct: bool = False
    num_tool_calls: int = 0
    next_question: str = ""
    turn_index: int = 1
    interactions: List[Turn] = Field(default_factory=list)  # type: ignore

    def render_history(self) -> str:
        """Render the game history as a formatted string.

        Returns:
            Formatted string showing all questions and responses.
        """
        return "\n\n".join(
            [
                f"Question #{i}: {turn.question}\nResponse #{i}: {turn.response}"
                for i, turn in enumerate(self.interactions, start=1)
            ]
        )


class TwentyQuestionsFlow(Flow[TwentyQuestionsGameState]):
    """CrewAI Flow for running a 20 Questions game.

    This flow coordinates the player and answerer agents through the game.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the flow with player, answerer, and optional search tool.

        Args:
            *args: Positional arguments to pass to Flow.
            **kwargs: Keyword arguments including player_llm, answer_llm, and search_tool.
        """
        self.player_llm = cast(CrewLLM, kwargs.pop("player_llm"))
        self.answer_llm = cast(CrewLLM, kwargs.pop("answer_llm"))
        self.search_tool = cast(Optional[SearchTool], kwargs.pop("search_tool", None))
        super().__init__(*args, **kwargs)

    @start("next_turn")
    def ask_question(self):
        """Generate the next question from the player agent."""
        agent = CrewAgent(
            role="Player in a game of 20 questions",
            goal="Minimize uncertainty and identify the hidden entity within 20 yes/no questions.",
            backstory="A focused reasoner who uses binary-partition questions to narrow down the remaining possibilities.",
            tools=[self.search_tool] if self.search_tool else [],
            llm=self.player_llm,
            max_iter=3,  # Maximum iterations of tool calls
            # Agent is instructed to use at most 1 search tool call per question.
            # but here it's allowed up to 3 chances.
            # Otherwise, the agent will be forced by CrewAI to give the "BEST Final answer"
            # which is not even a question that we want.
        )
        query = PLAYER_QUERY_TEMPLATE.format(
            history=self.state.render_history(),
            turn_index=self.state.turn_index,
            remaining_turns=20 - self.state.turn_index + 1,
            category=self.state.category,
        )

        result = agent.kickoff(query)
        console.print(f"[bold red]Player (Turn {self.state.turn_index}):[/bold red] {result.raw}")
        if self.search_tool is not None:
            self.state.num_tool_calls = self.search_tool.num_called
        self.state.next_question = result.raw

    @listen(ask_question)
    def answer_question(self):
        """Process the player's question and generate an answer."""
        query = ANSWERER_QUERY_TEMPLATE.format(
            answer=self.state.answer, next_question=self.state.next_question, category=self.state.category
        )
        # NOTE: We can also ground the answerer with a search tool.
        # But it would make the example too complicated for now.
        answerer_response = cast(AnswererResponse, self.answer_llm.call(query))  # type: ignore
        console.print(f"[bold red]Answerer (Turn {self.state.turn_index}):[/bold red] {answerer_response}")
        try:
            turn = Turn(question=self.state.next_question, response=answerer_response.yes_or_no)
            correct = answerer_response.correct
        except Exception as e:
            console.print(f"[bold red]Answerer Response Format Error: {e}[/bold red]")
            # Assuming n/a
            turn = Turn(question=self.state.next_question, response="n/a")
            correct = False
        self.state.interactions.append(turn)
        self.state.next_question = ""  # Reset the next question
        self.state.correct = correct

    @router(answer_question)
    def game_should_continue(self):
        """Determine if the game should continue or end.

        Returns:
            "game_over" if the game is finished, "next_turn" otherwise.
        """
        if self.state.correct:
            console.print(f"[bold red]Correct! You win![/bold red]")
            return "game_over"
        elif self.state.turn_index >= 20:
            console.print(
                f"[bold red]You've asked 20 questions and still haven't guessed the entity. You lose![/bold red]"
            )
            return "game_over"
        else:
            self.state.turn_index += 1
            console.print(f"[bold purple]Continue with turn #{self.state.turn_index}...[/bold purple]")
            return "next_turn"

    @listen("game_over")
    def finish(self):
        """Handle game completion."""
        console.print("The flow has reached the finished state.")
