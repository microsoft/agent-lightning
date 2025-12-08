# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import traceback
from typing import List, Optional, Tuple, TypedDict, cast

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, Field
from rich.console import Console

from agentlightning.adapter import TraceToMessages
from agentlightning.litagent import rollout
from agentlightning.reward import find_final_reward
from agentlightning.runner import LitAgentRunner
from agentlightning.store import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.types import Dataset, PromptTemplate

console = Console()


class JudgeResponse(BaseModel):
    reason: str = Field(description="The reason for the score. No more than 100 characters.")
    score: float = Field(description="The score for the match on a 0-1 scale. Be critical.")


class Room(TypedDict):
    id: str
    capacity: int
    equipment: List[str]
    accessible: bool
    distance_m: int
    booked: List[Tuple[str, str, int]]


class RoomStatus(Room):
    free: bool


class AvailableRooms(TypedDict):
    rooms: List[RoomStatus]


class RoomRequirement(TypedDict):
    date: str
    time: str
    duration_min: int
    attendees: int
    needs: List[str]
    accessible_required: bool


class RoomSelectionTask(TypedDict):
    id: str
    task_input: RoomRequirement
    expected_choice: str


TOOL_DEFINITIONS: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_rooms_and_availability",
            "description": "Return meeting rooms with capacity, equipment, accessibility, distance, and booked time slots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM 24h local"},
                    "duration_min": {"type": "integer", "description": "Meeting duration minutes"},
                },
                "required": ["date", "time", "duration_min"],
            },
        },
    },
]


def prompt_template_baseline() -> PromptTemplate:
    return PromptTemplate(
        template="Find a room on {date} at {time} for {duration_min} minutes, {attendees} attendees. Needs: {needs}. Accessible required: {accessible_required}",
        engine="f-string",
    )


def room_selection_grader(client: OpenAI, final_message: Optional[str], expected_choice: str) -> float:
    judge_prompt = (
        f"You are a strict grader of exact room choice."
        f"Task output:\n{final_message}\n\n"
        f"Task expected answer:\n{expected_choice}\n\n"
        f"Score the match on a 0-1 scale. Be critical.\n"
        f"Bear in mind that the score can be partially correct (between 0 and 1)."
    )
    judge = client.chat.completions.parse(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": judge_prompt},
        ],
        response_format=JudgeResponse,
        temperature=0.0,
    )

    judge_result = judge.choices[0].message.content
    console.print(f"[bold yellow]=== Judge ===[/bold yellow]")
    console.print(judge_result)

    judge_result_parsed = JudgeResponse.model_validate_json(judge_result)  # type: ignore

    console.print(f"[bold yellow]=== Judge Score ===[/bold yellow]")
    console.print(judge_result_parsed.score)
    return judge_result_parsed.score


@rollout
def room_selector(task: RoomSelectionTask, prompt_template: PromptTemplate) -> float:
    """An agent to select a room based on the given requirements.

    Oracle System Prompt (works with 100% accuracy with gpt-5 mini low reasoning effort):

        You are a scheduling assistant.
        Hard constraints: free for slot, capacity >= attendees, includes all required equipment,
        accessible==True if requested.
        Tie-break scoring (lower is better):
        1) capacity_slack = capacity - attendees (minimize)
        2) extra_equipment = provided_equipment_count - required_equipment_count (minimize)
        3) distance_m (minimize)
        4) fewer total booked blocks that day (minimize)
        Return No Room if no room is found that satisfies the constraints.
        Return strictly:
        final_choice: <ROOM_ID>
        reason: <one line stating the decisive criteria>

    Oracle User Prompt Template:

        Find a room on {task_input['date']} at {task_input['time']} for {task_input['duration_min']} minutes,
        {task_input['attendees']} attendees. Needs: {', '.join(task_input['needs']) or 'none'}.
        Accessible required: {task_input['accessible_required']}

    The current implementation greatly simply the oracle prompt and prompt template is provided by a parameter.
    The prompt template should be tuned by Agent-lightning's APO algorithm.
    It also should work with a very small model like gpt-4.1-nano.
    """

    client = OpenAI()
    model = "gpt-4.1-nano"

    user_message = prompt_template.format(**task["task_input"])

    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a scheduling assistant."},
        {
            "role": "user",
            "content": user_message,
        },
    ]

    console.print(f"[bold yellow]=== User Message ===[/bold yellow]")
    console.print(user_message)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
        # Minimize the randomness
        temperature=0.0,
        # Uncomment for gpt-5
        # reasoning_effort="low",
    )

    console.print(f"[bold yellow]=== Assistant Message ===[/bold yellow]")
    console.print(resp.choices[0].message)

    # Parse and process the tool calls
    tool_calls = resp.choices[0].message.tool_calls
    if tool_calls:

        tool_call_params: List[ChatCompletionMessageFunctionToolCallParam] = []
        tool_results: List[ChatCompletionToolMessageParam] = []
        for tc in tool_calls:
            if tc.type != "function":
                raise ValueError(f"Tool call is not a function: {tc}")
            if tc.function.name != "get_rooms_and_availability":
                raise ValueError(f"Tool call is not get_rooms_and_availability: {tc}")
            tool_call_params.append(
                ChatCompletionMessageFunctionToolCallParam(
                    id=tc.id,
                    type="function",
                    function={"name": tc.function.name, "arguments": tc.function.arguments},
                )
            )
            args = json.loads(tc.function.arguments)
            try:
                tool_output = get_rooms_and_availability(args["date"], args["time"], args["duration_min"])
            except Exception as e:
                tool_output = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            console.print(f"[bold yellow]=== Tool Message ===[/bold yellow]")
            console.print(tool_output)
            tool_results.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tc.id,
                    content=json.dumps(tool_output),
                )
            )

        # Update the messages for hte next call
        messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=resp.choices[0].message.content,
                tool_calls=tool_call_params,
            )
        )
        messages.extend(tool_results)

        next_resp = client.chat.completions.create(
            model=model,
            messages=messages,
            # Minimize the randomness
            temperature=0.0,
        )
        console.print(f"[bold yellow]=== Final Assistant Message ===[/bold yellow]")
        console.print(next_resp.choices[0].message.content)
        final_message = next_resp.choices[0].message.content

    else:
        final_message = resp.choices[0].message.content

    return room_selection_grader(client, final_message, task["expected_choice"])


# Local tool database (there might be multiple plausible fits)
ROOMS: List[Room] = [
    {
        "id": "Orion",
        "capacity": 4,
        "equipment": ["tv", "whiteboard"],
        "accessible": True,
        "distance_m": 12,
        "booked": [("2025-10-13", "10:00", 60), ("2025-10-13", "15:00", 30)],
    },
    {
        "id": "Lyra",
        "capacity": 10,
        "equipment": ["projector", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 30,
        "booked": [("2025-10-13", "09:30", 30), ("2025-10-13", "11:00", 60)],
    },
    {
        "id": "Vega",
        "capacity": 6,
        "equipment": ["tv"],
        "accessible": False,
        "distance_m": 22,
        "booked": [("2025-10-13", "14:00", 60)],
    },
    {
        "id": "Nova",
        "capacity": 12,
        "equipment": ["ledwall", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 45,
        "booked": [],
    },
    {
        "id": "Quark",
        "capacity": 8,
        "equipment": ["projector", "whiteboard"],
        "accessible": False,
        "distance_m": 18,
        "booked": [("2025-10-13", "10:30", 30)],
    },
    # Two extra to create harder ties
    {
        "id": "Atlas",
        "capacity": 6,
        "equipment": ["projector", "whiteboard"],
        "accessible": True,
        "distance_m": 10,
        "booked": [("2025-10-13", "09:00", 30), ("2025-10-13", "13:30", 30)],
    },
    {
        "id": "Pulse",
        "capacity": 8,
        "equipment": ["tv", "whiteboard", "confphone"],
        "accessible": True,
        "distance_m": 8,
        "booked": [("2025-10-13", "16:30", 30)],
    },
]


def overlaps(start: str, dur: int, other_start: str, other_dur: int) -> bool:
    def tmin(t: str):
        return int(t[:2]) * 60 + int(t[3:])

    a0, a1 = tmin(start), tmin(start) + dur
    b0, b1 = tmin(other_start), tmin(other_start) + other_dur
    return max(a0, b0) < min(a1, b1)


def get_rooms_and_availability(date: str, time_str: str, duration_min: int) -> AvailableRooms:
    avail: List[RoomStatus] = []
    for r in ROOMS:
        free = all(
            not (b_date == date and overlaps(time_str, duration_min, b_time, b_dur))
            for (b_date, b_time, b_dur) in r["booked"]
        )
        item: RoomStatus = {
            **r,
            "free": free,
        }
        avail.append(item)
    return {"rooms": avail}


def load_room_tasks() -> Dataset[RoomSelectionTask]:
    tasks: List[RoomSelectionTask] = []
    for line in open("room_tasks.jsonl"):
        task = json.loads(line)
        tasks.append(RoomSelectionTask(**task))
    return cast(Dataset[RoomSelectionTask], tasks)


async def debug_room_selector(limit: int = 1):
    # Prepare all the components to run the agent
    runner = LitAgentRunner[RoomSelectionTask](AgentOpsTracer())
    store = InMemoryLightningStore()
    prompt_template = prompt_template_baseline()
    tasks = load_room_tasks()
    with runner.run_context(agent=room_selector, store=store):
        for task in tasks:
            console.print("[bold green]=== Task ===[/bold green]", task, sep="\n")
            # Run the agent
            rollout = await runner.step(task, resources={"main_prompt": prompt_template})
            # Get the spans and convert them to messages
            # Useful for debugging and analysis
            spans = await store.query_spans(rollout.rollout_id)
            adapter = TraceToMessages()
            messages = adapter.adapt(spans)
            for message_idx, message in enumerate(messages):
                console.print(f"[bold purple]=== Postmortem Message #{message_idx} ===[/bold purple]")
                console.print(json.dumps(message))
            reward = find_final_reward(spans)
            console.print("[bold purple]=== Postmortem Reward ===[/bold purple]", reward, sep="\n")


if __name__ == "__main__":
    asyncio.run(debug_room_selector())
