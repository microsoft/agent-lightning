"""ScienceWorld agent — solves a text-based science task via a constrained-action LLM loop.

Loaded by ``agl_lite.controller.local_worker`` as
``examples.science_world.agents.sw_agent:SWAgent`` when the controller runs
in ``runner_type=local`` mode.

The class exposes one method, ``run(task)``, which:

  1. Boots a ScienceWorldEnv and loads the task / variation.
  2. Loops up to ``max_steps``; each turn shows the LLM the current
     observation + a numbered list of valid actions and asks it to pick one
     by index (``### ACTION: <n> ###``).
  3. Steps the env, POSTs a ``step`` event per turn, and a final
     ``episode_result`` event before returning.

Process exit code is the only terminal signal: non-zero → TERMINAL_FAILED.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 30
DEFAULT_ENV_STEP_LIMIT = 100
MAX_VALID_ACTIONS_SHOWN = 50
OBS_SNIPPET_CHARS = 240
ACTION_PATTERN = re.compile(r"###\s*ACTION:\s*(\d+)\s*###")

PROMPT_TEMPLATE = """You are playing a text-based science game. Solve the task by issuing actions one at a time.

TASK: {task_description}

CURRENT OBSERVATION:
{observation}

INVENTORY:
{inventory}

Choose ONE of the following actions by its number:
{action_list}

Respond with the action number in the format:
### ACTION: <number> ###"""


def _setup_logging() -> None:
    log_dir = os.environ.get("AGL_LOG_DIR")
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        log_path = Path(log_dir) / "agent.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def _post_event(event_type: str, data: dict[str, Any]) -> None:
    event_url = os.environ.get("AGL_EVENT_URL")
    if not event_url:
        log.warning("AGL_EVENT_URL not set — skipping %s event", event_type)
        return
    api_key = os.environ.get("OPENAI_API_KEY", "")
    try:
        resp = httpx.post(
            event_url,
            json={"event_type": event_type, "data": data},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception as e:
        log.error("Failed to post %s event: %s", event_type, e)


def _format_action_list(valid: list[dict[str, Any]]) -> str:
    shown = valid[:MAX_VALID_ACTIONS_SHOWN]
    return "\n".join(f"{i}. {a['action']}" for i, a in enumerate(shown))


def _parse_action_index(text: str, num_actions: int) -> int:
    m = ACTION_PATTERN.search(text or "")
    if not m:
        return 0
    idx = int(m.group(1))
    if 0 <= idx < num_actions:
        return idx
    return 0


def _stub_pick(_text_obs: str, _valid: list[dict[str, Any]]) -> int:
    """Stub policy: always action 0. Used when SW_STUB_LLM=1 for smoke tests."""
    return 0


class SWAgent:
    """ScienceWorld agent for agl-lite local runner mode.

    ``run()`` accepts the task dict that the controller injected from
    ``rollout.input`` and is expected to either complete normally or raise
    (which becomes a non-zero exit code).
    """

    async def run(self, task: dict[str, Any]) -> None:
        _setup_logging()
        task_name = task["task_name"]
        variation_idx = int(task["variation_idx"])
        simplification = task.get("simplification", "easy")
        max_steps = int(os.environ.get("SW_MAX_STEPS", str(DEFAULT_MAX_STEPS)))
        env_step_limit = int(os.environ.get("SW_ENV_STEP_LIMIT", str(DEFAULT_ENV_STEP_LIMIT)))
        stub_llm = os.environ.get("SW_STUB_LLM") == "1"

        log.info(
            "ScienceWorld task=%s variation=%d simplification=%s max_steps=%d stub=%s",
            task_name,
            variation_idx,
            simplification,
            max_steps,
            stub_llm,
        )

        from scienceworld import ScienceWorldEnv

        env = ScienceWorldEnv("", envStepLimit=env_step_limit)
        env.load(task_name, variation_idx, simplification)
        task_description = env.get_task_description()
        obs, info = env.reset()

        llm_client = None if stub_llm else _build_llm_client()
        model = os.environ.get("AGL_MODEL_NAME", "")

        final_score = float(info.get("score", 0.0))
        completed = False
        turn = 0

        for turn in range(max_steps):
            valid = env.get_valid_action_object_combinations_with_templates()
            if not valid:
                log.warning("No valid actions at turn %d — aborting episode", turn)
                break

            if stub_llm:
                action_idx = _stub_pick(obs, valid)
            else:
                assert llm_client is not None
                inventory = env.inventory()
                prompt = PROMPT_TEMPLATE.format(
                    task_description=task_description,
                    observation=obs,
                    inventory=inventory,
                    action_list=_format_action_list(valid),
                )
                response_text = await _call_llm(llm_client, model, prompt)
                action_idx = _parse_action_index(response_text, min(len(valid), MAX_VALID_ACTIONS_SHOWN))

            action_str = valid[action_idx]["action"]
            obs, step_reward, done, info = env.step(action_str)
            final_score = float(info.get("score", final_score))

            _post_event(
                "step",
                {
                    "turn": turn,
                    "action": action_str,
                    "reward": float(step_reward),
                    "score": final_score,
                    "done": bool(done),
                    "obs_snippet": (obs or "")[:OBS_SNIPPET_CHARS],
                },
            )

            if done:
                completed = True
                break

        _post_event(
            "episode_result",
            {
                "final_score": final_score,
                "num_turns": turn + 1,
                "completed": completed,
                "task_name": task_name,
                "variation_idx": variation_idx,
            },
        )
        log.info(
            "Episode done: score=%.2f turns=%d completed=%s",
            final_score,
            turn + 1,
            completed,
        )


def _build_llm_client() -> Any:
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is required (set by local controller)")
    api_key = os.environ.get("OPENAI_API_KEY", "token-abc123")

    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=base_url, api_key=api_key)


async def _call_llm(client: Any, model: str, prompt: str) -> str:
    temperature = float(os.environ.get("AGL_TEMPERATURE", "0.7"))
    max_tokens = int(os.environ.get("AGL_MAX_TOKENS", "256"))
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        log.error("LLM call failed: %s", e)
        return ""


def _entrypoint_for_local_smoke() -> None:
    """Allow running this module directly for ad-hoc debugging.

    Reads AGL_TASK_INPUT exactly like local_worker would, useful for
    iterating on the agent without spinning up the controller.
    """
    import asyncio

    raw = os.environ.get("AGL_TASK_INPUT")
    if not raw:
        print("AGL_TASK_INPUT not set", file=sys.stderr)
        sys.exit(2)
    task = json.loads(raw)
    asyncio.run(SWAgent().run(task))


if __name__ == "__main__":
    _entrypoint_for_local_smoke()
