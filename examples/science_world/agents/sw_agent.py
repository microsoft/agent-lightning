"""ScienceWorld agent — solves a text-based science task via a constrained-action LLM loop.

Loaded by ``agl_lite.controller.local_reconciler`` as
``examples.science_world.agents.sw_agent:SWAgent`` when the controller runs
in ``runner_type=local`` mode.

The class exposes one method, ``run()`` (no arguments). The local reconciler
injects the rollout input as environment variables (see ``local.env_map`` in
``train_sw_agent.py``) and provides ``AGL_OPENAI_BASE_URL`` / ``AGL_EVENT_URL``
/ ``AGL_KEY``. ``run()``:

  1. Boots a ScienceWorldEnv and loads the task / variation.
  2. Builds the initial user prompt once (task + first obs + first inventory
     + first action list) and keeps appending to a growing ``messages`` list:
     assistant turns are the model's raw replies, user turns are the next
     observation + inventory + valid-action list. This makes each turn's
     prompt a token-level prefix of the previous turn's ``prompt + response``,
     which is what ``rollout_bridge`` needs to merge multi-turn traces into a
     single trajectory row.
  3. Steps the env, then POSTs a single ``reward`` event = ``final_score / 100``
     for the VERL bridge to consume.

Process exit code is the only terminal signal: non-zero -> FAILED.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 30
DEFAULT_ENV_STEP_LIMIT = 100
DEFAULT_MAX_VALID_ACTIONS_SHOWN = 50
DEFAULT_OBS_SNIPPET_CHARS = 240
ACTION_PATTERN = re.compile(r"###\s*ACTION:\s*(\d+)\s*###")

INITIAL_PROMPT_TEMPLATE = """You are playing a text-based science game. Solve the task by issuing actions one at a time.

TASK: {task_description}

CURRENT OBSERVATION:
{observation}

INVENTORY:
{inventory}

Choose ONE of the following actions by its number:
{action_list}

Respond with EXACTLY the following, nothing else:
### ACTION: <number> ###
Do not include any reasoning, explanation, or other text.
"""

OBSERVATION_TEMPLATE = """OBSERVATION:
{observation}

INVENTORY:
{inventory}

Choose ONE of the following actions by its number:
{action_list}"""


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _format_action_list(valid: list[dict[str, Any]], max_shown: int) -> str:
    shown = valid[:max_shown]
    return "\n".join(f"{i}. {a['action']}" for i, a in enumerate(shown))


def _parse_action_index(text: str, num_actions: int) -> int:
    m = ACTION_PATTERN.fullmatch((text or "").strip())
    if not m:
        return 0
    idx = int(m.group(1))
    if 0 <= idx < num_actions:
        return idx
    return 0


class SWAgent:
    """ScienceWorld agent for agl-lite local runner mode."""

    async def run(self) -> None:
        _setup_logging()

        task_name = os.environ["TASK_NAME"]
        variation_idx = int(os.environ["VARIATION_IDX"])
        simplification = os.environ.get("SIMPLIFICATION", "easy")

        agl_key = os.environ["AGL_KEY"]
        event_url = os.environ["AGL_EVENT_URL"]
        openai_base_url = os.environ["AGL_OPENAI_BASE_URL"]

        max_steps = int(os.environ.get("SW_MAX_STEPS", str(DEFAULT_MAX_STEPS)))
        env_step_limit = int(os.environ.get("SW_ENV_STEP_LIMIT", str(DEFAULT_ENV_STEP_LIMIT)))
        max_valid_actions_shown = int(
            os.environ.get("SW_MAX_VALID_ACTIONS_SHOWN", str(DEFAULT_MAX_VALID_ACTIONS_SHOWN))
        )
        obs_snippet_chars = int(os.environ.get("SW_OBS_SNIPPET_CHARS", str(DEFAULT_OBS_SNIPPET_CHARS)))
        max_tokens = int(os.environ.get("AGL_MAX_TOKENS", "256"))

        log.info(
            "ScienceWorld task=%s variation=%d simplification=%s max_steps=%d",
            task_name,
            variation_idx,
            simplification,
            max_steps,
        )

        from openai import AsyncOpenAI
        from scienceworld import ScienceWorldEnv

        client = AsyncOpenAI(base_url=openai_base_url, api_key=agl_key, max_retries=6)

        env = ScienceWorldEnv("", envStepLimit=env_step_limit)
        env.load(task_name, variation_idx, simplification)
        task_description = env.get_task_description()
        obs, info = env.reset()
        inventory = env.inventory()
        valid = env.get_valid_action_object_combinations_with_templates()

        final_score = float(info.get("score", 0.0))
        completed = False
        turn = 0

        messages: list[dict[str, str]] = []
        if valid:
            messages.append(
                {
                    "role": "user",
                    "content": INITIAL_PROMPT_TEMPLATE.format(
                        task_description=task_description,
                        observation=obs,
                        inventory=inventory,
                        action_list=_format_action_list(valid, max_valid_actions_shown),
                    ),
                }
            )

        for turn in range(max_steps):
            if not valid:
                log.warning("No valid actions at turn %d — aborting episode", turn)
                break

            response_text = await self._call_llm(client, messages, max_tokens)
            action_idx = _parse_action_index(response_text, min(len(valid), max_valid_actions_shown))
            action_str = valid[action_idx]["action"]
            messages.append({"role": "assistant", "content": response_text})

            obs, _step_reward, done, info = env.step(action_str)
            final_score = float(info.get("score", final_score))

            log.info(
                "turn=%d action=%r score=%.2f done=%s obs=%r",
                turn,
                action_str,
                final_score,
                done,
                (obs or "")[:obs_snippet_chars],
            )

            if done:
                completed = True
                break

            inventory = env.inventory()
            valid = env.get_valid_action_object_combinations_with_templates()
            if valid and turn + 1 < max_steps:
                messages.append(
                    {
                        "role": "user",
                        "content": OBSERVATION_TEMPLATE.format(
                            observation=obs,
                            inventory=inventory,
                            action_list=_format_action_list(valid, max_valid_actions_shown),
                        ),
                    }
                )

        reward = max(0.0, min(1.0, final_score / 100.0))
        log.info(
            "Episode done: score=%.2f turns=%d completed=%s reward=%.3f",
            final_score,
            turn + 1,
            completed,
            reward,
        )

        httpx.post(
            event_url,
            json={"event_type": "reward", "data": {"value": reward}},
            headers={"Authorization": f"Bearer {agl_key}"},
            timeout=10.0,
        ).raise_for_status()

    @staticmethod
    async def _call_llm(client: Any, messages: list[dict[str, str]], max_tokens: int) -> str:
        try:
            completion = await client.chat.completions.create(
                model="auto",
                messages=messages,
                max_tokens=max_tokens,
                stop=["###\n", "###\n\n"],
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            log.error("LLM call failed: %s", e)
            return ""


if __name__ == "__main__":
    asyncio.run(SWAgent().run())
