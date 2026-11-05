# Copyright (c) Microsoft. All rights reserved.

"""Three arithmetic calls with local prefix checks and an explicit evidence seal."""

from __future__ import annotations

import os
from uuid import NAMESPACE_URL, uuid5

import httpx

from agentlightning.credit.client import ProcessCreditClient


def exact_integer(text: str, expected: int) -> bool:
    """Accept an integer answer only; never execute model text as Python."""
    try:
        return int(text.strip()) == expected and str(int(text.strip())) == text.strip()
    except ValueError:
        return False


class Agent:
    async def run(self):
        event_url = os.environ["AGL_EVENT_URL"]
        base, suffix = event_url.rsplit("/api/rollouts/", 1)
        rollout_id, _, attempt_id, _ = suffix.split("/")
        namespace = f"{base}/{rollout_id}/{attempt_id}"
        a, b, c, d = (int(os.environ[name]) for name in ("A", "B", "C", "D"))
        answers = (a + b, (a + b) * c, (a + b) * c - d)
        prompts = (f"Compute {a}+{b}.", f"Multiply the tool result by {c}.", f"Subtract {d} from the tool result.")
        headers = {"Authorization": f"Bearer {os.environ['AGL_KEY']}"}
        with httpx.Client(base_url=base, headers=headers, timeout=300) as client:
            rollout = client.get(f"/api/rollouts/{rollout_id}")
            rollout.raise_for_status()
            marker = rollout.json()["rollout"].get("metadata", {}).get("process_credit")
            credit = (
                ProcessCreditClient(client, rollout_id, attempt_id, policy_version=marker["policy_version"])
                if marker
                else None
            )
            messages = []
            actions = []
            correct = []
            for index, (prompt, expected) in enumerate(zip(prompts, answers, strict=True)):
                messages.append({"role": "user", "content": prompt + " Reply with only the integer."})
                body = {"model": "auto", "messages": list(messages), "temperature": 1, "top_p": 1, "max_tokens": 64}
                if credit:
                    completion = credit.complete(body, request_id=f"arithmetic-{index}")
                    if not 200 <= completion.status_code < 300:
                        break  # Captured HTTP failure is a valid environment failure, not missing policy evidence.
                    response = completion.body
                    actions.append(completion.event_id)
                else:
                    result = client.post(os.environ["AGL_OPENAI_BASE_URL"] + "/chat/completions", json=body)
                    result.raise_for_status()
                    response = result.json()
                text = response["choices"][0]["message"]["content"] or ""
                correct.append(exact_integer(text, expected))
                messages.extend(
                    ({"role": "assistant", "content": text}, {"role": "user", "content": f"Tool result: {expected}."})
                )
                if credit and index == 0:
                    credit.emit_process_reward(
                        0.2 if correct[0] else -0.2,
                        reward_id=uuid5(NAMESPACE_URL, namespace + "/prefix").hex,
                        target_event_id=actions[0],
                        evidence_through_event_id=actions[0],
                        source="integer-prefix-v1",
                    )
                if credit and index == 1:
                    # A retrospective label uses the second answer: never book it at target action 0.
                    credit.emit_process_reward(
                        0.1 if all(correct) else -0.1,
                        reward_id=uuid5(NAMESPACE_URL, namespace + "/delayed").hex,
                        target_event_id=actions[0],
                        evidence_through_event_id=actions[1],
                        source="integer-pair-v1",
                    )
            outcome = float(len(correct) == 3 and correct[-1])
            if credit:
                credit.emit_terminal_reward(
                    outcome, reward_id=uuid5(NAMESPACE_URL, namespace + "/terminal").hex, source="integer-final-v1"
                )
                credit.finalize()
            else:
                result = client.post(event_url, json={"event_type": "reward", "data": {"value": outcome}})
                result.raise_for_status()
