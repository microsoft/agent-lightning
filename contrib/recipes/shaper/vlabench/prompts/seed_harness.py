# Copyright (c) Microsoft. All rights reserved.


def build_context(history):
    if not history:
        return [{"type": "text", "text": "No previous actions (episode start)."}]
    parts = [{"type": "text", "text": "Previous rounds of execution:"}]
    for record in history:
        parts.append(
            {
                "type": "text",
                "text": (
                    "Round "
                    + str(record.get("round_index", 0) + 1)
                    + ": reasoning="
                    + str(record.get("planner_response", ""))
                    + " | subtask="
                    + str(record.get("command", ""))
                    + " | VLA steps="
                    + str(record.get("execution_steps", 0))
                ),
            }
        )
        observations = record.get("observation_after", [])
        for item in observations:
            parts.append(item)
    return parts
