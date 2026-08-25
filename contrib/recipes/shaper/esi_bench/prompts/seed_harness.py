# Copyright (c) Microsoft. All rights reserved.


def build_context(records):
    past = [record for record in records if record.get("record_kind") == "past"]
    current = next(
        (record for record in reversed(records) if record.get("record_kind") == "current"),
        {},
    )
    parts = []
    if current.get("call_kind") == "auxiliary_post_action":
        for item in current.get("observable_sequence", []):
            if item.get("content_kind") == "text":
                parts.append({"type": "text", "text": str(item.get("text", ""))})
            elif item.get("content_kind") == "observation":
                observation = item.get("observation", {})
                if isinstance(observation.get("full_frame"), dict):
                    parts.append(observation["full_frame"])
            elif item.get("content_kind") == "content_part" and isinstance(item.get("part"), dict):
                parts.append(item["part"])
        return parts
    if past:
        lines = []
        for record in past:
            lines.append(
                "Step "
                + str(record.get("step", 0))
                + ": action="
                + str(record.get("action", ""))
                + " answer="
                + str(record.get("answer", ""))
                + " confidence="
                + str(record.get("confidence", 0.0))
                + " result="
                + str(record.get("action_result_text", "{}"))
                + " reasoning="
                + str(record.get("reasoning", ""))
            )
        parts.append({"type": "text", "text": "Action history so far:\n" + "\n".join(lines)})
    for reference in current.get("reference_observations", []):
        parts.append({"type": "text", "text": str(reference.get("label", "QUESTION REFERENCE IMAGE"))})
        if isinstance(reference.get("full_frame"), dict):
            parts.append(reference["full_frame"])
    for record in past[-5:]:
        observation = record.get("observation", {})
        parts.append({"type": "text", "text": "[Past view from step " + str(record.get("step", 0)) + "]"})
        if isinstance(observation.get("full_frame"), dict):
            parts.append(observation["full_frame"])
        for extra_index, extra in enumerate(record.get("extra_observations", []), start=1):
            parts.append(
                {
                    "type": "text",
                    "text": (
                        "[Past extra view from step " + str(record.get("step", 0)) + " #" + str(extra_index) + "]"
                    ),
                }
            )
            if isinstance(extra.get("full_frame"), dict):
                parts.append(extra["full_frame"])
    observation = current.get("observation", {})
    parts.append({"type": "text", "text": "[CURRENT VIEW - step " + str(current.get("step", 1)) + "]"})
    if isinstance(observation.get("full_frame"), dict):
        parts.append(observation["full_frame"])
    return parts
