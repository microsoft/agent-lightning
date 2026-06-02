"""Hooks for the ScienceWorld example (local runner mode).

- on_enqueue: passthrough. The local controller already injects
  ``rollout.input`` into the worker subprocess as ``AGL_TASK_INPUT`` —
  no pod_spec, no K8s manifest to patch.

- on_succeeded: read the ``episode_result`` event posted by the agent,
  write a single scalar ``reward`` event = ``final_score / 100.0`` for
  the VERL bridge to consume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agl_lite.hooks import RolloutHooks, TraceWriter

if TYPE_CHECKING:
    from agl_lite.schemas import Rollout


class SWHooks(RolloutHooks):
    def on_succeeded(
        self,
        rollout: Rollout,
        events: dict[str, list[Any]],
        store: TraceWriter,
    ) -> None:
        episode = self._latest_episode_result(events)
        if episode is None:
            value = 0.0
            reason = "no episode_result event"
        else:
            final_score = float(episode.get("final_score", 0.0))
            value = max(0.0, min(1.0, final_score / 100.0))
            reason = "episode_completed" if episode.get("completed") else "max_steps_or_no_progress"

        attempt_id = rollout.last_attempt_id or "unknown"
        store.add_event(
            rollout.rollout_id,
            attempt_id,
            "reward",
            {
                "value": value,
                "reason": reason,
                "final_score": (episode or {}).get("final_score"),
                "num_turns": (episode or {}).get("num_turns"),
            },
        )

    @staticmethod
    def _latest_episode_result(events: dict[str, list[Any]]) -> dict[str, Any] | None:
        latest: dict[str, Any] | None = None
        for attempt_events in events.values():
            for evt in attempt_events:
                if evt.event_type == "episode_result":
                    latest = evt.data
        return latest
