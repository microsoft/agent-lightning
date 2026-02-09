# Copyright (c) Microsoft. All rights reserved.

"""
FlightRecorderAdapter - Import Audit Logs to LightningStore
=============================================================

Adapts Agent-OS Flight Recorder to Agent-Lightning store format.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class FlightRecorderAdapter:
    """
    Import Agent-OS Flight Recorder logs to LightningStore.

    Example:
        >>> from agent_os import FlightRecorder
        >>>
        >>> recorder = FlightRecorder()
        >>> adapter = FlightRecorderAdapter(recorder)
        >>>
        >>> # Import to Lightning store
        >>> adapter.import_to_store(lightning_store)
    """

    def __init__(
        self,
        flight_recorder: Any,
        *,
        trace_id_prefix: str = "agentos",
    ):
        """
        Initialize adapter.

        Args:
            flight_recorder: Agent-OS FlightRecorder
            trace_id_prefix: Prefix for trace IDs
        """
        self.recorder = flight_recorder
        self.trace_id_prefix = trace_id_prefix
        self._imported_count = 0

    def _convert_entry(self, entry: Any, index: int) -> Dict[str, Any]:
        """Convert Flight Recorder entry to span format."""
        entry_type = getattr(entry, "type", "unknown")
        timestamp = getattr(entry, "timestamp", datetime.now(timezone.utc))
        agent_id = getattr(entry, "agent_id", "unknown")

        span = {
            "span_id": f"{self.trace_id_prefix}-{index}",
            "trace_id": f"{self.trace_id_prefix}-{agent_id}",
            "name": f"agent_os.{entry_type}",
            "start_time": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
            "attributes": {
                "agent_os.entry_type": entry_type,
                "agent_os.agent_id": agent_id,
            },
        }

        # Add type-specific attributes
        if entry_type == "policy_check":
            span["attributes"].update(
                {
                    "agent_os.policy_name": getattr(entry, "policy_name", "unknown"),
                    "agent_os.policy_violated": getattr(entry, "violated", False),
                }
            )
        elif entry_type == "signal":
            span["attributes"].update(
                {
                    "agent_os.signal_type": getattr(entry, "signal", "unknown"),
                }
            )

        return span

    def get_spans(self) -> List[Dict[str, Any]]:
        """Get all entries as spans."""
        entries = []
        if hasattr(self.recorder, "get_entries"):
            entries = self.recorder.get_entries()
        elif hasattr(self.recorder, "entries"):
            entries = self.recorder.entries

        return [self._convert_entry(e, i) for i, e in enumerate(entries)]

    def import_to_store(self, store: Any) -> int:
        """
        Import spans to LightningStore.

        Args:
            store: LightningStore instance

        Returns:
            Number of spans imported
        """
        spans = self.get_spans()

        for span in spans:
            try:
                if hasattr(store, "emit_span"):
                    store.emit_span(span)
                elif hasattr(store, "add_span"):
                    store.add_span(span)
            except Exception as e:
                logger.error(f"Failed to import span: {e}")

        self._imported_count += len(spans)
        logger.info(f"Imported {len(spans)} spans to LightningStore")
        return len(spans)

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of policy violations."""
        spans = self.get_spans()
        violations = [s for s in spans if s["attributes"].get("agent_os.policy_violated", False)]
        return {
            "total_entries": len(spans),
            "total_violations": len(violations),
            "violation_rate": len(violations) / len(spans) if len(spans) > 0 else 0.0,
        }
