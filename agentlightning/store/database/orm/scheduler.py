# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from agentlightning.types.core import Rollout, Attempt
from .rollout import RolloutInDB
from .attempt import AttemptInDB
from .base import (
    DatabaseRuntimeError,
    RaceConditionError,
    NoRolloutToDequeueError,
)


class SchedulerInDB:

    def __init__(
        self, database: Database, table_rollouts: str, table_attempts: str,
    ) -> None:
        self._database = database
        self.table_rollouts = table_rollouts
        self.table_attempts = table_attempts

    def start_attempt_for_rollout(self, rollout: RolloutInDB) -> tuple[AttemptInDB, dict[str, Any]]:
        """Create a new AttemptInDB for the given RolloutInDB.
        Returns the new AttemptInDB and the list of fields updated in the RolloutInDB.
        """
        new_attempt = AttemptInDB(
            rollout_id=rollout.rollout_id,
            sequence_id=rollout.num_attempts + 1,
            status="preparing",
        )
        # Update the rollout's attempt count and latest attempt id
        rollout_to_update = {
            "num_attempts": rollout.num_attempts + 1,
            "latest_attempt_id": new_attempt.attempt_id,
            "status": "preparing",
            "enqueue_time": None,  # Clear enqueue time as it's being processed
        }
        rollout.update(rollout_to_update)

        return new_attempt, rollout_to_update

    async def dequeue_next_rollout_step(self) -> tuple[RolloutInDB, AttemptInDB]:
        """A single step to dequeue the next rollout and create its attempt."""
        # find the rollout with the earliest enqueue_time that is still queuing or requeuing
        # use atomic update status to preparing to avoid race conditions
        async with self._database.transaction():
            # Step 1: Select the row to update
            SELECT_QUERY = f"""
            SELECT *
            FROM {self.table_rollouts}
            WHERE status IN ('queuing', 'requeuing') AND enqueue_time IS NOT NULL
            ORDER BY enqueue_time ASC
            LIMIT 1;
            """
            row = await self._database.fetch_one(query=SELECT_QUERY) # type: ignore
            if row is None:
                raise NoRolloutToDequeueError("No rollout available to dequeue.")

            # Step 2: claim the rollout by updating its status to 'preparing'
            rollout_obj: RolloutInDB = RolloutInDB.from_record(row)
            current_status = rollout_obj.status # store current status for race condition check
            attempt_obj, rollout_update_fields = self.start_attempt_for_rollout(rollout_obj)

            update_result = await rollout_obj.update_in_db(
                self._database,
                self.table_rollouts,
                {"rollout_id": rollout_obj.rollout_id, "status": current_status},
                rollout_update_fields
            )
            if update_result is None: # no row was updated, another worker might have taken it
                raise RaceConditionError("Race condition detected while trying to dequeue rollout.")

            # Step 3: Insert the new attempt into the database
            await attempt_obj.insert_into_db(self._database, self.table_attempts)

            return rollout_obj, attempt_obj

    async def dequeue_next_rollout(self) -> tuple[RolloutInDB, AttemptInDB]:
        """Dequeue the next rollout to be processed based on FIFO scheduling.
        This is a placeholder implementation and should be replaced with actual database queries.
        """
        while True:
            try:
                return await self.dequeue_next_rollout_step()
            except RaceConditionError:
                # Another worker has taken the rollout, retry
                # print("Race condition detected, retrying dequeue operation.")
                # all_rollouts = await RolloutInDB.query_rollouts(self._database, self.table_rollouts)
                # print(f"Current rollouts in DB: {[r.model_dump() for r in all_rollouts]}")
                # raise DatabaseRuntimeError("Exceeded retry attempts due to race conditions.")
                continue # FIXME add max retry count
            except NoRolloutToDequeueError:
                # No rollout available to dequeue
                return None, None
            except Exception as e:
                logging.error(f"Unexpected error during dequeue operation: {e}")
                raise DatabaseRuntimeError(f"Unexpected error during dequeue operation: {e}")

