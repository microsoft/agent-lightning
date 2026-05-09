"""Tests for the reconciler — mock both AglLiteClient and K8s client."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agl_lite.client import AglLiteClient
from agl_lite.controller.config import ControllerSettings
from agl_lite.controller.job_builder import build_job_name
from agl_lite.controller.reconciler import Reconciler, _rollout_id_from_job
from agl_lite.schemas.rollout import Rollout, RolloutConfig, RolloutStatus


def _settings(**kwargs) -> ControllerSettings:
    defaults = {
        "base_url": "http://agl-lite:8000",
        "key": "test",
        "namespace": "default",
        "poll_interval": 1,
        "max_queue_time": 3600,
        "job_manifest_template": "deploy/controller/job-template.yaml.j2",
    }
    defaults.update(kwargs)
    return ControllerSettings(**defaults)


def _rollout(
    rollout_id: str = "r1",
    status: RolloutStatus = RolloutStatus.QUEUING,
    cancel_requested: bool = False,
    job_name: str | None = None,
    resources_id: str | None = None,
    created_at: float | None = None,
) -> Rollout:
    return Rollout(
        rollout_id=rollout_id,
        status=status,
        cancel_requested=cancel_requested,
        input={"task": "test"},
        config=RolloutConfig(),
        job_name=job_name,
        resources_id=resources_id,
        created_at=created_at or time.time(),
        updated_at=created_at or time.time(),
    )


def _job_dict(
    rollout_id: str = "r1",
    conditions: list[dict] | None = None,
) -> dict[str, Any]:
    """Minimal K8s Job dict."""
    job: dict[str, Any] = {
        "metadata": {
            "name": build_job_name(rollout_id),
            "namespace": "default",
            "labels": {
                "app.kubernetes.io/managed-by": "agl-lite",
                "agl-lite/rollout-id": rollout_id,
            },
        },
        "status": {},
    }
    if conditions:
        job["status"]["conditions"] = conditions
    return job


def _pod_dict(uid: str = "pod-uid-1", phase: str = "Succeeded", job_name: str = "agl-rollout-r1") -> dict[str, Any]:
    return {
        "metadata": {"uid": uid, "labels": {"job-name": job_name}},
        "status": {"phase": phase},
    }


class MockK8s:
    """Mock K8s client for testing."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}  # name → job dict
        self.pods: list[dict[str, Any]] = []
        self.create_job = AsyncMock(side_effect=self._create_job)
        self.delete_job = AsyncMock(side_effect=self._delete_job)
        self.get_job = AsyncMock(side_effect=self._get_job)
        self.list_jobs = AsyncMock(side_effect=self._list_jobs)
        self.list_pods = AsyncMock(side_effect=self._list_pods)
        self.watch_jobs = AsyncMock(side_effect=self._watch_jobs)
        self._watch_events: list[tuple[str, dict[str, Any]]] = []

    async def _create_job(self, manifest: dict) -> None:
        name = manifest["metadata"]["name"]
        self.jobs[name] = manifest

    async def _delete_job(self, name: str, namespace: str) -> None:
        self.jobs.pop(name, None)

    async def _get_job(self, name: str, namespace: str) -> dict | None:
        return self.jobs.get(name)

    async def _list_jobs(self, namespace: str, label_selector: str) -> list[dict]:
        return list(self.jobs.values())

    async def _list_pods(self, namespace: str, label_selector: str) -> list[dict]:
        return self.pods

    async def _watch_jobs(self, namespace: str, label_selector: str):
        """Return an async iterator over queued events, then block forever."""

        class _Watcher:
            def __init__(self, events):
                self._events = list(events)
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx < len(self._events):
                    event = self._events[self._idx]
                    self._idx += 1
                    return event
                # Block forever (simulates waiting for more events).
                await asyncio.sleep(3600)
                raise StopAsyncIteration

        return _Watcher(self._watch_events)


@pytest.fixture
def mock_k8s() -> MockK8s:
    return MockK8s()


@pytest.fixture
def mock_api() -> AsyncMock:
    api = AsyncMock(spec=AglLiteClient)
    api.query_rollouts = AsyncMock(return_value=[])
    api.patch_rollout = AsyncMock()
    return api


class TestReconcileCreateJobs:
    async def test_creates_job_for_queuing_rollout(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(
            side_effect=[
                [r],  # queuing rollouts
                [],  # running+cancelled
                [],  # running (crash recovery)
            ]
        )

        rec = Reconciler(mock_api, mock_k8s, _settings())
        await rec._reconcile_once()

        # Job should be created.
        mock_k8s.create_job.assert_called_once()
        manifest = mock_k8s.create_job.call_args[0][0]
        assert manifest["metadata"]["name"] == "agl-rollout-r1"

        # Rollout should be patched to running.
        mock_api.patch_rollout.assert_called()
        call_args = mock_api.patch_rollout.call_args_list[-1]
        assert call_args[0][0] == "r1"
        patch = call_args[0][1]
        assert patch.status == RolloutStatus.RUNNING

    async def test_skips_existing_job(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        """If Job already exists (crash recovery), just update status."""
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(side_effect=[[r], [], []])
        # Pre-create the Job.
        mock_k8s.jobs["agl-rollout-r1"] = _job_dict("r1")

        rec = Reconciler(mock_api, mock_k8s, _settings())
        await rec._reconcile_once()

        # Should NOT create a new Job.
        mock_k8s.create_job.assert_not_called()
        # Should patch to running.
        mock_api.patch_rollout.assert_called()

    async def test_job_creation_failure_stays_queuing(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(side_effect=[[r], [], []])
        mock_k8s.create_job = AsyncMock(side_effect=Exception("quota exceeded"))

        rec = Reconciler(mock_api, mock_k8s, _settings())
        await rec._reconcile_once()

        # Should NOT patch rollout (stays queuing, retry next cycle).
        mock_api.patch_rollout.assert_not_called()

    async def test_max_queue_time_exceeded(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        r = _rollout("r1", created_at=time.time() - 7200)  # 2 hours ago
        mock_api.query_rollouts = AsyncMock(side_effect=[[r], [], []])

        rec = Reconciler(mock_api, mock_k8s, _settings(max_queue_time=3600))
        await rec._reconcile_once()

        # Should be patched to terminal_failed.
        mock_api.patch_rollout.assert_called_once()
        patch = mock_api.patch_rollout.call_args[0][1]
        assert patch.status == RolloutStatus.TERMINAL_FAILED
        assert "max queue time" in patch.error_message


class TestPodCreationRateLimit:
    async def test_respects_max_pods_per_window(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        rollouts = [_rollout(f"r{i}") for i in range(3)]
        mock_api.query_rollouts = AsyncMock(side_effect=[rollouts, [], []])

        rec = Reconciler(mock_api, mock_k8s, _settings(max_pods_per_window=2, rate_limit_window_seconds=10))
        await rec._reconcile_once()

        assert mock_k8s.create_job.call_count == 2
        assert set(mock_k8s.jobs) == {"agl-rollout-r0", "agl-rollout-r1"}
        patched_ids = [call_args[0][0] for call_args in mock_api.patch_rollout.call_args_list]
        assert patched_ids == ["r0", "r1"]

    async def test_allows_creation_after_window_expires(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(side_effect=[[r], [], []])

        rec = Reconciler(mock_api, mock_k8s, _settings(max_pods_per_window=1, rate_limit_window_seconds=10))
        rec._record_pod_creation(now=time.monotonic() - 11)
        await rec._reconcile_once()

        mock_k8s.create_job.assert_called_once()
        assert len(rec._pod_creation_timestamps) == 1

    async def test_failed_creation_does_not_consume_capacity(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        rollouts = [_rollout("r1"), _rollout("r2")]
        mock_api.query_rollouts = AsyncMock(side_effect=[rollouts, [], []])

        async def create_job(manifest: dict) -> None:
            if manifest["metadata"]["name"] == "agl-rollout-r1":
                raise Exception("quota exceeded")
            await mock_k8s._create_job(manifest)

        mock_k8s.create_job = AsyncMock(side_effect=create_job)
        rec = Reconciler(mock_api, mock_k8s, _settings(max_pods_per_window=1, rate_limit_window_seconds=10))
        await rec._reconcile_once()

        assert mock_k8s.create_job.call_count == 2
        assert set(mock_k8s.jobs) == {"agl-rollout-r2"}
        mock_api.patch_rollout.assert_called_once()
        assert mock_api.patch_rollout.call_args[0][0] == "r2"
        assert len(rec._pod_creation_timestamps) == 1

    async def test_existing_job_status_repair_bypasses_rate_limit(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(side_effect=[[r], [], []])
        mock_k8s.jobs["agl-rollout-r1"] = _job_dict("r1")

        rec = Reconciler(mock_api, mock_k8s, _settings(max_pods_per_window=1, rate_limit_window_seconds=10))
        rec._record_pod_creation()
        await rec._reconcile_once()

        mock_k8s.create_job.assert_not_called()
        mock_api.patch_rollout.assert_called_once()
        patch = mock_api.patch_rollout.call_args[0][1]
        assert patch.status == RolloutStatus.RUNNING

    async def test_rate_limited_rollout_stays_queuing(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        r = _rollout("r1")
        mock_api.query_rollouts = AsyncMock(side_effect=[[r], [], []])

        rec = Reconciler(mock_api, mock_k8s, _settings(max_pods_per_window=1, rate_limit_window_seconds=10))
        rec._record_pod_creation()
        await rec._reconcile_once()

        mock_k8s.create_job.assert_not_called()
        mock_api.patch_rollout.assert_not_called()


class TestReconcileCancellation:
    async def test_cancel_queuing_rollout(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        r = _rollout("r1", cancel_requested=True)
        mock_api.query_rollouts = AsyncMock(side_effect=[[r], [], []])

        rec = Reconciler(mock_api, mock_k8s, _settings())
        await rec._reconcile_once()

        mock_api.patch_rollout.assert_called_once()
        patch = mock_api.patch_rollout.call_args[0][1]
        assert patch.status == RolloutStatus.CANCELLED

    async def test_cancel_running_rollout(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        r = _rollout("r1", status=RolloutStatus.RUNNING, cancel_requested=True, job_name="agl-rollout-r1")
        mock_api.query_rollouts = AsyncMock(
            side_effect=[
                [],  # queuing
                [r],  # running+cancelled
                [],  # running (crash recovery)
            ]
        )
        mock_k8s.jobs["agl-rollout-r1"] = _job_dict("r1")

        rec = Reconciler(mock_api, mock_k8s, _settings())
        await rec._reconcile_once()

        # Job should be deleted.
        mock_k8s.delete_job.assert_called_once_with("agl-rollout-r1", "default")
        # Rollout should be cancelled.
        mock_api.patch_rollout.assert_called()
        patch = mock_api.patch_rollout.call_args[0][1]
        assert patch.status == RolloutStatus.CANCELLED


class TestCrashRecovery:
    async def test_orphaned_running_rollout(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        """Running rollout whose Job no longer exists → terminal_failed."""
        r = _rollout("r1", status=RolloutStatus.RUNNING, job_name="agl-rollout-r1")
        mock_api.query_rollouts = AsyncMock(
            side_effect=[
                [],  # queuing
                [],  # running+cancelled
                [r],  # running (crash recovery)
            ]
        )
        # No jobs in K8s.

        rec = Reconciler(mock_api, mock_k8s, _settings())
        await rec._reconcile_once()

        mock_api.patch_rollout.assert_called_once()
        patch = mock_api.patch_rollout.call_args[0][1]
        assert patch.status == RolloutStatus.TERMINAL_FAILED
        assert "disappeared" in patch.error_message

    async def test_running_rollout_with_job_ok(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        """Running rollout with existing Job — no action needed."""
        r = _rollout("r1", status=RolloutStatus.RUNNING, job_name="agl-rollout-r1")
        mock_api.query_rollouts = AsyncMock(
            side_effect=[
                [],  # queuing
                [],  # running+cancelled
                [r],  # running (crash recovery)
            ]
        )
        mock_k8s.jobs["agl-rollout-r1"] = _job_dict("r1")

        rec = Reconciler(mock_api, mock_k8s, _settings())
        await rec._reconcile_once()

        mock_api.patch_rollout.assert_not_called()

    async def test_running_rollout_rechecks_job_when_list_is_stale(
        self,
        mock_api: AsyncMock,
        mock_k8s: MockK8s,
    ):
        """A stale list_jobs result must not mark a live Job as disappeared."""
        r = _rollout("r1", status=RolloutStatus.RUNNING, job_name="agl-rollout-r1")
        mock_api.query_rollouts = AsyncMock(
            side_effect=[
                [],  # queuing
                [],  # running+cancelled
                [r],  # running (crash recovery)
            ]
        )
        mock_k8s.jobs["agl-rollout-r1"] = _job_dict("r1")
        mock_k8s.list_jobs = AsyncMock(return_value=[])

        rec = Reconciler(mock_api, mock_k8s, _settings())
        await rec._reconcile_once()

        mock_k8s.get_job.assert_called_with("agl-rollout-r1", "default")
        mock_api.patch_rollout.assert_not_called()


class TestJobWatchEvents:
    async def test_job_complete(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        mock_k8s.pods = [_pod_dict(uid="pod-abc", phase="Succeeded")]

        rec = Reconciler(mock_api, mock_k8s, _settings())
        job = _job_dict("r1", conditions=[{"type": "Complete", "status": "True"}])
        await rec._handle_job_event(job)

        mock_api.patch_rollout.assert_called_once()
        patch = mock_api.patch_rollout.call_args[0][1]
        assert patch.status == RolloutStatus.SUCCEEDED
        assert patch.succeeded_attempt_id == "pod-abc"

    async def test_job_complete_no_pod(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        """Job completed but no succeeded pod found (GC race)."""
        mock_k8s.pods = []

        rec = Reconciler(mock_api, mock_k8s, _settings())
        job = _job_dict("r1", conditions=[{"type": "Complete", "status": "True"}])
        await rec._handle_job_event(job)

        patch = mock_api.patch_rollout.call_args[0][1]
        assert patch.status == RolloutStatus.SUCCEEDED
        assert patch.succeeded_attempt_id is None

    async def test_job_failed_backoff(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        rec = Reconciler(mock_api, mock_k8s, _settings())
        job = _job_dict(
            "r1",
            conditions=[
                {
                    "type": "Failed",
                    "status": "True",
                    "reason": "BackoffLimitExceeded",
                    "message": "Job has reached the specified backoff limit",
                },
            ],
        )
        await rec._handle_job_event(job)

        patch = mock_api.patch_rollout.call_args[0][1]
        assert patch.status == RolloutStatus.TERMINAL_FAILED
        assert "BackoffLimitExceeded" in patch.error_message

    async def test_job_failed_deadline(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        rec = Reconciler(mock_api, mock_k8s, _settings())
        job = _job_dict(
            "r1",
            conditions=[
                {"type": "Failed", "status": "True", "reason": "DeadlineExceeded", "message": ""},
            ],
        )
        await rec._handle_job_event(job)

        patch = mock_api.patch_rollout.call_args[0][1]
        assert patch.status == RolloutStatus.TERMINAL_FAILED
        assert "DeadlineExceeded" in patch.error_message

    async def test_ignores_non_true_conditions(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        rec = Reconciler(mock_api, mock_k8s, _settings())
        job = _job_dict(
            "r1",
            conditions=[
                {"type": "Complete", "status": "False"},
            ],
        )
        await rec._handle_job_event(job)

        mock_api.patch_rollout.assert_not_called()

    async def test_ignores_job_without_label(self, mock_api: AsyncMock, mock_k8s: MockK8s):
        rec = Reconciler(mock_api, mock_k8s, _settings())
        job = {"metadata": {"name": "some-job", "labels": {}}, "status": {}}
        await rec._handle_job_event(job)

        mock_api.patch_rollout.assert_not_called()


class TestHelpers:
    def test_rollout_id_from_job(self):
        job = _job_dict("r42")
        assert _rollout_id_from_job(job) == "r42"

    def test_rollout_id_from_job_missing(self):
        job = {"metadata": {"labels": {}}}
        assert _rollout_id_from_job(job) is None
