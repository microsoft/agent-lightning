"""Tests for error types."""

from agl_lite.schemas.errors import ConflictError, InvalidTransitionError, NotFoundError


class TestNotFoundError:
    def test_message(self):
        e = NotFoundError("Rollout", "r1")
        assert str(e) == "Rollout not found: r1"
        assert e.entity == "Rollout"
        assert e.entity_id == "r1"


class TestConflictError:
    def test_message(self):
        e = ConflictError("Rollout", "r1", expected=2, actual=3)
        assert "expected version 2" in str(e)
        assert "got 3" in str(e)


class TestInvalidTransitionError:
    def test_message(self):
        e = InvalidTransitionError("r1", "succeeded", "running")
        assert "cannot transition" in str(e)
        assert "succeeded → running" in str(e)
