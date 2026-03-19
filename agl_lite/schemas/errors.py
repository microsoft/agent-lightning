"""Error types for Store operations."""


class NotFoundError(Exception):
    """Raised when a requested entity does not exist."""

    def __init__(self, entity: str, entity_id: str) -> None:
        self.entity = entity
        self.entity_id = entity_id
        super().__init__(f"{entity} not found: {entity_id}")


class ConflictError(Exception):
    """Raised on optimistic locking failure (version mismatch)."""

    def __init__(self, entity: str, entity_id: str, expected: int, actual: int) -> None:
        self.entity = entity
        self.entity_id = entity_id
        self.expected = expected
        self.actual = actual
        super().__init__(f"{entity} {entity_id}: expected version {expected}, got {actual}")


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""

    def __init__(self, entity_id: str, from_status: str, to_status: str) -> None:
        self.entity_id = entity_id
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(f"Rollout {entity_id}: cannot transition {from_status} → {to_status}")
