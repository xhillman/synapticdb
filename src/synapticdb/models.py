"""Public pydantic models and the exception hierarchy.

Pure data definitions: no I/O, no imports from the rest of the package.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
EdgeOrigin = Literal["semantic", "temporal", "co_retrieval", "explicit"]
RecallSource = Literal["search", "association", "both"]


class SynapticError(Exception):
    """Base class for all synapticdb errors."""


class NotFoundError(SynapticError):
    """Unknown memory_id or query_id."""


class InvalidArgumentError(SynapticError):
    """Bad input, including repeated feedback for the same query_id."""


class EmbeddingError(SynapticError):
    """Embedding function missing, failed, or returned the wrong dimension."""


def unit_float(value: float, label: str) -> float:
    """Validate a weight, energy, score, or threshold on the closed unit scale.

    Lives here because every layer needs it and this is the module they all
    already import. Booleans are rejected rather than coerced: `True` is a
    caller mistake, not the weight 1.0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidArgumentError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise InvalidArgumentError(f"{label} must be between 0 and 1")
    return number


class _SerializableModel(BaseModel):
    """Give each public model one explicit JSON export interface."""

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary containing only JSON-compatible values."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Return the model encoded as JSON."""
        return self.model_dump_json()


class Memory(_SerializableModel):
    id: UUID
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    last_accessed_at: datetime
    access_count: NonNegativeInt = 0

    @field_validator("created_at", "last_accessed_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("timestamp must use UTC")
        return value.astimezone(timezone.utc)


class Recalled(_SerializableModel):
    memory: Memory
    # Blended ranking strength within this query. Comparable between results of
    # one recall, not between recalls: it is min-max normalized per query.
    score: UnitFloat
    # Evidence that this memory addresses the query, comparable across queries.
    # Threshold on this, not on `score`. An association has weak textual
    # evidence by construction — that is why search missed it — so it scores
    # low here even when the graph is right about it.
    confidence: UnitFloat
    via: RecallSource


class RecallResult(_SerializableModel):
    query_id: UUID
    memories: list[Recalled]
    maturity: UnitFloat
    latency_ms: NonNegativeFloat

    @property
    def associative(self) -> list[Recalled]:
        return [r for r in self.memories if r.via == "association"]


class Stats(_SerializableModel):
    memories: NonNegativeInt
    edges: NonNegativeInt
    edges_by_origin: dict[EdgeOrigin, NonNegativeInt]
    maturity: UnitFloat
    db_path: str
