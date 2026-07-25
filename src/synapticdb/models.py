"""Public pydantic models and the exception hierarchy.

Pure data definitions: no I/O, no imports from the rest of the package.
"""

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


class Memory(BaseModel):
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


class Recalled(BaseModel):
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


class RecallResult(BaseModel):
    query_id: UUID
    memories: list[Recalled]
    maturity: UnitFloat
    latency_ms: NonNegativeFloat

    @property
    def associative(self) -> list[Recalled]:
        return [r for r in self.memories if r.via == "association"]


class Stats(BaseModel):
    memories: NonNegativeInt
    edges: NonNegativeInt
    edges_by_origin: dict[EdgeOrigin, NonNegativeInt]
    maturity: UnitFloat
    db_path: str
