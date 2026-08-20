"""Retriever boundary and dependency-free smoke implementation."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from uuid import UUID

from synapticdb import SynapticDB
from synapticdb.embeddings import EmbeddingFunction
from synapticdb.learning import (
    RuntimePolicy,
    policy_parameters,
    runtime_policy,
)

from .contracts import MAX_FIXTURE_DIMENSIONS, MAX_MEMORY_COUNT, MAX_RECORD_CHARS, MAX_TOP_K
from .dataset import MemoryRecord

_TOKEN = re.compile(r"[a-z0-9_]+")
INGEST_EPOCH = datetime(2030, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Retrieval:
    ranked_ids: tuple[str, ...]
    path_benchmark_ids: tuple[str, ...] = ()
    query_id: str | None = None
    # Confidence per ranked ID, in the same order. Optional because a retriever
    # that cannot supply it simply forfeits calibration scoring rather than
    # failing. Scales need not match across retrievers: calibration only ever
    # compares one retriever against itself.
    scores: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.ranked_ids) > MAX_TOP_K or len(self.path_benchmark_ids) > MAX_MEMORY_COUNT:
            raise ValueError("retrieval evidence exceeds benchmark limits")
        if any(not benchmark_id for benchmark_id in self.ranked_ids + self.path_benchmark_ids):
            raise ValueError("retrieval IDs must be non-empty")
        if len(set(self.ranked_ids)) != len(self.ranked_ids):
            raise ValueError("ranked retrieval IDs must be unique")
        if self.scores and len(self.scores) != len(self.ranked_ids):
            raise ValueError("retrieval scores must align with ranked IDs")


class Retriever(Protocol):
    name: str

    def ingest(self, memories: Sequence[MemoryRecord], *, seed: int) -> None: ...

    def recall(self, text: str, *, top_k: int) -> Retrieval: ...

    def feedback(self, retrieval: Retrieval, *, positive: bool) -> None: ...

    def advance_to(self, moment: datetime) -> None:
        """Move the retriever's clock, so the harness can simulate elapsed time.

        A no-op for time-invariant retrievers. Implemented explicitly by each
        retriever rather than probed for, so a new implementation has to make a
        deliberate choice about time.
        """
        ...

    def close(self) -> None: ...


class FixtureRetriever:
    """Deterministic hashed-embedding retriever used only for CI plumbing."""

    name = "fixture"

    def __init__(self) -> None:
        self._docs: list[tuple[str, tuple[float, ...]]] = []

    def ingest(self, memories: Sequence[MemoryRecord], *, seed: int) -> None:
        del seed
        if not 1 <= len(memories) <= MAX_MEMORY_COUNT:
            raise ValueError(f"fixture ingest accepts 1 to {MAX_MEMORY_COUNT} memories")
        if len({memory.benchmark_id for memory in memories}) != len(memories):
            raise ValueError("fixture memory IDs must be unique")
        if any(not memory.content.strip() or len(memory.content) > MAX_RECORD_CHARS for memory in memories):
            raise ValueError("fixture memories require bounded non-empty content")
        self._docs = [(memory.benchmark_id, _fixture_embedding(memory.content)) for memory in memories]

    def recall(self, text: str, *, top_k: int) -> Retrieval:
        if not self._docs:
            raise RuntimeError("fixture retriever must be ingested before recall")
        if not text.strip() or len(text) > MAX_RECORD_CHARS or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError("recall requires non-empty text and a bounded top_k")
        query = _fixture_embedding(text)
        ranked = sorted(self._docs, key=lambda item: (-_dot(query, item[1]), item[0]))
        selected = ranked[:top_k]
        return Retrieval(
            tuple(memory_id for memory_id, _ in selected),
            scores=tuple(_dot(query, vector) for _, vector in selected),
        )

    def feedback(self, retrieval: Retrieval, *, positive: bool) -> None:
        if not isinstance(retrieval, Retrieval) or not isinstance(positive, bool):
            raise TypeError("feedback requires a Retrieval and boolean positive value")

    def advance_to(self, moment: datetime) -> None:
        # Time-invariant: no learning, no decay, so the clock cannot matter.
        del moment

    def close(self) -> None:
        return


@dataclass(frozen=True)
class SynapticConfig:
    """The full effective parameter set behind one benchmark run.

    Records every PRD §9 group rather than a hand-picked few, so a promoted
    record states exactly the configuration that produced it and a calibration
    sweep cannot be misread as a default-configuration result.
    """

    embedding: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SynapticRetriever:
    """Adapt the Phase 3 public API to benchmark-owned memory IDs."""

    name = "synaptic"

    def __init__(
        self,
        embedding_fn: EmbeddingFunction | None = None,
        *,
        embedding_name: str = "default",
        semantic_seed: tuple[float, int, float] | None = None,
        temporal_link: tuple[int, int, float] | None = None,
        overrides: Mapping[str, object] | None = None,
    ) -> None:
        if not embedding_name:
            raise ValueError("embedding_name must be non-empty")
        policy = _benchmark_policy(semantic_seed, temporal_link, overrides)
        self._memory = SynapticDB._with_policy(":memory:", embedding_fn, policy)
        parameters = policy_parameters(policy)
        self.config = SynapticConfig(
            embedding=embedding_name,
            params={key: _json_value(value) for key, value in sorted(parameters.items())},
        )
        self._benchmark_ids: dict[UUID, str] = {}
        self._ingested = False
        # None means "use the wall clock", which is how every record predating
        # the harness clock was produced. advance_to opts into simulated time.
        self._now: datetime | None = None

    def ingest(self, memories: Sequence[MemoryRecord], *, seed: int) -> None:
        del seed
        if self._ingested:
            raise RuntimeError("synaptic benchmark retriever only accepts one ingest")
        if not 1 <= len(memories) <= MAX_MEMORY_COUNT:
            raise ValueError(f"synaptic ingest accepts 1 to {MAX_MEMORY_COUNT} memories")
        if len({memory.benchmark_id for memory in memories}) != len(memories):
            raise ValueError("synaptic memory IDs must be unique")
        for record in memories:
            created_at = INGEST_EPOCH + timedelta(seconds=record.ingest_offset_seconds)
            memory = self._memory._remember_at(record.content, record.metadata, created_at)
            self._benchmark_ids[memory.id] = record.benchmark_id
        if len(self._benchmark_ids) != len(memories):
            raise RuntimeError("benchmark corpus contains duplicate memory content")
        self._ingested = True

    def recall(self, text: str, *, top_k: int) -> Retrieval:
        if not self._ingested:
            raise RuntimeError("synaptic retriever must be ingested before recall")
        if self._now is None:
            result = self._memory.recall(text, top_k=top_k)
        else:
            result = self._memory._recall_at(text, top_k, None, self._now)
        try:
            ranked = tuple(self._benchmark_ids[item.memory.id] for item in result.memories)
        except KeyError as exc:
            raise RuntimeError("SynapticDB returned an unknown benchmark memory") from exc
        path_ids = self._path_benchmark_ids(result.query_id)
        return Retrieval(
            ranked,
            path_ids,
            str(result.query_id),
            # Confidence, not score: calibration asks whether a number can be
            # thresholded across queries, and `score` is normalized within one.
            scores=tuple(item.confidence for item in result.memories),
        )

    def _path_benchmark_ids(self, query_id: UUID) -> tuple[str, ...]:
        query = self._memory._store.get_query(query_id)
        path_ids: list[str] = []
        seen: set[str] = set()
        for edge_id in query.path_edge_ids:
            edge = self._memory._store.get_edge(edge_id)
            self._append_path_memory_ids((edge.a, edge.b), path_ids, seen)
        return tuple(path_ids)

    def _append_path_memory_ids(
        self,
        memory_ids: tuple[UUID, UUID],
        path_ids: list[str],
        seen: set[str],
    ) -> None:
        for memory_id in memory_ids:
            benchmark_id = self._benchmark_ids.get(memory_id)
            if benchmark_id is None:
                raise RuntimeError("activation path contains an unknown benchmark memory")
            if benchmark_id in seen:
                continue
            path_ids.append(benchmark_id)
            seen.add(benchmark_id)

    def feedback(self, retrieval: Retrieval, *, positive: bool) -> None:
        if not isinstance(retrieval, Retrieval) or not isinstance(positive, bool):
            raise TypeError("feedback requires a Retrieval and boolean positive value")
        if retrieval.query_id is None:
            raise RuntimeError("synaptic feedback requires a retrieval carrying its query_id")
        query_id = UUID(retrieval.query_id)
        if self._now is None:
            self._memory.feedback(query_id, positive=positive)
            return
        self._memory._feedback_at(query_id, positive, self._now)

    def advance_to(self, moment: datetime) -> None:
        """Set the instant this retriever reads and writes the graph at.

        Until this is called the retriever uses the wall clock, which is what
        reproduces every record taken before the harness had a clock.
        """
        if not isinstance(moment, datetime) or moment.tzinfo is None:
            raise ValueError("advance_to requires an aware datetime")
        current = moment.astimezone(timezone.utc)
        if self._now is not None and current < self._now:
            raise ValueError("the benchmark clock only moves forward")
        self._now = current

    def close(self) -> None:
        self._memory.close()


def _benchmark_policy(
    semantic_seed: tuple[float, int, float] | None,
    temporal_link: tuple[int, int, float] | None,
    overrides: Mapping[str, object] | None,
) -> RuntimePolicy:
    """Build the complete policy before opening benchmark resources."""
    selected: dict[str, object] = {}
    if semantic_seed is not None:
        selected["semantic_seed"] = semantic_seed
    if temporal_link is not None:
        selected["temporal_link"] = temporal_link
    selected.update(overrides or {})
    return runtime_policy(selected)


def _json_value(value: object) -> Any:
    """Render one parameter for the report; tuples become JSON arrays."""
    return list(value) if isinstance(value, tuple) else value


def _fixture_embedding(text: str, dimensions: int = 128) -> tuple[float, ...]:
    if not text.strip():
        raise ValueError("fixture embedding text must be non-empty")
    if not 1 <= dimensions <= MAX_FIXTURE_DIMENSIONS:
        raise ValueError(f"dimensions must be between 1 and {MAX_FIXTURE_DIMENSIONS}")
    values = [0.0] * dimensions
    for token in _TOKEN.findall(text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        values[bucket] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return tuple(value / norm for value in values)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("dot-product vectors must have equal non-zero dimensions")
    return sum(a * b for a, b in zip(left, right, strict=True))
