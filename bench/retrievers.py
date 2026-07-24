"""Retriever boundary and dependency-free smoke implementation."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from uuid import UUID

from synapticdb import Synaptic
from synapticdb.embeddings import EmbeddingFunction
from synapticdb.learning import semantic_seed_config

from .contracts import MAX_FIXTURE_DIMENSIONS, MAX_MEMORY_COUNT, MAX_RECORD_CHARS, MAX_TOP_K
from .dataset import MemoryRecord

_TOKEN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class Retrieval:
    ranked_ids: tuple[str, ...]
    path_benchmark_ids: tuple[str, ...] = ()
    query_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.ranked_ids) > MAX_TOP_K or len(self.path_benchmark_ids) > MAX_MEMORY_COUNT:
            raise ValueError("retrieval evidence exceeds benchmark limits")
        if any(not benchmark_id for benchmark_id in self.ranked_ids + self.path_benchmark_ids):
            raise ValueError("retrieval IDs must be non-empty")
        if len(set(self.ranked_ids)) != len(self.ranked_ids):
            raise ValueError("ranked retrieval IDs must be unique")


class Retriever(Protocol):
    name: str

    def ingest(self, memories: Sequence[MemoryRecord], *, seed: int) -> None: ...

    def recall(self, text: str, *, top_k: int) -> Retrieval: ...

    def feedback(self, retrieval: Retrieval, *, positive: bool) -> None: ...

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
        return Retrieval(tuple(memory_id for memory_id, _ in ranked[:top_k]))

    def feedback(self, retrieval: Retrieval, *, positive: bool) -> None:
        if not isinstance(retrieval, Retrieval) or not isinstance(positive, bool):
            raise TypeError("feedback requires a Retrieval and boolean positive value")

    def close(self) -> None:
        return


@dataclass(frozen=True)
class SynapticConfig:
    embedding: str
    activation_blend_weight: float = 0.45
    semantic_seed_threshold: float = 0.6
    semantic_seed_max_links: int = 3
    semantic_seed_weight: float = 0.25

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
    ) -> None:
        if not embedding_name:
            raise ValueError("embedding_name must be non-empty")
        self._memory = Synaptic(":memory:", embedding_fn=embedding_fn)
        if semantic_seed is not None:
            self._memory._params["semantic_seed"] = semantic_seed
        semantic = semantic_seed_config(self._memory._params)
        self.config = SynapticConfig(
            embedding=embedding_name,
            semantic_seed_threshold=semantic.threshold,
            semantic_seed_max_links=semantic.max_links,
            semantic_seed_weight=semantic.initial_weight,
        )
        self._benchmark_ids: dict[UUID, str] = {}
        self._ingested = False

    def ingest(self, memories: Sequence[MemoryRecord], *, seed: int) -> None:
        # TODO(phase-5): PRD §10 requires a controlled ingestion schedule for
        # the 600 s temporal window, but Synaptic.remember() offers no
        # timestamp injection — ingesting in a tight loop would put every
        # consecutive pair in-window. Add a hook before temporal learning lands.
        del seed
        if self._ingested:
            raise RuntimeError("synaptic benchmark retriever only accepts one ingest")
        if not 1 <= len(memories) <= MAX_MEMORY_COUNT:
            raise ValueError(f"synaptic ingest accepts 1 to {MAX_MEMORY_COUNT} memories")
        if len({memory.benchmark_id for memory in memories}) != len(memories):
            raise ValueError("synaptic memory IDs must be unique")
        for record in memories:
            memory = self._memory.remember(record.content, record.metadata)
            self._benchmark_ids[memory.id] = record.benchmark_id
        if len(self._benchmark_ids) != len(memories):
            raise RuntimeError("benchmark corpus contains duplicate memory content")
        self._ingested = True

    def recall(self, text: str, *, top_k: int) -> Retrieval:
        if not self._ingested:
            raise RuntimeError("synaptic retriever must be ingested before recall")
        result = self._memory.recall(text, top_k=top_k)
        try:
            ranked = tuple(self._benchmark_ids[item.memory.id] for item in result.memories)
        except KeyError as exc:
            raise RuntimeError("Synaptic returned an unknown benchmark memory") from exc
        path_ids = self._path_benchmark_ids(result.query_id)
        return Retrieval(ranked, path_ids, str(result.query_id))

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
        # Phase 3 intentionally has no learning. Warm-up feedback is a no-op.

    def close(self) -> None:
        self._memory.close()


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
