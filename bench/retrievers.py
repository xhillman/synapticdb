"""Retriever boundary and dependency-free smoke implementation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Protocol, Sequence

from .dataset import MemoryRecord

_TOKEN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class Retrieval:
    ranked_ids: tuple[str, ...]
    path_benchmark_ids: tuple[str, ...] = ()
    query_id: str | None = None


class Retriever(Protocol):
    name: str

    def ingest(self, memories: Sequence[MemoryRecord], *, seed: int) -> None: ...

    def recall(self, text: str, *, top_k: int) -> Retrieval: ...

    def feedback(self, retrieval: Retrieval, *, positive: bool) -> None: ...


class FixtureRetriever:
    """Deterministic hashed-embedding retriever used only for CI plumbing."""

    name = "fixture"

    def __init__(self) -> None:
        self._docs: list[tuple[str, tuple[float, ...]]] = []

    def ingest(self, memories: Sequence[MemoryRecord], *, seed: int) -> None:
        del seed
        self._docs = [(memory.benchmark_id, _fixture_embedding(memory.content)) for memory in memories]

    def recall(self, text: str, *, top_k: int) -> Retrieval:
        query = _fixture_embedding(text)
        ranked = sorted(self._docs, key=lambda item: (-_dot(query, item[1]), item[0]))
        return Retrieval(tuple(memory_id for memory_id, _ in ranked[:top_k]))

    def feedback(self, retrieval: Retrieval, *, positive: bool) -> None:
        del retrieval, positive


def _fixture_embedding(text: str, dimensions: int = 128) -> tuple[float, ...]:
    values = [0.0] * dimensions
    for token in _TOKEN.findall(text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        values[bucket] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return tuple(value / norm for value in values)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))
