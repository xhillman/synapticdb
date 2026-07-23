"""Public synchronous Synaptic API."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from uuid import UUID

from synapticdb.confidence import ConfidenceCache, GraphMetrics
from synapticdb.embeddings import Embedder, EmbeddingFunction
from synapticdb.models import InvalidArgumentError, Memory, Recalled, RecallResult, Stats
from synapticdb.retrieval import RankedHit, min_max_normalize, reciprocal_rank_fusion
from synapticdb.store import GraphSummary, Store

# TODO(phase-4): candidate depth (PRD §9 parameter 2) is also defined in
# store.py; give it a single source of truth when the blend lands.
_CANDIDATE_LIMIT = 40
_MAX_FILTER_KEYS = 64
_MAX_TEXT_CHARS = 1_000_000
_MAX_TOP_K = 100


class Synaptic:
    """Own one memory database and its retrieval resources."""

    def __init__(
        self,
        db_path: str | Path,
        embedding_fn: EmbeddingFunction | None = None,
    ) -> None:
        self._store = Store(db_path)
        self._confidence = ConfidenceCache()
        self._closed = False
        try:
            self._embedder = Embedder(
                embedding_fn,
                expected_dimension=self._store.embedding_dimension(),
            )
        except Exception:
            self._store.close()
            self._closed = True
            raise

    def remember(
        self,
        content: str,
        metadata: Mapping[str, object] | None = None,
    ) -> Memory:
        self._require_open()
        text = _bounded_text(content, "content")
        stored_metadata = _metadata(metadata)
        embedding = self._embedder.embed(text)
        memory = self._store.insert_memory(text, stored_metadata, embedding)
        self._confidence.invalidate()
        return memory

    def recall(
        self,
        query: str,
        *,
        top_k: int = 10,
        where: Mapping[str, object] | None = None,
    ) -> RecallResult:
        self._require_open()
        started = time.perf_counter()
        text = _bounded_text(query, "query")
        result_limit = _top_k(top_k)
        filters = _where_filter(where)
        embedding = self._embedder.embed(text)
        keyword = self._store.keyword_search(text, limit=_CANDIDATE_LIMIT)
        semantic = self._store.semantic_search(embedding, limit=_CANDIDATE_LIMIT)
        fused = reciprocal_rank_fusion(
            (
                tuple(hit.memory_id for hit in keyword),
                tuple(hit.memory_id for hit in semantic),
            )
        )
        fused_scores = min_max_normalize({hit.memory_id: hit.score for hit in fused})
        selected_ids, selected_scores = self._select_results(fused, fused_scores, filters, result_limit)
        maturity = self._maturity(self._store.graph_summary())
        # Fusion-only results carry energy 1.0 (PRD §6.6); Phase 4 replaces
        # these with real activation energies.
        query_row, memories = self._store.record_recall(
            text,
            selected_ids,
            {memory_id: 1.0 for memory_id in selected_ids},
            (),
        )
        recalled = [Recalled(memory=memory, score=selected_scores[memory.id], via="search") for memory in memories]
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RecallResult(
            query_id=query_row.id,
            memories=recalled,
            maturity=maturity,
            latency_ms=latency_ms,
        )

    def forget(self, memory_id: UUID) -> None:
        self._require_open()
        if not isinstance(memory_id, UUID):
            raise InvalidArgumentError("memory_id must be a UUID")
        self._store.forget_memory(memory_id)
        self._confidence.invalidate()

    def stats(self) -> Stats:
        self._require_open()
        summary = self._store.graph_summary()
        return Stats(
            memories=summary.memory_count,
            edges=summary.edge_count,
            edges_by_origin=summary.edges_by_origin(),
            maturity=self._maturity(summary),
            db_path=self._store.db_path,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._store.close()
        self._closed = True

    def __enter__(self) -> Synaptic:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _select_results(
        self,
        fused: tuple[RankedHit, ...],
        scores: Mapping[UUID, float],
        filters: Mapping[str, object],
        limit: int,
    ) -> tuple[tuple[UUID, ...], dict[UUID, float]]:
        ranked_ids = tuple(hit.memory_id for hit in fused)
        memories = self._store.get_memories(ranked_ids)
        selected_ids: list[UUID] = []
        for memory in memories:
            if _metadata_matches(memory.metadata, filters):
                selected_ids.append(memory.id)
            if len(selected_ids) == limit:
                break
        identifiers = tuple(selected_ids)
        return identifiers, {memory_id: scores[memory_id] for memory_id in identifiers}

    def _maturity(self, summary: GraphSummary) -> float:
        metrics = GraphMetrics(
            node_count=summary.memory_count,
            edge_count=summary.edge_count,
            average_edge_weight=summary.average_edge_weight,
            average_reinforcement_count=summary.average_reinforcement_count,
        )
        return self._confidence.get(metrics)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Synaptic instance is closed")


def _bounded_text(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise InvalidArgumentError(f"{label} must be a string")
    if not value.strip():
        raise InvalidArgumentError(f"{label} must not be blank")
    if len(value) > _MAX_TEXT_CHARS:
        raise InvalidArgumentError(f"{label} exceeds {_MAX_TEXT_CHARS} characters")
    return value


def _top_k(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= _MAX_TOP_K:
        raise InvalidArgumentError(f"top_k must be an integer between 1 and {_MAX_TOP_K}")
    return value


def _metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping) or not all(isinstance(key, str) for key in metadata):
        raise InvalidArgumentError("metadata must be a mapping with string keys")
    return dict(metadata)


def _where_filter(where: Mapping[str, object] | None) -> dict[str, object]:
    if where is None:
        return {}
    if not isinstance(where, Mapping) or not all(isinstance(key, str) for key in where):
        raise InvalidArgumentError("where must be a mapping with string keys")
    if len(where) > _MAX_FILTER_KEYS:
        raise InvalidArgumentError(f"where accepts at most {_MAX_FILTER_KEYS} keys")
    try:
        json.dumps(dict(where), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InvalidArgumentError("where values must be JSON-serializable") from exc
    return dict(where)


def _metadata_matches(metadata: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    return all(key in metadata and metadata[key] == value for key, value in filters.items())
