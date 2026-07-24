"""Public synchronous Synaptic API."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from uuid import UUID

from synapticdb.activation import (
    ACTIVATION_SEED_COUNT,
    ActivationResult,
    Neighbor,
    spread_activation,
)
from synapticdb.confidence import ConfidenceCache, GraphMetrics
from synapticdb.embeddings import Embedder, EmbeddingFunction
from synapticdb.learning import (
    ParameterValue,
    default_parameters,
    semantic_seed_config,
    semantic_seed_ids,
)
from synapticdb.models import InvalidArgumentError, Memory, Recalled, RecallResult, Stats
from synapticdb.retrieval import (
    BlendedHit,
    blend_rankings,
    min_max_normalize,
    reciprocal_rank_fusion,
)
from synapticdb.store import CANDIDATE_LIMIT, EdgeSeed, GraphSummary, Store

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
        self._params: dict[str, ParameterValue] = default_parameters()
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
        config = semantic_seed_config(self._params)
        candidates = self._store.semantic_search(embedding, limit=config.max_links)
        candidate_scores = tuple((hit.memory_id, hit.score) for hit in candidates)
        linked_ids = semantic_seed_ids(candidate_scores, config)
        seeds = tuple(EdgeSeed(memory_id, config.initial_weight, "semantic") for memory_id in linked_ids)
        result = self._store.insert_memory_with_edges(text, stored_metadata, embedding, seeds)
        if result.inserted:
            self._confidence.invalidate()
        return result.memory

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
        ranked, activation, maturity = self._rank_candidates(text, embedding)
        selected_ids, selected = self._select_results(ranked, filters, result_limit)
        energies = _result_energies(selected_ids, activation)
        query_row, memories = self._store.record_recall(
            text,
            selected_ids,
            energies,
            activation.path_edge_ids,
        )
        recalled = [
            Recalled(memory=memory, score=selected[memory.id].score, via=selected[memory.id].via)
            for memory in memories
        ]
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RecallResult(
            query_id=query_row.id,
            memories=recalled,
            maturity=maturity,
            latency_ms=latency_ms,
        )

    def _rank_candidates(
        self,
        text: str,
        embedding: Sequence[float],
    ) -> tuple[tuple[BlendedHit, ...], ActivationResult, float]:
        keyword = self._store.keyword_search(text, limit=CANDIDATE_LIMIT)
        semantic = self._store.semantic_search(embedding, limit=CANDIDATE_LIMIT)
        fused = reciprocal_rank_fusion(
            (
                tuple(hit.memory_id for hit in keyword),
                tuple(hit.memory_id for hit in semantic),
            )
        )
        fused_scores = min_max_normalize({hit.memory_id: hit.score for hit in fused})
        maturity = self._maturity(self._store.graph_summary())
        seeds = tuple(
            (hit.memory_id, fused_scores[hit.memory_id])
            for hit in fused[:ACTIVATION_SEED_COUNT]
        )
        activation = spread_activation(seeds, self._activation_neighbors)
        activation_scores = {hit.memory_id: hit.score for hit in activation.hits}
        ranked = blend_rankings(fused_scores, activation_scores, maturity)
        return ranked, activation, maturity

    def _activation_neighbors(self, memory_id: UUID) -> tuple[Neighbor, ...]:
        # TODO(phase-5): PRD §5.2 spreads via *effective* edge weight (§6.4);
        # edge.weight is the stored, undecayed value. Switch to the decayed
        # weight when lazy decay lands, or activation ignores edge aging.
        neighbors: list[Neighbor] = []
        for edge in self._store.list_edges_for_node(memory_id):
            target_id = edge.b if edge.a == memory_id else edge.a
            neighbors.append(Neighbor(target_id, edge.id, edge.weight))
        return tuple(neighbors)

    def _select_results(
        self,
        ranked: tuple[BlendedHit, ...],
        filters: Mapping[str, object],
        limit: int,
    ) -> tuple[tuple[UUID, ...], dict[UUID, BlendedHit]]:
        ranked_ids = tuple(hit.memory_id for hit in ranked)
        memories = self._store.get_memories(ranked_ids)
        hits_by_id = {hit.memory_id: hit for hit in ranked}
        selected_ids: list[UUID] = []
        for memory in memories:
            if _metadata_matches(memory.metadata, filters):
                selected_ids.append(memory.id)
            if len(selected_ids) == limit:
                break
        identifiers = tuple(selected_ids)
        return identifiers, {memory_id: hits_by_id[memory_id] for memory_id in identifiers}

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


def _result_energies(
    result_ids: Sequence[UUID],
    activation: ActivationResult,
) -> dict[UUID, float]:
    activation_energies = {hit.memory_id: hit.energy for hit in activation.hits}
    return {memory_id: activation_energies.get(memory_id, 1.0) for memory_id in result_ids}
