"""Public synchronous SynapticDB API."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import TracebackType
from uuid import UUID

from synapticdb.activation import ActivationResult, Neighbor, spread_activation
from synapticdb.confidence import ConfidenceCache, GraphMetrics
from synapticdb.embeddings import Embedder, EmbeddingFunction
from synapticdb.learning import (
    DEFAULT_RUNTIME_POLICY,
    LINKED_RESULT_COUNT,
    RuntimePolicy,
    co_retrieval_pairs,
    negative_feedback_weight,
    positive_feedback_seed,
    semantic_seed_ids,
    unordered_pairs,
)
from synapticdb.models import (
    InvalidArgumentError,
    Memory,
    Recalled,
    RecallResult,
    Stats,
    unit_float,
)
from synapticdb.retrieval import (
    BlendedHit,
    blend_rankings,
    min_max_normalize,
    reciprocal_rank_fusion,
)
from synapticdb.store import (
    Edge,
    EdgeSeed,
    GraphSummary,
    PairSeed,
    QueryRow,
    Store,
)

_MAX_FILTER_KEYS = 64
_MAX_TEXT_CHARS = 1_000_000
_MAX_TOP_K = 100
# Result pairs are bounded by top_k, but a query row's path_edge_ids is not, so
# feedback needs its own ceiling. C(100, 2) covers every pair a maximal recall
# can produce; anything past that is a corrupt or hand-built query row.
_MAX_FEEDBACK_EDGES = _MAX_TOP_K * (_MAX_TOP_K - 1) // 2


@dataclass(frozen=True, slots=True)
class _FeedbackTarget:
    """One edge feedback will update, with its row when already loaded."""

    first_id: UUID
    second_id: UUID
    edge: Edge | None


class SynapticDB:
    """Own one memory database and its retrieval resources."""

    def __init__(
        self,
        db_path: str | Path,
        embedding_fn: EmbeddingFunction | None = None,
    ) -> None:
        self._initialize(db_path, embedding_fn, DEFAULT_RUNTIME_POLICY)

    @classmethod
    def _with_policy(
        cls,
        db_path: str | Path,
        embedding_fn: EmbeddingFunction | None,
        policy: RuntimePolicy,
    ) -> SynapticDB:
        """Build an internal runtime with one complete policy."""
        instance = cls.__new__(cls)
        instance._initialize(db_path, embedding_fn, policy)
        return instance

    def _initialize(
        self,
        db_path: str | Path,
        embedding_fn: EmbeddingFunction | None,
        policy: RuntimePolicy,
    ) -> None:
        self._store = Store(db_path)
        self._confidence = ConfidenceCache()
        self._policy = policy
        try:
            self._embedder = Embedder(
                embedding_fn,
                expected_dimension=self._store.embedding_dimension(),
            )
        except Exception:
            self._store.close()
            raise

    def store(
        self,
        content: str,
        metadata: Mapping[str, object] | None = None,
    ) -> Memory:
        """Store one memory and link it to what was stored around it.

        Content is deduplicated by hash, so storing the same text twice
        returns the first memory and creates no second copy or new edges.
        Storing also grows the association graph: memories written close
        together in time gain a temporal edge, and every hundredth insert runs
        the maintenance pass that prunes edges decayed below the floor.
        """
        return self._store_at(content, metadata, datetime.now(timezone.utc))

    def _store_at(
        self,
        content: str,
        metadata: Mapping[str, object] | None,
        created_at: datetime,
    ) -> Memory:
        self._require_open()
        text = _bounded_text(content, "content")
        stored_metadata = _metadata(metadata)
        embedding = self._embedder.embed(text)
        seeds = self._load_store_edge_seeds(embedding, created_at)
        result = self._store.insert_memory_with_edges(
            text,
            stored_metadata,
            embedding,
            seeds,
            created_at=created_at,
            half_life_days=self._half_life_days(),
        )
        if result.inserted:
            self._confidence.invalidate()
            self._maintain_if_due(result.store_count, created_at)
        return result.memory

    def _maintain_if_due(self, store_count: int, now: datetime) -> None:
        """Run the PRD §6.5 pass every Nth insert, at this insert's instant.

        Errors are not caught. A prune that cannot run leaves the graph
        accumulating dead edges indefinitely, which is worse than a loud
        failure; the memory itself is already committed either way.
        """
        if store_count % self._policy.maintenance_interval != 0:
            return
        decay = self._policy.decay
        pruned = self._store.prune_weak_edges(
            decay.prune_threshold,
            now=now,
            half_life_days=float(decay.half_life_days),
        )
        self._store.expire_queries(now=now)
        if pruned:
            self._confidence.invalidate()

    def _load_store_edge_seeds(
        self,
        embedding: Sequence[float],
        created_at: datetime,
    ) -> tuple[EdgeSeed, ...]:
        rate = self._policy.co_retrieval.reinforcement_rate
        # Semantic seeding is off by default (disabled on benchmark evidence);
        # only runs the extra semantic_search when explicitly re-enabled.
        semantic = self._policy.semantic_seed
        semantic_seeds: tuple[EdgeSeed, ...] = ()
        if semantic is not None:
            candidates = self._store.semantic_search(embedding, limit=semantic.max_links)
            scores = tuple((hit.memory_id, hit.score) for hit in candidates)
            semantic_ids = semantic_seed_ids(scores, semantic)
            semantic_seeds = tuple(
                EdgeSeed(memory_id, semantic.initial_weight, "semantic", rate) for memory_id in semantic_ids
            )
        temporal = self._policy.temporal_link
        temporal_ids = self._store.recent_memory_ids(
            before=created_at,
            window_seconds=temporal.window_seconds,
            limit=temporal.max_links,
        )
        temporal_seeds = tuple(
            EdgeSeed(memory_id, temporal.initial_weight, "temporal", rate) for memory_id in temporal_ids
        )
        return semantic_seeds + temporal_seeds

    def recall(
        self,
        query: str,
        *,
        top_k: int = 10,
        where: Mapping[str, object] | None = None,
        min_confidence: float = 0.0,
    ) -> RecallResult:
        """Retrieve memories, optionally dropping those below a confidence floor.

        `min_confidence` filters on `Recalled.confidence`, the absolute
        similarity between query and memory, so one threshold behaves the same
        across queries. Raising it lets a recall return **fewer than `top_k`
        results, including none at all** — which is how a caller distinguishes
        "no good answer" from "here are ten weak ones".

        Associations carry low confidence by construction, so a high floor
        returns direct matches only.
        """
        return self._recall_at(query, top_k, where, datetime.now(timezone.utc), min_confidence)

    def _recall_at(
        self,
        query: str,
        top_k: int,
        where: Mapping[str, object] | None,
        now: datetime,
        min_confidence: float = 0.0,
    ) -> RecallResult:
        """Run one recall against a single read time.

        Every edge weight in this recall decays to `now`, so a query cannot see
        one neighbor aged to a different instant than the next.
        """
        self._require_open()
        floor = unit_float(min_confidence, "min_confidence")
        started = time.perf_counter()
        text = _bounded_text(query, "query")
        result_limit = _top_k(top_k)
        filters = _where_filter(where)
        embedding = self._embedder.embed(text)
        ranked, activation, maturity = self._rank_candidates(text, embedding, now)
        selected_ids, selected = self._select_results(ranked, filters, result_limit)
        # Cosine against the query, looked up for the selected results rather
        # than taken from semantic_search: a result can arrive through keyword
        # search or activation without entering the semantic top-k.
        similarities = self._store.similarities(embedding, selected_ids)
        confidences = {
            # A negative cosine means unrelated, not anti-relevant.
            memory_id: max(0.0, min(1.0, similarities.get(memory_id, 0.0)))
            for memory_id in selected_ids
        }
        # Filter before learning, not after: an association the caller rejected
        # as too weak should not also be reinforced as though it were useful.
        selected_ids = tuple(memory_id for memory_id in selected_ids if confidences[memory_id] >= floor)
        energies = _recall_energies(selected_ids, activation)
        pair_seeds = self._co_retrieval_seeds(selected_ids)
        query_row, memories = self._store.record_recall(
            text,
            selected_ids,
            energies,
            activation.path_edge_ids,
            recorded_at=now,
            pair_seeds=pair_seeds,
            half_life_days=self._half_life_days(),
        )
        if pair_seeds:
            self._confidence.invalidate()
        recalled = [
            Recalled(
                memory=memory,
                score=selected[memory.id].score,
                confidence=confidences[memory.id],
                via=selected[memory.id].via,
            )
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
        now: datetime,
    ) -> tuple[tuple[BlendedHit, ...], ActivationResult, float]:
        fusion = self._policy.fusion
        spreading = self._policy.activation
        keyword = self._store.keyword_search(text, limit=fusion.candidate_depth)
        semantic = self._store.semantic_search(embedding, limit=fusion.candidate_depth)
        fused = reciprocal_rank_fusion(
            (
                tuple(hit.memory_id for hit in keyword),
                tuple(hit.memory_id for hit in semantic),
            ),
            k=fusion.rrf_k,
        )
        fused_scores = min_max_normalize({hit.memory_id: hit.score for hit in fused})
        summary = self._store.graph_summary(half_life_days=self._half_life_days(), now=now)
        maturity = self._maturity(summary)
        seeds = tuple((hit.memory_id, fused_scores[hit.memory_id]) for hit in fused[: spreading.seeds])
        activation = spread_activation(
            seeds,
            partial(self._activation_neighbors_at, now),
            spreading,
        )
        activation_scores = {hit.memory_id: hit.score for hit in activation.hits}
        ranked = blend_rankings(
            fused_scores,
            activation_scores,
            maturity,
            self._policy.activation_blend_weight,
            seed_ids=tuple(memory_id for memory_id, _ in seeds),
        )
        return ranked, activation, maturity

    def _activation_neighbors_at(self, now: datetime, memory_id: UUID) -> tuple[Neighbor, ...]:
        # PRD §5.2 spreads energy through the *effective* edge weight (§6.4),
        # so an edge that has not been reinforced in months carries less.
        edges = self._store.list_edges_for_node(
            memory_id,
            half_life_days=self._half_life_days(),
            now=now,
        )
        neighbors: list[Neighbor] = []
        for edge in edges:
            target_id = edge.b if edge.a == memory_id else edge.a
            neighbors.append(Neighbor(target_id, edge.id, edge.effective_weight))
        return tuple(neighbors)

    def _co_retrieval_seeds(self, result_ids: Sequence[UUID]) -> tuple[PairSeed, ...]:
        """Build the PRD §6.3 edges linking the top results of one recall."""
        config = self._policy.co_retrieval
        return tuple(
            PairSeed(
                first_id,
                second_id,
                config.initial_weight,
                "co_retrieval",
                config.reinforcement_rate,
            )
            for first_id, second_id in co_retrieval_pairs(result_ids)
        )

    def _half_life_days(self) -> float:
        return float(self._policy.decay.half_life_days)

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

    def feedback(self, query_id: UUID, *, positive: bool = True) -> None:
        """Apply explicit feedback for one recall (PRD §6.6).

        Positive feedback strengthens — and may create — the edges among the
        results and along the association paths that produced them. Negative
        feedback weakens the same edges without creating any.
        """
        self._feedback_at(query_id, positive, datetime.now(timezone.utc))

    def _feedback_at(self, query_id: UUID, positive: bool, now: datetime) -> None:
        """Apply feedback against a single read time, as recall does."""
        self._require_open()
        if not isinstance(query_id, UUID):
            raise InvalidArgumentError("query_id must be a UUID")
        if not isinstance(positive, bool):
            raise InvalidArgumentError("positive must be a boolean")
        query = self._store.get_query(query_id)
        pairs = self._feedback_pairs(query, now)
        rate = self._policy.feedback_rate
        seeds, updates = self._feedback_updates(pairs, query.energies, rate, positive, now)
        self._store.apply_feedback(
            query_id,
            1 if positive else -1,
            pair_seeds=seeds,
            weight_updates=updates,
            reinforced_at=now,
            half_life_days=self._half_life_days(),
        )
        self._confidence.invalidate()

    def _feedback_pairs(self, query: QueryRow, now: datetime) -> tuple[_FeedbackTarget, ...]:
        """Return one target per distinct edge among the results and paths.

        A query row outlives the memories it names, so pair only the results
        that still exist: feedback on a partly forgotten recall applies to what
        remains rather than failing.

        Result pairs are bounded to the top LINKED_RESULT_COUNT, amending PRD
        §6.6. As written, §6.6 pairs every entry of `result_ids`, which is
        C(top_k, 2): 45 edges per event at top_k 10, and 4,950 at 100. Because
        `top_k` is a caller-controlled per-call argument, one
        `recall(top_k=100)` followed by `feedback()` would write nearly five
        thousand edges — write amplification driven by an API caller, against
        co-retrieval's deliberately fixed 10.

        Measured on the chained profile: unbounded feedback built 943 learned
        edges against 150 real ones and cost two associative hits; bounding it
        cut that to 177 and recovered one. See the 2026-07-25 entry in
        dev/v0-dev-plan.md.

        Path edges are not bounded here. They already exist — activation
        traversed them — so reinforcing one grows nothing.
        """
        targets: dict[frozenset[UUID], _FeedbackTarget] = {}
        surviving = self._store.existing_memory_ids(query.result_ids)
        for first_id, second_id in unordered_pairs(surviving[:LINKED_RESULT_COUNT]):
            targets[frozenset((first_id, second_id))] = _FeedbackTarget(first_id, second_id, None)
        path_edges = self._store.edges_by_ids(
            query.path_edge_ids,
            half_life_days=self._half_life_days(),
            now=now,
        )
        # A path edge can also be a result pair; PRD §6.6 updates each edge, so
        # the later entry replaces the earlier one and carries its known weight.
        for edge in path_edges:
            targets[frozenset((edge.a, edge.b))] = _FeedbackTarget(edge.a, edge.b, edge)
        if len(targets) > _MAX_FEEDBACK_EDGES:
            raise InvalidArgumentError(f"feedback touches at most {_MAX_FEEDBACK_EDGES} edges")
        return tuple(targets.values())

    def _feedback_updates(
        self,
        targets: Sequence[_FeedbackTarget],
        energies: Mapping[UUID, float],
        rate: float,
        positive: bool,
        now: datetime,
    ) -> tuple[tuple[PairSeed, ...], tuple[tuple[str, float], ...]]:
        """Split targets into positive create-or-reinforce seeds and negative rewrites."""
        seeds: list[PairSeed] = []
        updates: list[tuple[str, float]] = []
        for target in targets:
            first = energies.get(target.first_id)
            second = energies.get(target.second_id)
            if first is None or second is None:
                # The memory was forgotten after the recall; nothing to weight.
                continue
            if positive:
                weight, reinforce_rate = positive_feedback_seed(first, second, rate)
                seeds.append(PairSeed(target.first_id, target.second_id, weight, "co_retrieval", reinforce_rate))
                continue
            # Read at `now`, so the weakening builds on the weight this
            # feedback saw rather than one decayed to the wall clock.
            edge = target.edge or self._store.get_edge_between(
                target.first_id,
                target.second_id,
                half_life_days=self._half_life_days(),
                now=now,
            )
            if edge is None:
                # Negative feedback never creates an edge (PRD §6.6).
                continue
            updates.append((edge.id, negative_feedback_weight(edge.effective_weight, first, second, rate)))
        return tuple(seeds), tuple(updates)

    def connect(self, first_id: UUID, second_id: UUID) -> None:
        """Assert an explicit link between two memories (PRD §3, §9 group 15).

        The only user-created edge origin. Idempotent, and never weakens a link
        the graph already rates more highly.
        """
        self._connect_at(first_id, second_id, datetime.now(timezone.utc))

    def _connect_at(self, first_id: UUID, second_id: UUID, now: datetime) -> None:
        """Assert an explicit link against a single read time."""
        self._require_open()
        if not isinstance(first_id, UUID) or not isinstance(second_id, UUID):
            raise InvalidArgumentError("connect requires two UUID memory_ids")
        self._store.assert_edge(
            first_id,
            second_id,
            self._policy.connect_weight,
            "explicit",
            asserted_at=now,
            half_life_days=self._half_life_days(),
        )
        self._confidence.invalidate()

    def forget(self, memory_id: UUID) -> None:
        """Delete one memory and every edge touching it.

        Raises NotFoundError for an unknown id. Edge removal is a foreign-key
        cascade, so the graph cannot keep an edge pointing at a memory that no
        longer exists.
        """
        self._require_open()
        if not isinstance(memory_id, UUID):
            raise InvalidArgumentError("memory_id must be a UUID")
        self._store.forget_memory(memory_id)
        self._confidence.invalidate()

    def stats(self) -> Stats:
        """Report memory and edge counts, edges by origin, and graph maturity.

        Between writes this reuses the previous scan, so the decay reflected in
        the reported averages can lag by the time since that scan. Any write
        refreshes it.
        """
        self._require_open()
        summary = self._store.graph_summary(half_life_days=self._half_life_days())
        return Stats(
            memories=summary.memory_count,
            edges=summary.edge_count,
            edges_by_origin=summary.edges_by_origin(),
            maturity=self._maturity(summary),
            db_path=self._store.db_path,
        )

    def close(self) -> None:
        """Release the database connection. Calling this twice is harmless."""
        self._store.close()

    def __enter__(self) -> SynapticDB:
        """Enter a context that closes the database on exit."""
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the database, whether the block succeeded or raised."""
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
        if self._store.closed:
            raise RuntimeError("SynapticDB instance is closed")


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


def _recall_energies(
    result_ids: Sequence[UUID],
    activation: ActivationResult,
) -> dict[UUID, float]:
    """Return the energies feedback (PRD §6.6) needs to weight its updates.

    Covers every result — 1.0 where activation never reached one — and every
    activated node besides. Activation records a path edge only after
    energizing both of its endpoints, so storing the activated nodes is what
    makes `e_i · e_j` defined for a path edge whose endpoint the top-k
    ranking left out.
    """
    energies = {hit.memory_id: hit.energy for hit in activation.hits}
    for memory_id in result_ids:
        energies.setdefault(memory_id, 1.0)
    return energies
