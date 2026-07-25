"""SQLite persistence, full-text search, and in-process vector search."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType
from typing import Literal, cast
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray

from synapticdb.learning import DEFAULT_HALF_LIFE_DAYS, decayed_weight, reinforce_weight
from synapticdb.models import EdgeOrigin, EmbeddingError, InvalidArgumentError, Memory, NotFoundError

CANDIDATE_LIMIT = 40
_EDGE_LOOKUP_LIMIT = 400
_MEMORY_EDGE_LIMIT = 80
# PRD §6.3 pairs the top 5 results of a recall, so C(5, 2) = 10 edges.
_RECALL_EDGE_LIMIT = 10
# PRD §6.6 pairs *every* result, so a maximal top_k of 100 gives C(100, 2).
_FEEDBACK_EDGE_LIMIT = 4950
_EXPIRE_BATCH_LIMIT = 1000
_QUERY_TOKEN_LIMIT = 64
_TEMPORAL_LINK_LIMIT = 40
_TEMPORAL_WINDOW_LIMIT = 86_400
_EDGE_ORIGINS = frozenset({"semantic", "temporal", "co_retrieval", "explicit"})
_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_SCHEMA_VERSION = 1

FloatVector = NDArray[np.float32]
FloatMatrix = NDArray[np.float32]
FeedbackValue = Literal[-1, 1]


@dataclass(frozen=True, slots=True)
class Edge:
    id: str
    a: UUID
    b: UUID
    weight: float
    origin: EdgeOrigin
    created_at: datetime
    last_reinforced_at: datetime
    reinforcement_count: int
    # PRD section 6.4: `weight` is the stored value, `effective_weight` is that
    # value decayed to the read time. Consumers that rank, spread, or prune
    # edges use `effective_weight`; only writers use `weight`.
    #
    # Excluded from equality: it is a view of the row at one instant, not part
    # of the row. Two reads of one edge differ here by the microseconds between
    # them, and that must not make them unequal edges.
    effective_weight: float = field(compare=False)


@dataclass(frozen=True, slots=True)
class EdgeSeed:
    memory_id: UUID
    weight: float
    origin: EdgeOrigin
    reinforce_rate: float | None = None


@dataclass(frozen=True, slots=True)
class PairSeed:
    """One edge between two existing memories.

    The free-standing sibling of EdgeSeed, which is anchored to the memory
    being inserted and so names only the other endpoint.
    """

    first_id: UUID
    second_id: UUID
    weight: float
    origin: EdgeOrigin
    reinforce_rate: float | None = None


@dataclass(frozen=True, slots=True)
class MemoryInsert:
    memory: Memory
    inserted: bool


@dataclass(frozen=True, slots=True)
class QueryRow:
    id: UUID
    text: str
    created_at: datetime
    result_ids: tuple[UUID, ...]
    energies: dict[UUID, float]
    path_edge_ids: tuple[str, ...]
    feedback: FeedbackValue | None


@dataclass(frozen=True, slots=True)
class SearchHit:
    memory_id: UUID
    score: float


@dataclass(frozen=True, slots=True)
class GraphSummary:
    memory_count: int
    edge_count: int
    average_edge_weight: float
    average_reinforcement_count: float
    semantic_edges: int
    temporal_edges: int
    co_retrieval_edges: int
    explicit_edges: int

    def edges_by_origin(self) -> dict[EdgeOrigin, int]:
        return {
            "semantic": self.semantic_edges,
            "temporal": self.temporal_edges,
            "co_retrieval": self.co_retrieval_edges,
            "explicit": self.explicit_edges,
        }


@dataclass(frozen=True, slots=True)
class _PreparedQuery:
    identifier: UUID
    text: str
    created_at: datetime
    result_ids: tuple[str, ...]
    energies: dict[str, float]
    path_edge_ids: tuple[str, ...]


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY, content TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE, metadata TEXT NOT NULL DEFAULT '{}',
    embedding BLOB NOT NULL, created_at TEXT NOT NULL, last_accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, content='memories', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY, a TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    b TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    weight REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    origin TEXT NOT NULL CHECK (origin IN ('semantic', 'temporal', 'co_retrieval', 'explicit')),
    created_at TEXT NOT NULL, last_reinforced_at TEXT NOT NULL,
    reinforcement_count INTEGER NOT NULL DEFAULT 0 CHECK (reinforcement_count >= 0),
    UNIQUE (a, b), CHECK (a < b)
);
CREATE INDEX IF NOT EXISTS idx_edges_a ON edges(a);
CREATE INDEX IF NOT EXISTS idx_edges_b ON edges(b);

CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY, text TEXT NOT NULL, created_at TEXT NOT NULL,
    result_ids TEXT NOT NULL, energies TEXT NOT NULL, path_edge_ids TEXT NOT NULL,
    feedback INTEGER CHECK (feedback IS NULL OR feedback IN (-1, 1))
);

PRAGMA user_version = 1;
"""

# Decay in SQL, so ORDER BY and AVG rank by effective weight rather than by the
# stored value. julianday() yields the elapsed span in days directly; the two
# bound parameters are the read time and the half-life, in that order.
_DECAYED_WEIGHT_SQL = "synaptic_decayed_weight(weight, julianday(?) - julianday(last_reinforced_at), ?)"


class Store:
    """Own one SQLite connection and its aligned vector cache."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._connection = sqlite3.connect(self.db_path, timeout=5.0)
        self._connection.row_factory = sqlite3.Row
        self._vector_ids: list[str] | None = None
        self._vector_matrix: FloatMatrix | None = None
        self._closed = False
        try:
            self._initialize_database()
        except Exception:
            self._connection.close()
            self._closed = True
            raise

    def _initialize_database(self) -> None:
        fts_existed = self._object_exists("table", "memories_fts")
        version_row = self._connection.execute("PRAGMA user_version").fetchone()
        version = 0 if version_row is None else int(version_row[0])
        if version not in (0, _SCHEMA_VERSION):
            raise RuntimeError(f"unsupported database schema version: {version}")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA foreign_keys = ON")
        foreign_keys = self._connection.execute("PRAGMA foreign_keys").fetchone()
        if foreign_keys is None or foreign_keys[0] != 1:
            raise RuntimeError("SQLite foreign-key enforcement is unavailable")
        # Register the decay law itself rather than restating it in SQL: SQLite
        # math functions are a build-time option, and one implementation cannot
        # drift from the Python one.
        self._connection.create_function(
            "synaptic_decayed_weight",
            3,
            _sql_decayed_weight,
            deterministic=True,
        )
        self._connection.executescript(_SCHEMA_SQL)
        if not fts_existed:
            self._connection.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
        self._connection.commit()

    def _object_exists(self, object_type: str, name: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type, name),
        ).fetchone()
        return row is not None

    def close(self) -> None:
        if self._closed:
            return
        self._connection.close()
        self._closed = True
        self._vector_ids = None
        self._vector_matrix = None

    def __enter__(self) -> Store:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def insert_memory(
        self,
        content: str,
        metadata: Mapping[str, object],
        embedding: Sequence[float],
        *,
        memory_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> Memory:
        result = self.insert_memory_with_edges(
            content,
            metadata,
            embedding,
            (),
            memory_id=memory_id,
            created_at=created_at,
        )
        return result.memory

    def insert_memory_with_edges(
        self,
        content: str,
        metadata: Mapping[str, object],
        embedding: Sequence[float],
        edge_seeds: Sequence[EdgeSeed],
        *,
        memory_id: UUID | None = None,
        created_at: datetime | None = None,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> MemoryInsert:
        """Insert one memory and its bounded initial edges atomically."""
        self._require_open()
        normalized_content = _normalize_content(content)
        content_hash = hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()
        metadata_text = _prepare_metadata(metadata)
        vector = _prepare_embedding(embedding)
        seeds = _prepare_edge_seeds(edge_seeds)
        timestamp = _require_utc(created_at or _utc_now(), "created_at")
        identifier = memory_id or uuid4()
        with self._connection:
            self._ensure_embedding_dimension(vector.size)
            existing = self._connection.execute(
                "SELECT * FROM memories WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if existing is not None:
                return MemoryInsert(_memory_from_row(existing), False)
            self._require_memory_ids(tuple(str(seed.memory_id) for seed in seeds))
            self._connection.execute(
                """
                INSERT INTO memories (
                    id, content, content_hash, metadata, embedding,
                    created_at, last_accessed_at, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    str(identifier),
                    content,
                    content_hash,
                    metadata_text,
                    vector.tobytes(order="C"),
                    timestamp.isoformat(),
                    timestamp.isoformat(),
                ),
            )
            row = self._required_memory_row(identifier)
            for seed in seeds:
                self._write_edge(identifier, seed, timestamp, half_life_days)
        self._append_vector_cache(identifier, vector)
        return MemoryInsert(_memory_from_row(row), True)

    def get_memory(self, memory_id: UUID) -> Memory:
        self._require_open()
        return _memory_from_row(self._required_memory_row(memory_id))

    def get_memories(self, memory_ids: Sequence[UUID]) -> tuple[Memory, ...]:
        self._require_open()
        identifiers = _uuid_texts(memory_ids, "memory_ids")
        if not identifiers:
            return ()
        placeholders = ",".join("?" for _ in identifiers)
        rows = self._connection.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            identifiers,
        ).fetchall()
        by_id = {row["id"]: _memory_from_row(row) for row in rows}
        missing = [identifier for identifier in identifiers if identifier not in by_id]
        if missing:
            raise NotFoundError(f"unknown memory_id: {missing[0]}")
        return tuple(by_id[identifier] for identifier in identifiers)

    def recent_memory_ids(
        self,
        *,
        before: datetime,
        window_seconds: int,
        limit: int,
    ) -> tuple[UUID, ...]:
        """Return the most recent bounded memory IDs inside a UTC window."""
        self._require_open()
        timestamp = _require_utc(before, "before")
        window = _positive_limit(window_seconds, "temporal window")
        result_limit = _positive_limit(limit, "temporal limit")
        if window > _TEMPORAL_WINDOW_LIMIT or result_limit > _TEMPORAL_LINK_LIMIT:
            raise InvalidArgumentError("temporal lookup exceeds its bounded limits")
        cutoff = timestamp - timedelta(seconds=window)
        rows = self._connection.execute(
            """
            SELECT id FROM memories
            WHERE created_at >= ? AND created_at <= ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (cutoff.isoformat(), timestamp.isoformat(), result_limit),
        ).fetchall()
        return tuple(UUID(row["id"]) for row in rows)

    def forget_memory(self, memory_id: UUID) -> None:
        self._require_open()
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM memories WHERE id = ?",
                (str(memory_id),),
            )
            if cursor.rowcount != 1:
                raise NotFoundError(f"unknown memory_id: {memory_id}")
        self._drop_vector_cache(memory_id)

    def bump_access(
        self,
        memory_ids: Sequence[UUID],
        *,
        accessed_at: datetime | None = None,
    ) -> None:
        self._require_open()
        identifiers = _uuid_texts(memory_ids, "memory_ids")
        if not identifiers:
            return
        timestamp = _require_utc(accessed_at or _utc_now(), "accessed_at")
        placeholders = ",".join("?" for _ in identifiers)
        with self._connection:
            self._require_memory_ids(identifiers)
            self._connection.execute(
                f"""
                UPDATE memories
                SET access_count = access_count + 1, last_accessed_at = ?
                WHERE id IN ({placeholders})
                """,
                (timestamp.isoformat(), *identifiers),
            )

    def embedding_dimension(self) -> int | None:
        self._require_open()
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = 'embedding_dim'",
        ).fetchone()
        return None if row is None else int(row["value"])

    def graph_summary(
        self,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: datetime | None = None,
    ) -> GraphSummary:
        self._require_open()
        read_time = _require_utc(now or _utc_now(), "now")
        # The average uses effective weights (PRD §6.4), so graph confidence
        # reports what the graph is worth now rather than what it once held.
        row = self._connection.execute(
            f"""
            SELECT
                (SELECT COUNT(*) FROM memories) AS memory_count,
                COUNT(*) AS edge_count,
                COALESCE(AVG({_DECAYED_WEIGHT_SQL}), 0.0) AS average_edge_weight,
                COALESCE(AVG(reinforcement_count), 0.0) AS average_reinforcement_count,
                COALESCE(SUM(origin = 'semantic'), 0) AS semantic_edges,
                COALESCE(SUM(origin = 'temporal'), 0) AS temporal_edges,
                COALESCE(SUM(origin = 'co_retrieval'), 0) AS co_retrieval_edges,
                COALESCE(SUM(origin = 'explicit'), 0) AS explicit_edges
            FROM edges
            """,
            (read_time.isoformat(), half_life_days),
        ).fetchone()
        if row is None:
            raise RuntimeError("graph summary query returned no row")
        return GraphSummary(
            memory_count=int(row["memory_count"]),
            edge_count=int(row["edge_count"]),
            average_edge_weight=float(row["average_edge_weight"]),
            average_reinforcement_count=float(row["average_reinforcement_count"]),
            semantic_edges=int(row["semantic_edges"]),
            temporal_edges=int(row["temporal_edges"]),
            co_retrieval_edges=int(row["co_retrieval_edges"]),
            explicit_edges=int(row["explicit_edges"]),
        )

    def assert_edge(
        self,
        first_id: UUID,
        second_id: UUID,
        weight: float,
        origin: EdgeOrigin,
        *,
        asserted_at: datetime | None = None,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> Edge:
        """Declare an edge worth at least `weight`, creating or raising it.

        The third way to write an edge, beside insert_edge (create, leave an
        existing one alone) and _write_edge (create or reinforce). An assertion
        never weakens: it compares against the *effective* weight, so a heavy
        but long-decayed edge is raised rather than left stale.

        `reinforcement_count` is untouched on purpose. That counter feeds graph
        confidence as evidence of accumulated use, and an idempotent assertion
        must not let a caller inflate maturity by repeating it.
        """
        self._require_open()
        a, b = _canonical_pair(first_id, second_id)
        edge_id = _edge_id(a, b)
        asserted_weight = _unit_float(weight, "weight")
        stored_origin = _edge_origin(origin)
        timestamp = _require_utc(asserted_at or _utc_now(), "asserted_at")
        with self._connection:
            self._require_memory_ids((str(a), str(b)))
            existing = self._connection.execute(
                "SELECT * FROM edges WHERE id = ?",
                (edge_id,),
            ).fetchone()
            if existing is None:
                self._write_edge(a, EdgeSeed(b, asserted_weight, stored_origin), timestamp, half_life_days)
            else:
                current = _edge_from_row(existing, half_life_days, timestamp)
                self._connection.execute(
                    "UPDATE edges SET weight = ?, origin = ?, last_reinforced_at = ? WHERE id = ?",
                    (
                        max(current.effective_weight, asserted_weight),
                        stored_origin,
                        timestamp.isoformat(),
                        edge_id,
                    ),
                )
            row = self._required_edge_row(edge_id)
        return _edge_from_row(row, half_life_days, timestamp)

    def insert_edge(
        self,
        first_id: UUID,
        second_id: UUID,
        weight: float,
        origin: EdgeOrigin,
        *,
        created_at: datetime | None = None,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> Edge:
        """Create the canonical edge; if it already exists, return it unchanged.

        Reinforcement (weight change, count bump, decay reset) is a separate,
        deliberate act: see bulk_update_edge_weights.
        """
        self._require_open()
        a, b = _canonical_pair(first_id, second_id)
        edge_id = _edge_id(a, b)
        stored_weight = _unit_float(weight, "weight")
        stored_origin = _edge_origin(origin)
        timestamp = _require_utc(created_at or _utc_now(), "created_at")
        with self._connection:
            self._require_memory_ids((str(a), str(b)))
            seed = EdgeSeed(b, stored_weight, stored_origin)
            self._write_edge(a, seed, timestamp, half_life_days)
            row = self._required_edge_row(edge_id)
        return _edge_from_row(row, half_life_days, timestamp)

    def _write_edge(
        self,
        memory_id: UUID,
        seed: EdgeSeed,
        timestamp: datetime,
        half_life_days: float,
    ) -> None:
        a, b = _canonical_pair(memory_id, seed.memory_id)
        edge_id = _edge_id(a, b)
        cursor = self._connection.execute(
            """
            INSERT INTO edges (
                id, a, b, weight, origin, created_at,
                last_reinforced_at, reinforcement_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(id) DO NOTHING
            """,
            (
                edge_id,
                str(a),
                str(b),
                seed.weight,
                seed.origin,
                timestamp.isoformat(),
                timestamp.isoformat(),
            ),
        )
        if cursor.rowcount not in (0, 1):
            raise RuntimeError("edge insert returned an invalid row count")
        if cursor.rowcount == 1 or seed.reinforce_rate is None:
            return
        row = self._required_edge_row(edge_id)
        # PRD section 6.4: decay is lazy, but this UPDATE already rewrites the
        # weight and resets last_reinforced_at, so the aged value must be the
        # base of the reinforcement. Reinforcing the stored value instead would
        # silently cancel every day of decay the edge had accrued.
        aged = _decay_to(float(row["weight"]), row["last_reinforced_at"], timestamp, half_life_days)
        weight = reinforce_weight(aged, seed.reinforce_rate)
        update = self._connection.execute(
            """
            UPDATE edges
            SET weight = ?, last_reinforced_at = ?,
                reinforcement_count = reinforcement_count + 1
            WHERE id = ?
            """,
            (weight, timestamp.isoformat(), edge_id),
        )
        if update.rowcount != 1:
            raise RuntimeError("edge reinforcement did not update one row")

    def get_edge(
        self,
        edge_id: str,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: datetime | None = None,
    ) -> Edge:
        self._require_open()
        read_time = _require_utc(now or _utc_now(), "now")
        return _edge_from_row(self._required_edge_row(edge_id), half_life_days, read_time)

    def get_edge_between(
        self,
        first_id: UUID,
        second_id: UUID,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: datetime | None = None,
    ) -> Edge | None:
        self._require_open()
        read_time = _require_utc(now or _utc_now(), "now")
        a, b = _canonical_pair(first_id, second_id)
        row = self._connection.execute(
            "SELECT * FROM edges WHERE id = ?",
            (_edge_id(a, b),),
        ).fetchone()
        return None if row is None else _edge_from_row(row, half_life_days, read_time)

    def list_edges_for_node(
        self,
        memory_id: UUID,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: datetime | None = None,
    ) -> tuple[Edge, ...]:
        self._require_open()
        self._require_memory_ids((str(memory_id),))
        read_time = _require_utc(now or _utc_now(), "now")
        # A hub node past the limit degrades to its strongest edges rather
        # than failing recall; co-retrieval learning makes dense hubs normal.
        # "Strongest" ranks by effective weight (PRD §6.4), so a stale heavy
        # edge cannot crowd out a fresh one before the caller ever sees it.
        rows = self._connection.execute(
            f"""
            SELECT * FROM edges WHERE a = ? OR b = ?
            ORDER BY {_DECAYED_WEIGHT_SQL} DESC, rowid
            LIMIT ?
            """,
            (
                str(memory_id),
                str(memory_id),
                read_time.isoformat(),
                half_life_days,
                _EDGE_LOOKUP_LIMIT,
            ),
        ).fetchall()
        return tuple(_edge_from_row(row, half_life_days, read_time) for row in rows)

    def bulk_update_edge_weights(
        self,
        updates: Sequence[tuple[str, float]],
        *,
        reinforced_at: datetime | None = None,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> tuple[Edge, ...]:
        """Write caller-computed weights and reset their decay clocks.

        Callers derive the new weight from Edge.effective_weight, not from
        Edge.weight: this method stores exactly what it is given.
        """
        self._require_open()
        prepared = _prepare_weight_updates(updates)
        if not prepared:
            return ()
        timestamp = _require_utc(reinforced_at or _utc_now(), "reinforced_at")
        with self._connection:
            self._write_weight_updates(prepared, timestamp)
            rows = self._edge_rows_by_ids(tuple(edge_id for edge_id, _ in prepared))
        return tuple(_edge_from_row(row, half_life_days, timestamp) for row in rows)

    def _write_weight_updates(
        self,
        updates: Sequence[tuple[str, float]],
        timestamp: datetime,
    ) -> None:
        """Store each weight and reset its decay clock, inside the caller's transaction."""
        for edge_id, weight in updates:
            cursor = self._connection.execute(
                """
                UPDATE edges
                SET weight = ?, last_reinforced_at = ?,
                    reinforcement_count = reinforcement_count + 1
                WHERE id = ?
                """,
                (weight, timestamp.isoformat(), edge_id),
            )
            if cursor.rowcount != 1:
                raise NotFoundError(f"unknown edge_id: {edge_id}")

    def save_query(
        self,
        text: str,
        result_ids: Sequence[UUID],
        energies: Mapping[UUID, float],
        path_edge_ids: Sequence[str],
        *,
        query_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> QueryRow:
        self._require_open()
        prepared = _prepare_query(text, result_ids, energies, path_edge_ids, query_id, created_at)
        with self._connection:
            self._insert_query(prepared)
            row = self._required_query_row(prepared.identifier)
        return _query_from_row(row)

    def record_recall(
        self,
        text: str,
        result_ids: Sequence[UUID],
        energies: Mapping[UUID, float],
        path_edge_ids: Sequence[str],
        *,
        query_id: UUID | None = None,
        recorded_at: datetime | None = None,
        pair_seeds: Sequence[PairSeed] = (),
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> tuple[QueryRow, tuple[Memory, ...]]:
        """Persist query evidence, access bumps, and learned edges in one transaction.

        A recall is one event: if its co-retrieval edges cannot be written, the
        query row it learned them from must not survive either.
        """
        self._require_open()
        timestamp = _require_utc(recorded_at or _utc_now(), "recorded_at")
        prepared = _prepare_query(text, result_ids, energies, path_edge_ids, query_id, timestamp)
        seeds = _prepare_pair_seeds(pair_seeds, _RECALL_EDGE_LIMIT)
        identifiers = prepared.result_ids
        with self._connection:
            self._require_memory_ids(identifiers)
            self._insert_query(prepared)
            self._bump_accessed(identifiers, timestamp)
            self._write_pair_seeds(seeds, timestamp, half_life_days)
            query_row = self._required_query_row(prepared.identifier)
            memory_rows = tuple(self._required_memory_row(UUID(identifier)) for identifier in identifiers)
        return _query_from_row(query_row), tuple(_memory_from_row(row) for row in memory_rows)

    def _bump_accessed(self, identifiers: Sequence[str], timestamp: datetime) -> None:
        if not identifiers:
            return
        placeholders = ",".join("?" for _ in identifiers)
        self._connection.execute(
            f"""
            UPDATE memories
            SET access_count = access_count + 1, last_accessed_at = ?
            WHERE id IN ({placeholders})
            """,
            (timestamp.isoformat(), *identifiers),
        )

    def _write_pair_seeds(
        self,
        seeds: Sequence[PairSeed],
        timestamp: datetime,
        half_life_days: float,
    ) -> None:
        """Create or reinforce each learned edge inside the caller's transaction."""
        for seed in seeds:
            self._require_memory_ids((str(seed.first_id), str(seed.second_id)))
            edge_seed = EdgeSeed(seed.second_id, seed.weight, seed.origin, seed.reinforce_rate)
            self._write_edge(seed.first_id, edge_seed, timestamp, half_life_days)

    def get_query(self, query_id: UUID) -> QueryRow:
        self._require_open()
        return _query_from_row(self._required_query_row(query_id))

    def existing_memory_ids(self, memory_ids: Sequence[UUID]) -> tuple[UUID, ...]:
        """Return the subset of IDs still present, preserving the caller's order.

        Unlike get_memories, a missing ID is not an error: a query row outlives
        the memories it names, and forget() may have removed one since.
        """
        self._require_open()
        identifiers = _uuid_texts(memory_ids, "memory_ids")
        if not identifiers:
            return ()
        placeholders = ",".join("?" for _ in identifiers)
        rows = self._connection.execute(
            f"SELECT id FROM memories WHERE id IN ({placeholders})",
            identifiers,
        ).fetchall()
        present = {row["id"] for row in rows}
        return tuple(UUID(identifier) for identifier in identifiers if identifier in present)

    def edges_by_ids(
        self,
        edge_ids: Sequence[str],
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: datetime | None = None,
    ) -> tuple[Edge, ...]:
        """Return the edges that still exist, skipping any that do not.

        Unlike get_edge, a missing row is not an error: a query row records the
        path edges it traversed, and forget() may have cascaded one away before
        the caller acts on that record.
        """
        self._require_open()
        identifiers = _string_ids(edge_ids, "edge_ids")
        if not identifiers:
            return ()
        read_time = _require_utc(now or _utc_now(), "now")
        placeholders = ",".join("?" for _ in identifiers)
        rows = self._connection.execute(
            f"SELECT * FROM edges WHERE id IN ({placeholders})",
            identifiers,
        ).fetchall()
        by_id = {row["id"]: row for row in rows}
        return tuple(
            _edge_from_row(by_id[edge_id], half_life_days, read_time) for edge_id in identifiers if edge_id in by_id
        )

    def apply_feedback(
        self,
        query_id: UUID,
        feedback: FeedbackValue,
        *,
        pair_seeds: Sequence[PairSeed] = (),
        weight_updates: Sequence[tuple[str, float]] = (),
        reinforced_at: datetime | None = None,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> QueryRow:
        """Record explicit feedback and its edge updates in one transaction.

        Either the graph learns from the feedback and the query is marked as
        answered, or neither happens: a partially applied update would let a
        second call re-apply the surviving half.
        """
        self._require_open()
        if feedback not in (-1, 1):
            raise InvalidArgumentError("feedback must be -1 or 1")
        seeds = _prepare_pair_seeds(pair_seeds, _FEEDBACK_EDGE_LIMIT)
        updates = _prepare_weight_updates(weight_updates)
        if len(updates) > _FEEDBACK_EDGE_LIMIT:
            raise InvalidArgumentError(f"this write accepts at most {_FEEDBACK_EDGE_LIMIT} edges")
        timestamp = _require_utc(reinforced_at or _utc_now(), "reinforced_at")
        with self._connection:
            self._require_unanswered_query(query_id)
            self._write_pair_seeds(seeds, timestamp, half_life_days)
            self._write_weight_updates(updates, timestamp)
            self._connection.execute(
                "UPDATE queries SET feedback = ? WHERE id = ?",
                (feedback, str(query_id)),
            )
            updated = self._required_query_row(query_id)
        return _query_from_row(updated)

    def _require_unanswered_query(self, query_id: UUID) -> sqlite3.Row:
        row = self._required_query_row(query_id)
        if row["feedback"] is not None:
            raise InvalidArgumentError(f"feedback already recorded for query_id: {query_id}")
        return row

    def set_query_feedback(self, query_id: UUID, feedback: FeedbackValue) -> QueryRow:
        self._require_open()
        if feedback not in (-1, 1):
            raise InvalidArgumentError("feedback must be -1 or 1")
        with self._connection:
            self._require_unanswered_query(query_id)
            self._connection.execute(
                "UPDATE queries SET feedback = ? WHERE id = ?",
                (feedback, str(query_id)),
            )
            updated = self._required_query_row(query_id)
        return _query_from_row(updated)

    def expire_queries(
        self,
        *,
        now: datetime | None = None,
        limit: int = _EXPIRE_BATCH_LIMIT,
    ) -> int:
        self._require_open()
        row_limit = _positive_limit(limit, "expire limit")
        timestamp = _require_utc(now or _utc_now(), "now")
        pending_cutoff = (timestamp - timedelta(days=7)).isoformat()
        completed_cutoff = (timestamp - timedelta(days=30)).isoformat()
        with self._connection:
            cursor = self._connection.execute(
                """
                DELETE FROM queries WHERE id IN (
                    SELECT id FROM queries
                    WHERE (feedback IS NULL AND created_at < ?)
                       OR (feedback IS NOT NULL AND created_at < ?)
                    ORDER BY created_at, rowid
                    LIMIT ?
                )
                """,
                (pending_cutoff, completed_cutoff, row_limit),
            )
        return cursor.rowcount

    def keyword_search(self, query: str, limit: int = CANDIDATE_LIMIT) -> tuple[SearchHit, ...]:
        # bm25 ties break on rowid (insertion order), never on id: ids are
        # random UUIDs, so an id tie-break reorders equal-score results on
        # every reingest and makes benchmark runs irreproducible.
        self._require_open()
        result_limit = _positive_limit(limit, "keyword limit")
        expression = _fts_expression(query)
        if expression is None:
            return ()
        rows = self._connection.execute(
            """
            SELECT memories.id, bm25(memories_fts) AS rank
            FROM memories_fts
            JOIN memories ON memories.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ?
            ORDER BY rank, memories.rowid
            LIMIT ?
            """,
            (expression, result_limit),
        ).fetchall()
        return tuple(SearchHit(UUID(row["id"]), -float(row["rank"])) for row in rows)

    def semantic_search(
        self,
        query_embedding: Sequence[float],
        limit: int = CANDIDATE_LIMIT,
    ) -> tuple[SearchHit, ...]:
        self._require_open()
        result_limit = _positive_limit(limit, "semantic limit")
        vector = _prepare_embedding(query_embedding)
        dimension = self.embedding_dimension()
        if dimension is None:
            return ()
        if vector.size != dimension:
            raise EmbeddingError(f"embedding dimension {vector.size} does not match stored dimension {dimension}")
        identifiers, matrix = self._vector_cache(dimension)
        scores = matrix @ (vector / _vector_norm(vector))
        ranked = np.argsort(-scores, kind="stable")[:result_limit]
        return tuple(SearchHit(UUID(identifiers[int(index)]), float(scores[int(index)])) for index in ranked)

    def _required_memory_row(self, memory_id: UUID) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM memories WHERE id = ?",
            (str(memory_id),),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"unknown memory_id: {memory_id}")
        return cast(sqlite3.Row, row)

    def _insert_query(self, query: _PreparedQuery) -> None:
        self._connection.execute(
            """
            INSERT INTO queries (
                id, text, created_at, result_ids, energies,
                path_edge_ids, feedback
            ) VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                str(query.identifier),
                query.text,
                query.created_at.isoformat(),
                _encode_json(list(query.result_ids), "result_ids"),
                _encode_json(query.energies, "energies"),
                _encode_json(list(query.path_edge_ids), "path_edge_ids"),
            ),
        )

    def _required_edge_row(self, edge_id: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM edges WHERE id = ?",
            (edge_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"unknown edge_id: {edge_id}")
        return cast(sqlite3.Row, row)

    def _required_query_row(self, query_id: UUID) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM queries WHERE id = ?",
            (str(query_id),),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"unknown query_id: {query_id}")
        return cast(sqlite3.Row, row)

    def _require_memory_ids(self, identifiers: Sequence[str]) -> None:
        if not identifiers:
            return
        placeholders = ",".join("?" for _ in identifiers)
        row = self._connection.execute(
            f"SELECT COUNT(*) AS count FROM memories WHERE id IN ({placeholders})",
            tuple(identifiers),
        ).fetchone()
        found = 0 if row is None else int(row["count"])
        if found != len(set(identifiers)):
            raise NotFoundError("one or more memory_ids are unknown")

    def _edge_rows_by_ids(self, edge_ids: tuple[str, ...]) -> tuple[sqlite3.Row, ...]:
        placeholders = ",".join("?" for _ in edge_ids)
        rows = self._connection.execute(
            f"SELECT * FROM edges WHERE id IN ({placeholders})",
            edge_ids,
        ).fetchall()
        by_id = {row["id"]: row for row in rows}
        return tuple(by_id[edge_id] for edge_id in edge_ids)

    def _ensure_embedding_dimension(self, dimension: int) -> None:
        stored = self.embedding_dimension()
        if stored is None:
            self._connection.execute(
                "INSERT INTO meta (key, value) VALUES ('embedding_dim', ?)",
                (str(dimension),),
            )
            return
        if stored != dimension:
            raise EmbeddingError(f"embedding dimension {dimension} does not match stored dimension {stored}")

    def _vector_cache(self, dimension: int) -> tuple[list[str], FloatMatrix]:
        if self._vector_ids is None or self._vector_matrix is None:
            rows = self._connection.execute("SELECT id, embedding FROM memories ORDER BY rowid").fetchall()
            matrix = np.empty((len(rows), dimension), dtype=np.float32)
            identifiers: list[str] = []
            for index, row in enumerate(rows):
                vector = np.frombuffer(row["embedding"], dtype="<f4")
                matrix[index] = vector / _vector_norm(vector)
                identifiers.append(row["id"])
            self._vector_ids = identifiers
            self._vector_matrix = matrix
        return self._vector_ids, self._vector_matrix

    def _append_vector_cache(self, memory_id: UUID, vector: FloatVector) -> None:
        if self._vector_matrix is None or self._vector_ids is None:
            return
        unit_vector = vector / _vector_norm(vector)
        matrix = np.vstack((self._vector_matrix, unit_vector[np.newaxis, :]))
        self._vector_ids = [*self._vector_ids, str(memory_id)]
        self._vector_matrix = matrix

    def _drop_vector_cache(self, memory_id: UUID) -> None:
        if self._vector_matrix is None or self._vector_ids is None:
            return
        try:
            index = self._vector_ids.index(str(memory_id))
        except ValueError:
            self._vector_ids = None
            self._vector_matrix = None
            return
        identifiers = [*self._vector_ids]
        identifiers.pop(index)
        self._vector_ids = identifiers
        self._vector_matrix = np.delete(self._vector_matrix, index, axis=0)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("store is closed")


def _normalize_content(content: str) -> str:
    if not isinstance(content, str):
        raise InvalidArgumentError("content must be a string")
    normalized = unicodedata.normalize("NFKC", content)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise InvalidArgumentError("content must not be blank")
    return normalized


def _encode_json(value: object, label: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as error:
        raise InvalidArgumentError(f"{label} must be JSON-serializable") from error


def _prepare_metadata(metadata: Mapping[str, object]) -> str:
    if not isinstance(metadata, Mapping):
        raise InvalidArgumentError("metadata must be a mapping")
    if not all(isinstance(key, str) for key in metadata):
        raise InvalidArgumentError("metadata keys must be strings")
    return _encode_json(dict(metadata), "metadata")


def _prepare_weight_updates(updates: Sequence[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    prepared = tuple((edge_id, _unit_float(weight, "weight")) for edge_id, weight in updates)
    if not all(isinstance(edge_id, str) and edge_id for edge_id, _ in prepared):
        raise InvalidArgumentError("edge ids must be non-empty strings")
    if len({edge_id for edge_id, _ in prepared}) != len(prepared):
        raise InvalidArgumentError("edge updates must not contain duplicate ids")
    return prepared


def _prepare_pair_seeds(pair_seeds: Sequence[PairSeed], limit: int) -> tuple[PairSeed, ...]:
    seeds = tuple(pair_seeds)
    if len(seeds) > limit:
        raise InvalidArgumentError(f"this write accepts at most {limit} edges")
    if not all(isinstance(seed, PairSeed) for seed in seeds):
        raise InvalidArgumentError("pair seeds must be PairSeed values")
    endpoints = tuple((seed.first_id, seed.second_id) for seed in seeds)
    if not all(isinstance(value, UUID) for pair in endpoints for value in pair):
        raise InvalidArgumentError("pair seed memory IDs must be UUID values")
    # Compare canonically: (a, b) and (b, a) address the same edge row, so
    # accepting both would make one write silently reinforce the other.
    canonical = tuple(frozenset(pair) for pair in endpoints)
    if len(set(canonical)) != len(canonical):
        raise InvalidArgumentError("pair seeds must not repeat an edge")
    return tuple(
        PairSeed(
            seed.first_id,
            seed.second_id,
            _unit_float(seed.weight, "weight"),
            _edge_origin(seed.origin),
            None if seed.reinforce_rate is None else _unit_float(seed.reinforce_rate, "reinforce rate"),
        )
        for seed in seeds
    )


def _prepare_edge_seeds(edge_seeds: Sequence[EdgeSeed]) -> tuple[EdgeSeed, ...]:
    seeds = tuple(edge_seeds)
    if len(seeds) > _MEMORY_EDGE_LIMIT:
        raise InvalidArgumentError(f"a memory accepts at most {_MEMORY_EDGE_LIMIT} initial edges")
    if not all(isinstance(seed, EdgeSeed) for seed in seeds):
        raise InvalidArgumentError("edge seeds must be EdgeSeed values")
    if not all(isinstance(seed.memory_id, UUID) for seed in seeds):
        raise InvalidArgumentError("edge seed memory IDs must be UUID values")
    mechanism_keys = tuple((seed.memory_id, seed.origin) for seed in seeds)
    if len(set(mechanism_keys)) != len(mechanism_keys):
        raise InvalidArgumentError("edge seeds must be unique per memory and origin")
    return tuple(
        EdgeSeed(
            seed.memory_id,
            _unit_float(seed.weight, "weight"),
            _edge_origin(seed.origin),
            None if seed.reinforce_rate is None else _unit_float(seed.reinforce_rate, "reinforce rate"),
        )
        for seed in seeds
    )


def _prepare_embedding(embedding: Sequence[float]) -> FloatVector:
    try:
        # Out-of-range values overflow to inf here; the finite check below rejects them.
        with np.errstate(over="ignore"):
            vector = np.ascontiguousarray(np.asarray(embedding, dtype="<f4"))
    except (TypeError, ValueError) as error:
        raise EmbeddingError("embedding must be a one-dimensional numeric sequence") from error
    if vector.ndim != 1 or vector.size == 0:
        raise EmbeddingError("embedding must be a non-empty one-dimensional sequence")
    if not bool(np.isfinite(vector).all()):
        raise EmbeddingError("embedding values must be finite")
    if _vector_norm(vector) == 0.0:
        raise EmbeddingError("embedding must not be a zero vector")
    return vector


def _vector_norm(vector: FloatVector) -> float:
    return float(np.linalg.norm(vector.astype(np.float64, copy=False)))


def _memory_from_row(row: sqlite3.Row) -> Memory:
    return Memory(
        id=UUID(row["id"]),
        content=row["content"],
        metadata=json.loads(row["metadata"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]),
        access_count=row["access_count"],
    )


def _decay_to(weight: float, last_reinforced_at: str, now: datetime, half_life_days: float) -> float:
    """Decay one stored weight from its ISO timestamp to the given read time."""
    elapsed = (now - datetime.fromisoformat(last_reinforced_at)).total_seconds() / 86_400.0
    return decayed_weight(weight, elapsed, half_life_days)


def _sql_decayed_weight(
    weight: float | None,
    days_elapsed: float | None,
    half_life_days: float | None,
) -> float:
    """Expose the PRD section 6.4 decay law to SQLite."""
    if weight is None or days_elapsed is None or half_life_days is None:
        # julianday() returns NULL for a timestamp it cannot parse. Every
        # timestamp is written by _require_utc, so a NULL here means the row is
        # corrupt and must not be silently ranked as weightless.
        raise RuntimeError("edge row contains an unreadable weight or timestamp")
    return decayed_weight(weight, days_elapsed, half_life_days)


def _edge_from_row(row: sqlite3.Row, half_life_days: float, now: datetime) -> Edge:
    stored_weight = float(row["weight"])
    return Edge(
        id=row["id"],
        a=UUID(row["a"]),
        b=UUID(row["b"]),
        weight=stored_weight,
        origin=cast(EdgeOrigin, row["origin"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        last_reinforced_at=datetime.fromisoformat(row["last_reinforced_at"]),
        reinforcement_count=int(row["reinforcement_count"]),
        effective_weight=_decay_to(stored_weight, row["last_reinforced_at"], now, half_life_days),
    )


def _query_from_row(row: sqlite3.Row) -> QueryRow:
    return QueryRow(
        id=UUID(row["id"]),
        text=row["text"],
        created_at=datetime.fromisoformat(row["created_at"]),
        result_ids=tuple(UUID(item) for item in json.loads(row["result_ids"])),
        energies={UUID(key): float(value) for key, value in json.loads(row["energies"]).items()},
        path_edge_ids=tuple(json.loads(row["path_edge_ids"])),
        feedback=cast("FeedbackValue | None", row["feedback"]),
    )


def _uuid_texts(values: Sequence[UUID], label: str) -> tuple[str, ...]:
    if not all(isinstance(value, UUID) for value in values):
        raise InvalidArgumentError(f"{label} must contain UUID values")
    return tuple(str(value) for value in values)


def _string_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    prepared = tuple(values)
    if not all(isinstance(value, str) and value for value in prepared):
        raise InvalidArgumentError(f"{label} must contain non-empty strings")
    return prepared


def _prepare_energies(energies: Mapping[UUID, float]) -> dict[str, float]:
    prepared: dict[str, float] = {}
    for memory_id, energy in energies.items():
        if not isinstance(memory_id, UUID):
            raise InvalidArgumentError("energy keys must be UUID values")
        try:
            value = float(energy)
        except (TypeError, ValueError) as error:
            raise InvalidArgumentError("energies must be numeric") from error
        if not math.isfinite(value) or value < 0.0:
            raise InvalidArgumentError("energies must be finite nonnegative values")
        prepared[str(memory_id)] = value
    return prepared


def _prepare_query(
    text: str,
    result_ids: Sequence[UUID],
    energies: Mapping[UUID, float],
    path_edge_ids: Sequence[str],
    query_id: UUID | None,
    created_at: datetime | None,
) -> _PreparedQuery:
    results = _uuid_texts(result_ids, "result_ids")
    stored_energies = _prepare_energies(energies)
    paths = _string_ids(path_edge_ids, "path_edge_ids")
    _require_result_energies(results, stored_energies)
    identifier = query_id or uuid4()
    timestamp = _require_utc(created_at or _utc_now(), "created_at")
    return _PreparedQuery(identifier, text, timestamp, results, stored_energies, paths)


def _require_result_energies(result_ids: tuple[str, ...], energies: Mapping[str, float]) -> None:
    missing = [memory_id for memory_id in result_ids if memory_id not in energies]
    if missing:
        raise InvalidArgumentError(f"missing energy for result_id: {missing[0]}")


def _canonical_pair(first_id: UUID, second_id: UUID) -> tuple[UUID, UUID]:
    if first_id == second_id:
        raise InvalidArgumentError("an edge requires two different memory_ids")
    if str(first_id) < str(second_id):
        return first_id, second_id
    return second_id, first_id


def _edge_id(a: UUID, b: UUID) -> str:
    return hashlib.sha256(f"{a}{b}".encode("ascii")).hexdigest()


def _edge_origin(origin: EdgeOrigin) -> EdgeOrigin:
    if origin not in _EDGE_ORIGINS:
        raise InvalidArgumentError(f"unknown edge origin: {origin}")
    return origin


def _unit_float(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InvalidArgumentError(f"{label} must be numeric") from error
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise InvalidArgumentError(f"{label} must be a finite value between 0 and 1")
    return number


def _require_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise InvalidArgumentError(f"{label} must use UTC")
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _positive_limit(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidArgumentError(f"{label} must be a positive integer")
    return value


def _fts_expression(query: str) -> str | None:
    if not isinstance(query, str):
        raise InvalidArgumentError("query must be a string")
    tokens = _WORD_PATTERN.findall(unicodedata.normalize("NFKC", query))[:_QUERY_TOKEN_LIMIT]
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)
