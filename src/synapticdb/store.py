"""SQLite persistence, full-text search, and in-process vector search."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType
from typing import Literal, cast
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray

from synapticdb.models import EdgeOrigin, EmbeddingError, InvalidArgumentError, Memory, NotFoundError

_CANDIDATE_LIMIT = 40
_EXPIRE_BATCH_LIMIT = 1000
_QUERY_TOKEN_LIMIT = 64
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
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, content='memories', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    a TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    b TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    weight REAL NOT NULL CHECK (weight >= 0.0 AND weight <= 1.0),
    origin TEXT NOT NULL CHECK (
        origin IN ('semantic', 'temporal', 'co_retrieval', 'explicit')
    ),
    created_at TEXT NOT NULL,
    last_reinforced_at TEXT NOT NULL,
    reinforcement_count INTEGER NOT NULL DEFAULT 0
        CHECK (reinforcement_count >= 0),
    UNIQUE (a, b),
    CHECK (a < b)
);
CREATE INDEX IF NOT EXISTS idx_edges_a ON edges(a);
CREATE INDEX IF NOT EXISTS idx_edges_b ON edges(b);

CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    result_ids TEXT NOT NULL,
    energies TEXT NOT NULL,
    path_edge_ids TEXT NOT NULL,
    feedback INTEGER CHECK (feedback IS NULL OR feedback IN (-1, 1))
);

PRAGMA user_version = 1;
"""


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
        self._require_open()
        normalized_content = _normalize_content(content)
        content_hash = hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()
        metadata_text = _prepare_metadata(metadata)
        vector = _prepare_embedding(embedding)
        timestamp = _require_utc(created_at or _utc_now(), "created_at")
        identifier = memory_id or uuid4()
        row: sqlite3.Row
        inserted = False
        with self._connection:
            self._ensure_embedding_dimension(vector.size)
            existing = self._connection.execute(
                "SELECT * FROM memories WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if existing is not None:
                return _memory_from_row(existing)
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
            inserted = True
        if inserted:
            self._append_vector_cache(identifier, vector)
        return _memory_from_row(row)

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

    def graph_summary(self) -> GraphSummary:
        self._require_open()
        row = self._connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM memories) AS memory_count,
                COUNT(*) AS edge_count,
                COALESCE(AVG(weight), 0.0) AS average_edge_weight,
                COALESCE(AVG(reinforcement_count), 0.0) AS average_reinforcement_count,
                COALESCE(SUM(origin = 'semantic'), 0) AS semantic_edges,
                COALESCE(SUM(origin = 'temporal'), 0) AS temporal_edges,
                COALESCE(SUM(origin = 'co_retrieval'), 0) AS co_retrieval_edges,
                COALESCE(SUM(origin = 'explicit'), 0) AS explicit_edges
            FROM edges
            """
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

    def insert_edge(
        self,
        first_id: UUID,
        second_id: UUID,
        weight: float,
        origin: EdgeOrigin,
        *,
        created_at: datetime | None = None,
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
            self._connection.execute(
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
                    stored_weight,
                    stored_origin,
                    timestamp.isoformat(),
                    timestamp.isoformat(),
                ),
            )
            row = self._required_edge_row(edge_id)
        return _edge_from_row(row)

    def get_edge(self, edge_id: str) -> Edge:
        self._require_open()
        return _edge_from_row(self._required_edge_row(edge_id))

    def get_edge_between(self, first_id: UUID, second_id: UUID) -> Edge | None:
        self._require_open()
        a, b = _canonical_pair(first_id, second_id)
        row = self._connection.execute(
            "SELECT * FROM edges WHERE id = ?",
            (_edge_id(a, b),),
        ).fetchone()
        return None if row is None else _edge_from_row(row)

    def list_edges_for_node(self, memory_id: UUID) -> tuple[Edge, ...]:
        self._require_open()
        self._require_memory_ids((str(memory_id),))
        rows = self._connection.execute(
            "SELECT * FROM edges WHERE a = ? OR b = ? ORDER BY id",
            (str(memory_id), str(memory_id)),
        ).fetchall()
        return tuple(_edge_from_row(row) for row in rows)

    def bulk_update_edge_weights(
        self,
        updates: Sequence[tuple[str, float]],
        *,
        reinforced_at: datetime | None = None,
    ) -> tuple[Edge, ...]:
        self._require_open()
        prepared = tuple((edge_id, _unit_float(weight, "weight")) for edge_id, weight in updates)
        if not all(isinstance(edge_id, str) and edge_id for edge_id, _ in prepared):
            raise InvalidArgumentError("edge ids must be non-empty strings")
        if len({edge_id for edge_id, _ in prepared}) != len(prepared):
            raise InvalidArgumentError("edge updates must not contain duplicate ids")
        if not prepared:
            return ()
        timestamp = _require_utc(reinforced_at or _utc_now(), "reinforced_at")
        with self._connection:
            for edge_id, weight in prepared:
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
            rows = self._edge_rows_by_ids(tuple(edge_id for edge_id, _ in prepared))
        return tuple(_edge_from_row(row) for row in rows)

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
    ) -> tuple[QueryRow, tuple[Memory, ...]]:
        """Persist query evidence and update returned memories in one transaction."""
        self._require_open()
        timestamp = _require_utc(recorded_at or _utc_now(), "recorded_at")
        prepared = _prepare_query(text, result_ids, energies, path_edge_ids, query_id, timestamp)
        identifiers = prepared.result_ids
        with self._connection:
            self._require_memory_ids(identifiers)
            self._insert_query(prepared)
            if identifiers:
                placeholders = ",".join("?" for _ in identifiers)
                self._connection.execute(
                    f"""
                    UPDATE memories
                    SET access_count = access_count + 1, last_accessed_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    (timestamp.isoformat(), *identifiers),
                )
            query_row = self._required_query_row(prepared.identifier)
            memory_rows = tuple(self._required_memory_row(UUID(identifier)) for identifier in identifiers)
        return _query_from_row(query_row), tuple(_memory_from_row(row) for row in memory_rows)

    def get_query(self, query_id: UUID) -> QueryRow:
        self._require_open()
        return _query_from_row(self._required_query_row(query_id))

    def set_query_feedback(self, query_id: UUID, feedback: FeedbackValue) -> QueryRow:
        self._require_open()
        if feedback not in (-1, 1):
            raise InvalidArgumentError("feedback must be -1 or 1")
        with self._connection:
            row = self._required_query_row(query_id)
            if row["feedback"] is not None:
                raise InvalidArgumentError(f"feedback already recorded for query_id: {query_id}")
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
                    ORDER BY created_at, id
                    LIMIT ?
                )
                """,
                (pending_cutoff, completed_cutoff, row_limit),
            )
        return cursor.rowcount

    def keyword_search(self, query: str, limit: int = _CANDIDATE_LIMIT) -> tuple[SearchHit, ...]:
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
            ORDER BY rank, memories.id
            LIMIT ?
            """,
            (expression, result_limit),
        ).fetchall()
        return tuple(SearchHit(UUID(row["id"]), -float(row["rank"])) for row in rows)

    def semantic_search(
        self,
        query_embedding: Sequence[float],
        limit: int = _CANDIDATE_LIMIT,
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


def _edge_from_row(row: sqlite3.Row) -> Edge:
    return Edge(
        id=row["id"],
        a=UUID(row["a"]),
        b=UUID(row["b"]),
        weight=float(row["weight"]),
        origin=cast(EdgeOrigin, row["origin"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        last_reinforced_at=datetime.fromisoformat(row["last_reinforced_at"]),
        reinforcement_count=int(row["reinforcement_count"]),
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
