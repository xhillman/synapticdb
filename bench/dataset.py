"""Load and validate frozen JSONL benchmark inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .contracts import (
    MAX_DATA_FILE_BYTES,
    MAX_MANIFEST_FILES,
    MAX_MEMORY_COUNT,
    MAX_QUERY_COUNT,
    MAX_RECORD_CHARS,
    MAX_SCHEDULE_COUNT,
    MAX_WARMUP_COUNT,
)


class DatasetError(ValueError):
    """Raised when benchmark data violates its integrity contract."""


_ALLOWED_MANIFEST_FILES = frozenset(
    {
        "chains.md",
        "memories.jsonl",
        "queries.jsonl",
        "schedule.jsonl",
        "warmup.jsonl",
    }
)


@dataclass(frozen=True)
class MemoryRecord:
    benchmark_id: str
    content: str
    ingest_offset_seconds: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    label: str
    text: str
    expected_ids: tuple[str, ...]
    intermediate_ids: tuple[str, ...]
    reviewer_note: str = ""


@dataclass(frozen=True)
class WarmupEvent:
    text: str
    positive: bool
    # Set when the event deliberately traverses a holdout chain. The declared
    # memory must be a bridge, never that chain's target — enforced in
    # _validate, so a hand-edited fixture cannot teach the benchmark its answer.
    chain_query_id: str | None = None
    expected_memory_id: str | None = None


@dataclass(frozen=True)
class BenchmarkDataset:
    memories: tuple[MemoryRecord, ...]
    queries: tuple[QueryRecord, ...]
    warmup: tuple[WarmupEvent, ...]
    fingerprint: str

    @property
    def direct_total(self) -> int:
        return sum(query.label == "direct" for query in self.queries)

    @property
    def associative_total(self) -> int:
        return sum(query.label == "associative" for query in self.queries)


def load_dataset(data_dir: str | Path, *, expected_counts: tuple[int, int, int] | None = None) -> BenchmarkDataset:
    """Load memories, queries, and optional warm-up events from ``data_dir``."""
    root = Path(data_dir)
    manifest = _verify_manifest(root)
    memory_rows = _read_jsonl(root / "memories.jsonl", max_rows=MAX_MEMORY_COUNT)
    query_rows = _read_jsonl(root / "queries.jsonl", max_rows=MAX_QUERY_COUNT)
    schedule_rows = _read_jsonl(root / "schedule.jsonl", max_rows=MAX_SCHEDULE_COUNT)
    warmup_path = root / "warmup.jsonl"
    warmup_rows = _read_jsonl(warmup_path, max_rows=MAX_WARMUP_COUNT) if warmup_path.exists() else []

    schedule = _schedule(schedule_rows)
    raw_memory_ids = {_text(row, "memory_id", "memory", index) for index, row in enumerate(memory_rows, 1)}
    if set(schedule) != raw_memory_ids:
        raise DatasetError("ingestion schedule IDs do not exactly match memory IDs")
    memories = tuple(_memory(row, index, schedule) for index, row in enumerate(memory_rows, 1))
    queries = tuple(_query(row, index) for index, row in enumerate(query_rows, 1))
    warmup = tuple(_warmup(row, index) for index, row in enumerate(warmup_rows, 1))
    _validate(memories, queries, warmup)

    counts = (len(memories), sum(q.label == "direct" for q in queries), sum(q.label == "associative" for q in queries))
    manifest_counts = manifest.get("counts")
    if not isinstance(manifest_counts, dict):
        raise DatasetError("dataset manifest requires a counts object")
    declared = (
        manifest_counts.get("memories"),
        manifest_counts.get("direct_queries"),
        manifest_counts.get("associative_queries"),
    )
    if counts != declared or len(warmup) != manifest_counts.get("warmup_events"):
        raise DatasetError(f"dataset counts do not match manifest: {counts}, warmup={len(warmup)}")
    if expected_counts is not None and counts != expected_counts:
        raise DatasetError(f"dataset counts {counts} do not match expected {expected_counts}")
    return BenchmarkDataset(memories, queries, warmup, _fingerprint(root, warmup_path.exists()))


def _read_jsonl(path: Path, *, max_rows: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetError(f"missing dataset file: {path}")
    _validate_file_size(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if line_number > max_rows:
                raise DatasetError(f"dataset file exceeds {max_rows} rows: {path}")
            if len(raw) > MAX_RECORD_CHARS:
                raise DatasetError(f"record exceeds {MAX_RECORD_CHARS} characters at {path}:{line_number}")
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"row at {path}:{line_number} must be an object")
            rows.append(row)
    if not rows:
        raise DatasetError(f"dataset file is empty: {path}")
    return rows


def _text(row: dict[str, Any], key: str, kind: str, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{kind} row {index} requires non-empty {key}")
    return value.strip()


def _ids(row: dict[str, Any], key: str, kind: str, index: int) -> tuple[str, ...]:
    value = row.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise DatasetError(f"{kind} row {index} requires a string list for {key}")
    return tuple(value)


def _memory(row: dict[str, Any], index: int, schedule: dict[str, int]) -> MemoryRecord:
    # linked_memory_ids and all other annotations remain evaluator-only.
    benchmark_id = _text(row, "memory_id", "memory", index)
    return MemoryRecord(
        benchmark_id=benchmark_id,
        content=_text(row, "content", "memory", index),
        ingest_offset_seconds=schedule.get(benchmark_id, -1),
        metadata={
            key: value
            for key, value in row.items()
            if key not in {"memory_id", "content", "created_at", "linked_memory_ids"}
        },
    )


def _query(row: dict[str, Any], index: int) -> QueryRecord:
    label = _text(row, "label", "query", index)
    if label not in {"direct", "associative"}:
        raise DatasetError(f"query row {index} has invalid label {label!r}")
    return QueryRecord(
        query_id=_text(row, "query_id", "query", index),
        label=label,
        text=_text(row, "text", "query", index),
        expected_ids=_ids(row, "expected_relevant_node_ids", "query", index),
        intermediate_ids=_ids(row, "required_intermediate_node_ids", "query", index),
        reviewer_note=str(row.get("reviewer_note", "")),
    )


def _warmup(row: dict[str, Any], index: int) -> WarmupEvent:
    positive = row.get("positive")
    if not isinstance(positive, bool):
        raise DatasetError(f"warmup row {index} requires boolean positive")
    chain_query_id = row.get("chain_query_id")
    expected_memory_id = row.get("expected_memory_id")
    if (chain_query_id is None) != (expected_memory_id is None):
        raise DatasetError(f"warmup row {index} needs both chain_query_id and expected_memory_id")
    return WarmupEvent(
        text=_text(row, "text", "warmup", index),
        positive=positive,
        chain_query_id=chain_query_id,
        expected_memory_id=expected_memory_id,
    )


def _schedule(rows: list[dict[str, Any]]) -> dict[str, int]:
    schedule: dict[str, int] = {}
    previous = -1
    for index, row in enumerate(rows, 1):
        benchmark_id = _text(row, "memory_id", "schedule", index)
        offset = row.get("ingest_offset_seconds")
        if not isinstance(offset, int) or offset < 0 or offset <= previous:
            raise DatasetError(f"schedule row {index} requires an increasing non-negative offset")
        if benchmark_id in schedule:
            raise DatasetError(f"duplicate schedule memory_id: {benchmark_id}")
        schedule[benchmark_id] = offset
        previous = offset
    return schedule


def _validate(
    memories: tuple[MemoryRecord, ...], queries: tuple[QueryRecord, ...], warmup: tuple[WarmupEvent, ...]
) -> None:
    memory_ids = [memory.benchmark_id for memory in memories]
    query_ids = [query.query_id for query in queries]
    if len(memory_ids) != len(set(memory_ids)):
        raise DatasetError("duplicate memory_id")
    if len(query_ids) != len(set(query_ids)):
        raise DatasetError("duplicate query_id")
    labels = {query.label for query in queries}
    if labels != {"direct", "associative"}:
        raise DatasetError("dataset must include direct and associative queries")
    known = set(memory_ids)
    if any(memory.ingest_offset_seconds < 0 for memory in memories):
        raise DatasetError("ingestion schedule does not match every memory")
    for query in queries:
        if not query.expected_ids or not set(query.expected_ids) <= known:
            raise DatasetError(f"query {query.query_id} has missing or unknown expected IDs")
        if query.label == "associative" and not query.intermediate_ids:
            raise DatasetError(f"associative query {query.query_id} has no intermediate IDs")
        if not set(query.intermediate_ids) <= known:
            raise DatasetError(f"query {query.query_id} has unknown intermediate IDs")
    holdout_texts = {query.text.casefold() for query in queries}
    if any(event.text.casefold() in holdout_texts for event in warmup):
        raise DatasetError("warmup and holdout query texts must be disjoint")
    _validate_no_leakage(queries, warmup, known)


def _validate_no_leakage(
    queries: tuple[QueryRecord, ...],
    warmup: tuple[WarmupEvent, ...],
    known: set[str],
) -> None:
    """A chain-traversing warm-up must surface a bridge, never the target.

    The generator enforces this too. Checking it again on load means a
    hand-edited fixture cannot slip a leaked answer past the harness.
    """
    targets = {query.query_id: set(query.expected_ids) for query in queries}
    for event in warmup:
        if event.chain_query_id is None:
            continue
        if event.chain_query_id not in targets:
            raise DatasetError(f"warmup references unknown chain {event.chain_query_id}")
        if event.expected_memory_id not in known:
            raise DatasetError(f"warmup references unknown memory {event.expected_memory_id}")
        if event.expected_memory_id in targets[event.chain_query_id]:
            raise DatasetError(f"warmup would surface the holdout target for {event.chain_query_id}")


def _fingerprint(root: Path, include_warmup: bool) -> str:
    digest = hashlib.sha256()
    names: Iterable[str] = (
        ("memories.jsonl", "queries.jsonl", "schedule.jsonl", "warmup.jsonl")
        if include_warmup
        else ("memories.jsonl", "queries.jsonl", "schedule.jsonl")
    )
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _validate_file_size(path: Path) -> None:
    size = path.stat().st_size
    if size > MAX_DATA_FILE_BYTES:
        raise DatasetError(f"dataset file exceeds {MAX_DATA_FILE_BYTES} bytes: {path}")


def _verify_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise DatasetError(f"missing dataset manifest: {path}")
    _validate_file_size(path)
    try:
        manifest = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        expected = manifest["sha256"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DatasetError(f"invalid dataset manifest: {path}") from exc
    if not isinstance(expected, dict) or not 1 <= len(expected) <= MAX_MANIFEST_FILES:
        raise DatasetError(f"manifest requires 1 to {MAX_MANIFEST_FILES} checksums: {path}")
    for name, checksum in expected.items():
        if not isinstance(name, str) or not isinstance(checksum, str):
            raise DatasetError(f"manifest checksums must map strings to strings: {path}")
        if name not in _ALLOWED_MANIFEST_FILES:
            raise DatasetError(f"manifest contains unsupported file: {name}")
        target = root / name
        if not target.is_file():
            raise DatasetError(f"manifest file is missing: {target}")
        _validate_file_size(target)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != checksum:
            raise DatasetError(f"dataset checksum mismatch: {target}")
    return manifest
