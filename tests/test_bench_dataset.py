import hashlib
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

from bench.dataset import DatasetError, load_dataset
from bench.generate_dataset import build_full

ROOT = Path(__file__).parents[1]


def test_full_dataset_contract_and_gold_isolation() -> None:
    dataset = load_dataset(ROOT / "bench/data/full", expected_counts=(500, 25, 25))

    assert len(dataset.warmup) == 6
    assert sum(event.positive for event in dataset.warmup) == 5
    assert len(dataset.fingerprint) == 64
    assert [memory.benchmark_id for memory in dataset.memories[:3]] == [
        "mem-0001",
        "mem-0002",
        "mem-0003",
    ]
    assert all("linked_memory_ids" not in memory.metadata for memory in dataset.memories)
    assert all("created_at" not in memory.metadata for memory in dataset.memories)
    assert not hasattr(dataset.memories[0], "linked_memory_ids")
    assert all(left.ingest_offset_seconds < right.ingest_offset_seconds for left, right in pairwise(dataset.memories))
    gaps = {right.ingest_offset_seconds - left.ingest_offset_seconds for left, right in pairwise(dataset.memories)}
    assert gaps == {120, 900}
    assert Counter(memory.metadata["memory_type"] for memory in dataset.memories) == {
        "factual": 125,
        "episodic": 125,
        "procedural": 125,
        "contextual": 125,
    }
    assert Counter(memory.metadata["source"] for memory in dataset.memories) == {
        "incident-log": 125,
        "runbook": 125,
        "service-ticket": 125,
        "field-note": 125,
    }


def test_smoke_dataset_contract() -> None:
    dataset = load_dataset(ROOT / "bench/data/smoke", expected_counts=(50, 5, 5))
    assert len(dataset.queries) == 10
    assert {query.label for query in dataset.queries} == {"direct", "associative"}
    type_counts = Counter(memory.metadata["memory_type"] for memory in dataset.memories)
    assert max(type_counts.values()) - min(type_counts.values()) <= 1


def test_synthetic_generator_is_deterministic() -> None:
    assert build_full() == build_full()


def test_manifest_detects_dataset_drift(tmp_path: Path) -> None:
    (tmp_path / "memories.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "schedule.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sha256": {"memories.jsonl": "bad"}}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="checksum mismatch"):
        load_dataset(tmp_path)


def test_dataset_rejects_more_than_the_bounded_memory_rows(tmp_path: Path) -> None:
    payload = "{}\n" * 501
    (tmp_path / "memories.jsonl").write_text(payload, encoding="utf-8")
    checksum = hashlib.sha256(payload.encode()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sha256": {"memories.jsonl": checksum}}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="exceeds 500 rows"):
        load_dataset(tmp_path)


def test_manifest_cannot_reference_files_outside_dataset(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sha256": {"../outside.jsonl": "unused"}}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="unsupported file"):
        load_dataset(tmp_path)
