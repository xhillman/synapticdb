import hashlib
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

from bench.dataset import DatasetError, WarmupEvent, load_dataset
from bench.generate_dataset import CHAIN_SPECS, build_chained, build_full
from bench.generate_dataset import _validate_no_leakage as generator_leakage_guard

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


def test_every_chain_carries_a_distinct_warm_up_question() -> None:
    holdout = {spec[5].casefold() for spec in CHAIN_SPECS}
    warmups = [spec[7] for spec in CHAIN_SPECS]
    assert len(warmups) == len(CHAIN_SPECS)
    # Different question, same chain: that is what makes warming legitimate
    # rather than teaching the answer.
    assert all(text.casefold() not in holdout for text in warmups)
    assert len({text.casefold() for text in warmups}) == len(warmups)


def test_chained_profile_shares_the_corpus_and_differs_only_in_warm_up() -> None:
    full = load_dataset(ROOT / "bench/data/full", expected_counts=(500, 25, 25))
    chained = load_dataset(ROOT / "bench/data/chained", expected_counts=(500, 25, 25))
    assert full.memories == chained.memories
    assert full.queries == chained.queries
    assert full.warmup != chained.warmup
    # A different fingerprint is correct and required: it is a different
    # dataset, so its records can never be mistaken for full's.
    assert full.fingerprint != chained.fingerprint
    linked = [event for event in chained.warmup if event.chain_query_id is not None]
    assert len(linked) == 25


def test_generator_refuses_a_warm_up_that_surfaces_the_target() -> None:
    queries = [
        {
            "query_id": "Q-026",
            "label": "associative",
            "text": "why does it fail?",
            "expected_relevant_node_ids": ["mem-0004"],
            "required_intermediate_node_ids": ["mem-0002"],
        }
    ]
    leaking = [{"text": "what changed?", "positive": True, "chain_query_id": "Q-026", "expected_memory_id": "mem-0004"}]
    with pytest.raises(RuntimeError, match="surface the holdout target"):
        generator_leakage_guard(queries, leaking)

    safe = [{"text": "what changed?", "positive": True, "chain_query_id": "Q-026", "expected_memory_id": "mem-0002"}]
    generator_leakage_guard(queries, safe)


def test_loader_refuses_a_warm_up_that_surfaces_the_target(tmp_path: Path) -> None:
    chained = load_dataset(ROOT / "bench/data/chained", expected_counts=(500, 25, 25))
    target_query = next(query for query in chained.queries if query.label == "associative")
    leaking = WarmupEvent(
        text="a question that gives it away",
        positive=True,
        chain_query_id=target_query.query_id,
        expected_memory_id=target_query.expected_ids[0],
    )
    from bench.dataset import _validate

    with pytest.raises(DatasetError, match="surface the holdout target"):
        _validate(chained.memories, chained.queries, (leaking,))


def test_build_chained_keeps_a_negative_feedback_event() -> None:
    warmup = build_chained()[3]
    # Negative feedback must stay exercised, or half of PRD section 6.6 goes
    # unmeasured on the profile built to measure feedback.
    assert any(not row["positive"] for row in warmup)
    assert sum(row["positive"] for row in warmup) == len(CHAIN_SPECS)
