from collections.abc import Sequence
from pathlib import Path

import pytest

from bench.dataset import MemoryRecord, load_dataset
from bench.protocol import run_benchmark
from bench.retrievers import FixtureRetriever, SynapticRetriever, _fixture_embedding
from synapticdb import InvalidArgumentError
from synapticdb.learning import default_parameters

ROOT = Path(__file__).parents[1]


def schedule_embedding(text: str) -> Sequence[float]:
    return (
        float("first" in text),
        float("second" in text),
        float("third" in text),
    )


def test_synaptic_retriever_runs_the_smoke_benchmark_without_models() -> None:
    dataset = load_dataset(ROOT / "bench/data/smoke", expected_counts=(50, 5, 5))
    report = run_benchmark(
        dataset,
        baseline_factory=FixtureRetriever,
        candidate_factory=lambda: SynapticRetriever(_fixture_embedding, embedding_name="fixture"),
        required_unique_wins=0,
    )
    assert report.candidate_name == "synaptic"
    assert report.candidate_config["embedding"] == "fixture"
    recorded = report.candidate_config["params"]
    assert recorded["activation_blend_weight"] == 0.45
    assert recorded["semantic_seed"] is None
    assert recorded["temporal_link"] == [600, 3, 0.2]
    # The record names the whole budget, not a chosen subset.
    assert set(recorded) == set(default_parameters())
    assert len(report.runs[0].queries) == 10


def test_synaptic_retriever_returns_activation_path_memory_ids() -> None:
    retriever = SynapticRetriever(_fixture_embedding, embedding_name="fixture")
    records = (
        MemoryRecord("first", "alpha anchor", 0),
        MemoryRecord("second", "beta related", 1),
    )
    try:
        retriever.ingest(records, seed=1337)
        first_id, second_id = retriever._benchmark_ids
        retriever._memory._store.insert_edge(first_id, second_id, 1.0, "explicit")

        result = retriever.recall("alpha", top_k=2)

        assert set(result.path_benchmark_ids) == {"first", "second"}
    finally:
        retriever.close()


def test_synaptic_retriever_applies_benchmark_semantic_parameters() -> None:
    retriever = SynapticRetriever(
        _fixture_embedding,
        embedding_name="fixture",
        semantic_seed=(0.8, 1, 0.4),
        temporal_link=(300, 2, 0.1),
    )
    try:
        # The report records the full effective budget, so a promoted record
        # can never be misread as a default-configuration run.
        params = retriever.config.params
        assert params["semantic_seed"] == [0.8, 1, 0.4]
        assert params["temporal_link"] == [300, 2, 0.1]
        assert set(params) == set(default_parameters())
    finally:
        retriever.close()


def test_synaptic_retriever_applies_and_validates_parameter_overrides() -> None:
    retriever = SynapticRetriever(
        _fixture_embedding,
        embedding_name="fixture",
        overrides={"activation_blend_weight": 0.9},
    )
    try:
        assert retriever.config.params["activation_blend_weight"] == 0.9
        assert retriever._memory._params["activation_blend_weight"] == 0.9
    finally:
        retriever.close()

    with pytest.raises(ValueError, match="unknown parameter"):
        SynapticRetriever(_fixture_embedding, embedding_name="fixture", overrides={"nope": 1})
    # A malformed override must fail at construction, not midway through a run.
    with pytest.raises(InvalidArgumentError):
        SynapticRetriever(
            _fixture_embedding,
            embedding_name="fixture",
            overrides={"activation_blend_weight": 5.0},
        )


def test_synaptic_retriever_uses_controlled_ingestion_schedule() -> None:
    retriever = SynapticRetriever(schedule_embedding, embedding_name="schedule")
    records = (
        MemoryRecord("first", "first memory", 0),
        MemoryRecord("second", "second memory", 600),
        MemoryRecord("third", "third memory", 1201),
    )
    try:
        retriever.ingest(records, seed=1337)
        first_id, second_id, third_id = retriever._benchmark_ids
        assert retriever._memory._store.get_edge_between(first_id, second_id) is not None
        assert retriever._memory._store.get_edge_between(second_id, third_id) is None
    finally:
        retriever.close()
