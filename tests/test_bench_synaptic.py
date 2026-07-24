from pathlib import Path

from bench.dataset import MemoryRecord, load_dataset
from bench.protocol import run_benchmark
from bench.retrievers import FixtureRetriever, SynapticRetriever, _fixture_embedding

ROOT = Path(__file__).parents[1]


def test_synaptic_retriever_runs_the_smoke_benchmark_without_models() -> None:
    dataset = load_dataset(ROOT / "bench/data/smoke", expected_counts=(50, 5, 5))
    report = run_benchmark(
        dataset,
        baseline_factory=FixtureRetriever,
        candidate_factory=lambda: SynapticRetriever(_fixture_embedding, embedding_name="fixture"),
        required_unique_wins=0,
    )
    assert report.candidate_name == "synaptic"
    assert report.candidate_config == {
        "embedding": "fixture",
        "activation_blend_weight": 0.45,
    }
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
