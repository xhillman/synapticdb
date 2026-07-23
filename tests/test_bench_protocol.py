from collections.abc import Sequence
from pathlib import Path

import pytest

from bench.contracts import MAX_SEED_COUNT
from bench.dataset import MemoryRecord, load_dataset
from bench.protocol import run_benchmark
from bench.reporting import render_markdown, write_report
from bench.retrievers import FixtureRetriever, Retrieval, _dot, _fixture_embedding

ROOT = Path(__file__).parents[1]


class ScriptedRetriever:
    def __init__(self, name: str, results: dict[str, Retrieval]) -> None:
        self.name = name
        self.results = results
        self.ingested: Sequence[MemoryRecord] = ()
        self.feedback_count = 0
        self.feedback_values: list[bool] = []
        self.closed = False

    def ingest(self, memories: Sequence[MemoryRecord], *, seed: int) -> None:
        del seed
        self.ingested = memories

    def recall(self, text: str, *, top_k: int) -> Retrieval:
        result = self.results.get(text, Retrieval(()))
        return Retrieval(result.ranked_ids[:top_k], result.path_benchmark_ids, result.query_id)

    def feedback(self, retrieval: Retrieval, *, positive: bool) -> None:
        del retrieval
        self.feedback_count += 1
        self.feedback_values.append(positive)

    def close(self) -> None:
        self.closed = True


def test_smoke_benchmark_runs_end_to_end_without_model_dependencies() -> None:
    dataset = load_dataset(ROOT / "bench/data/smoke", expected_counts=(50, 5, 5))
    report = run_benchmark(
        dataset,
        baseline_factory=FixtureRetriever,
        candidate_factory=FixtureRetriever,
        required_unique_wins=0,
    )
    assert report.passed
    assert len(report.runs[0].queries) == 10
    assert "| 1337 |" in render_markdown(report)


def test_fixture_embeddings_are_deterministic() -> None:
    assert _fixture_embedding("SQLite WAL") == _fixture_embedding("SQLite WAL")
    assert _fixture_embedding("SQLite WAL") != _fixture_embedding("banana")


def test_fixture_math_rejects_unbounded_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        _fixture_embedding("bounded input", dimensions=0)
    with pytest.raises(ValueError, match="equal non-zero"):
        _dot((1.0,), (1.0, 2.0))


def test_warmup_uses_query_level_positive_and_negative_feedback() -> None:
    dataset = load_dataset(ROOT / "bench/data/smoke", expected_counts=(50, 5, 5))
    baseline = ScriptedRetriever("baseline", {})
    candidate = ScriptedRetriever("candidate", {})
    run_benchmark(
        dataset,
        baseline_factory=lambda: baseline,
        candidate_factory=lambda: candidate,
        required_unique_wins=0,
    )
    assert baseline.feedback_values == [True, True, True, True, True, False]
    assert candidate.feedback_values == baseline.feedback_values
    assert baseline.closed
    assert candidate.closed


def test_unique_win_requires_target_baseline_miss_and_path_evidence() -> None:
    dataset = load_dataset(ROOT / "bench/data/smoke", expected_counts=(50, 5, 5))
    baseline_results: dict[str, Retrieval] = {}
    candidate_results: dict[str, Retrieval] = {}
    for query in dataset.queries:
        if query.label == "direct":
            baseline_results[query.text] = Retrieval((query.expected_ids[0],))
            candidate_results[query.text] = Retrieval((query.expected_ids[0],))
        else:
            baseline_results[query.text] = Retrieval(("mem-0050",))
            candidate_results[query.text] = Retrieval(
                (query.expected_ids[0],),
                path_benchmark_ids=(query.intermediate_ids[0],),
            )

    report = run_benchmark(
        dataset,
        baseline_factory=lambda: ScriptedRetriever("baseline", baseline_results),
        candidate_factory=lambda: ScriptedRetriever("candidate", candidate_results),
        required_unique_wins=5,
    )
    run = report.runs[0]
    assert run.associative_unique_wins == 5
    assert run.direct_parity
    assert report.passed


def test_direct_parity_allows_only_one_loss_on_25_query_corpus() -> None:
    dataset = load_dataset(ROOT / "bench/data/full", expected_counts=(500, 25, 25))
    baseline_results = {
        query.text: Retrieval((query.expected_ids[0],)) for query in dataset.queries if query.label == "direct"
    }
    candidate_results = dict(baseline_results)
    direct_queries = [query for query in dataset.queries if query.label == "direct"]
    candidate_results[direct_queries[0].text] = Retrieval(())

    report = run_benchmark(
        dataset,
        baseline_factory=lambda: ScriptedRetriever("baseline", baseline_results),
        candidate_factory=lambda: ScriptedRetriever("candidate", candidate_results),
        required_unique_wins=0,
    )
    assert report.runs[0].baseline_direct_hits == 25
    assert report.runs[0].candidate_direct_hits == 24
    assert report.runs[0].direct_parity


def test_every_seed_must_pass() -> None:
    dataset = load_dataset(ROOT / "bench/data/smoke", expected_counts=(50, 5, 5))
    report = run_benchmark(
        dataset,
        baseline_factory=FixtureRetriever,
        candidate_factory=FixtureRetriever,
        seeds=(7, 19, 42),
        required_unique_wins=0,
    )
    assert len(report.runs) == 3
    assert report.passed


def test_benchmark_rejects_more_than_the_seed_limit() -> None:
    dataset = load_dataset(ROOT / "bench/data/smoke", expected_counts=(50, 5, 5))
    seeds = tuple(range(MAX_SEED_COUNT + 1))
    with pytest.raises(ValueError, match="unique values"):
        run_benchmark(
            dataset,
            baseline_factory=FixtureRetriever,
            candidate_factory=None,
            seeds=seeds,
            required_unique_wins=0,
        )


def test_report_writer_emits_json_and_markdown(tmp_path: Path) -> None:
    dataset = load_dataset(ROOT / "bench/data/smoke", expected_counts=(50, 5, 5))
    report = run_benchmark(
        dataset,
        baseline_factory=FixtureRetriever,
        candidate_factory=FixtureRetriever,
        required_unique_wins=0,
    )
    json_path, markdown_path = write_report(report, tmp_path, run_id="fixed")
    assert '"dataset_fingerprint"' in json_path.read_text(encoding="utf-8")
    assert "# SynapticDB Benchmark" in markdown_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="filename-safe"):
        write_report(report, tmp_path, run_id="../outside")
