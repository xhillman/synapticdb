from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pytest

from bench.contracts import MAX_DIVERSITY_PASSES, MAX_SEED_COUNT, MAX_SIMULATED_DAYS
from bench.dataset import BenchmarkDataset, MemoryRecord, load_dataset
from bench.protocol import Timeline, run_benchmark
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
        self.recalls: list[str] = []
        self.advances: list[datetime] = []

    def ingest(self, memories: Sequence[MemoryRecord], *, seed: int) -> None:
        del seed
        self.ingested = memories

    def recall(self, text: str, *, top_k: int) -> Retrieval:
        self.recalls.append(text)
        result = self.results.get(text, Retrieval(()))
        return Retrieval(result.ranked_ids[:top_k], result.path_benchmark_ids, result.query_id)

    def feedback(self, retrieval: Retrieval, *, positive: bool) -> None:
        del retrieval
        self.feedback_count += 1
        self.feedback_values.append(positive)

    def advance_to(self, moment: datetime) -> None:
        self.advances.append(moment)

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


def _smoke() -> BenchmarkDataset:
    return load_dataset(ROOT / "bench/data/smoke", expected_counts=(50, 5, 5))


def test_no_measurement_requested_leaves_the_clock_untouched() -> None:
    dataset = _smoke()
    baseline = ScriptedRetriever("baseline", {})
    candidate = ScriptedRetriever("candidate", {})
    report = run_benchmark(
        dataset,
        baseline_factory=lambda: baseline,
        candidate_factory=lambda: candidate,
        required_unique_wins=0,
    )
    # Default behaviour must stay exactly what every pre-clock record used.
    assert baseline.advances == []
    assert candidate.advances == []
    assert report.runs[0].measurements == ()


def test_trajectory_scores_a_cold_instance_that_is_never_warmed() -> None:
    dataset = _smoke()
    built: list[ScriptedRetriever] = []

    def candidate_factory() -> ScriptedRetriever:
        retriever = ScriptedRetriever("candidate", {})
        built.append(retriever)
        return retriever

    run_benchmark(
        dataset,
        baseline_factory=lambda: ScriptedRetriever("baseline", {}),
        candidate_factory=candidate_factory,
        required_unique_wins=0,
        measures=frozenset({"trajectory"}),
    )

    warm, cold = built[0], built[1]
    # The cold instance answers the holdout only: no warm-up, no feedback.
    assert cold.feedback_count == 0
    assert warm.feedback_count == len(dataset.warmup)
    assert len(cold.recalls) == len(dataset.queries)


def test_trajectory_gate_fails_when_warming_loses_a_hit() -> None:
    dataset = _smoke()
    associative = [query for query in dataset.queries if query.label == "associative"]
    target = associative[0]
    answer = Retrieval((target.expected_ids[0],), target.intermediate_ids[:1], "q")
    built: list[ScriptedRetriever] = []

    def candidate_factory() -> ScriptedRetriever:
        # The first instance built is the warm one and answers nothing; the
        # second is cold and answers correctly, so warming loses a hit.
        retriever = ScriptedRetriever("candidate", {} if not built else {target.text: answer})
        built.append(retriever)
        return retriever

    report = run_benchmark(
        dataset,
        baseline_factory=lambda: ScriptedRetriever("baseline", {}),
        candidate_factory=candidate_factory,
        required_unique_wins=0,
        measures=frozenset({"trajectory"}),
    )

    trajectory = report.runs[0].measurements[0]
    assert trajectory.name == "trajectory"
    assert (trajectory.before, trajectory.after) == (1, 0)
    assert not trajectory.passed
    assert not report.passed


class NarrowingRetriever(ScriptedRetriever):
    """Returns fewer distinct memories over time: forever, or once then steady."""

    def __init__(self, name: str, pool: Sequence[str], *, compounding: bool) -> None:
        super().__init__(name, {})
        self.pool = list(pool)
        self.compounding = compounding
        self._seen: dict[str, int] = {}

    def recall(self, text: str, *, top_k: int) -> Retrieval:
        # How many times this query has come round is the pass index. Counting
        # repeats per text rather than tracking a first query keeps warm-up
        # events, which never repeat, from being mistaken for a pass boundary.
        self._seen[text] = self._seen.get(text, 0) + 1
        passes = self._seen[text] - 1
        narrowed = passes if self.compounding else min(passes, 1)
        width = max(1, len(self.pool) - narrowed)
        return Retrieval(tuple(self.pool[:width])[:top_k])


def _diversity_series(compounding: bool, passes: int) -> tuple[int, ...]:
    dataset = _smoke()
    # Smaller than top_k, so narrowing is visible rather than clipped by it.
    pool = [f"mem-{index:04d}" for index in range(1, 9)]
    report = run_benchmark(
        dataset,
        baseline_factory=lambda: ScriptedRetriever("baseline", {}),
        candidate_factory=lambda: NarrowingRetriever("candidate", pool, compounding=compounding),
        required_unique_wins=0,
        measures=frozenset({"diversity"}),
        diversity_passes=passes,
    )
    return report.runs[0].measurements[0].series


def test_diversity_series_separates_compounding_from_settling() -> None:
    compounding = _diversity_series(compounding=True, passes=4)
    settling = _diversity_series(compounding=False, passes=4)

    # Both fail the gate. The gate alone cannot tell them apart, which is why
    # the series exists: one keeps falling, the other stops.
    assert compounding[-1] < compounding[0]
    assert settling[-1] < settling[0]
    assert len(set(compounding)) == len(compounding)
    assert settling[1] == settling[-1]


def test_diversity_defaults_to_the_original_two_pass_probe() -> None:
    dataset = _smoke()
    report = run_benchmark(
        dataset,
        baseline_factory=FixtureRetriever,
        candidate_factory=FixtureRetriever,
        required_unique_wins=0,
        measures=frozenset({"diversity"}),
    )
    measurement = report.runs[0].measurements[0]
    assert len(measurement.series) == 2
    assert (measurement.before, measurement.after) == (measurement.series[0], measurement.series[-1])


@pytest.mark.parametrize("passes", [1, 0, MAX_DIVERSITY_PASSES + 1])
def test_diversity_passes_are_bounded(passes: int) -> None:
    with pytest.raises(ValueError, match="diversity passes"):
        run_benchmark(
            _smoke(),
            baseline_factory=FixtureRetriever,
            candidate_factory=FixtureRetriever,
            required_unique_wins=0,
            diversity_passes=passes,
        )


class AnsweringRetriever(ScriptedRetriever):
    """Answers every answerable query at a fixed rank, with a fixed confidence.

    `distractor_score` is what it reports for a question with no answer — the
    knob that separates a calibrated system from an overconfident one.
    """

    def __init__(
        self,
        name: str,
        dataset: BenchmarkDataset,
        *,
        rank: int,
        answer_score: float,
        distractor_score: float,
    ) -> None:
        super().__init__(name, {})
        self.answers = {q.text: q.expected_ids for q in dataset.queries if q.expected_ids}
        self.rank = rank
        self.answer_score = answer_score
        self.distractor_score = distractor_score

    def recall(self, text: str, *, top_k: int) -> Retrieval:
        filler = [f"filler-{index}" for index in range(top_k)]
        expected = self.answers.get(text)
        if expected is None:
            return Retrieval(tuple(filler), scores=tuple([self.distractor_score] * len(filler)))
        ranked = [*filler[: self.rank - 1], expected[0], *filler[self.rank - 1 :]][:top_k]
        return Retrieval(tuple(ranked), scores=tuple([self.answer_score] * len(ranked)))


def _run(candidate: ScriptedRetriever, dataset: BenchmarkDataset, **kwargs: object) -> object:
    return run_benchmark(
        dataset,
        baseline_factory=lambda: ScriptedRetriever("baseline", {}),
        candidate_factory=lambda: candidate,
        required_unique_wins=0,
        **kwargs,  # type: ignore[arg-type]
    )


def test_reciprocal_rank_reflects_where_the_answer_landed() -> None:
    dataset = _smoke()
    for rank, expected in ((1, 1.0), (4, 0.25)):
        candidate = AnsweringRetriever("c", dataset, rank=rank, answer_score=0.9, distractor_score=0.1)
        report = _run(candidate, dataset)
        assert report.runs[0].mrr == pytest.approx(expected)  # type: ignore[attr-defined]

    # Never returning the answer scores zero, not "no data".
    blind = ScriptedRetriever("blind", {})
    assert _run(blind, dataset).runs[0].mrr == 0.0  # type: ignore[attr-defined]


def test_chain_coverage_counts_the_reasoning_path_returned() -> None:
    dataset = _smoke()
    associative = [q for q in dataset.queries if q.label == "associative"]
    both = {q.text: Retrieval(q.intermediate_ids[:2] + q.expected_ids) for q in associative}
    one = {q.text: Retrieval(q.intermediate_ids[:1] + q.expected_ids) for q in associative}
    assert _run(ScriptedRetriever("c", both), dataset).runs[0].mean_intermediate_coverage == 1.0  # type: ignore[attr-defined]
    assert _run(ScriptedRetriever("c", one), dataset).runs[0].mean_intermediate_coverage == 0.5  # type: ignore[attr-defined]


def test_the_mrr_ratchet_fails_a_regression() -> None:
    dataset = _smoke()
    strong = AnsweringRetriever("c", dataset, rank=1, answer_score=0.9, distractor_score=0.1)
    weak = AnsweringRetriever("c", dataset, rank=5, answer_score=0.9, distractor_score=0.1)

    assert _run(strong, dataset, mrr_floor=1.0).passed  # type: ignore[attr-defined]
    # 0.2 against a floor of 1.0: the ratchet must refuse it.
    assert not _run(weak, dataset, mrr_floor=1.0).passed  # type: ignore[attr-defined]
    # Absent a floor the gate is silent, not vacuously green.
    assert _run(weak, dataset).runs[0].mrr == pytest.approx(0.2)  # type: ignore[attr-defined]


def _chained() -> BenchmarkDataset:
    return load_dataset(ROOT / "bench/data/chained", expected_counts=(500, 25, 25))


def _auc_for(answer_score: float, distractor_score: float) -> float | None:
    dataset = _chained()
    candidate = AnsweringRetriever("c", dataset, rank=1, answer_score=answer_score, distractor_score=distractor_score)
    report = _run(candidate, dataset)
    auc: float | None = report.runs[0].confidence_auc  # type: ignore[attr-defined]
    return auc


def test_confidence_auc_measures_separation_in_both_directions() -> None:
    # Perfect separation, none, and inverted.
    assert _auc_for(0.9, 0.1) == pytest.approx(1.0)
    assert _auc_for(0.5, 0.5) == pytest.approx(0.5)
    assert _auc_for(0.1, 0.9) == pytest.approx(0.0)


def test_the_confidence_gate_fails_a_system_that_cannot_be_thresholded() -> None:
    dataset = _chained()
    separating = AnsweringRetriever("c", dataset, rank=1, answer_score=0.9, distractor_score=0.1)
    overlapping = AnsweringRetriever("c", dataset, rank=1, answer_score=0.5, distractor_score=0.5)

    assert _run(separating, dataset).passed  # type: ignore[attr-defined]
    # An AUC of 0.5 is no signal: a threshold cannot separate the two, and the
    # run must say so rather than reporting a number nobody checks.
    assert not _run(overlapping, dataset).passed  # type: ignore[attr-defined]


def test_confidence_gate_is_absent_rather_than_passing_without_distractors() -> None:
    dataset = _smoke()
    candidate = AnsweringRetriever("c", dataset, rank=1, answer_score=0.5, distractor_score=0.5)
    run = _run(candidate, dataset).runs[0]  # type: ignore[attr-defined]
    # The smoke profile has no distractors, so there is nothing to measure.
    assert run.confidence_auc is None
    assert run.passed


def test_measurement_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown measurement"):
        run_benchmark(
            _smoke(),
            baseline_factory=FixtureRetriever,
            candidate_factory=FixtureRetriever,
            required_unique_wins=0,
            measures=frozenset({"vibes"}),
        )


def test_simulated_spans_are_bounded() -> None:
    with pytest.raises(ValueError, match="between 0 and"):
        Timeline(warmup_span_days=-1.0)
    with pytest.raises(ValueError, match="between 0 and"):
        Timeline(decay_probe_days=MAX_SIMULATED_DAYS + 1)


def test_warmup_spreads_events_across_the_configured_span() -> None:
    dataset = _smoke()
    candidate = ScriptedRetriever("candidate", {})
    run_benchmark(
        dataset,
        baseline_factory=lambda: ScriptedRetriever("baseline", {}),
        candidate_factory=lambda: candidate,
        required_unique_wins=0,
        timeline=Timeline(warmup_span_days=30.0),
    )
    # Differing instants are the whole point: uniformly aged edges make decay a
    # constant multiplier, which cannot reorder anything.
    warmup_instants = candidate.advances[: len(dataset.warmup)]
    assert len(set(warmup_instants)) > 1
    assert warmup_instants == sorted(warmup_instants)


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
