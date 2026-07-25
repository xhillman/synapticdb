"""Benchmark execution, scoring, and merge-gate rules."""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from importlib import metadata
from pathlib import Path
from statistics import median
from typing import Any

from .contracts import (
    MAX_DIVERSITY_PASSES,
    MAX_MEMORY_COUNT,
    MAX_QUERY_COUNT,
    MAX_SEED_COUNT,
    MAX_SIMULATED_DAYS,
    MAX_TOP_K,
    MAX_WARMUP_COUNT,
    MIN_CONFIDENCE_AUC,
)
from .dataset import BenchmarkDataset, QueryRecord
from .retrievers import INGEST_EPOCH, Retrieval, Retriever

RetrieverFactory = Callable[[], Retriever]
KNOWN_MEASURES = frozenset({"trajectory", "decay", "diversity"})


@dataclass(frozen=True)
class QueryScore:
    query_id: str
    label: str
    baseline_hit: bool
    candidate_hit: bool
    baseline_rank: int | None
    candidate_rank: int | None
    path_present: bool
    unique_win: bool
    # Rank quality, which recall@10 cannot express: rank 1 and rank 10 are the
    # same hit but not the same answer. 0.0 when the target never appears.
    reciprocal_rank: float = 0.0
    # Share of the query's annotated reasoning path that came back. The dataset
    # has always carried these IDs; until now they were only a boolean flag.
    intermediate_coverage: float = 0.0
    # Top score the candidate returned. Meaningless for ranking, but it is what
    # an agent would threshold on, and the only way to catch a system that is
    # confident about a question with no answer.
    candidate_top_score: float | None = None


@dataclass(frozen=True, slots=True)
class Timeline:
    """Simulated spans the harness replays events across.

    All zero means "use the wall clock", which is how every record predating
    the harness clock was produced.
    """

    warmup_span_days: float = 0.0
    query_offset_days: float = 0.0
    decay_probe_days: float = 0.0

    def __post_init__(self) -> None:
        spans = (self.warmup_span_days, self.query_offset_days, self.decay_probe_days)
        if any(not 0.0 <= span <= MAX_SIMULATED_DAYS for span in spans):
            raise ValueError(f"simulated spans must be between 0 and {MAX_SIMULATED_DAYS} days")

    @property
    def simulated(self) -> bool:
        return any((self.warmup_span_days, self.query_offset_days, self.decay_probe_days))


# The wall-clock timeline every record predating the harness clock was made on.
WALL_CLOCK = Timeline()


@dataclass(frozen=True)
class Measurement:
    """One directional gate: a relationship the PRD claims must hold."""

    name: str
    before: int
    after: int
    passed: bool
    # Every reading behind before/after, when a gate takes more than two. The
    # gate is a verdict; the series is the evidence that distinguishes a value
    # settling once from one compounding downward.
    series: tuple[int, ...] = ()


@dataclass(frozen=True)
class SeedResult:
    seed: int
    baseline_direct_hits: int
    candidate_direct_hits: int
    baseline_associative_hits: int
    candidate_associative_hits: int
    associative_unique_wins: int
    direct_parity: bool
    baseline_reproduced: bool
    passed: bool
    elapsed_seconds: float
    queries: tuple[QueryScore, ...]
    # Empty unless a measurement was requested, so existing records still parse.
    measurements: tuple[Measurement, ...] = ()
    # Over answerable queries only. Distractors have no rank to reciprocate.
    mrr: float = 0.0
    mean_intermediate_coverage: float = 0.0
    # median(correct-answer score) - median(distractor top score). Positive
    # means the system scores a real answer above anything it finds for a
    # question that has none. None when the profile carries no distractors.
    score_separation: float | None = None
    # Pre-registered gate: MIN_CONFIDENCE_AUC. None when the profile has no
    # distractors, which is an absent measurement rather than a passing one.
    confidence_auc: float | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    dataset_fingerprint: str
    baseline_name: str
    candidate_name: str
    top_k: int
    direct_total: int
    associative_total: int
    required_unique_wins: int
    direct_tolerance: float
    expected_baseline_hits: tuple[int, int] | None
    passed: bool
    runs: tuple[SeedResult, ...]
    baseline_config: dict[str, Any]
    candidate_config: dict[str, Any]
    environment: dict[str, str]
    # The simulated spans this run used. All zero means the wall clock, so a
    # record can never be misread as having aged a graph it did not.
    timeline: dict[str, float] = field(default_factory=dict)
    # The MRR the run had to match or beat, and the record it came from.
    # None when nothing was supplied: an absent gate, not a passing one.
    mrr_floor: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Execution:
    result: SeedResult
    baseline_name: str
    candidate_name: str
    baseline_config: dict[str, Any]
    candidate_config: dict[str, Any]


def run_benchmark(
    dataset: BenchmarkDataset,
    *,
    baseline_factory: RetrieverFactory,
    candidate_factory: RetrieverFactory | None,
    seeds: tuple[int, ...] = (1337,),
    top_k: int = 10,
    required_unique_wins: int = 10,
    direct_tolerance: float = 0.05,
    expected_baseline_hits: tuple[int, int] | None = None,
    timeline: Timeline = WALL_CLOCK,
    measures: frozenset[str] = frozenset(),
    diversity_passes: int = 2,
    mrr_floor: float | None = None,
) -> BenchmarkReport:
    """Run ingest → warm-up → holdout evaluation for each bounded seed."""
    _validate_run_inputs(dataset, seeds, top_k, required_unique_wins, direct_tolerance, expected_baseline_hits)
    unknown = measures - KNOWN_MEASURES
    if unknown:
        raise ValueError(f"unknown measurement: {sorted(unknown)[0]}")
    if not 2 <= diversity_passes <= MAX_DIVERSITY_PASSES:
        raise ValueError(f"diversity passes must be between 2 and {MAX_DIVERSITY_PASSES}")
    executions = tuple(
        _execute_seed(
            dataset,
            seed,
            baseline_factory,
            candidate_factory,
            top_k,
            required_unique_wins,
            direct_tolerance,
            expected_baseline_hits,
            timeline,
            measures,
            diversity_passes,
            mrr_floor,
        )
        for seed in seeds
    )
    _validate_execution_identity(executions)
    first = executions[0]
    return BenchmarkReport(
        dataset_fingerprint=dataset.fingerprint,
        baseline_name=first.baseline_name,
        candidate_name=first.candidate_name,
        top_k=top_k,
        direct_total=dataset.direct_total,
        associative_total=dataset.associative_total,
        required_unique_wins=required_unique_wins,
        direct_tolerance=direct_tolerance,
        expected_baseline_hits=expected_baseline_hits,
        passed=all(execution.result.passed for execution in executions),
        runs=tuple(execution.result for execution in executions),
        baseline_config=first.baseline_config,
        candidate_config=first.candidate_config,
        environment=_environment(),
        timeline=asdict(timeline),
        mrr_floor=mrr_floor,
    )


def _execute_seed(
    dataset: BenchmarkDataset,
    seed: int,
    baseline_factory: RetrieverFactory,
    candidate_factory: RetrieverFactory | None,
    top_k: int,
    required_unique_wins: int,
    direct_tolerance: float,
    expected_baseline_hits: tuple[int, int] | None,
    timeline: Timeline = WALL_CLOCK,
    measures: frozenset[str] = frozenset(),
    diversity_passes: int = 2,
    mrr_floor: float | None = None,
) -> _Execution:
    started = time.perf_counter()
    with ExitStack() as resources:
        baseline = baseline_factory()
        resources.callback(baseline.close)
        candidate = baseline if candidate_factory is None else candidate_factory()
        if candidate is not baseline:
            resources.callback(candidate.close)
        baseline.ingest(dataset.memories, seed=seed)
        if candidate is not baseline:
            candidate.ingest(dataset.memories, seed=seed)
        cold = None
        if "trajectory" in measures:
            cold = _cold_associative_hits(dataset, seed, candidate_factory, top_k, resources, timeline)
        _warm(baseline, dataset, top_k, timeline)
        if candidate is not baseline:
            _warm(candidate, dataset, top_k, timeline)
        _advance(baseline, candidate, timeline.warmup_span_days + timeline.query_offset_days)
        baseline_results = [baseline.recall(query.text, top_k=top_k) for query in dataset.queries]
        candidate_results = (
            baseline_results
            if candidate is baseline
            else [candidate.recall(query.text, top_k=top_k) for query in dataset.queries]
        )
        measurements = _measure(
            dataset,
            candidate,
            candidate_results,
            top_k=top_k,
            timeline=timeline,
            measures=measures,
            cold_associative_hits=cold,
            diversity_passes=diversity_passes,
        )
        result = _score(
            dataset,
            seed,
            baseline_results,
            candidate_results,
            top_k=top_k,
            required_unique_wins=required_unique_wins,
            direct_tolerance=direct_tolerance,
            expected_baseline_hits=expected_baseline_hits,
            elapsed=time.perf_counter() - started,
            measurements=measurements,
            mrr_floor=mrr_floor,
        )
        return _Execution(
            result,
            baseline.name,
            candidate.name,
            _retriever_config(baseline),
            _retriever_config(candidate),
        )


def _validate_run_inputs(
    dataset: BenchmarkDataset,
    seeds: tuple[int, ...],
    top_k: int,
    required_unique_wins: int,
    direct_tolerance: float,
    expected_baseline_hits: tuple[int, int] | None,
) -> None:
    if not 1 <= len(seeds) <= MAX_SEED_COUNT or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must contain 1 to {MAX_SEED_COUNT} unique values")
    if not all(isinstance(seed, int) for seed in seeds):
        raise TypeError("every seed must be an integer")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    if not 0 <= required_unique_wins <= dataset.associative_total:
        raise ValueError("required_unique_wins exceeds the associative query count")
    if not 0.0 <= direct_tolerance <= 1.0:
        raise ValueError("direct_tolerance must be between 0 and 1")
    bounded_dataset = (
        1 <= len(dataset.memories) <= MAX_MEMORY_COUNT
        and 1 <= len(dataset.queries) <= MAX_QUERY_COUNT
        and len(dataset.warmup) <= MAX_WARMUP_COUNT
        and dataset.direct_total > 0
        and dataset.associative_total > 0
    )
    if not bounded_dataset:
        raise ValueError("dataset must contain bounded direct and associative inputs")
    _validate_expected_hits(dataset, expected_baseline_hits)


def _validate_expected_hits(dataset: BenchmarkDataset, expected: tuple[int, int] | None) -> None:
    if expected is None:
        return
    if len(expected) != 2:
        raise ValueError("expected_baseline_hits requires direct and associative counts")
    if not 0 <= expected[0] <= dataset.direct_total:
        raise ValueError("expected direct baseline hits exceed dataset totals")
    if not 0 <= expected[1] <= dataset.associative_total:
        raise ValueError("expected associative baseline hits exceed dataset totals")


def _validate_execution_identity(executions: tuple[_Execution, ...]) -> None:
    first = executions[0]
    if any(execution.baseline_name != first.baseline_name for execution in executions):
        raise RuntimeError("baseline identity changed between seeds")
    if any(execution.candidate_name != first.candidate_name for execution in executions):
        raise RuntimeError("candidate identity changed between seeds")
    if any(execution.baseline_config != first.baseline_config for execution in executions):
        raise RuntimeError("baseline configuration changed between seeds")
    if any(execution.candidate_config != first.candidate_config for execution in executions):
        raise RuntimeError("candidate configuration changed between seeds")


def _warm(
    retriever: Retriever,
    dataset: BenchmarkDataset,
    top_k: int,
    timeline: Timeline = WALL_CLOCK,
) -> None:
    """Replay the warm-up events, spread across the configured span.

    Spreading matters: uniformly aged edges make decay a constant multiplier,
    which cannot reorder anything. Differing ages are what make it a signal.
    """
    events = dataset.warmup
    for index, event in enumerate(events):
        if timeline.warmup_span_days and len(events) > 1:
            _advance(retriever, retriever, timeline.warmup_span_days * index / (len(events) - 1))
        result = retriever.recall(event.text, top_k=top_k)
        retriever.feedback(result, positive=event.positive)


def _advance(first: Retriever, second: Retriever, days: float) -> None:
    """Move both retrievers to the same instant past the ingest epoch."""
    if not days:
        return
    moment = INGEST_EPOCH + timedelta(days=days)
    first.advance_to(moment)
    if second is not first:
        second.advance_to(moment)


def _cold_associative_hits(
    dataset: BenchmarkDataset,
    seed: int,
    candidate_factory: RetrieverFactory | None,
    top_k: int,
    resources: ExitStack,
    timeline: Timeline,
) -> int | None:
    """Score the holdout on an unwarmed instance, for the trajectory gate.

    A separate instance rather than an earlier pass on the same one: recall
    itself writes co-retrieval edges, so a first pass cannot be repeated as a
    clean baseline.

    The cold instance is advanced to the *same instant* the warm one answers
    at. Skipping that compares a fresh graph against an aged one and charges
    the difference to warming: at a 37-day offset a temporal edge has decayed
    to 43% of its weight, which swamps whatever the warm-up did.
    """
    if candidate_factory is None:
        return None
    cold = candidate_factory()
    resources.callback(cold.close)
    cold.ingest(dataset.memories, seed=seed)
    _advance(cold, cold, timeline.warmup_span_days + timeline.query_offset_days)
    results = [cold.recall(query.text, top_k=top_k) for query in dataset.queries]
    return _associative_hits(dataset, results, top_k)


def _associative_hits(dataset: BenchmarkDataset, results: Sequence[Retrieval], top_k: int) -> int:
    hits = 0
    for query, retrieval in zip(dataset.queries, results, strict=True):
        if query.label != "associative":
            continue
        hits += _first_rank(retrieval.ranked_ids[:top_k], set(query.expected_ids)) is not None
    return hits


def _direct_hits(dataset: BenchmarkDataset, results: Sequence[Retrieval], top_k: int) -> int:
    hits = 0
    for query, retrieval in zip(dataset.queries, results, strict=True):
        if query.label != "direct":
            continue
        hits += _first_rank(retrieval.ranked_ids[:top_k], set(query.expected_ids)) is not None
    return hits


def _distinct_memories(results: Sequence[Retrieval], top_k: int) -> int:
    return len({memory_id for retrieval in results for memory_id in retrieval.ranked_ids[:top_k]})


def _measure(
    dataset: BenchmarkDataset,
    candidate: Retriever,
    warm_results: Sequence[Retrieval],
    *,
    top_k: int,
    timeline: Timeline,
    measures: frozenset[str],
    cold_associative_hits: int | None,
    diversity_passes: int,
) -> tuple[Measurement, ...]:
    """Run the requested directional gates against the warmed candidate."""
    measurements: list[Measurement] = []
    if "trajectory" in measures and cold_associative_hits is not None:
        warm = _associative_hits(dataset, warm_results, top_k)
        measurements.append(Measurement("trajectory", cold_associative_hits, warm, warm >= cold_associative_hits))
    if "diversity" in measures:
        series = [_distinct_memories(warm_results, top_k)]
        for _ in range(diversity_passes - 1):
            repeat = [candidate.recall(query.text, top_k=top_k) for query in dataset.queries]
            series.append(_distinct_memories(repeat, top_k))
        measurements.append(Measurement("diversity", series[0], series[-1], series[-1] >= series[0], tuple(series)))
    if "decay" in measures and timeline.decay_probe_days:
        _advance(
            candidate,
            candidate,
            timeline.warmup_span_days + timeline.query_offset_days + timeline.decay_probe_days,
        )
        aged = [candidate.recall(query.text, top_k=top_k) for query in dataset.queries]
        before = _direct_hits(dataset, warm_results, top_k)
        after = _direct_hits(dataset, aged, top_k)
        measurements.append(Measurement("decay_direct_recall", before, after, after >= before))
    return tuple(measurements)


def _score(
    dataset: BenchmarkDataset,
    seed: int,
    baseline: list[Retrieval],
    candidate: list[Retrieval],
    *,
    top_k: int,
    required_unique_wins: int,
    direct_tolerance: float,
    expected_baseline_hits: tuple[int, int] | None,
    elapsed: float,
    measurements: tuple[Measurement, ...] = (),
    mrr_floor: float | None = None,
) -> SeedResult:
    expected_count = len(dataset.queries)
    if len(baseline) != expected_count or len(candidate) != expected_count:
        raise RuntimeError("retrievers must return exactly one result per query")
    scores = tuple(
        _score_query(query, base, contender, top_k)
        for query, base, contender in zip(dataset.queries, baseline, candidate, strict=True)
    )
    base_direct = sum(score.baseline_hit for score in scores if score.label == "direct")
    candidate_direct = sum(score.candidate_hit for score in scores if score.label == "direct")
    base_associative = sum(score.baseline_hit for score in scores if score.label == "associative")
    candidate_associative = sum(score.candidate_hit for score in scores if score.label == "associative")
    wins = sum(score.unique_win for score in scores)
    answerable = tuple(score for score in scores if score.label != "distractor")
    mrr = _mean(score.reciprocal_rank for score in answerable)
    coverage = _mean(score.intermediate_coverage for score in answerable if score.label == "associative")
    separation = _score_separation(scores)
    auc = _confidence_auc(scores)
    base_rate = base_direct / dataset.direct_total
    candidate_rate = candidate_direct / dataset.direct_total
    parity = candidate_rate + direct_tolerance >= base_rate
    reproduced = expected_baseline_hits is None or (base_direct, base_associative) == expected_baseline_hits
    return SeedResult(
        seed=seed,
        baseline_direct_hits=base_direct,
        candidate_direct_hits=candidate_direct,
        baseline_associative_hits=base_associative,
        candidate_associative_hits=candidate_associative,
        associative_unique_wins=wins,
        direct_parity=parity,
        baseline_reproduced=reproduced,
        # A directional gate is a gate: a violated relationship fails the run.
        # Pre-registered before the first run: MRR must not regress against the
        # reference record, and a real answer must outscore the best guess at an
        # unanswerable question. Coverage is reported, never gated — we cannot
        # yet argue which direction is good.
        passed=(
            reproduced
            and parity
            and wins >= required_unique_wins
            and all(measurement.passed for measurement in measurements)
            and (mrr_floor is None or mrr >= mrr_floor)
            and (separation is None or separation > 0.0)
            and (auc is None or auc >= MIN_CONFIDENCE_AUC)
        ),
        elapsed_seconds=elapsed,
        queries=scores,
        measurements=measurements,
        mrr=mrr,
        mean_intermediate_coverage=coverage,
        score_separation=separation,
        confidence_auc=auc,
    )


def _mean(values: Iterable[float]) -> float:
    collected = tuple(values)
    return sum(collected) / len(collected) if collected else 0.0


def _confidence_auc(scores: Sequence[QueryScore]) -> float | None:
    """How often a correct answer outscores a question that has none.

    The Mann-Whitney statistic: 1.0 is perfect separation, 0.5 is no signal,
    ties count as half. Threshold-free on purpose — it asks whether *any*
    threshold could work, rather than assuming one.
    """
    answered, distractors = _confidence_samples(scores)
    if not answered or not distractors:
        return None
    wins = sum(
        1.0 if answer > distractor else 0.5 if answer == distractor else 0.0
        for answer in answered
        for distractor in distractors
    )
    return wins / (len(answered) * len(distractors))


def _confidence_samples(scores: Sequence[QueryScore]) -> tuple[list[float], list[float]]:
    answered = [
        score.candidate_top_score
        for score in scores
        if score.label != "distractor" and score.candidate_hit and score.candidate_top_score is not None
    ]
    distractors = [
        score.candidate_top_score
        for score in scores
        if score.label == "distractor" and score.candidate_top_score is not None
    ]
    return answered, distractors


def _score_separation(scores: Sequence[QueryScore]) -> float | None:
    """How far a real answer outscores the best guess at an unanswerable one.

    Positive means the score an agent would threshold on carries information.
    None when the profile has no distractors, or when the retriever supplies no
    scores — an absent measurement, not a passing one.
    """
    answered, distractors = _confidence_samples(scores)
    if not answered or not distractors:
        return None
    return median(answered) - median(distractors)


def _score_query(query: QueryRecord, baseline: Retrieval, candidate: Retrieval, top_k: int) -> QueryScore:
    expected = set(query.expected_ids)
    baseline_rank = _first_rank(baseline.ranked_ids[:top_k], expected)
    candidate_rank = _first_rank(candidate.ranked_ids[:top_k], expected)
    path_present = bool(set(query.intermediate_ids) & set(candidate.path_benchmark_ids))
    unique = bool(
        query.label == "associative" and candidate_rank is not None and baseline_rank is None and path_present
    )
    returned = set(candidate.ranked_ids[:top_k])
    coverage = (
        len(set(query.intermediate_ids) & returned) / len(query.intermediate_ids) if query.intermediate_ids else 0.0
    )
    return QueryScore(
        query_id=query.query_id,
        label=query.label,
        baseline_hit=baseline_rank is not None,
        candidate_hit=candidate_rank is not None,
        baseline_rank=baseline_rank,
        candidate_rank=candidate_rank,
        path_present=path_present,
        unique_win=unique,
        reciprocal_rank=0.0 if candidate_rank is None else 1.0 / candidate_rank,
        intermediate_coverage=coverage,
        candidate_top_score=candidate.scores[0] if candidate.scores else None,
    )


def _first_rank(ranked_ids: tuple[str, ...], expected: set[str]) -> int | None:
    return next((rank for rank, node_id in enumerate(ranked_ids, 1) if node_id in expected), None)


def _retriever_config(retriever: Retriever) -> dict[str, Any]:
    config = getattr(retriever, "config", None)
    return config.to_dict() if config is not None and hasattr(config, "to_dict") else {}


def _git_commit() -> str | None:
    """Identify the code under test; suffix -dirty when the tree has changes."""
    cwd = Path(__file__).resolve().parent
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
            cwd=cwd,
        )
        status = subprocess.run(
            ("git", "status", "--porcelain"),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
            cwd=cwd,
        )
    except OSError:
        return None
    sha = revision.stdout.strip()
    if revision.returncode != 0 or not sha:
        return None
    if status.returncode == 0 and status.stdout.strip():
        return f"{sha}-dirty"
    return sha


def _environment() -> dict[str, str]:
    values = {"python": sys.version.split()[0], "platform": platform.platform()}
    commit = _git_commit()
    if commit is not None:
        values["commit"] = commit
    for package in (
        "faiss-cpu",
        "numpy",
        "rank-bm25",
        "scikit-learn",
        "sentence-transformers",
        "torch",
        "transformers",
    ):
        try:
            values[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return values
