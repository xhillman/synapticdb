"""Benchmark execution, scoring, and merge-gate rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
import platform
import sys
import time
from typing import Any, Callable

from .dataset import BenchmarkDataset, QueryRecord
from .retrievers import Retriever, Retrieval


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_benchmark(
    dataset: BenchmarkDataset,
    *,
    baseline_factory: Callable[[], Retriever],
    candidate_factory: Callable[[], Retriever],
    seeds: tuple[int, ...] = (1337,),
    top_k: int = 10,
    required_unique_wins: int = 10,
    direct_tolerance: float = 0.05,
    expected_baseline_hits: tuple[int, int] | None = None,
) -> BenchmarkReport:
    """Run ingest → warm-up → holdout evaluation for each fixed seed."""
    if not seeds or top_k <= 0:
        raise ValueError("at least one seed and a positive top_k are required")
    runs: list[SeedResult] = []
    baseline_name = candidate_name = ""
    baseline_config: dict[str, Any] = {}
    candidate_config: dict[str, Any] = {}
    for seed in seeds:
        started = time.perf_counter()
        baseline = baseline_factory()
        same_retriever = candidate_factory is baseline_factory
        candidate = baseline if same_retriever else candidate_factory()
        baseline_name, candidate_name = baseline.name, candidate.name
        baseline_config = _retriever_config(baseline)
        candidate_config = _retriever_config(candidate)
        baseline.ingest(dataset.memories, seed=seed)
        if not same_retriever:
            candidate.ingest(dataset.memories, seed=seed)
        _warm(baseline, dataset, top_k)
        if not same_retriever:
            _warm(candidate, dataset, top_k)
        baseline_results = [baseline.recall(query.text, top_k=top_k) for query in dataset.queries]
        candidate_results = baseline_results if same_retriever else [
            candidate.recall(query.text, top_k=top_k) for query in dataset.queries
        ]
        runs.append(
            _score(
                dataset,
                seed,
                baseline_results,
                candidate_results,
                top_k=top_k,
                required_unique_wins=required_unique_wins,
                direct_tolerance=direct_tolerance,
                expected_baseline_hits=expected_baseline_hits,
                elapsed=time.perf_counter() - started,
            )
        )
    return BenchmarkReport(
        dataset_fingerprint=dataset.fingerprint,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        top_k=top_k,
        direct_total=dataset.direct_total,
        associative_total=dataset.associative_total,
        required_unique_wins=required_unique_wins,
        direct_tolerance=direct_tolerance,
        expected_baseline_hits=expected_baseline_hits,
        passed=all(run.passed for run in runs),
        runs=tuple(runs),
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        environment=_environment(),
    )


def _warm(retriever: Retriever, dataset: BenchmarkDataset, top_k: int) -> None:
    for event in dataset.warmup:
        result = retriever.recall(event.text, top_k=top_k)
        retriever.feedback(result, positive=event.positive)


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
) -> SeedResult:
    scores = tuple(
        _score_query(query, base, contender, top_k)
        for query, base, contender in zip(dataset.queries, baseline, candidate)
    )
    base_direct = sum(score.baseline_hit for score in scores if score.label == "direct")
    candidate_direct = sum(score.candidate_hit for score in scores if score.label == "direct")
    base_associative = sum(score.baseline_hit for score in scores if score.label == "associative")
    candidate_associative = sum(score.candidate_hit for score in scores if score.label == "associative")
    wins = sum(score.unique_win for score in scores)
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
        passed=reproduced and parity and wins >= required_unique_wins,
        elapsed_seconds=elapsed,
        queries=scores,
    )


def _score_query(query: QueryRecord, baseline: Retrieval, candidate: Retrieval, top_k: int) -> QueryScore:
    expected = set(query.expected_ids)
    baseline_rank = _first_rank(baseline.ranked_ids[:top_k], expected)
    candidate_rank = _first_rank(candidate.ranked_ids[:top_k], expected)
    path_present = bool(set(query.intermediate_ids) & set(candidate.path_benchmark_ids))
    unique = bool(
        query.label == "associative"
        and candidate_rank is not None
        and baseline_rank is None
        and path_present
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
    )


def _first_rank(ranked_ids: tuple[str, ...], expected: set[str]) -> int | None:
    return next((rank for rank, node_id in enumerate(ranked_ids, 1) if node_id in expected), None)


def _retriever_config(retriever: Retriever) -> dict[str, Any]:
    config = getattr(retriever, "config", None)
    return config.to_dict() if config is not None and hasattr(config, "to_dict") else {}


def _environment() -> dict[str, str]:
    values = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("faiss-cpu", "numpy", "rank-bm25", "scikit-learn", "sentence-transformers", "torch", "transformers"):
        try:
            values[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return values
