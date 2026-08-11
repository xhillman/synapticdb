import json
import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from bench.__main__ import _parse_overrides
from bench.dataset import MemoryRecord, load_dataset
from bench.protocol import run_benchmark
from bench.retrievers import (
    INGEST_EPOCH,
    FixtureRetriever,
    SynapticRetriever,
    _fixture_embedding,
)
from synapticdb import InvalidArgumentError
from synapticdb.learning import ParameterValue, default_parameters

ROOT = Path(__file__).parents[1]

EXPECTED_REPORTED_PARAMS: dict[str, object] = {
    "activation_blend_weight": 0.45,
    "activation_decay": 0.2,
    "activation_max_steps": 5,
    "activation_min_energy": 0.05,
    "activation_seeds": 5,
    "candidate_depth": 40,
    "co_retrieval": [0.05, 0.05],
    "connect_weight": 0.5,
    "decay_and_prune": [30, 0.02],
    "feedback_rate": 0.15,
    "hop_bonus": 0.15,
    "maintenance_interval": 100,
    "rrf_k": 60,
    "seed_penalty": 0.2,
    "semantic_seed": None,
    "temporal_link": [600, 3, 0.2],
    "top_k": 10,
}

VALID_PARAMETER_OVERRIDES: tuple[tuple[str, ParameterValue, object], ...] = (
    ("top_k", 7, 7),
    ("candidate_depth", 30, 30),
    ("rrf_k", 50, 50),
    ("activation_seeds", 4, 4),
    ("activation_max_steps", 4, 4),
    ("activation_decay", 0.3, 0.3),
    ("activation_min_energy", 0.1, 0.1),
    ("hop_bonus", 0.25, 0.25),
    ("seed_penalty", 0.15, 0.15),
    ("activation_blend_weight", 0.6, 0.6),
    ("semantic_seed", (0.7, 2, 0.3), [0.7, 2, 0.3]),
    ("temporal_link", (300, 2, 0.1), [300, 2, 0.1]),
    ("co_retrieval", (0.1, 0.2), [0.1, 0.2]),
    ("feedback_rate", 0.2, 0.2),
    ("connect_weight", 0.6, 0.6),
    ("decay_and_prune", (60, 0.03), [60, 0.03]),
    ("maintenance_interval", 200, 200),
)


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
    assert recorded == EXPECTED_REPORTED_PARAMS
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


@pytest.mark.parametrize(("key", "value", "recorded"), VALID_PARAMETER_OVERRIDES)
def test_synaptic_retriever_accepts_and_records_each_parameter(
    key: str,
    value: ParameterValue,
    recorded: object,
) -> None:
    retriever = SynapticRetriever(
        _fixture_embedding,
        embedding_name="fixture",
        overrides={key: value},
    )
    try:
        assert retriever.config.params[key] == recorded
        assert set(retriever.config.params) == set(default_parameters())
    finally:
        retriever.close()


def test_parameter_overrides_take_precedence_over_focused_benchmark_options() -> None:
    retriever = SynapticRetriever(
        _fixture_embedding,
        embedding_name="fixture",
        semantic_seed=(0.8, 1, 0.4),
        temporal_link=(300, 2, 0.1),
        overrides={
            "semantic_seed": (0.7, 2, 0.3),
            "temporal_link": (900, 4, 0.25),
        },
    )
    try:
        assert retriever.config.params["semantic_seed"] == [0.7, 2, 0.3]
        assert retriever.config.params["temporal_link"] == [900, 4, 0.25]
    finally:
        retriever.close()


def test_synaptic_retriever_rejects_unknown_and_invalid_parameters_during_construction() -> None:
    with pytest.raises(ValueError, match=r"^unknown parameter: nope$"):
        SynapticRetriever(_fixture_embedding, embedding_name="fixture", overrides={"nope": 1})
    with pytest.raises(InvalidArgumentError, match=r"^activation blend weight must be between 0 and 1$"):
        SynapticRetriever(
            _fixture_embedding,
            embedding_name="fixture",
            overrides={"activation_blend_weight": 5.0},
        )


def test_parse_overrides_accepts_the_complete_parameter_budget() -> None:
    entries = tuple(f"{key}={json.dumps(value)}" for key, value in default_parameters().items())
    assert _parse_overrides(entries) == default_parameters()


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (("broken",), "--param expects KEY=JSON, got: broken"),
        (("nope=1",), "unknown parameter: nope"),
        (("top_k=10", "top_k=5"), "parameter repeated: top_k"),
        (("top_k={broken",), "--param top_k needs a JSON value, got: {broken"),
    ],
)
def test_parse_overrides_preserves_boundary_errors(entries: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _parse_overrides(entries)


def test_advance_to_moves_the_instant_recall_records() -> None:
    retriever = SynapticRetriever(_fixture_embedding, embedding_name="fixture")
    records = (
        MemoryRecord("first", "alpha anchor", 0),
        MemoryRecord("second", "alpha related", 120),
    )
    moment = INGEST_EPOCH + timedelta(days=45)
    try:
        retriever.ingest(records, seed=1337)
        retriever.advance_to(moment)
        result = retriever.recall("alpha", top_k=2)
        stored = retriever._memory._store.get_query(UUID(result.query_id or ""))
        assert stored.created_at == moment
    finally:
        retriever.close()


def test_the_benchmark_clock_only_moves_forward() -> None:
    retriever = SynapticRetriever(_fixture_embedding, embedding_name="fixture")
    try:
        retriever.advance_to(INGEST_EPOCH + timedelta(days=10))
        with pytest.raises(ValueError, match="only moves forward"):
            retriever.advance_to(INGEST_EPOCH + timedelta(days=9))
        with pytest.raises(ValueError, match="aware datetime"):
            retriever.advance_to(datetime(2030, 1, 1))
    finally:
        retriever.close()


def test_spreading_the_warm_up_leaves_edges_at_differing_ages() -> None:
    retriever = SynapticRetriever(_fixture_embedding, embedding_name="fixture")
    # Enough memories that one recall's co-retrieval cannot re-touch every
    # edge: on a tiny corpus it does, and every age collapses to the last
    # recall's instant.
    records = tuple(MemoryRecord(f"m{index}", f"alpha memory {index}", index * 120) for index in range(12))
    try:
        retriever.ingest(records, seed=1337)
        for day in (10, 40):
            retriever.advance_to(INGEST_EPOCH + timedelta(days=day))
            retriever.recall("alpha", top_k=4)
        store = retriever._memory._store
        stamps = {
            edge.last_reinforced_at
            for memory_id in retriever._benchmark_ids
            for edge in store.list_edges_for_node(memory_id)
        }
        # Uniform ages make decay a constant multiplier that cannot reorder
        # anything; differing ages are what make it measurable.
        assert len(stamps) > 1
    finally:
        retriever.close()


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
