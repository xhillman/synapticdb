import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bench.baseline import BaselineConfig
from bench.dataset import BenchmarkDataset, load_dataset
from bench.oracle import BaselineOracle, OracleError, load_baseline_oracle

ROOT = Path(__file__).parents[1]
ORACLES = ROOT / "bench/oracles"


@pytest.mark.parametrize(
    ("profile", "query_count", "digest"),
    [
        ("full", 50, "b81a2f0e9540fa5e1779ef888d548e0d50c0641329f6d4e7cd15828c418b6c92"),
        ("chained", 62, "e0f8c4ac3edcc3b2d82cd73c888b2e0aafd35ca5effb1c8faa266cb40ab48fa7"),
    ],
)
def test_frozen_baseline_is_bound_to_the_dataset(profile: str, query_count: int, digest: str) -> None:
    dataset = load_dataset(ROOT / f"bench/data/{profile}", expected_counts=(500, 25, 25))
    oracle = load_baseline_oracle(ORACLES / f"{profile}.json", dataset)

    assert oracle.dataset_fingerprint == dataset.fingerprint
    assert oracle.baseline_name == "baseline"
    assert oracle.baseline_config == BaselineConfig()
    assert oracle.top_k == 10
    assert oracle.rankings_sha256 == digest
    assert tuple(row.query_id for row in oracle.rankings) == tuple(query.query_id for query in dataset.queries)
    assert len(oracle.rankings) == query_count
    assert all(len(row.ranked_ids) == oracle.top_k for row in oracle.rankings)
    assert _hits(oracle, dataset, "direct") == 25
    assert _hits(oracle, dataset, "associative") == 10


def test_full_and_chained_share_answerable_rankings() -> None:
    full_dataset = load_dataset(ROOT / "bench/data/full", expected_counts=(500, 25, 25))
    chained_dataset = load_dataset(ROOT / "bench/data/chained", expected_counts=(500, 25, 25))
    full = load_baseline_oracle(ORACLES / "full.json", full_dataset)
    chained = load_baseline_oracle(ORACLES / "chained.json", chained_dataset)
    full_query_ids = {query.query_id for query in full_dataset.queries}

    assert tuple(row.ranked_ids for row in full.rankings) == tuple(
        row.ranked_ids for row in chained.rankings if row.query_id in full_query_ids
    )


Mutation = Callable[[dict[str, Any]], None]


def _wrong_schema(payload: dict[str, Any]) -> None:
    payload["schema_version"] = 2


def _wrong_fingerprint(payload: dict[str, Any]) -> None:
    payload["dataset_fingerprint"] = "0" * 64


def _wrong_config(payload: dict[str, Any]) -> None:
    payload["baseline_config"]["final_top_k"] = 9


def _dirty_commit(payload: dict[str, Any]) -> None:
    payload["environment"]["commit"] += "-dirty"


def _missing_query(payload: dict[str, Any]) -> None:
    payload["rankings"].pop()


def _reordered_queries(payload: dict[str, Any]) -> None:
    payload["rankings"][0], payload["rankings"][1] = payload["rankings"][1], payload["rankings"][0]


def _short_ranking(payload: dict[str, Any]) -> None:
    payload["rankings"][0]["ranked_ids"].pop()


def _duplicate_memory(payload: dict[str, Any]) -> None:
    payload["rankings"][0]["ranked_ids"][1] = payload["rankings"][0]["ranked_ids"][0]


def _unknown_memory(payload: dict[str, Any]) -> None:
    payload["rankings"][0]["ranked_ids"][0] = "mem-unknown"


def _wrong_digest(payload: dict[str, Any]) -> None:
    payload["rankings_sha256"] = "0" * 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_wrong_schema, "schema_version"),
        (_wrong_fingerprint, "dataset fingerprint"),
        (_wrong_config, "configuration"),
        (_dirty_commit, "clean 40-character SHA"),
        (_missing_query, "query coverage"),
        (_reordered_queries, "query IDs"),
        (_short_ranking, "exactly 10 memory IDs"),
        (_duplicate_memory, "memory IDs must be unique"),
        (_unknown_memory, "unknown memory ID"),
        (_wrong_digest, "rankings digest"),
    ],
)
def test_oracle_rejects_drift(tmp_path: Path, mutate: Mutation, message: str) -> None:
    dataset = load_dataset(ROOT / "bench/data/full", expected_counts=(500, 25, 25))
    payload = json.loads((ORACLES / "full.json").read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OracleError, match=message):
        load_baseline_oracle(path, dataset)


def test_oracle_rejects_duplicate_fields_and_invalid_numbers(tmp_path: Path) -> None:
    dataset = load_dataset(ROOT / "bench/data/full", expected_counts=(500, 25, 25))
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    invalid_number = tmp_path / "invalid-number.json"
    invalid_number.write_text('{"schema_version": NaN}', encoding="utf-8")

    with pytest.raises(OracleError, match="duplicate field"):
        load_baseline_oracle(duplicate, dataset)
    with pytest.raises(OracleError, match="invalid number"):
        load_baseline_oracle(invalid_number, dataset)


def _hits(oracle: BaselineOracle, dataset: BenchmarkDataset, label: str) -> int:
    return sum(
        bool(set(query.expected_ids) & set(oracle.ranked_ids(query.query_id)))
        for query in dataset.queries
        if query.label == label
    )
