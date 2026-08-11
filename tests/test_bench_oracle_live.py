import os
from pathlib import Path

from bench.baseline import LockedBaseline
from bench.dataset import load_dataset
from bench.oracle import load_baseline_oracle

ROOT = Path(__file__).parents[1]


def test_live_baseline_matches_both_frozen_oracles() -> None:
    if os.environ.get("SYNAPTICDB_RUN_LIVE_BASELINE") != "1":
        import pytest

        pytest.skip("set SYNAPTICDB_RUN_LIVE_BASELINE=1 to verify the model-backed baseline")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    full = load_dataset(ROOT / "bench/data/full", expected_counts=(500, 25, 25))
    chained = load_dataset(ROOT / "bench/data/chained", expected_counts=(500, 25, 25))
    baseline = LockedBaseline()
    try:
        baseline.ingest(full.memories, seed=1337)
        for profile, dataset in (("full", full), ("chained", chained)):
            oracle = load_baseline_oracle(ROOT / f"bench/oracles/{profile}.json", dataset)
            actual = tuple(baseline.recall(query.text, top_k=oracle.top_k).ranked_ids for query in dataset.queries)
            assert actual == tuple(row.ranked_ids for row in oracle.rankings)
    finally:
        baseline.close()


if __name__ == "__main__":
    test_live_baseline_matches_both_frozen_oracles()
    print("live baseline matches the full and chained frozen oracles")
