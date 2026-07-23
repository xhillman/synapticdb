import pytest

from bench.baseline import BaselineConfig, _rrf, _sigmoid, _sort


def test_rrf_rewards_overlap_and_uses_one_based_ranks() -> None:
    scores = _rrf([("a", 9.0), ("b", 8.0)], [("b", 1.0), ("c", 0.5)], 60)
    assert scores["a"] == 1 / 61
    assert scores["b"] == 1 / 62 + 1 / 61
    assert scores["b"] > scores["a"]


def test_baseline_helpers_are_deterministic() -> None:
    assert _sort([("b", 1.0), ("a", 1.0)]) == [("a", 1.0), ("b", 1.0)]
    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(-1_000.0) == 0.0
    assert _sigmoid(1_000.0) == 1.0


def test_baseline_configuration_enforces_resource_and_weight_contracts() -> None:
    with pytest.raises(ValueError, match="positive"):
        BaselineConfig(final_top_k=0)
    with pytest.raises(ValueError, match="sum to 1"):
        BaselineConfig(cross_encoder_weight=0.5)
