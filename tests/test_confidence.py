import pytest

from synapticdb.confidence import ConfidenceCache, GraphMetrics, graph_confidence


def test_confidence_is_zero_for_empty_or_edgeless_graph() -> None:
    assert graph_confidence(GraphMetrics(0, 0, 0.0, 0.0)) == 0.0
    assert graph_confidence(GraphMetrics(500, 0, 0.0, 0.0)) == 0.0


def test_confidence_combines_the_four_components() -> None:
    metrics = GraphMetrics(
        node_count=500,
        edge_count=4000,
        average_edge_weight=0.6,
        average_reinforcement_count=10.0,
    )
    assert graph_confidence(metrics) == pytest.approx(1.0)


def test_confidence_interpolates_within_bands() -> None:
    metrics = GraphMetrics(
        node_count=125,
        edge_count=250,
        average_edge_weight=0.475,
        average_reinforcement_count=6.0,
    )
    assert graph_confidence(metrics) == pytest.approx(0.285)


def test_confidence_cache_requires_invalidation_before_recompute() -> None:
    cache = ConfidenceCache()
    cold = GraphMetrics(10, 1, 0.2, 0.0)
    mature = GraphMetrics(500, 4000, 0.6, 10.0)
    assert cache.get(cold) == 0.0
    assert cache.get(mature) == 0.0
    cache.invalidate()
    assert cache.get(mature) == 1.0


def test_graph_metrics_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        GraphMetrics(-1, 0, 0.0, 0.0)
    with pytest.raises(ValueError):
        GraphMetrics(1, 1, float("nan"), 0.0)
    with pytest.raises(ValueError):
        GraphMetrics(1, 1, 0.5, float("inf"))
