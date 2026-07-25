"""Graph confidence calculation and instance-local caching."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GraphMetrics:
    node_count: int
    edge_count: int
    average_edge_weight: float
    average_reinforcement_count: float

    def __post_init__(self) -> None:
        counts_are_valid = (
            isinstance(self.node_count, int)
            and not isinstance(self.node_count, bool)
            and self.node_count >= 0
            and isinstance(self.edge_count, int)
            and not isinstance(self.edge_count, bool)
            and self.edge_count >= 0
        )
        values_are_valid = (
            math.isfinite(self.average_edge_weight)
            and 0.0 <= self.average_edge_weight <= 1.0
            and math.isfinite(self.average_reinforcement_count)
            and self.average_reinforcement_count >= 0.0
        )
        if not counts_are_valid or not values_are_valid:
            raise ValueError("graph metrics must contain non-negative counts and a unit edge weight")


class ConfidenceCache:
    """Cache confidence until a successful graph-affecting write invalidates it.

    This cache has no key: correctness depends on every graph write calling
    invalidate(). TODO(phase-5): co-retrieval reinforcement (inside recall),
    feedback edge updates, and connect() all write the graph and must
    invalidate; a missed call means maturity goes silently stale.

    Edge decay (PRD §6.4) also moves confidence, and time passing fires no
    invalidate(). A long-lived instance that never writes therefore reports the
    maturity it computed at its first recall. Accepted for v0: instances are
    short-lived, and every remember() drops the cached value anyway.
    """

    def __init__(self) -> None:
        self._value: float | None = None

    def get(self, metrics: GraphMetrics) -> float:
        if self._value is None:
            self._value = graph_confidence(metrics)
        return self._value

    def invalidate(self) -> None:
        self._value = None


def graph_confidence(metrics: GraphMetrics) -> float:
    """Return the weighted four-component graph confidence."""
    if metrics.node_count == 0 or metrics.edge_count == 0:
        return 0.0
    average_edges = metrics.edge_count / metrics.node_count
    confidence = (
        0.2 * _node_density(metrics.node_count)
        + 0.3 * _edge_density(average_edges)
        + 0.3 * _edge_quality(metrics.average_edge_weight)
        + 0.2 * _reinforcement(metrics.average_reinforcement_count)
    )
    return min(1.0, max(0.0, confidence))


def _node_density(count: int) -> float:
    if count < 50:
        return 0.0
    if count < 200:
        return ((count - 50) / 150.0) * 0.5
    if count < 500:
        return 0.5 + ((count - 200) / 300.0) * 0.3
    return 1.0


def _edge_density(average: float) -> float:
    if average < 1.0:
        return 0.0
    if average < 3.0:
        return ((average - 1.0) / 2.0) * 0.5
    if average < 8.0:
        return 0.5 + ((average - 3.0) / 5.0) * 0.4
    return 1.0


def _edge_quality(average: float) -> float:
    if average < 0.35:
        return 0.0
    if average < 0.6:
        return ((average - 0.35) / 0.25) * 0.6
    return 1.0


def _reinforcement(average: float) -> float:
    if average < 2.0:
        return 0.0
    if average < 10.0:
        return ((average - 2.0) / 8.0) * 0.7
    return 1.0
