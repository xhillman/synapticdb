from collections.abc import Sequence
from itertools import pairwise
from uuid import UUID, uuid4

import pytest

from synapticdb import InvalidArgumentError
from synapticdb.activation import Neighbor, spread_activation


class Graph:
    def __init__(self, neighbors: dict[UUID, Sequence[Neighbor]]) -> None:
        self.neighbors = neighbors
        self.lookups: dict[UUID, int] = {}

    def lookup(self, memory_id: UUID) -> Sequence[Neighbor]:
        self.lookups[memory_id] = self.lookups.get(memory_id, 0) + 1
        return self.neighbors.get(memory_id, ())


def test_activation_propagates_energy_and_applies_scoring_rules() -> None:
    seed, middle, end = uuid4(), uuid4(), uuid4()
    graph = Graph(
        {
            seed: (Neighbor(middle, "seed-middle", 1.0),),
            middle: (Neighbor(end, "middle-end", 1.0),),
        }
    )

    result = spread_activation(((seed, 1.0),), graph.lookup)
    hits = {hit.memory_id: hit for hit in result.hits}

    assert hits[seed].energy == pytest.approx(1.0)
    assert hits[seed].score == pytest.approx(0.8)
    assert hits[middle].energy == pytest.approx(0.8)
    assert hits[middle].score == pytest.approx(0.8 * 1.15)
    assert hits[end].energy == pytest.approx(0.64)
    assert hits[end].score == pytest.approx(0.64 * 1.30)
    assert hits[end].hops == 2
    assert result.path_edge_ids == ("seed-middle", "middle-end")


def test_activation_prunes_below_minimum_energy() -> None:
    seed, target = uuid4(), uuid4()
    graph = Graph({seed: (Neighbor(target, "weak", 0.05),)})

    result = spread_activation(((seed, 1.0),), graph.lookup)

    assert [hit.memory_id for hit in result.hits] == [seed]
    assert result.path_edge_ids == ()
    assert target not in graph.lookups


def test_activation_loop_guard_propagates_each_node_once() -> None:
    first, second = uuid4(), uuid4()
    graph = Graph(
        {
            first: (Neighbor(second, "edge", 1.0),),
            second: (Neighbor(first, "edge", 1.0),),
        }
    )

    result = spread_activation(((first, 1.0),), graph.lookup)

    assert graph.lookups == {first: 1, second: 1}
    assert {hit.memory_id: hit.energy for hit in result.hits} == pytest.approx(
        {first: 1.0, second: 0.8}
    )
    assert result.path_edge_ids == ("edge",)


def test_activation_stops_after_five_steps() -> None:
    nodes = tuple(uuid4() for _ in range(7))
    graph = Graph(
        {
            source: (Neighbor(target, f"edge-{index}", 1.0),)
            for index, (source, target) in enumerate(pairwise(nodes))
        }
    )

    result = spread_activation(((nodes[0], 1.0),), graph.lookup)
    activated = {hit.memory_id for hit in result.hits}

    assert nodes[5] in activated
    assert nodes[6] not in activated
    assert len(result.path_edge_ids) == 5


def test_activation_records_each_energy_improving_edge() -> None:
    first, second, target = uuid4(), uuid4(), uuid4()
    graph = Graph(
        {
            first: (Neighbor(target, "lower", 0.5),),
            second: (Neighbor(target, "higher", 0.8),),
        }
    )

    result = spread_activation(((first, 1.0), (second, 0.9)), graph.lookup)
    hits = {hit.memory_id: hit for hit in result.hits}

    assert hits[target].energy == pytest.approx(0.576)
    assert result.path_edge_ids == ("lower", "higher")


def test_activation_rejects_more_than_five_seeds() -> None:
    seeds = tuple((uuid4(), 1.0) for _ in range(6))
    with pytest.raises(InvalidArgumentError, match="at most 5 seeds"):
        spread_activation(seeds, lambda _memory_id: ())


def test_activation_degrades_on_a_dense_hub_instead_of_raising() -> None:
    seed = uuid4()
    neighbors = tuple(Neighbor(uuid4(), f"edge-{index}", 1.0) for index in range(500))

    result = spread_activation(((seed, 1.0),), lambda _memory_id: neighbors)

    # Truncated to the first 400 neighbors, then capped at 400 total
    # activated nodes (seed included) — never an exception mid-recall.
    assert len(result.hits) == 400
    assert {hit.memory_id for hit in result.hits} <= {seed, *(n.memory_id for n in neighbors[:400])}
