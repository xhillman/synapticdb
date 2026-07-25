"""Pure spreading activation over a bounded neighbor lookup."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

from synapticdb.models import InvalidArgumentError, unit_float

ACTIVATION_DECAY = 0.2
ACTIVATION_HOP_BONUS = 0.15
ACTIVATION_MAX_STEPS = 5
ACTIVATION_MIN_ENERGY = 0.05
ACTIVATION_SEED_COUNT = 5
ACTIVATION_SEED_PENALTY = 0.2
_MAX_ACTIVATED_NODES = 400
_MAX_NEIGHBORS_PER_NODE = 400


@dataclass(frozen=True, slots=True)
class ActivationConfig:
    """The PRD §5.2 spreading parameters, as one explicit argument.

    Field defaults are the module constants above, which remain the single
    source of the locked holdout values. Passing a non-default instance is how
    the benchmark sweeps PRD §9 groups 4 through 9.
    """

    seeds: int = ACTIVATION_SEED_COUNT
    max_steps: int = ACTIVATION_MAX_STEPS
    decay: float = ACTIVATION_DECAY
    min_energy: float = ACTIVATION_MIN_ENERGY
    hop_bonus: float = ACTIVATION_HOP_BONUS
    seed_penalty: float = ACTIVATION_SEED_PENALTY


DEFAULT_ACTIVATION = ActivationConfig()


@dataclass(frozen=True, slots=True)
class Neighbor:
    memory_id: UUID
    edge_id: str
    weight: float


@dataclass(frozen=True, slots=True)
class ActivationHit:
    memory_id: UUID
    energy: float
    score: float
    hops: int


@dataclass(frozen=True, slots=True)
class ActivationResult:
    hits: tuple[ActivationHit, ...]
    path_edge_ids: tuple[str, ...]


NeighborLookup = Callable[[UUID], Sequence[Neighbor]]
Seed = tuple[UUID, float]


def spread_activation(
    seeds: Sequence[Seed],
    neighbor_lookup: NeighborLookup,
    config: ActivationConfig = DEFAULT_ACTIVATION,
) -> ActivationResult:
    """Spread seed energy through each reached node at most once."""
    prepared = _prepare_seeds(seeds, config)
    if not prepared:
        return ActivationResult((), ())
    energies = {memory_id: energy for memory_id, energy in prepared}
    hops = {memory_id: 0 for memory_id, _ in prepared}
    discovery_order = {memory_id: index for index, (memory_id, _) in enumerate(prepared)}
    seed_ids = frozenset(energies)
    frontier = [memory_id for memory_id, energy in prepared if energy >= config.min_energy]
    propagated: set[UUID] = set()
    path_ids: list[str] = []
    path_set: set[str] = set()
    # NOTE: propagation cascades within a step — a frontier node processed
    # later in a step sees energy improvements made earlier in the same step
    # (Gauss-Seidel), whereas the PRD §5.2 pseudocode reads as a per-step
    # snapshot. Deterministic, at most mildly energy-inflating; first suspect
    # if bench calibration cannot reproduce the locked holdout config.
    for _ in range(config.max_steps):
        if not frontier:
            break
        frontier = _propagate_frontier(
            frontier,
            neighbor_lookup,
            energies,
            hops,
            discovery_order,
            propagated,
            path_ids,
            path_set,
            config,
        )
    hits = _score_hits(energies, hops, discovery_order, seed_ids, config)
    return ActivationResult(hits, tuple(path_ids))


def _propagate_frontier(
    frontier: Sequence[UUID],
    neighbor_lookup: NeighborLookup,
    energies: dict[UUID, float],
    hops: dict[UUID, int],
    discovery_order: dict[UUID, int],
    propagated: set[UUID],
    path_ids: list[str],
    path_set: set[str],
    config: ActivationConfig,
) -> list[UUID]:
    next_ids: set[UUID] = set()
    for source_id in frontier:
        if source_id in propagated:
            continue
        propagated.add(source_id)
        neighbors = _prepare_neighbors(neighbor_lookup(source_id))
        reached = _propagate_source(
            source_id,
            neighbors,
            energies,
            hops,
            discovery_order,
            propagated,
            path_ids,
            path_set,
            config,
        )
        next_ids.update(reached)
    return sorted(next_ids, key=lambda memory_id: (-energies[memory_id], discovery_order[memory_id]))


def _propagate_source(
    source_id: UUID,
    neighbors: Sequence[Neighbor],
    energies: dict[UUID, float],
    hops: dict[UUID, int],
    discovery_order: dict[UUID, int],
    propagated: set[UUID],
    path_ids: list[str],
    path_set: set[str],
    config: ActivationConfig,
) -> set[UUID]:
    reached: set[UUID] = set()
    for neighbor in neighbors:
        energy = energies[source_id] * neighbor.weight * (1.0 - config.decay)
        if energy < config.min_energy or energy <= energies.get(neighbor.memory_id, -1.0):
            continue
        if neighbor.memory_id not in discovery_order and len(discovery_order) >= _MAX_ACTIVATED_NODES:
            # Node budget reached: keep improving known nodes, discover no more.
            continue
        _record_improvement(
            source_id,
            neighbor,
            energy,
            energies,
            hops,
            discovery_order,
            path_ids,
            path_set,
        )
        if neighbor.memory_id not in propagated:
            reached.add(neighbor.memory_id)
    return reached


def _record_improvement(
    source_id: UUID,
    neighbor: Neighbor,
    energy: float,
    energies: dict[UUID, float],
    hops: dict[UUID, int],
    discovery_order: dict[UUID, int],
    path_ids: list[str],
    path_set: set[str],
) -> None:
    if neighbor.memory_id not in discovery_order:
        discovery_order[neighbor.memory_id] = len(discovery_order)
    energies[neighbor.memory_id] = energy
    hops[neighbor.memory_id] = hops[source_id] + 1
    if neighbor.edge_id not in path_set:
        path_ids.append(neighbor.edge_id)
        path_set.add(neighbor.edge_id)


def _score_hits(
    energies: dict[UUID, float],
    hops: dict[UUID, int],
    discovery_order: dict[UUID, int],
    seed_ids: frozenset[UUID],
    config: ActivationConfig,
) -> tuple[ActivationHit, ...]:
    hits: list[ActivationHit] = []
    for memory_id, energy in energies.items():
        if energy < config.min_energy:
            continue
        hop_count = hops[memory_id]
        multiplier = 1.0 - config.seed_penalty
        if memory_id not in seed_ids:
            multiplier = 1.0 + config.hop_bonus * hop_count
        hits.append(ActivationHit(memory_id, energy, energy * multiplier, hop_count))
    hits.sort(key=lambda hit: (-hit.score, discovery_order[hit.memory_id]))
    return tuple(hits)


def _prepare_seeds(seeds: Sequence[Seed], config: ActivationConfig) -> tuple[Seed, ...]:
    prepared = tuple(seeds)
    if len(prepared) > config.seeds:
        raise InvalidArgumentError(f"activation accepts at most {config.seeds} seeds")
    identifiers: list[UUID] = []
    for memory_id, energy in prepared:
        if not isinstance(memory_id, UUID):
            raise InvalidArgumentError("activation seed IDs must be UUID values")
        unit_float(energy, "activation seed energy")
        identifiers.append(memory_id)
    if len(set(identifiers)) != len(identifiers):
        raise InvalidArgumentError("activation seed IDs must be unique")
    return tuple((memory_id, float(energy)) for memory_id, energy in prepared)


def _prepare_neighbors(neighbors: Sequence[Neighbor]) -> tuple[Neighbor, ...]:
    # Degrade at scale: consider only the first neighbors the lookup supplies
    # (the store already returns strongest-first) instead of failing recall.
    prepared = tuple(neighbors)[:_MAX_NEIGHBORS_PER_NODE]
    for neighbor in prepared:
        if not isinstance(neighbor, Neighbor):
            raise InvalidArgumentError("neighbor lookup must return Neighbor values")
        if not isinstance(neighbor.memory_id, UUID) or not neighbor.edge_id:
            raise InvalidArgumentError("activation neighbors require a UUID and edge ID")
        unit_float(neighbor.weight, "activation edge weight")
    return prepared
