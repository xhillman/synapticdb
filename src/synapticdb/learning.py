"""Pure parameter and weight calculations for graph learning."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from synapticdb.models import InvalidArgumentError

ParameterValue = float | int | tuple[float | int, ...]


@dataclass(frozen=True, slots=True)
class SemanticSeedConfig:
    threshold: float
    max_links: int
    initial_weight: float


def default_parameters() -> dict[str, ParameterValue]:
    """Return the 17 private parameter groups from PRD section 9."""
    return {
        "top_k": 10,
        "candidate_depth": 40,
        "rrf_k": 60,
        "activation_seeds": 5,
        "activation_max_steps": 5,
        "activation_decay": 0.2,
        "activation_min_energy": 0.05,
        "hop_bonus": 0.15,
        "seed_penalty": 0.2,
        "activation_blend_weight": 0.45,
        "semantic_seed": (0.6, 3, 0.25),
        "temporal_link": (600, 3, 0.2),
        "co_retrieval": (0.05, 0.05),
        "feedback_rate": 0.15,
        "connect_weight": 0.5,
        "decay_and_prune": (30, 0.02),
        "maintenance_interval": 100,
    }


def semantic_seed_config(params: Mapping[str, ParameterValue]) -> SemanticSeedConfig:
    """Read and validate the semantic seed parameter group."""
    value = params.get("semantic_seed")
    if not isinstance(value, tuple) or len(value) != 3:
        raise InvalidArgumentError("semantic_seed must contain threshold, max links, and weight")
    threshold = _unit_float(value[0], "semantic seed threshold")
    max_links = value[1]
    weight = _unit_float(value[2], "semantic seed weight")
    if isinstance(max_links, bool) or not isinstance(max_links, int) or not 1 <= max_links <= 40:
        raise InvalidArgumentError("semantic seed max links must be between 1 and 40")
    return SemanticSeedConfig(threshold, max_links, weight)


def semantic_seed_ids(
    candidates: Sequence[tuple[UUID, float]],
    config: SemanticSeedConfig,
) -> tuple[UUID, ...]:
    """Select bounded, unique candidates that meet the similarity threshold."""
    if len(candidates) > config.max_links:
        raise InvalidArgumentError("semantic candidates exceed the configured maximum")
    selected: list[UUID] = []
    seen: set[UUID] = set()
    for memory_id, similarity in candidates:
        if not isinstance(memory_id, UUID):
            raise InvalidArgumentError("semantic candidate IDs must be UUID values")
        score = _similarity_float(similarity)
        if score < config.threshold or memory_id in seen:
            continue
        selected.append(memory_id)
        seen.add(memory_id)
    return tuple(selected)


def reinforce_weight(weight: float, rate: float) -> float:
    """Return one passive reinforcement update."""
    stored_weight = _unit_float(weight, "edge weight")
    reinforcement_rate = _unit_float(rate, "reinforcement rate")
    return stored_weight + reinforcement_rate * (1.0 - stored_weight)


def _unit_float(value: float | int, label: str) -> float:
    if isinstance(value, bool):
        raise InvalidArgumentError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InvalidArgumentError(f"{label} must be numeric") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise InvalidArgumentError(f"{label} must be between 0 and 1")
    return number


def _similarity_float(value: float) -> float:
    if isinstance(value, bool):
        raise InvalidArgumentError("semantic similarity must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InvalidArgumentError("semantic similarity must be numeric") from error
    if not math.isfinite(number) or not -1.000001 <= number <= 1.000001:
        raise InvalidArgumentError("semantic similarity must be between -1 and 1")
    return min(1.0, max(-1.0, number))
