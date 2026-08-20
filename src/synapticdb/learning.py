"""Pure parameter and weight calculations for graph learning."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import cast
from uuid import UUID

from synapticdb.activation import ActivationConfig
from synapticdb.models import InvalidArgumentError, unit_float

# PRD section 6.4 half-life, in days. Store methods default to this value so a
# caller that does not sweep parameters still applies the specified decay.
DEFAULT_HALF_LIFE_DAYS = 30.0
_MAX_HALF_LIFE_DAYS = 3650
_SECONDS_PER_DAY = 86_400.0

# How many of a recall's top results are close enough to link. Both learning
# mechanisms that create edges from results use it: co-retrieval (PRD §6.3) and
# explicit feedback (§6.6, as amended — see api.SynapticDB._feedback_pairs). One
# constant because it is one idea, and letting the two drift apart is what
# produced the write amplification the amendment fixes.
#
# Separate from ACTIVATION_SEED_COUNT, which happens to share the value 5: this
# bounds what a recall learns, that bounds where activation starts.
LINKED_RESULT_COUNT = 5

# PRD §6.6: positive feedback creates a missing edge at 0.05·e_i·e_j, never
# below 0.02 — the same value §6.5 prunes at, so a created edge starts at or
# above the survival line rather than being born already collectable.
_CO_RETRIEVAL_SEED_WEIGHT = 0.05
_FEEDBACK_EDGE_FLOOR = 0.02

# Calibration values for semantic seeding (PRD §6.1 / §9 group 11). Semantic
# seeding is DISABLED by default (see DEFAULT_RUNTIME_POLICY): the full-corpus
# benchmark showed it contributes +0 associative unique wins at every threshold
# from 0.6 to 0.85 — embedding similarity is orthogonal to the benchmark's
# associative chains, so its edges only duplicate what vector search already
# ranks together. This tuple re-enables it for calibration / future A/Bs by
# building an internal policy override. See bench/README.md for the evidence.
SEMANTIC_SEED_CALIBRATION: tuple[float, int, float] = (0.6, 3, 0.25)


@dataclass(frozen=True, slots=True)
class SemanticSeedConfig:
    threshold: float
    max_links: int
    initial_weight: float


@dataclass(frozen=True, slots=True)
class TemporalLinkConfig:
    window_seconds: int
    max_links: int
    initial_weight: float


@dataclass(frozen=True, slots=True)
class FusionConfig:
    candidate_depth: int
    rrf_k: int


@dataclass(frozen=True, slots=True)
class CoRetrievalConfig:
    initial_weight: float
    reinforcement_rate: float


@dataclass(frozen=True, slots=True)
class DecayConfig:
    half_life_days: int
    prune_threshold: float


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    """The complete validated policy used by one SynapticDB runtime."""

    top_k: int
    fusion: FusionConfig
    activation: ActivationConfig
    activation_blend_weight: float
    semantic_seed: SemanticSeedConfig | None
    temporal_link: TemporalLinkConfig
    co_retrieval: CoRetrievalConfig
    feedback_rate: float
    connect_weight: float
    decay: DecayConfig
    maintenance_interval: int


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


def unordered_pairs(memory_ids: Sequence[UUID]) -> tuple[tuple[UUID, UUID], ...]:
    """Return every unordered pair among unique memory IDs, in a stable order."""
    selected = tuple(memory_ids)
    if not all(isinstance(memory_id, UUID) for memory_id in selected):
        raise InvalidArgumentError("memory IDs must be UUID values")
    if len(set(selected)) != len(selected):
        raise InvalidArgumentError("memory IDs must be unique")
    pairs: list[tuple[UUID, UUID]] = []
    for first_index, first_id in enumerate(selected):
        for second_id in selected[first_index + 1 :]:
            pairs.append((first_id, second_id))
    return tuple(pairs)


def co_retrieval_pairs(result_ids: Sequence[UUID]) -> tuple[tuple[UUID, UUID], ...]:
    """Return every unordered pair among the top results of one recall.

    Bounded by construction: slicing before pairing means at most
    LINKED_RESULT_COUNT results yield at most 10 pairs, whatever the
    caller passes.
    """
    return unordered_pairs(tuple(result_ids)[:LINKED_RESULT_COUNT])


def reinforce_weight(weight: float, rate: float) -> float:
    """Return one passive reinforcement update."""
    stored_weight = unit_float(weight, "edge weight")
    reinforcement_rate = unit_float(rate, "reinforcement rate")
    return stored_weight + reinforcement_rate * (1.0 - stored_weight)


def positive_feedback_seed(
    first_energy: float,
    second_energy: float,
    rate: float,
) -> tuple[float, float]:
    """Return the (initial weight, reinforcement rate) for one positive update.

    PRD §6.6 creates a missing edge at 0.05·e_i·e_j with a 0.02 floor, and
    reinforces an existing one by rate·e_i·e_j·(1-w). Passing the second value
    as a PairSeed reinforce_rate produces that update exactly.
    """
    scale = _energy_product(first_energy, second_energy)
    reinforcement_rate = unit_float(rate, "feedback rate") * scale
    initial_weight = max(_FEEDBACK_EDGE_FLOOR, _CO_RETRIEVAL_SEED_WEIGHT * scale)
    return initial_weight, reinforcement_rate


def negative_feedback_weight(
    weight: float,
    first_energy: float,
    second_energy: float,
    rate: float,
) -> float:
    """Return the weakened weight for one negative update.

    Callers pass the *effective* weight: like every other write path, negative
    feedback builds on the decayed value rather than the stored one. The result
    cannot go negative, since rate·e_i·e_j never exceeds the rate itself.
    """
    current = unit_float(weight, "edge weight")
    scale = _energy_product(first_energy, second_energy)
    reduction = unit_float(rate, "feedback rate") * scale
    return current * (1.0 - reduction)


def _energy_product(first_energy: float, second_energy: float) -> float:
    first = unit_float(first_energy, "energy")
    second = unit_float(second_energy, "energy")
    return first * second


def decayed_weight(weight: float, days_elapsed: float, half_life_days: float) -> float:
    """Return the PRD section 6.4 effective weight for one elapsed span.

    Elapsed days below zero clamp to zero: decay only ever reduces a weight.
    A row written with a future timestamp is therefore treated as brand new
    rather than amplified, which keeps the result inside [0, 1].
    """
    stored_weight = unit_float(weight, "edge weight")
    half_life = _positive_days(half_life_days, "decay half-life days")
    elapsed = _finite_float(days_elapsed, "elapsed days")
    if elapsed <= 0.0:
        return stored_weight
    half_lives_elapsed = elapsed / half_life
    decay_multiplier = math.pow(2.0, -half_lives_elapsed)
    return stored_weight * decay_multiplier


def effective_weight(
    weight: float,
    last_reinforced_at: datetime,
    now: datetime,
    half_life_days: float,
) -> float:
    """Return the decayed weight of an edge last reinforced at a known time."""
    reinforced = _utc_datetime(last_reinforced_at, "last_reinforced_at")
    read_time = _utc_datetime(now, "now")
    days_elapsed = (read_time - reinforced).total_seconds() / _SECONDS_PER_DAY
    return decayed_weight(weight, days_elapsed, half_life_days)


def _single_parameter(value: object, key: str) -> float | int:
    """Validate one raw scalar parameter at the benchmark boundary."""
    if value is None or isinstance(value, tuple):
        raise InvalidArgumentError(f"{key} must be a single number")
    return cast(float | int, value)


def _group_parameter(
    value: object,
    key: str,
    size: int,
    contents: str,
) -> tuple[float | int, ...]:
    """Validate one raw grouped parameter at the benchmark boundary."""
    if not isinstance(value, tuple) or len(value) != size:
        raise InvalidArgumentError(f"{key} must contain {contents}")
    return cast(tuple[float | int, ...], value)


def _validated_unit(value: float | int, label: str) -> float:
    unit_float(value, label)
    return value


def _unit_parameter(parameters: Mapping[str, object], key: str, label: str) -> float:
    value = _single_parameter(parameters[key], key)
    return _validated_unit(value, label)


def _fusion_parameter(parameters: Mapping[str, object]) -> FusionConfig:
    # retrieval._MAX_RANKING_LENGTH rejects a ranking longer than 100, so the
    # policy fails here instead of deep inside reciprocal_rank_fusion.
    depth = _bounded_int(_single_parameter(parameters["candidate_depth"], "candidate_depth"), "candidate depth", 100)
    rrf_k = _bounded_int(_single_parameter(parameters["rrf_k"], "rrf_k"), "rrf k", 10_000)
    return FusionConfig(depth, rrf_k)


def _activation_parameter(parameters: Mapping[str, object]) -> ActivationConfig:
    return ActivationConfig(
        seeds=_bounded_int(
            _single_parameter(parameters["activation_seeds"], "activation_seeds"), "activation seeds", 100
        ),
        max_steps=_bounded_int(
            _single_parameter(parameters["activation_max_steps"], "activation_max_steps"),
            "activation max steps",
            20,
        ),
        decay=_unit_parameter(parameters, "activation_decay", "activation decay"),
        min_energy=_unit_parameter(parameters, "activation_min_energy", "activation min energy"),
        hop_bonus=_unit_parameter(parameters, "hop_bonus", "hop bonus"),
        seed_penalty=_unit_parameter(parameters, "seed_penalty", "seed penalty"),
    )


def _semantic_seed_parameter(value: object) -> SemanticSeedConfig | None:
    if value is None:
        return None
    group = _group_parameter(value, "semantic_seed", 3, "threshold, max links, and weight")
    return SemanticSeedConfig(
        _validated_unit(group[0], "semantic seed threshold"),
        _bounded_int(group[1], "semantic seed max links", 40),
        _validated_unit(group[2], "semantic seed weight"),
    )


def _temporal_link_parameter(value: object) -> TemporalLinkConfig:
    group = _group_parameter(value, "temporal_link", 3, "window, max links, and weight")
    return TemporalLinkConfig(
        _bounded_int(group[0], "temporal window", 86_400),
        _bounded_int(group[1], "temporal max links", 40),
        _validated_unit(group[2], "temporal link weight"),
    )


def _co_retrieval_parameter(value: object) -> CoRetrievalConfig:
    group = _group_parameter(value, "co_retrieval", 2, "initial weight and reinforcement rate")
    return CoRetrievalConfig(
        _validated_unit(group[0], "co-retrieval initial weight"),
        _validated_unit(group[1], "passive reinforcement rate"),
    )


def _decay_parameter(value: object) -> DecayConfig:
    group = _group_parameter(value, "decay_and_prune", 2, "half-life days and prune threshold")
    return DecayConfig(
        _bounded_int(group[0], "decay half-life days", _MAX_HALF_LIFE_DAYS),
        _validated_unit(group[1], "prune threshold"),
    )


def policy_parameters(policy: RuntimePolicy) -> dict[str, object]:
    """Return the stable 17-key benchmark representation of a policy."""
    return {
        "top_k": policy.top_k,
        "candidate_depth": policy.fusion.candidate_depth,
        "rrf_k": policy.fusion.rrf_k,
        "activation_seeds": policy.activation.seeds,
        "activation_max_steps": policy.activation.max_steps,
        "activation_decay": policy.activation.decay,
        "activation_min_energy": policy.activation.min_energy,
        "hop_bonus": policy.activation.hop_bonus,
        "seed_penalty": policy.activation.seed_penalty,
        "activation_blend_weight": policy.activation_blend_weight,
        "semantic_seed": None
        if policy.semantic_seed is None
        else (policy.semantic_seed.threshold, policy.semantic_seed.max_links, policy.semantic_seed.initial_weight),
        "temporal_link": (
            policy.temporal_link.window_seconds,
            policy.temporal_link.max_links,
            policy.temporal_link.initial_weight,
        ),
        "co_retrieval": (policy.co_retrieval.initial_weight, policy.co_retrieval.reinforcement_rate),
        "feedback_rate": policy.feedback_rate,
        "connect_weight": policy.connect_weight,
        "decay_and_prune": (policy.decay.half_life_days, policy.decay.prune_threshold),
        "maintenance_interval": policy.maintenance_interval,
    }


DEFAULT_RUNTIME_POLICY = RuntimePolicy(
    top_k=10,
    fusion=FusionConfig(40, 60),
    activation=ActivationConfig(),
    activation_blend_weight=0.45,
    semantic_seed=None,
    temporal_link=TemporalLinkConfig(600, 3, 0.2),
    co_retrieval=CoRetrievalConfig(0.05, 0.05),
    feedback_rate=0.15,
    connect_weight=0.5,
    decay=DecayConfig(30, 0.02),
    maintenance_interval=100,
)


def default_parameters() -> dict[str, object]:
    """Return the 17-key benchmark representation of the default policy."""
    return policy_parameters(DEFAULT_RUNTIME_POLICY)


def runtime_policy(overrides: Mapping[str, object] | None = None) -> RuntimePolicy:
    """Build one complete policy from bounded benchmark overrides."""
    parameters = default_parameters()
    for key, value in (overrides or {}).items():
        if key not in parameters:
            raise ValueError(f"unknown parameter: {key}")
        parameters[key] = value
    return RuntimePolicy(
        top_k=_bounded_int(_single_parameter(parameters["top_k"], "top_k"), "top_k", 100),
        fusion=_fusion_parameter(parameters),
        activation=_activation_parameter(parameters),
        activation_blend_weight=_unit_parameter(parameters, "activation_blend_weight", "activation blend weight"),
        semantic_seed=_semantic_seed_parameter(parameters["semantic_seed"]),
        temporal_link=_temporal_link_parameter(parameters["temporal_link"]),
        co_retrieval=_co_retrieval_parameter(parameters["co_retrieval"]),
        feedback_rate=_unit_parameter(parameters, "feedback_rate", "feedback rate"),
        connect_weight=_unit_parameter(parameters, "connect_weight", "connect weight"),
        decay=_decay_parameter(parameters["decay_and_prune"]),
        maintenance_interval=_bounded_int(
            _single_parameter(parameters["maintenance_interval"], "maintenance_interval"),
            "maintenance interval",
            100_000,
        ),
    )


def _bounded_int(value: float | int, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise InvalidArgumentError(f"{label} must be between 1 and {maximum}")
    return value


def _positive_days(value: float, label: str) -> float:
    number = _finite_float(value, label)
    if not 0.0 < number <= float(_MAX_HALF_LIFE_DAYS):
        raise InvalidArgumentError(f"{label} must be between 0 and {_MAX_HALF_LIFE_DAYS}")
    return number


def _finite_float(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise InvalidArgumentError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidArgumentError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise InvalidArgumentError(f"{label} must be finite")
    return number


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime):
        raise InvalidArgumentError(f"{label} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise InvalidArgumentError(f"{label} must use UTC")
    return value.astimezone(timezone.utc)


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
