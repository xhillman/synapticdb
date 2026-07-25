"""Pure retrieval ranking helpers."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from synapticdb.models import InvalidArgumentError, RecallSource

RRF_K = 60
ACTIVATION_BLEND_WEIGHT = 0.45
# TODO(phase-5): this rejects rankings longer than 100, so a bench sweep of
# candidate depth (PRD §9 parameter 2) past 100 would raise here.
_MAX_RANKING_LENGTH = 100
_MAX_SOURCE_COUNT = 4


@dataclass(frozen=True, slots=True)
class RankedHit:
    memory_id: UUID
    score: float


@dataclass(frozen=True, slots=True)
class BlendedHit:
    memory_id: UUID
    score: float
    via: RecallSource


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[UUID]],
    *,
    k: int = RRF_K,
) -> tuple[RankedHit, ...]:
    """Fuse bounded ranked ID lists with standard reciprocal rank fusion."""
    if not 1 <= len(rankings) <= _MAX_SOURCE_COUNT:
        raise InvalidArgumentError(f"RRF requires 1 to {_MAX_SOURCE_COUNT} rankings")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise InvalidArgumentError("RRF k must be positive")
    scores: dict[UUID, float] = {}
    first_seen: dict[UUID, int] = {}
    seen_count = 0
    for ranking in rankings:
        if len(ranking) > _MAX_RANKING_LENGTH:
            raise InvalidArgumentError(f"each RRF ranking is limited to {_MAX_RANKING_LENGTH} results")
        if any(not isinstance(memory_id, UUID) for memory_id in ranking):
            raise InvalidArgumentError("each RRF ranking must contain UUID values")
        if len(set(ranking)) != len(ranking):
            raise InvalidArgumentError("each RRF ranking must contain unique memory IDs")
        for rank, memory_id in enumerate(ranking, 1):
            if memory_id not in first_seen:
                first_seen[memory_id] = seen_count
                seen_count += 1
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=lambda memory_id: (-scores[memory_id], first_seen[memory_id]))
    return tuple(RankedHit(memory_id, scores[memory_id]) for memory_id in ordered)


def min_max_normalize(scores: Mapping[UUID, float]) -> dict[UUID, float]:
    """Normalize finite scores to [0, 1], treating an equal band as fully relevant."""
    if len(scores) > _MAX_SOURCE_COUNT * _MAX_RANKING_LENGTH:
        raise InvalidArgumentError("score collection exceeds the retrieval limit")
    if not scores:
        return {}
    if any(not isinstance(memory_id, UUID) for memory_id in scores):
        raise InvalidArgumentError("score keys must be UUID values")
    try:
        values = tuple(float(value) for value in scores.values())
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidArgumentError("scores must be numeric") from exc
    if any(not math.isfinite(value) for value in values):
        raise InvalidArgumentError("scores must be finite")
    low = min(values)
    high = max(values)
    if high == low:
        return {memory_id: 1.0 for memory_id in scores}
    width = high - low
    return {memory_id: (float(score) - low) / width for memory_id, score in scores.items()}


def blend_rankings(
    fusion_scores: Mapping[UUID, float],
    activation_scores: Mapping[UUID, float],
    confidence: float,
    blend_weight: float = ACTIVATION_BLEND_WEIGHT,
    seed_ids: Collection[UUID] = (),
) -> tuple[BlendedHit, ...]:
    """Blend normalized source scores and attribute each result.

    `blend_weight` is the alpha ceiling from PRD §5.3: the share of the final
    score activation can claim at full graph confidence. Actual alpha scales it
    by maturity, so a cold graph degrades to pure hybrid search.

    `seed_ids` are the fusion results activation started from. They are dropped
    before normalizing, because a seed is not a discovery: it holds full seed
    energy by construction, so leaving it in the distribution pins the scale at
    its value and compresses every genuine discovery toward zero. A seed keeps
    its fusion score and is attributed to search, which is where it came from.
    """
    maturity = _unit_float(confidence, "graph confidence")
    weight = _unit_float(blend_weight, "activation blend weight")
    normalized_fusion = min_max_normalize(fusion_scores)
    if maturity == 0.0 or weight == 0.0:
        return tuple(BlendedHit(memory_id, score, "search") for memory_id, score in normalized_fusion.items())
    seeds = frozenset(seed_ids)
    discovered = {
        memory_id: score for memory_id, score in activation_scores.items() if memory_id not in seeds
    }
    normalized_activation = min_max_normalize(discovered)
    alpha = weight * maturity
    ordered_ids = tuple(dict.fromkeys((*normalized_fusion, *normalized_activation)))
    blended = [
        BlendedHit(
            memory_id,
            (1.0 - alpha) * normalized_fusion.get(memory_id, 0.0)
            + alpha * normalized_activation.get(memory_id, 0.0),
            _recall_source(memory_id, normalized_fusion, normalized_activation),
        )
        for memory_id in ordered_ids
    ]
    source_order = {memory_id: index for index, memory_id in enumerate(ordered_ids)}
    blended.sort(key=lambda hit: (-hit.score, source_order[hit.memory_id]))
    return tuple(blended)


def _recall_source(
    memory_id: UUID,
    fusion_scores: Mapping[UUID, float],
    activation_scores: Mapping[UUID, float],
) -> RecallSource:
    in_fusion = memory_id in fusion_scores
    in_activation = memory_id in activation_scores
    if in_fusion and in_activation:
        return "both"
    if in_activation:
        return "association"
    return "search"


def _unit_float(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidArgumentError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise InvalidArgumentError(f"{label} must be between 0 and 1")
    return number
