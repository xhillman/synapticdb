"""Pure retrieval ranking helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from synapticdb.models import InvalidArgumentError

RRF_K = 60
# TODO(phase-5): this rejects rankings longer than 100, so a bench sweep of
# candidate depth (PRD §9 parameter 2) past 100 would raise here.
_MAX_RANKING_LENGTH = 100
_MAX_SOURCE_COUNT = 4


@dataclass(frozen=True, slots=True)
class RankedHit:
    memory_id: UUID
    score: float


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
