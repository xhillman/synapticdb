from collections.abc import Sequence
from typing import cast
from uuid import UUID, uuid4

import pytest

from synapticdb import InvalidArgumentError
from synapticdb.retrieval import min_max_normalize, reciprocal_rank_fusion


def test_rrf_rewards_overlap_without_an_extra_bonus() -> None:
    first, overlap, last = uuid4(), uuid4(), uuid4()
    fused = reciprocal_rank_fusion(((first, overlap), (overlap, last)))
    assert [hit.memory_id for hit in fused] == [overlap, first, last]
    expected = 1 / 62 + 1 / 61
    assert fused[0].score == pytest.approx(expected)


def test_rrf_preserves_first_seen_order_for_equal_scores() -> None:
    first, second = uuid4(), uuid4()
    fused = reciprocal_rank_fusion(((first,), (second,)))
    assert [hit.memory_id for hit in fused] == [first, second]


def test_rrf_rejects_duplicate_ids_within_one_source() -> None:
    memory_id = uuid4()
    with pytest.raises(InvalidArgumentError, match="unique"):
        reciprocal_rank_fusion(((memory_id, memory_id),))


def test_rrf_rejects_non_uuid_ids_and_boolean_k() -> None:
    invalid_ranking = cast(Sequence[UUID], ("memory",))
    with pytest.raises(InvalidArgumentError, match="UUID"):
        reciprocal_rank_fusion((invalid_ranking,))
    with pytest.raises(InvalidArgumentError, match="positive"):
        reciprocal_rank_fusion(((uuid4(),),), k=True)


def test_min_max_normalize_handles_empty_equal_and_varied_scores() -> None:
    low, middle, high = uuid4(), uuid4(), uuid4()
    assert min_max_normalize({}) == {}
    assert min_max_normalize({low: 2.0}) == {low: 1.0}
    assert min_max_normalize({low: 2.0, high: 2.0}) == {low: 1.0, high: 1.0}
    normalized = min_max_normalize({low: 2.0, middle: 3.0, high: 4.0})
    assert normalized == {low: 0.0, middle: 0.5, high: 1.0}
